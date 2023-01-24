# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   freq_train_pipeline.py
    Time:        2022/10/25 10:49:14
    Editor:      Figo
-----------------------------------
'''

import cox
import os
import json
import torch
import argparse
import torchvision as tv
import torch.nn.functional as F

from tqdm import tqdm
from model.freqvae import freqvae_cifar
from default import RESUME_DEFAULT
from utils.helps_visual import recon_images
from utils.helps_train import parse_train_tools
from utils.helps_data import load_freq_dataset, load_dataset
from utils.focal_frequency_loss import FocalFrequencyLoss as FFL
from utils.helps import amp_pha_recon, check_kwargs, set_random_seed, writr_down_the_args_each_exp
from torch.utils.data import DataLoader
from robustness.model_utils import make_and_restore_model
from robustness.tools.helpers import AverageMeter, accuracy

Image2Tensor = tv.transforms.PILToTensor()


def train_loop(args, epoch, model, optim, loader):
    
    model.train()
    loop_type = "train"
    Losses, Loss_kl, Loss_re = AverageMeter(), AverageMeter(), AverageMeter()
    iterator = tqdm(enumerate(loader), total=len(loader))
    print("="*20, "Start Epoch {} Training!".format(epoch), "="*20)

    for _, (image, label) in iterator:
        batch_size = label.shape[0]
        ori_imiage, amplitude, phase = image[0].cuda(), image[1].cuda(), image[2].cuda()
        
        if args.freqvae == "phase":
            recon_phase, mu, logvar, _ = model(phase)
            recon_image = amp_pha_recon(amplitude, recon_phase)
            original, reconstruction = phase, recon_phase
        
        elif args.freqvae == "amplitude":
            recon_amplitude, mu, logvar, _ = model(amplitude)
            recon_image = amp_pha_recon(recon_amplitude, phase)
            original, reconstruction = amplitude, recon_amplitude
        
        def kl_loss(mu, logvar):
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_kl /= batch_size * 3 * args.dim
            return loss_kl
        
        loss_kl = kl_loss(mu, logvar)                           # VAE: align with the normal Gauss distribution
        
        """
            #! Reconstruction error calculation:
            :: Strategy one: the reconstruction-phase and original-amplitude composite a new image, which align with original image in pixel-level
            >>> recon_image = amp_pha_recon(recon_amplitude, phase)
            >>> loss_re = F.mse_loss(recon_image, image)

            :: Strategy two: the reconstruction-phase align with original phase through mse loss
            >>> loss_re = F.mse_loss(recon_phase, phase)

            :: Strategy three: the reconstruction-phase and original-amplitude composite a new image in frequency-level, which align with the frequency 
            :: of original image through Frequency-Focal-loss @ https://github.com/EndlessSora/focal-frequency-loss
            :: Ref: Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021. <https://arxiv.org/pdf/2012.12821.pdf>

            >>> recon_image = amp_pha_recon(recon_amplitude, phase)
            >>> ffl = FFL(loss_weight=1.0, alpha=1.0)       # initialize nn.Module class
            >>> loss_re = ffl(recon_image, image)           # calculate focal frequency loss
        """
        
        if args.strategy == 1:
            loss_re = F.mse_loss(recon_image, ori_imiage)
        elif args.strategy == 2:
            loss_re = F.mse_loss(reconstruction, original)
        elif args.strategy == 3:
            ffl = FFL(loss_weight=1.0, alpha=1.0)               # initialize nn.Module class
            loss_re = ffl(recon_image, ori_imiage)              # calculate focal frequency loss
        
        re = args.re
        kl = args.kl * 0
        loss = re * loss_re + kl * loss_kl

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        Losses.update(loss.data.item(), batch_size)
        Loss_kl.update(loss_kl.data.item(), batch_size) 
        Loss_re.update(loss_re.data.item(), batch_size)

        """ ---------- iterator desc ---------- """
        desc = (
            '{} Epoch:{} | Loss {loss.avg:.4f} | '
            'Loss_re {loss_re.avg:.4f} | Loss_kl {loss_kl.avg:.3f} |'
            .format(loop_type, epoch, loss=Losses, loss_re=Loss_re, loss_kl=Loss_kl)
        )
        iterator.set_description(desc)
        iterator.refresh()


def test_loop(args, epoch, model, FVAE, loader):
    
    model.eval()
    FVAE.eval()
    loop_type = "test"
    iterator = tqdm(enumerate(loader), total=len(loader))
    top1, top5 = AverageMeter(), AverageMeter()
    
    for _, (image, label) in iterator:
        batch_size = label.shape[0]
        amplitude, phase, targets = image[1].cuda(), image[2].cuda(), label.cuda()
        
        if args.freqvae == "phase":
            recon_phase, _, _, _ = FVAE(phase)
            recon_image = amp_pha_recon(amplitude, recon_phase)
        elif args.freqvae == "amplitude":
            recon_amplitude, _, _, _ = FVAE(amplitude)
            recon_image = amp_pha_recon(recon_amplitude, phase)
        
        outputs = model(recon_image, with_latent=False, with_image=False)
        
        maxk = min(5, outputs.shape[-1])
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, maxk))
        prec1, prec5 = prec1[0], prec5[0]
        
        """ Update the AverageMete """
        top1.update(prec1, batch_size)
        top5.update(prec5, batch_size)

        """ ---------- iterator desc ---------- """
        top1_acc, top5_acc = top1.avg, top5.avg
        desc = (
            '{} Epoch:{} | NatPrec1 {top1_acc:.3f} | NatPrec5 {top5_acc:.3f} |'
            .format(loop_type, epoch, top1_acc=top1_acc, top5_acc=top5_acc)
        )
        iterator.set_description(desc)
        iterator.refresh()

    return top1_acc


def train(args, model_bag, loader, optimizer, scheduler, writer):
    """
        :: model_bag: contains well-trained classification model, branch-freqvae model
        :: loader: train_loader, test_loader, all types with `DataLoader.FreqDataset`
    """
    
    best_acc, start_epoch = 0, 0
    trainloader, testloader = loader
    model, Fvae = model_bag

    def save_checkpoints(path):
        checkpoints = {  # checkpoints
            'model': Fvae.state_dict(), 
            "freqvae": args.freqvae, 
            'epoch': epoch,
            'best_acc': best_acc
        }
        torch.save(checkpoints, path)
        print("==> Finish saving the checkpoints in epoch {}!".format(epoch))

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loop(args, epoch, Fvae, optimizer, trainloader)  # train stage
        if scheduler is not None: scheduler.step()
        with torch.no_grad():
            test_acc = test_loop(args, epoch, model, Fvae, testloader)  # test stage
            if epoch % 5 == 0:
                recon_images(args, epoch, batch_size=64, dataset=testloader.dataset, model=Fvae, writer=writer)
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoints(folder + "/checkpoints.best.pth")
            save_checkpoints(folder + "/checkpoints.latest.pth")
    
    writr_down_the_args_each_exp(args, folder)


def get_argparse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Train FVAE')
    #! args of experiment settings
    parser.add_argument('--seed', default=666, type=int, help='seed')
    parser.add_argument('--exp_name', default='None', type=str, help='training freqvae model')
    parser.add_argument('--freqvae', default="phase", type=str, help="The branch of FreqVAE model")
    #! args of training FreqVAE
    parser.add_argument('--dim', default=512, type=int, help='CNN_embed_dim')
    parser.add_argument('--fdim', default=32, type=int, help='features dim')
    parser.add_argument('--re', default=5, type=float, help='reconstruction weight')
    parser.add_argument('--kl', default=0.1, type=float, help='kl weight') 
    args = parser.parse_args()
    return args


def main():
    #! Set default kwargs
    args = get_argparse()
    with open("./config/config_train_freqvae.json", "r") as file:
        kwargs = json.load(file)['kwargs']
    kwargs = check_kwargs(kwargs)
    vars(args).update(kwargs)
    set_random_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    print("=> {}-{} freqvae training...".format(args.freqvae, args.dim))

    #! Load the clean dataset and get freq dataset
    T = tv.transforms.Compose([Image2Tensor])
    dataset = load_dataset(args.dataset, types="custom", transform_train=T, transform_test=T)
    train_set, test_set = load_freq_dataset(dataset, transform_train=T, transform_test=T)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    print("==> Train loader is OK, got {} train images".format(len(train_loader.dataset)))
    print("==> Test loader is OK, got {} test images".format(len(test_loader.dataset)))

    #! Load the FreqVAE model 
    CNN_embed_dim, feature_dim = args.dim, args.fdim
    model = freqvae_cifar(feature_dim, CNN_embed_dim)
    model = torch.nn.DataParallel(model).cuda()
    print("==> FreqVAE Model is OK!")

    #! Load the classifier model
    print("==> Load well trained Model...")
    checkpoint = "-".join([args.net_state, args.net_type, args.dataset])
    feature_extractor, _ = make_and_restore_model(
        arch=args.net_type, dataset=dataset, resume_path=RESUME_DEFAULT[checkpoint]
    )
    print("==> {} is OK...".format(checkpoint))

    #! Set the Criterion Optimizer & Scheduler
    _, optimizer, scheduler = parse_train_tools(args, model)
    print("==> Load Criterion: {} || Optimizer: {} || Scheduler: {}".format(
        args.criterion.upper(), args.optimizer.upper(), args.scheduler.upper()
    ))

    #! Set the checkpoints folder
    global folder
    name = "-".join([args.freqvae, str(args.dim)])
    folder = os.path.join(args.outf, name)
    if not os.path.exists(folder):
        os.mkdir(folder)

    #! Main train pipeline
    store = cox.store.Store("./checkpoints/", exp_id=name)
    writer = store.tensorboard
    train(args, (feature_extractor, model), (train_loader, test_loader), optimizer, scheduler, writer)


if __name__ == "__main__":

    #! command shell: python train_pipeline.py --freqvae phase --dim 512
    main()