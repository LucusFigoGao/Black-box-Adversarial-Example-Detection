# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   prd_nn_ijcai2023.py
    Time:        2022/12/16 20:55:02
    Editor:      Figo
-----------------------------------
'''

import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv

from model.freqvae import freqvae_cifar as PixelVAE
from torch.utils.data import DataLoader
from utils.helps_visual import visual_image
from utils.helps_load import get_dataset_path
from utils.helps_data import load_dataset, load_data_from_file
from frd_nn_ijcai2023 import Container, Detector, triple_pair_dataset, get_feature_maps
from default import PRD_DEFAULT, RESUME_DEFAULT
from robustness.tools.helpers import AverageMeter
from robustness.model_utils import make_and_restore_model
from robustness.data_augmentation import TEST_TRANSFORMS_DEFAULT


SAVE_DIR = "Image/PRD"
Image2Tensor, Tensor2Image = tv.transforms.ToTensor(), tv.transforms.ToPILImage()
BASE_LAYER_DIM = {"layer2": 128, "layer3": 256, "layer4": 512, "layer4-v2": 512} # base model 
ONLINE_LAYER_DIM = {"layer2": 320, "layer3": 640, "layer4": 640, "layer4-v2": 512}  # online model


class make_dataset:
    def __init__(self, data, label, transform) -> None:
        self.data = data
        self.target = label
        self.transform = transform
    
    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        from PIL import Image
        image, label = self.data[index], self.target[index]
        if image.mean() <= 5.0:
            image = np.uint8(image*255)
        image = Image.fromarray(np.uint8(image.transpose(1, 2, 0)))
        image = self.transform(image)
        return image, label
    

def load_prd_model(args):
    model_params = torch.load(PRD_DEFAULT[args.dataset])['model']
    model_params_off_parallel = model_params
    model = PixelVAE(d=args.fdim, z=args.dim, with_classifier=False)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_params_off_parallel)
    print("=> Pixel-VAE is OK...")
    return model


def load_model_and_dataset(args):
    
    #! Load pixel-vae model
    model = load_prd_model(args)
    
    #! Load original classification-task model
    print("=> Load well-trained classification-task model...")
    dataset = load_dataset(args.dataset, "default")


    #! Load victim model as backbone(base)
    if args.version == "base":
        ckpt_name = '-'.join([args.victim_model[0], args.victim_model[1], args.dataset])
        feature_extractor, _ = make_and_restore_model(arch=args.victim_model[1], dataset=dataset, resume_path=RESUME_DEFAULT[ckpt_name])
        print("=> {} is OK...".format(ckpt_name))
    
    #! Load pre-trained model as backbone(online)
    if args.version == "online":
        ckpt_name = '-'.join([args.victim_model[0], 'wideresnet28', args.dataset])
        feature_extractor, _ = make_and_restore_model(arch='wideresnet28', dataset=dataset, resume_path=RESUME_DEFAULT[ckpt_name])
        print("=> ====================================|")
        print("=> Here we use wideresnet28 as backbone!")
        print("=> ====================================|")
        print("=> {} is OK...".format(ckpt_name))

    #! Load train dataset (Victim model)
    Transtest = tv.transforms.Compose([Image2Tensor])
    print("Load train dataset (Victim model)")
    AEset, _ = load_data_from_file(args.fae)
    CEset, label = load_data_from_file(args.fce)
    NEset, _ = load_data_from_file(args.fne)
    AEset = make_dataset(AEset, label, Transtest)
    CEset = make_dataset(CEset, label, Transtest)
    NEset = make_dataset(NEset, label, Transtest)
    train_dataset = (AEset, CEset, NEset, label)
    print("=> Got {} under-test image".format(len(AEset)))

    #! Load test dataset (Hacker model)
    print("Load test dataset (Hacker model)")
    AEset, _ = load_data_from_file(args.fae_test)
    CEset, label = load_data_from_file(args.fce_test)
    NEset, _ = load_data_from_file(args.fne_test)
    AEset = make_dataset(AEset, label, Transtest)
    CEset = make_dataset(CEset, label, Transtest)
    NEset = make_dataset(NEset, label, Transtest)
    test_dataset = (AEset, CEset, NEset, label)

    return model, feature_extractor, (train_dataset, test_dataset)


def prd_reconstruction(model, loader):
    model.eval()
    ImagePairs = Container()
    with torch.no_grad():
        for idx, (image, label) in enumerate(loader):                
            image, label = image.cuda(), label.cuda().view(-1, )
            recon_image, _, _, _ = model(image)
            if idx == 0:
                rnd = np.random.randint(0, len(label), 1)[0]
                if not os.path.exists(SAVE_DIR):
                    os.mkdir(SAVE_DIR)
                savdir = os.path.join(SAVE_DIR, "pixel-vae{}.png".format(rnd))
                visual_image(
                    [
                        image[rnd].cpu().data.permute(1, 2, 0).numpy(), 
                        recon_image[rnd].cpu().data.permute(1, 2, 0).numpy()
                    ], savdir, padding=2
                )
            ImagePairs.update([image, recon_image, label])
    return ImagePairs


def reconstruction_pipeline(args, layer, dataset, nature=True):
    #! Stage one: Get image pair for (CEs, AEs, NEs) through CD-VAE reconstruction.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    triple_pair = prd_reconstruction(model, loader)
    print(">>> Finished pixel-vae reconstruction...")
    
    #! Stage two: Get reconstruction pair in feature space. (original model, specific layer)
    GetFeaDataset = triple_pair_dataset(triple_pair, TEST_TRANSFORMS_DEFAULT(32))
    GetFeaLoader = DataLoader(GetFeaDataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    FeaSavings = get_feature_maps(feature_extractor, GetFeaLoader, layer, is_tqdm=True, Nature=nature)
    print(">>> Finished feature maps extraction...")        
    return FeaSavings


def make_detection_features(args, layer, clean_data, noisy_data, adv_data):
    global Logit_Save
    Logit_Save = os.path.join(args.outf, file_name)
    if not os.path.exists(Logit_Save): os.mkdir(Logit_Save)
    print("=> Pixel-VAE(nn) file {} saved in {}".format(file_name, Logit_Save))

    if os.path.exists(Logit_Save + '/BLogits.npy') \
        and os.path.exists(Logit_Save + '/BLabel.npy') \
            and os.path.exists(Logit_Save + '/Size.npy'):
        print('Load Logits from: {}'.format(Logit_Save))
        Size = np.load(Logit_Save + '/Size.npy')
        BLogits = np.load(Logit_Save+'/BLogits.npy')
        BLabel = np.load(Logit_Save+'/BLabel.npy')
    
    else:
        print('Generate Logits from source: {}'.format(file_name))

        #! Generate logits for three datasets (AEs, CEs, NEs)
        print('>>> Clean Data...')
        CPFLogits, CPLogits, CLabel = reconstruction_pipeline(args, layer, clean_data, nature=True)
        
        print('>>> Noisy Data...')
        NsyPFLogits, NsyPLogits, NsyLabel = reconstruction_pipeline(args, layer, noisy_data, nature=True)
        
        print('>>> Adversarial Data...')
        AdvPFLogits, AdvLabel = reconstruction_pipeline(args, layer, adv_data, nature=False)

        Size = [CPFLogits.shape[1], CPLogits.shape[1], AdvPFLogits.shape[1]]
        BLogits = np.concatenate((CPFLogits, CPLogits, NsyPFLogits, NsyPLogits, AdvPFLogits), axis=1)
        BLabel = np.concatenate((CLabel, NsyLabel, AdvLabel), axis=0)
        print('\nSave Logits in: {}'.format(Logit_Save))
        np.save(Logit_Save + '/Size.npy', Size)
        np.save(Logit_Save + '/BLogits.npy', BLogits)
        np.save(Logit_Save + '/BLabel.npy',BLabel)
    
    TS0 = int(np.floor(args.TSR * Size[0]))
    TS1 = int(np.floor(args.TSR * Size[1]))
    TS2 = int(np.floor(args.TSR * Size[2]))
    CPFindex, CPindex, Advindex = np.random.permutation(Size[0]), np.random.permutation(Size[1]), np.random.permutation(Size[2])
    TrainIndex = np.concatenate((
        CPFindex[0:TS0], CPindex[0:TS1]+Size[0], CPFindex[0:TS0]+Size[0]+Size[1], \
            CPindex[0:TS1]+2*Size[0]+Size[1], Advindex[0:TS2]+2*Size[1]+2*Size[0]), axis=0)
    TestIndex = np.concatenate((
        CPFindex[TS0:], CPindex[TS1:]+Size[0], CPFindex[TS0:]+Size[0]+Size[1], \
            CPindex[TS1:]+2*Size[0]+Size[1], Advindex[TS2:]+2*Size[1]+2*Size[0]), axis=0)

    return BLogits, BLabel, TrainIndex, TestIndex


def get_auroc(output, target_var):
    Bioutput = torch.zeros([output.shape[0], 2])
    Bioutput[:, 0] = torch.max(output[:, 0:2], 1)[0]
    Bioutput[:, 1] = output[:, 2]
    target_var[np.nonzero(target_var.cpu().numpy() == 1)] = 0
    target_var[np.nonzero(target_var.cpu().numpy() == 2)] = 1
    Bioutput = torch.nn.Softmax(dim=1)(Bioutput)
    
    y_pred = Bioutput.cpu().data.max(1)[1].numpy().astype(np.float64)
    Bioutput = Bioutput.cpu().detach().numpy().astype(np.float64)[:, 1]
    Y = target_var.cpu().detach().numpy().astype(np.float64)
    
    num_samples = Y.shape[0]

    from sklearn.metrics import accuracy_score, roc_auc_score
    auroc, acc = roc_auc_score(Y, Bioutput), accuracy_score(Y, y_pred)
    
    l1 = open('%s/confidence_TMP_In.txt' % Logit_Save, 'w')
    l2 = open('%s/confidence_TMP_Out.txt' % Logit_Save, 'w')
    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-Bioutput[i]))
        else:
            l2.write("{}\n".format(-Bioutput[i]))
    l1.flush()
    l2.flush()
    # results = callog.metric(Logit_Save, ['TMP'])
    return auroc, acc*100


def train_pipeline(args, BLogits, BLabel, DualMOdel, criterion, optimizer, scheduler, TrainIndex, ValIndex):
    best_auroc = 0.0
    for epoch in range(args.epoch):
        TrainDetector(args, epoch, BLogits, BLabel, DualMOdel, criterion, optimizer, TrainIndex)
        
        def save_ckpt():
            ckpt = {
                "model": DualMOdel.state_dict(), 
                "epoch": epoch, 
                "auroc": best_auroc, 
                "acc": best_acc
            }
            torch.save(ckpt, Logit_Save+"/checkpoint.best.pt")
        
        if epoch % 5 == 0:
            auroc, acc = TestDetector(args, BLogits, BLabel, DualMOdel, ValIndex)
            if auroc >= best_auroc:
                best_auroc, best_acc = auroc, acc
                save_ckpt()
        
        if scheduler is not None:
            scheduler.step()
    print("=> Finished training!")


def TrainDetector(args, epoch, BLogits, BLabel, model, criterion, optimizer, TrainIndex):
    model.train()
    losses = AverageMeter()
    TotalDataScale = len(TrainIndex)
    TrainIndex = TrainIndex[np.random.permutation(TotalDataScale)]
    i= 0
    open('%s/confidence_TMP_In.txt' % Logit_Save, 'w').close()
    open('%s/confidence_TMP_Out.txt' % Logit_Save, 'w').close()
    while not len(TrainIndex) == 0:

        if len(TrainIndex) < args.TB:
            FDx = torch.from_numpy(BLogits[1, TrainIndex[0: ]].astype(dtype=np.float32))
            PDx = torch.from_numpy(BLogits[0, TrainIndex[0: ]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[TrainIndex[0: ]])
        else:
            PDx = torch.from_numpy(BLogits[0, TrainIndex[0: args.TB]].astype(dtype=np.float32))
            FDx = torch.from_numpy(BLogits[1, TrainIndex[0: args.TB]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[TrainIndex[0: args.TB]])
        
        PDx = torch.autograd.Variable(PDx.cuda())
        FDx = torch.autograd.Variable(FDx.cuda())
        target_var = torch.autograd.Variable(target.cuda()).long()
        output = model(PDx, FDx)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.float()
        loss = loss.float()
        auroc, acc = get_auroc(output, target_var)
        losses.update(loss.data, PDx.shape[0])

        if i % args.print_freq == 0:
            if (i + 1) * args.TB > TotalDataScale:
                r = '\rEpoch: [{0}][{1}/{2}] Loss:{loss.avg:.4f} || AUROC:{auroc:.3f}% || ACC:{acc:.3f}% '.format(
                    epoch, TotalDataScale, TotalDataScale, loss=losses, auroc=auroc*100, acc=acc)
            else:
                r = '\rEpoch: [{0}][{1}/{2}] Loss:{loss.avg:.4f} || AUROC:{auroc:.3f}% || ACC:{acc:.3f}% '.format(
                    epoch, (i + 1) * args.TB, TotalDataScale, loss=losses, auroc=auroc*100, acc=acc)
            sys.stdout.write(r)
        i += 1
        TrainIndex = TrainIndex[args.TB:]


def TestDetector(args, BLogits, BLabel, DualMOdel, ValIndex):
    Index = ValIndex[np.random.permutation(ValIndex.shape[0])]
    open('%s/confidence_TMP_In.txt' % Logit_Save, 'w').close()
    open('%s/confidence_TMP_Out.txt' % Logit_Save, 'w').close()
    i, TotalDataScale = 0, len(Index)

    DualMOdel.eval()

    while not len(Index) == 0:
        if len(Index) < args.TB:
            FDx = torch.from_numpy(BLogits[1, Index[0:]].astype(dtype=np.float32))
            PDx = torch.from_numpy(BLogits[0, Index[0:]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[Index[0:]])
        else:
            PDx = torch.from_numpy(BLogits[0, Index[0:args.TB]].astype(dtype=np.float32))
            FDx = torch.from_numpy(BLogits[1, Index[0:args.TB]].astype(dtype=np.float32))
            target = torch.from_numpy(BLabel[Index[0:args.TB]])

        PDx = torch.autograd.Variable(PDx.cuda())
        FDx = torch.autograd.Variable(FDx.cuda())
        target_var = torch.autograd.Variable(target.cuda()).long()
        
        output = DualMOdel(PDx, FDx).float()
        Index = Index[args.TB:]
        auroc, acc = get_auroc(output, target_var)

        if (i + 1) * args.TB > TotalDataScale:
            r = '\rEpoch: [Test][{0}/{1}]  AUROC:{auroc:.3f}% || ACC:{acc:.3f}% '.format(
                    TotalDataScale, TotalDataScale, auroc=auroc*100, acc=acc)
        else:
            r = '\rEpoch: [Test][{0}/{1}]  AUROC:{auroc:.3f}% || ACC:{acc:.3f}% '.format(
                    (i + 1) * args.TB, TotalDataScale, auroc=auroc*100, acc=acc)
        sys.stdout.write(r)
        i += 1
    
    return auroc, acc


def main(args):

    #! Load model and dataset
    get_dataset_path(args)

    LAYER_DIM = BASE_LAYER_DIM if args.version == "base" else ONLINE_LAYER_DIM

    #! make dirs to save pixel-vae features
    if not os.path.exists(args.outf) and args.outf is not None:
        os.mkdir(args.outf)
    print("=> Make dirs ({}) to save pixel-vae features".format(args.outf))

    """
        :: #! Load dataset(CEs, AEs, NEs, label)
        :: #! Load well-trained Pixel-VAE-model
        :: #! Load well-trained classification-task model
    """
    global model, feature_extractor
    model, feature_extractor, dataset = load_model_and_dataset(args)
    
    #! get detection model
    detector = Detector(num_classes=LAYER_DIM[args.layer], C_Number=3)
    detector.cuda()
    for p in detector.parameters():
        p.requires_grad_()
    
    #! Define optimizer, scheduler and criterion
    optimizer = torch.optim.SGD(detector.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = None

    #! Generate detected dataset
    train_dataset, test_dataset = dataset
    adv_data_train, clean_data_train, noisy_data_train, _ = train_dataset
    adv_data_test, clean_data_test, noisy_data_test, _ = test_dataset

    #! Train detector stage
    global file_name
    victim_file = "{}:{}-".format(args.framework, args.layer) + "-".join([
        args.victim_model[0], args.victim_model[1], args.dataset, 
        args.adv_type_train, args.adv_norm_train, str(args.adv_parameter_train)
    ])
    file_name = victim_file
    print(f"=> {file_name}")
    BLogits, BLabel, TrainIndex, TestIndex = make_detection_features(args, args.layer, clean_data_train, noisy_data_train, adv_data_train)
    
    if not os.path.exists(Logit_Save+"/checkpoint.best.pt"):
        train_pipeline(args, BLogits, BLabel, detector, criterion, optimizer, scheduler, TrainIndex, TestIndex)
    
    kwargs = torch.load(Logit_Save+"/checkpoint.best.pt")
    auroc, acc, ckpt = kwargs['auroc'], kwargs['acc'], kwargs['model']
    print("=> (Self) The detector performance for {}: AUROC:{:.4f}% || ACC:{:.4f}%".format(
        file_name, auroc*100, acc
    ))

    #! Test detector stage
    hacker_file = "{}:{}-".format(args.framework, args.layer) + "-".join([
        args.hacker_model[0], args.hacker_model[1], args.dataset, 
        args.adv_type_test, args.adv_norm_test, str(args.adv_parameter_test)
    ])
    file_name = hacker_file
    print(f"=> {file_name}")
    detector.load_state_dict(ckpt)
    BLogits, BLabel, TrainIndex, TestIndex = make_detection_features(args, args.layer, clean_data_test, noisy_data_test, adv_data_test)
    auroc, acc = TestDetector(args, BLogits, BLabel, detector, TestIndex)

    #! Record
    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(args.outf+"/readme.txt", "a") as files:
        files.write("="*13 + "Records " + "="*13 + "\n")
        files.write("=> recording time: {}\n".format(time_now))
        files.write("=> {}\n".format(victim_file))
        files.write("=> {}\n".format(hacker_file))
        files.write("=> Best Pixel-VAE-Detector: ROC-AUC score:{:.4f} | ACC score:{:.4f}\n".format(auroc*100, acc))
    files.close()


if __name__ == "__main__":
    
    import json
    import argparse

    parser = argparse.ArgumentParser(description="Pixel detection")
    args = parser.parse_args()
    with open("config/config_prd.json", "r") as file:
        kwargs = json.load(file)['kwargs']
    kwargs['victim_model'] = ['normal', "resnet18"]
    kwargs['hacker_model'] = ['normal', "resnet18"]
    
    kwargs["adv_type_train"] = "pgd"
    kwargs["adv_norm_train"] = "l2"
    kwargs["adv_parameter_train"] = "1.0"
    
    kwargs["adv_type_test"] = "pgd"
    kwargs["adv_norm_test"] = "l2"
    kwargs["adv_parameter_test"] = "1.0"

    vars(args).update(kwargs)

    #! Load model and dataset
    get_dataset_path(args)

    model, feature_extractor, dataset = load_model_and_dataset(args)
    adv_data_train, clean_data_train, noisy_data_train, label_train = dataset[0]

    loader = DataLoader(adv_data_train, batch_size=args.batch_size, shuffle=False)
    triple_pair = prd_reconstruction(model, loader)
    print(">>> Finished pixel reconstruction...")

    loader = DataLoader(clean_data_train, batch_size=args.batch_size, shuffle=False)
    triple_pair = prd_reconstruction(model, loader)
    print(">>> Finished pixel reconstruction...")

    loader = DataLoader(noisy_data_train, batch_size=args.batch_size, shuffle=False)
    triple_pair = prd_reconstruction(model, loader)
    print(">>> Finished pixel reconstruction...")