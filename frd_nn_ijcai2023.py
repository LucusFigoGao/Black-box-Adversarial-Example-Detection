# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   frd_nn_ijcai2023.py
    Time:        2022/11/20 21:47:51
    Editor:      Figo
-----------------------------------
'''

"""
    :: Stage one: Get image pair for CEs, AEs, NEs through FreqVAE reconstruction.
    :: Stage two: Calculate reconstruction error in feature space. (original model, specific layer)
    :: Stage three: Train a binary classifier, default LR model.
"""

import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv

from torch.utils.data import DataLoader
from utils.helps import cal_accuracy, amp_pha_recon
from utils.helps_data import load_dataset, load_freq_dataset
from utils.helps_load import get_dataset_path, load_ckpt_freqvae_v1
from robustness.tools.helpers import AverageMeter
from robustness.main import make_and_restore_model
from robustness.data_augmentation import TEST_TRANSFORMS_DEFAULT
from default import FRD_RESUME_DEFAULT, RESUME_DEFAULT


Image2Tensor, Tensor2Image = tv.transforms.ToTensor(), tv.transforms.ToPILImage()
BASE_LAYER_DIM = {"layer2": 128, "layer3": 256, "layer4": 512, "layer4-v2": 512} # base model 
ONLINE_LAYER_DIM = {"layer2": 320, "layer3": 640, "layer4": 640, "layer4-v2": 512}  # online model


class triple_pair_dataset(nn.Module):
    """
        :: [image, recon_image, label]: [torch.FloatTensor, torch.FloatTensor torch.LongTensor]
        :: image: range(0, 1)
        :: #! First turn to PIL.Image and then apply tv.transforms
    """
    def __init__(self, dataset, transforms=None):
        super(triple_pair_dataset, self).__init__()
        self.image, self.recon_image, self.targets = dataset.values
        self.transforms = tv.transforms.Compose([
            Tensor2Image, 
            transforms if transforms is not None else Image2Tensor
        ])
    
    def __getitem__(self, index):
        image, recon_image, label = self.image[index], self.recon_image[index], self.targets[index]
        image = self.transforms(image)
        recon_image = self.transforms(recon_image)
        return image, recon_image, label

    def __len__(self):
        return len(self.targets)


class Detector(nn.Module):
    def __init__(self, num_classes=512, C_Number=3):
        super(Detector, self).__init__()
        self.Relu = nn.ReLU(inplace=True)
        self.Linear1 = nn.Linear(2*num_classes, 2*num_classes)
        self.Linear = nn.Linear(2*num_classes, C_Number)
        self.SM = nn.Softmax(dim=1)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, FDx, PDx):
        output = torch.cat((FDx, PDx),1)
        output = self.Linear1(output)
        output = self.Relu(output)
        output = self.Linear(output)
        return output


class Container(nn.Module):
    def __init__(self):
        super(Container, self).__init__()
        self.reset()
    
    def reset(self):
        self.length = 0
        self.values = []
    
    def update(self, vals, dim=0):
        if self.length == 0:
            self.values.extend(vals)
        else:
            for idx, (v, V) in enumerate(zip(vals, self.values)):
                self.values[idx] = torch.cat([V, v], dim=dim)
        self.length += len(vals[-1])


def load_model_and_dataset(args):
    
    #! Load freqvae model through function version-1.0
    model = load_ckpt_freqvae_v1(args, FRD_RESUME_DEFAULT[args.dataset])
    
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

    #! Load under-test dataset
    """
        :: dataset iter each rounds:
        >>> [(image, amplitude, phase), label]
        >>> [(torch.Tensor, torch.Tensor, torch.Tensor), torch.LongTensor]
    """
    Transtest = tv.transforms.Compose([Image2Tensor])
    _, AEset = load_freq_dataset(args.fae, only_val=True, transform_test=Transtest)
    _, CEset = load_freq_dataset(args.fce, only_val=True, transform_test=Transtest)
    _, NEset = load_freq_dataset(args.fne, only_val=True, transform_test=Transtest)
    train_dataset = (AEset, CEset, NEset)

    _, AEset = load_freq_dataset(args.fae_test, only_val=True, transform_test=Transtest)
    _, CEset = load_freq_dataset(args.fce_test, only_val=True, transform_test=Transtest)
    _, NEset = load_freq_dataset(args.fne_test, only_val=True, transform_test=Transtest)
    test_dataset = (AEset, CEset, NEset)

    return model, feature_extractor, (train_dataset, test_dataset)


def freqvae_reconstruction(model, loader, choice):
    """
        :: model: Well trained freqvae model
        :: loader: DataLoader.FreqDataset: [(image, amplitude, phase), label]
        :: #! Return: Container.values: List[torch.Tensor, torch.Tensor, torch.LongTensor]
    """
    if choice == "joint":
        fvae_amplitude, fvae_phase = model
        fvae_amplitude.eval()
        fvae_phase.eval()
    
    elif choice == "amplitude":
        fvae_amplitude = model
        fvae_amplitude.eval()
    
    elif choice == "phase":
        fvae_phase = model
        fvae_phase.eval()
      
    with torch.no_grad():
        triple_pair = Container()
        for (image, amplitude, phase), label in loader:
            image, amplitude, phase = image.cuda(), amplitude.cuda(), phase.cuda()
            if choice in ["joint", "phase"]:
                recon_phase, _, _, _ = fvae_phase(phase)                    # reconstruct the phase
            if choice in ["joint", "amplitude"]:
                recon_amplitude, _, _, _ = fvae_amplitude(amplitude)        # reconstruct the amplitude
        
            if choice == "joint":
                recon_image = amp_pha_recon(recon_amplitude, phase)
            elif choice == "amplitude":
                recon_image = amp_pha_recon(recon_amplitude, phase)
            elif choice == "phase":
                recon_image = amp_pha_recon(amplitude, recon_phase)
            
            #! save a triple pair
            triple_pair.update([image, recon_image, label])

        return triple_pair


def get_feature_maps(model, loader, locate=None, is_tqdm=False, Nature=False):
    """
        :: loader: iters (image, recon_image, label) each round
        :: return feature maps of specific layer
    """
    from tqdm import tqdm

    Fsaving, Lsaving = Container(), Container()
    iterator = tqdm(enumerate(loader), total=len(loader)) if is_tqdm else enumerate(loader)

    """
        :: BothTwoCorrectLogits: (original, reconstruction) images correctly predicted by both model
        :: OnlyOneCorrectLogits: (original, reconstruction) images has different prediction by two models
    """

    with torch.no_grad():
        model.eval()
        for _, (image, recon_image, label) in iterator:
            
            #! model forward to get the feature maps
            n_feed, targets = len(label), label.cuda()
            inputs = torch.cat([image, recon_image], dim=0).cuda()
            logits_all, _, features_all = model(inputs.cuda(), with_latent=True, with_image=False)

            #! get the split logits
            logits, recon_logits = torch.split(logits_all, n_feed, dim=0)
            features, recon_features = torch.split(features_all[locate], n_feed, dim=0)

            global C
            _, C, H, W = features.shape
            
            features, recon_features = features.reshape(n_feed, C, H*W).mean(dim=-1), \
                                       recon_features.reshape(n_feed, C, H*W).mean(dim=-1)

            #! calculate the accuracy
            equal_flag, _ = cal_accuracy(logits, targets)
            recon_equal_flag, _ = cal_accuracy(recon_logits, targets)

            #! save the logits of specific layer

            if Nature == True:
                SameIdx = torch.where(equal_flag - recon_equal_flag == 0)[0]
                DiffIdx = torch.where(equal_flag - recon_equal_flag != 0)[0]

                BTC_logits, OOC_logits = torch.zeros((2, len(SameIdx), C)), torch.zeros((2, len(DiffIdx), C))

                BTC_logits[0, :], BTC_logits[1, :], BTC_targets = features[SameIdx], recon_features[SameIdx], label[SameIdx]
                OOC_logits[0, :], OOC_logits[1, :], OOC_targets = features[DiffIdx], recon_features[DiffIdx], label[DiffIdx]
                
                Fsaving.update([BTC_logits, OOC_logits], dim=1)
                Lsaving.update([BTC_targets, OOC_targets, equal_flag, recon_equal_flag])
            
            else:
                BTC_logits = torch.zeros((2, len(label), C))
                BTC_logits[0, :], BTC_logits[1, :] = features, recon_features
                Fsaving.update([BTC_logits], dim=1)
                Lsaving.update([label, equal_flag, recon_equal_flag])
    
    totoal_correct = Lsaving.values[-2].sum()
    recon_total_correct = Lsaving.values[-1].sum()
    total_number = Lsaving.length
    print(Lsaving.values[-2], Lsaving.values[-1], Lsaving.length)
    print("=> The test accuracy original image is:{:.2f}%".format(100. * totoal_correct / total_number))
    print("=> The test accuracy reconstruction image is:{:.2f}%".format(100. * recon_total_correct / total_number))
    
    if Nature:
        Label = torch.cat((torch.zeros(Fsaving.values[0].shape[1]), torch.ones(Fsaving.values[1].shape[1])))
        print(Fsaving.values[0].shape, Fsaving.values[1].shape, Label.shape)
        return Fsaving.values[0], Fsaving.values[1], Label
    else:
        Label = torch.ones(Fsaving.values[0].shape[1]) * 2
        print(Fsaving.values[0].shape, Lsaving.values[0].shape)
        return Fsaving.values[0], Label


def reconstruction_pipeline(args, layer, dataset, nature=True):
    #! Stage one: Get image pair for (CEs, AEs, NEs) through FreqVAE reconstruction.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    triple_pair = freqvae_reconstruction(model, loader, choice=args.freqvae)
    print(">>> Finished freqvae reconstruction...")
    
    #! Stage two: Get reconstruction pair in feature space. (original model, specific layer)
    GetFeaDataset = triple_pair_dataset(triple_pair, TEST_TRANSFORMS_DEFAULT(32))
    GetFeaLoader = DataLoader(GetFeaDataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    FeaSavings = get_feature_maps(feature_extractor, GetFeaLoader, layer, is_tqdm=True, Nature=nature)
    print(">>> Finished feature maps extraction...")        
    return FeaSavings


def make_detection_features(args, layer, clean_data, noisy_data, adv_data):
    global Logit_Save
    Logit_Save = os.path.join(args.outf, file_name)
    if not os.path.exists(Logit_Save): os.mkdir(Logit_Save)
    print("=> FRD(nn) file {} saved in {}".format(file_name, Logit_Save))

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
    AUROC, ACC = [], []

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
        AUROC.append(auroc)
        ACC.append(acc)

        if (i + 1) * args.TB > TotalDataScale:
            r = '\rEpoch: [Test][{0}/{1}]  AUROC:{auroc:.3f}% || ACC:{acc:.3f}% '.format(
                    TotalDataScale, TotalDataScale, auroc=auroc*100, acc=acc)
        else:
            r = '\rEpoch: [Test][{0}/{1}]  AUROC:{auroc:.3f}% || ACC:{acc:.3f}% '.format(
                    (i + 1) * args.TB, TotalDataScale, auroc=auroc*100, acc=acc)
        sys.stdout.write(r)
        i += 1
    
    return sum(AUROC)/len(AUROC), sum(ACC)/len(ACC)


def main(args):

    LAYER_DIM = BASE_LAYER_DIM if args.version == "base" else ONLINE_LAYER_DIM

    #! Load model and dataset
    get_dataset_path(args)

    #! make dirs to save frd features
    if not os.path.exists(args.outf) and args.outf is not None:
        os.mkdir(args.outf)
    print("=> Make dirs ({}) to save frd features".format(args.outf))

    """
        :: #! Load dataset(CEs, AEs, NEs, label)
        :: #! Load well-trained FreqVAE-model
        :: #! Load well-trained classification-task model
    """
    global model, feature_extractor
    model, feature_extractor, dataset = load_model_and_dataset(args)

    #! Generate detected dataset
    train_dataset, test_dataset = dataset
    adv_data_train, clean_data_train, noisy_data_train = train_dataset
    adv_data_test, clean_data_test, noisy_data_test = test_dataset

    #! Train detector stage
    global file_name
    victim_file = "{}:{}-".format(args.freqvae, args.layer) + "-".join([
        args.victim_model[0], args.victim_model[1], args.dataset, 
        args.adv_type_train, args.adv_norm_train, str(args.adv_parameter_train)
    ])
    file_name = victim_file
    print(f"=> {file_name}")
    BLogits, BLabel, TrainIndex, TestIndex = make_detection_features(args, args.layer, clean_data_train, noisy_data_train, adv_data_train)
    
    #! get detection model
    detector = Detector(num_classes=LAYER_DIM[args.layer], C_Number=3)
    detector.cuda()
    for p in detector.parameters():
        p.requires_grad_()
    
    #! Define optimizer, scheduler and criterion
    optimizer = torch.optim.SGD(detector.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = None

    if not os.path.exists(Logit_Save+"/checkpoint.best.pt"):
        train_pipeline(args, BLogits, BLabel, detector, criterion, optimizer, scheduler, TrainIndex, TestIndex)
    
    kwargs = torch.load(Logit_Save+"/checkpoint.best.pt")
    auroc, acc, ckpt = kwargs['auroc'], kwargs['acc'], kwargs['model']
    print("=> (Self) The detector performance for {}: AUROC:{:.4f}% || ACC:{:.4f}%".format(
        file_name, auroc*100, acc
    ))

    #! Test detector stage
    hacker_file = "{}:{}-".format(args.freqvae, args.layer) + "-".join([
        args.hacker_model[0], args.hacker_model[1], args.dataset, 
        args.adv_type_test, args.adv_norm_test, str(args.adv_parameter_test)
    ])
    file_name = hacker_file
    print(f"=> {file_name}")
    detector.load_state_dict(ckpt)
    args.TSR, args.TB = 0.0, 512
    BLogits, BLabel, TrainIndex, TestIndex = make_detection_features(args, args.layer, clean_data_test, noisy_data_test, adv_data_test)
    auroc, acc = TestDetector(args, BLogits, BLabel, detector, TestIndex)

    #! Record
    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(args.outf+"/readme.txt", "a") as files:
        files.write("="*13 + "Records " + "="*13 + "\n")
        files.write("=> recording time: {}\n".format(time_now))
        files.write("=> {}\n".format(victim_file))
        files.write("=> {}\n".format(hacker_file))
        files.write("=> Best FRD-Detector: ROC-AUC score:{:.4f} | ACC score:{:.4f}\n".format(auroc*100, acc))
    files.close()
