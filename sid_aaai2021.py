# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   sid_aaai2021.py
    Time:        2022/11/20 22:01:39
    Editor:      Figo
-----------------------------------
'''

import os
import sys
import time
import dill
import torch
import numpy as np
import torch.nn as nn

from model.fresnet import FRESNET
from utils.helps_model import reload_model
from utils.helps_load import get_dataset_path
from utils.helps_data import load_dataset, easy_load_model_dataset_for_test, load_data_from_file
from robustness.main import make_and_restore_model
from robustness.tools.helpers import AverageMeter
from default import SID_DUAL_RESUME, RESUME_DEFAULT


class DualCIFAR4(nn.Module):
    """ ModelCIFAR model """
    def __init__(self, num_classes=10, C_Number=3):
        super(DualCIFAR4, self).__init__()
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


def reload_model_for_training(args, model):
    net_type = args.victim_model[1]
    kwargs = {"num_classes": args.num_classes}
    replacement = FRESNET[net_type](**kwargs)
    model = reload_model(model, replacement)
    print("=> Successfully reload model (sid-f{})...".format(net_type))
    return model


def get_model(args):

    print("==> Load Pixel/Frequency-trained Model...")
    #* Load model from model zoo and replaced by sid-model @https://github.com/JinyuTian/SID

    net_state = args.victim_model[0]
    net_type = args.victim_model[1]

    ds = load_dataset(args.dataset)
    ckpt = "-".join([net_state, net_type, args.dataset])
    Pmodel, checkpoint = make_and_restore_model(arch=net_type, dataset=ds, resume_path=RESUME_DEFAULT[ckpt])
    print("==> Pixel-trained Victim Model is OK...")

    #! Load dual model from model zoo
    model, _ = easy_load_model_dataset_for_test(args.victim_model[1], args.dataset)
    Fmodel = reload_model_for_training(args, model)
    ckpt = "-".join(['dual', net_type, args.dataset])

    if os.path.isfile(SID_DUAL_RESUME[ckpt]):
        print("=> loading checkpoint '{}'".format(SID_DUAL_RESUME[ckpt]))
        checkpoint = torch.load(SID_DUAL_RESUME[ckpt], pickle_module=dill)
        
        # Makes us able to load models saved with legacy versions
        state_dict_path = 'model'
        if not ('model' in checkpoint): state_dict_path = 'state_dict'

        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}
        Fmodel.load_state_dict(sd)
        print("=> loaded checkpoint '{}' (epoch {})".format(SID_DUAL_RESUME[ckpt], checkpoint['epoch']))
    print("==> Frequency trained Model is OK...")
    return Pmodel, Fmodel


def get_data(args):
    #! Load train dataset (Victim model)
    print("Load train dataset (Victim model)")
    AEset, _ = load_data_from_file(args.fae)
    CEset, label = load_data_from_file(args.fce)
    NEset, _ = load_data_from_file(args.fne)
    AEset, CEset, NEset, label = torch.tensor(AEset, dtype=torch.float32), \
                                 torch.tensor(CEset, dtype=torch.float32), \
                                 torch.tensor(NEset, dtype=torch.float32), \
                                 torch.tensor(label, dtype=torch.long)
    train_ds = (AEset, CEset, NEset, label)
    print("=> Got {} under-test image".format(len(AEset)))

    #! Load test dataset (Hacker model)
    print("Load test dataset (Hacker model)")
    AEset, _ = load_data_from_file(args.fae_test)
    CEset, label = load_data_from_file(args.fce_test)
    NEset, _ = load_data_from_file(args.fne_test)

    AEset, CEset, NEset, label = torch.tensor(AEset, dtype=torch.float32), \
                                 torch.tensor(CEset, dtype=torch.float32), \
                                 torch.tensor(NEset, dtype=torch.float32), \
                                 torch.tensor(label, dtype=torch.long)
    test_ds = (AEset, CEset, NEset, label)
    print("=> Got {} under-test image".format(len(AEset)))
    
    return train_ds, test_ds


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, pred


def GetLogitValues(Data, label, PDmodel, FDmodel, args, Natural=True):
    """
        :: Data: The AEs, CEs, or NEs data with available type: numpy.ndarray, torch.Tensor
        :: PDmodel: Pixel Domain trained model
        :: FDmodel: Frequency Domain trained model

        Variable:
        :: PFCrtImgs:   Clean images correctly predicted by both PDmodel and FDmodel
        :: PCrtImgs:    Clean images correctly predicted by the PDmodel only

        :: PFNsyImgs:   Noisy images correctly predicted by both PDmodel and FDmodel
        :: PNsyImgs:    Noisy images correctly predicted by both PDmodel only

        :: PFAdvImgs:   Adver images successfully attacked both the PDmodel and the FDmodel
        :: PAdvImgs:    Adver images successfully attacked the PDmodel only

    """

    if isinstance(Data, torch.Tensor): Data = Data.numpy()
    if isinstance(label,torch.Tensor): label = label.numpy()
    FDmodel.eval()
    PDmodel.eval()

    #! Initialize
    TrainIndex = np.arange(0, Data.shape[0])            # (0, total_number)
    Scale = len(TrainIndex)                             # total_number
    PFLogits = np.zeros([2, 2, args.num_classes])       # (2, 2, 10)
    PLogits = np.zeros([2, 2, args.num_classes])        # (2, 2, 10)
    PF_Targets = np.zeros([2])                          # (2)
    P_Targets = np.zeros([2])                           # (2)
    Flag = np.zeros([2])                                # (2)

    #! iters dataset
    i = 0
    while not len(TrainIndex) == 0:                     # args.TB: Train batch size
        if len(TrainIndex) < args.TB:
            TempIndex = TrainIndex[0:]
            inputs = torch.from_numpy(Data[TempIndex])
            target = torch.from_numpy(label[TempIndex])
        else:
            TempIndex = TrainIndex[0: args.TB]
            inputs = torch.from_numpy(Data[TempIndex])
            target = torch.from_numpy(label[TempIndex])

        input_var = torch.autograd.Variable(inputs.cuda()).float()
        target_var = torch.autograd.Variable(target.cuda()).long()
        
        def model_forward(model, inputs, targets):
            outputs = model(inputs, with_image=False, with_latent=False)
            prec1, pred = accuracy(outputs.data, targets)
            return outputs.detach().cpu().numpy(), pred
        
        FDx, Fpred = model_forward(FDmodel, input_var, target_var)
        PDx, Ppred = model_forward(PDmodel, input_var, target_var)

        """
            :: Here we make a judgement of each data point which satisfies, 
            :: 1. the data is correctly predicted (successfully attacked) by both PDmodel and FDmodel
            :: 2. the data is correctly predicted (successfully attacked) by the PDmodel only
            :: and save the outputs of both two models. (2 * 2 = 4)
        """
        if Natural == True:
            PFCrtImgsIdx = np.nonzero((Ppred - Fpred).cpu().numpy() == 0)[1]        #! same prediction
            PCrtImgsIdx = np.nonzero((Ppred - Fpred).cpu().numpy())[1]              #! different prediction

            TPFLogits = np.zeros([2, len(PFCrtImgsIdx), PDx.shape[-1]])             # (2, N1, 10)
            TPLogits = np.zeros([2, len(PCrtImgsIdx), PDx.shape[-1]])               # (2, N2, 10)  N1 + N2 = total_number
            
            #! same prediction
            TPFLogits[0,:], TPFLogits[1,:], TPF_Targets = PDx[PFCrtImgsIdx], FDx[PFCrtImgsIdx], target.numpy()[PFCrtImgsIdx]
            #! different prediction
            TPLogits[0,:], TPLogits[1,:], TP_Targets = PDx[PCrtImgsIdx], FDx[PCrtImgsIdx], target.numpy()[PCrtImgsIdx]

            #! Turn to savings, but i don't know why the author added a zero-mat before each values?
            PFLogits = np.concatenate((PFLogits, TPFLogits), axis=1)                # (2, 2, 10) + (2, N1, 10) --> (2, 2+N1, 10)
            PLogits = np.concatenate((PLogits, TPLogits), axis=1)                   # (2, 2, 10) + (2, N2, 10) --> (2, 2+N2, 10)
            PF_Targets = np.concatenate((PF_Targets, TPF_Targets))                  # (2) + (N1) --> (2+N1)
            P_Targets = np.concatenate((P_Targets, TP_Targets))                     # (2) + (N2) --> (2+N2)

        else:
            """
                :: Here we save all the outputs values, no matter the prediction results of two models
            """
            TPFLogits = np.zeros([2, PDx.shape[0], PDx.shape[-1]])
            TPFLogits[0,:], TPFLogits[1,:], TPF_Targets = PDx, FDx, target.numpy()

            PFLogits = np.concatenate((PFLogits, TPFLogits), axis=1)
            PF_Targets = np.concatenate((PF_Targets, TPF_Targets))
        
        i = i + 1
        TrainIndex = TrainIndex[args.TB:]
    
    """
    #! Initialize phase
        >>> PFLogits = np.zeros([2, 2, args.num_classes])     # (2, 2, 10)
        >>> PLogits = np.zeros([2, 2, args.num_classes])      # (2, 2, 10)
        >>> PF_Targets = np.zeros([2])                      # (2)
        >>> P_Targets = np.zeros([2])                       # (2)
        >>> Flag = np.zeros([2])                            # (2)
    
    #! Return phase
        Natrual == True: (clean data, noisy data)
        :: PFLogits: logits, both 2 models correctly predict, PFLogits[0] from Pixel domain, PFLogits[1] from Frequency domain
        :: PLogits: logits, only Pixel model correctly predicts, PLogits[0] from Pixel domain, PLogits[1] from Frequency domain
        :: Label: label, range(0, 1, 2) for detection task. Here, 
        >>> PFLogits --> label 0; 
        >>> PLogits --> label 1;
        >>> PLogits(Normal=False) ---> label 2
        :: PF_Targets: targets, range(0, 10), from PDmodel and FDmodel predict
        :: P_Targets: targets, range(0, 10), from PDmodel predicts
    """
    
    if Natural == True:
        PFLogits = PFLogits[:, 2:, ]                        # (2, N1, 10)
        PLogits = PLogits[:, 2:, ]                          # (2, N2, 10)
        PF_Targets = PF_Targets[2:]                         # (N1)
        P_Targets = P_Targets[2:]                           # (N2)
        Flag = Flag[2:]                                     # (N1)
        #! N1 detection data with label 0, N2 detection data with label 1
        Label = np.concatenate((np.zeros(PFLogits.shape[1]), np.ones(PLogits.shape[1])))
        return PFLogits, PLogits, Label, PF_Targets, P_Targets, Flag
    
    else:
        PFLogits = PFLogits[:, 2:, ]                        # (2, N1, 10)
        PF_Targets = PF_Targets[2:]                         # (N1)
        Label = 2 * np.ones(PFLogits.shape[1])              # N1 detection data with label 2
        Flag = Flag[2:]                                     # (N1)
        return PFLogits, Label, PF_Targets, Flag


def GetStackLogitValues(clean_data, noisy_data, adv_data, label, PDmodel, FDmodel, args):
    
    global Logit_Save
    Logit_Save = os.path.join(args.outf, file_name)
    if not os.path.exists(Logit_Save): os.mkdir(Logit_Save)
    print("=> SID file {} saved in {}".format(file_name, Logit_Save))

    if os.path.exists(Logit_Save + '/BLogits.npy') \
        and os.path.exists(Logit_Save + '/BLabel.npy') \
            and os.path.exists(Logit_Save + '/CTarget.npy') \
                and os.path.exists(Logit_Save + '/CLogits.npy') \
                    and os.path.exists(Logit_Save + '/Size.npy'):
        print('Load Logits from: {}'.format(Logit_Save))
        Size = np.load(Logit_Save + '/Size.npy')
        BLogits = np.load(Logit_Save+'/BLogits.npy')
        BLabel = np.load(Logit_Save+'/BLabel.npy')
        CTarget = np.load(Logit_Save+'/CTarget.npy')
        CLogits = np.load(Logit_Save+'/CLogits.npy')
    
    elif args.retrain:
        print('Generate Logits from source: {}'.format('-'.join([
            args.net_state, args.net_type, args.dataset, 
            args.adv_type, args.adv_norm, args.adv_parameter
        ])))
        
        #! Generate logits for three datasets (AEs, CEs, NEs)
        print('>>> Clean Data...')
        CPFLogits, CPLogits, CLabel, CPF_Targets, CP_Targets, CCrtFlag = GetLogitValues(clean_data, label, PDmodel, FDmodel, args, Natural=True)
        INDEX = np.argmax(CPFLogits[0, :, :], 1) - np.argmax(CPFLogits[1, :, :],1)
        np.sum(INDEX)

        print('>>> Noisy Data...')
        NsyPFLogits, NsyPLogits, NsyLabel, NsyPF_Targets, NsyP_Targets, NCrtFlag = GetLogitValues(noisy_data, label, PDmodel, FDmodel, args, Natural=True)
        INDEX = np.argmax(NsyPFLogits[0, :, :], 1)-np.argmax(NsyPFLogits[1, :, :], 1)
        np.sum(INDEX)

        print('>>> Adversarial Data...')
        AdvPFLogits, AdvLabel, AdvPF_Targets, ACrtFlag = GetLogitValues(adv_data, label, PDmodel, FDmodel, args, Natural=False)
        np.size(np.nonzero(np.argmax(AdvPFLogits[0, :, :], 1) - label.numpy())) / label.shape[0]
        
        
        Size = [CPFLogits.shape[1], CPLogits.shape[1], AdvPFLogits.shape[1]]
        BLogits = np.concatenate((CPFLogits, CPLogits, NsyPFLogits, NsyPLogits, AdvPFLogits), axis=1)
        BLabel = np.concatenate((CLabel, NsyLabel, AdvLabel), axis=0)
        CTarget = np.concatenate((CPF_Targets, CP_Targets), axis=0)
        CLogits = np.concatenate((CPFLogits, CPLogits,), axis=1)
        print('\nSave Logits in: {}'.format(Logit_Save))
        np.save(Logit_Save + '/Size.npy', Size)
        np.save(Logit_Save + '/BLogits.npy', BLogits)
        np.save(Logit_Save+'/BLabel.npy',BLabel)
        np.save(Logit_Save+'/CTarget.npy',CTarget)
        np.save(Logit_Save+'/CLogits.npy',CLogits)

    else:
        print('Generate Logits from source: {}'.format(file_name))

        #! Generate logits for three datasets (AEs, CEs, NEs)
        print('>>> Clean Data...')
        CPFLogits, CPLogits, CLabel, CPF_Targets, CP_Targets, CCrtFlag = GetLogitValues(clean_data, label, PDmodel, FDmodel, args, Natural=True)
        
        print('>>> Noisy Data...')
        NsyPFLogits, NsyPLogits, NsyLabel, NsyPF_Targets, NsyP_Targets, NCrtFlag = GetLogitValues(noisy_data, label, PDmodel, FDmodel, args, Natural=True)
        
        print('>>> Adversarial Data...')
        AdvPFLogits, AdvLabel, AdvPF_Targets, ACrtFlag = GetLogitValues(adv_data, label, PDmodel, FDmodel, args, Natural=False)

        Size = [CPFLogits.shape[1], CPLogits.shape[1], AdvPFLogits.shape[1]]
        BLogits = np.concatenate((CPFLogits, CPLogits, NsyPFLogits, NsyPLogits, AdvPFLogits), axis=1)
        BLabel = np.concatenate((CLabel, NsyLabel, AdvLabel), axis=0)
        CTarget = np.concatenate((CPF_Targets, CP_Targets), axis=0)
        CLogits = np.concatenate((CPFLogits, CPLogits,), axis=1)
        print('\nSave Logits in: {}'.format(Logit_Save))
        np.save(Logit_Save + '/Size.npy', Size)
        np.save(Logit_Save + '/BLogits.npy', BLogits)
        np.save(Logit_Save + '/BLabel.npy',BLabel)
        np.save(Logit_Save + '/CTarget.npy',CTarget)
        np.save(Logit_Save + '/CLogits.npy',CLogits)
    
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

    return BLogits, BLabel, CTarget, CLogits, TrainIndex, TestIndex


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

    global file_name

    #! Load model and dataset
    get_dataset_path(args)

    #! make dirs to save sid features
    if not os.path.exists(args.outf) and args.outf is not None:
        os.mkdir(args.outf)
    print("=> Make dirs ({}) to save sid features".format(args.outf))

    #! Load Pixel/Frequency-trained model
    PDmodel, FDmodel = get_model(args)
    PDmodel.cuda().eval()
    FDmodel.cuda().eval()
    print("=> Pixel/Frequency-trained model is OK...")
    
    #! get detection model
    DualMOdel = DualCIFAR4(num_classes=args.num_classes, C_Number=3)
    DualMOdel.cuda()
    for p in DualMOdel.parameters():
        p.requires_grad_()
    
    #! Define optimizer, scheduler and criterion
    optimizer = torch.optim.SGD(
            DualMOdel.parameters(), args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = None

    #! Generate detected dataset
    train_dataset, test_dataset = get_data(args)
    adv_data_train, clean_data_train, noisy_data_train, label_train = train_dataset
    adv_data_test, clean_data_test, noisy_data_test, label_test = test_dataset

    #! Train detector stage
    victim_file = "sid:" + "-".join([
        args.victim_model[0], args.victim_model[1], args.dataset, 
        args.adv_type_train, args.adv_norm_train, str(args.adv_parameter_train)
    ])
    file_name = victim_file
    print(f"=> {file_name}")
   
    print("=> Detector Training Phase")
    BLogits, BLabel, _, _, TrainIndex, ValIndex = GetStackLogitValues(
        clean_data_train, noisy_data_train, adv_data_train, label_train, PDmodel, FDmodel, args
    )
    for x in [BLogits, BLabel, TrainIndex, ValIndex]: print(x.shape)

    if not os.path.exists(Logit_Save+"/checkpoint.best.pt"):
        train_pipeline(args, BLogits, BLabel, DualMOdel, criterion, optimizer, scheduler, TrainIndex, ValIndex)

    #! Get the performance for detector
    kwargs = torch.load(Logit_Save+"/checkpoint.best.pt")
    auroc, acc, ckpt = kwargs['auroc'], kwargs['acc'], kwargs['model']
    print("=> The detector performance for {}: AUROC:{:.4f}% || ACC:{:.4f}%".format(
        file_name, auroc*100, acc
    ))

    #! Test detector stage
    hacker_file = "sid:" + "-".join([
        args.hacker_model[0], args.hacker_model[1], args.dataset, 
        args.adv_type_test, args.adv_norm_test, str(args.adv_parameter_test)
    ])
    file_name = hacker_file
    print(f"=> {file_name}")
    DualMOdel.load_state_dict(ckpt)
    print("=> Load well-trained dual model")
    
    args.TSR, args.TB = 0.0, 640
    BLogits, BLabel, _, _, TrainIndex, TestIndex = GetStackLogitValues(
        clean_data_test, noisy_data_test, adv_data_test, label_test, PDmodel, FDmodel, args
    )
    auroc, acc = TestDetector(args, BLogits, BLabel, DualMOdel, TestIndex)

    #! Record
    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(args.outf+"/readme.txt", "a") as files:
        files.write("="*13 + "Records " + "="*13 + "\n")
        files.write("=> recording time: {}\n".format(time_now))
        files.write("=> {}\n".format(victim_file))
        files.write("=> {}\n".format(hacker_file))
        files.write("=> Best SID-Detector: ROC-AUC score:{:.4f} | ACC score:{:.4f}\n".format(auroc*100, acc))
    files.close()
