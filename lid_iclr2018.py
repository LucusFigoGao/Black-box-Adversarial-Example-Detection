# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   lid_iclr2018.py
    Time:        2022/11/18 13:23:15
    Editor:      Figo
-----------------------------------
'''

"""
    To get LID score: 
    :: Stage one: Get image from CEs, AEs, NEs
    :: Stage two: Calculate LID score in feature space. (original model, specific layer)
    :: Stage three: Train a binary classifier, default LR model.
"""

import os
import time
import torch
import numpy as np

from torch.autograd import Variable
from scipy.spatial.distance import cdist
from default import RESUME_DEFAULT
from utils.helps import cal_accuracy
from utils.helps_load import get_dataset_path
from utils.helps_detect import Container, detect
from utils.helps_data import load_dataset, load_data_from_file
from robustness.model_utils import make_and_restore_model


# lid of a batch of query points X
OVERLAP_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 90]

def mle_batch(data, batch, k):
    '''
    commpute lid score using data & batch with k-neighbors
    return: a: computed LID score
    '''
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def get_LID(model, test_clean_data, test_adv_data, test_noisy_data, test_label, num_output):
    '''
    Compute LID score on adversarial samples
    return: LID score
    '''
    model.eval()  
    total = 0
    batch_size = 100
    label_container = Container()
    
    LID, LID_adv, LID_noisy = [], [], []    
    overlap_list = OVERLAP_LIST
    for i in overlap_list:
        LID.append([])
        LID_adv.append([])
        LID_noisy.append([])
        
    for data_index in range(int(np.floor(test_clean_data.shape[0]/batch_size))):
        data = test_clean_data[total : total + batch_size].cuda()
        adv_data = test_adv_data[total : total + batch_size].cuda()
        noisy_data = test_noisy_data[total : total + batch_size].cuda()
        target = test_label[total : total + batch_size].cuda()

        total += batch_size
        data, target = Variable(data, volatile=True), Variable(target)
        
        output, out_features = model.feature_list(data)
        X_act = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].shape[0], out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2).cpu().data.numpy()
            X_act.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].shape[0], -1)))
        equal_flag_clean, _ = cal_accuracy(output, target)
        
        output, out_features = model.feature_list(Variable(adv_data, volatile=True))
        X_act_adv = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].shape[0], out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2).cpu().data.numpy()
            X_act_adv.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].shape[0], -1)))
        equal_flag_adver, _ = cal_accuracy(output, target)

        output, out_features = model.feature_list(Variable(noisy_data, volatile=True))
        X_act_noisy = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].shape[0], out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2).cpu().data.numpy()
            X_act_noisy.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].shape[0], -1)))
        equal_flag_noise, _ = cal_accuracy(output, target)
        
        print("==> Batch[{}]: Finished extracting the features from model...".format(data_index))
        
        # LID
        list_counter = 0 
        for overlap in overlap_list:
            LID_list = []
            LID_adv_list = []
            LID_noisy_list = []

            for j in range(num_output):
                lid_score = mle_batch(X_act[j], X_act[j], k = overlap)
                lid_score = lid_score.reshape((lid_score.shape[0], -1))
                lid_adv_score = mle_batch(X_act[j], X_act_adv[j], k = overlap)
                lid_adv_score = lid_adv_score.reshape((lid_adv_score.shape[0], -1))
                lid_noisy_score = mle_batch(X_act[j], X_act_noisy[j], k = overlap)
                lid_noisy_score = lid_noisy_score.reshape((lid_noisy_score.shape[0], -1))
                
                LID_list.append(lid_score)
                LID_adv_list.append(lid_adv_score)
                LID_noisy_list.append(lid_noisy_score)

            LID_concat = LID_list[0]
            LID_adv_concat = LID_adv_list[0]
            LID_noisy_concat = LID_noisy_list[0]

            for i in range(1, num_output):
                LID_concat = np.concatenate((LID_concat, LID_list[i]), axis=1)
                LID_adv_concat = np.concatenate((LID_adv_concat, LID_adv_list[i]), axis=1)
                LID_noisy_concat = np.concatenate((LID_noisy_concat, LID_noisy_list[i]), axis=1)
                
            LID[list_counter].extend(LID_concat)
            LID_adv[list_counter].extend(LID_adv_concat)
            LID_noisy[list_counter].extend(LID_noisy_concat)
            list_counter += 1
            print(">>> ({}) Finished calculating the lid scores...".format(overlap))

        label_container.update([equal_flag_clean, equal_flag_adver, equal_flag_noise])

    correct_clean, correct_adver, correct_noisy = label_container.values[0].sum(),\
                                                  label_container.values[1].sum(),\
                                                  label_container.values[2].sum()
    total_number = label_container.length
    print("=> Test accuracy of clean data is:{:.2f}%".format(100. * correct_clean / total_number))
    print("=> Test accuracy of adversarial data is:{:.2f}%".format(100. * correct_adver / total_number))
    print("=> Test accuracy of noisy data is:{:.2f}%".format(100. * correct_noisy / total_number))

    return LID, LID_adv, LID_noisy


def merge_and_generate_labels(X_pos, X_neg):
    """
        :: merge positve and nagative artifact and generate labels
        :: return: X: merged samples, 2D ndarray
                   y: generated labels (0/1): 2D ndarray same size as X
        :: #! this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


def load_characteristics(file_name):
    """
    Load the calculated scores
    return: data and label of input score
    """
    X, Y = None, None
    
    data = np.load(file_name)
    
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1] # labels only need to load once
         
    return X, Y

# function for applying LID detector

class DetectionDataset:
    def __init__(self, data, label) -> None:
        self.data = data
        self.label = label


def load_model_and_dataset(args):
    #! Load train loader from clean data
    dataset = load_dataset(args.dataset, "default")
    train_loader, _ = dataset.make_loaders(workers=0, batch_size=args.batch_size, data_aug=False)
    print("==> Train loader is OK, got {} train images".format(len(train_loader.dataset)))

    #! Load original victim model
    print("=> Load well trained classifier model...")
    ckpt_name = '-'.join([args.victim_model[0], args.victim_model[1], args.dataset])
    model, _ = make_and_restore_model(arch=args.victim_model[1], dataset=dataset, resume_path=RESUME_DEFAULT[ckpt_name])
    print("=> {} is OK...".format(ckpt_name))
    
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
    return model, (train_ds, test_ds), train_loader


def detection_stage(args):
    print('evaluate the LID estimator')
    score_list = ['lid-{}'.format(idx) for idx in OVERLAP_LIST]
    print('load train data: {}'.format(args.outf))

    best_auroc_score, best_lid_name, acc_score, best_lr = 0, None, 0, None
    for score in score_list:
        files = os.path.join(args.outf, "{}:{}.npy".format(score, file_name))
        print("=> name of LID score file is:{}".format(files))

        total_X, total_Y = load_characteristics(files)
        dataset = DetectionDataset(total_X, total_Y)
        lr, auc_score, acc, scaler = detect(dataset)

        if auc_score >= best_auroc_score:
            best_auroc_score, acc_score = auc_score, acc
            best_lid_name, best_lr = score, lr
        
        print("Best LID-Detector [{}]: ROC-AUC score:{} | ACC score:{}".format(best_lid_name, best_auroc_score, acc_score))
    return best_lr, best_lid_name


def main(args):
    
    #! Load model and dataset
    get_dataset_path(args)

    #! make dirs to save lid features
    if not os.path.exists(args.outf) and args.outf is not None:
        os.mkdir(args.outf)
    print("=> Make dirs ({}) to save lid features".format(args.outf))

    model, dataset, _ = load_model_and_dataset(args)
    model.cuda().eval()

    #! dataset
    train_dataset, test_dataset = dataset
    adv_data_train, clean_data_train, noisy_data_train, label_train = train_dataset
    adv_data_test, clean_data_test, noisy_data_test, label_test = test_dataset

    def set_information_for_feature_extaction():
        temp_x = Variable(torch.rand(2, 3, 32, 32)).cuda()
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        return feature_list, num_output
    
    def lid_save(LID, LID_adv, LID_noisy, file_name):
        list_counter = 0
        for overlap in OVERLAP_LIST:
            Save_LID = np.asarray(LID[list_counter], dtype=np.float32)
            Save_LID_adv = np.asarray(LID_adv[list_counter], dtype=np.float32)
            Save_LID_noisy = np.asarray(LID_noisy[list_counter], dtype=np.float32)
            Save_LID_pos = np.concatenate((Save_LID, Save_LID_noisy))
            LID_data, LID_labels = merge_and_generate_labels(Save_LID_adv, Save_LID_pos)
            
            #! LID socre saving
            files = os.path.join(args.outf, "lid-{}:{}.npy".format(overlap, file_name))
            LID_data = np.concatenate((LID_data, LID_labels), axis=1)
            np.save(files, LID_data)
            list_counter += 1
    
    
    global file_name
    victim_file = "-".join([
        args.victim_model[0], args.victim_model[1], args.dataset, 
        args.adv_type_train, args.adv_norm_train, str(args.adv_parameter_train)
    ])
    file_name = victim_file
    print("=> name of LID score file is:{}".format(file_name))
    _, num_output = set_information_for_feature_extaction()
    
    #! Train Lid detector --- feature extraction
    for overlap in OVERLAP_LIST:
        check_path = os.path.join(args.outf, "lid-{}:{}.npy".format(overlap, file_name))
        if os.path.exists(check_path): continue
        else:
            LID, LID_adv, LID_noisy = get_LID(model, clean_data_train, adv_data_train, noisy_data_train, label_train, num_output)
            lid_save(LID, LID_adv, LID_noisy, file_name)
            break
    
    #! Detection inference stage:
    lr_predictor, overlap = detection_stage(args)
    overlap_for_test = int(overlap.split('-')[-1])
    print(f"=> Best Lid detector for {victim_file} is:{overlap_for_test}")


    hacker_file = "-".join([
        args.hacker_model[0], args.hacker_model[1], args.dataset, 
        args.adv_type_test, args.adv_norm_test, str(args.adv_parameter_test)
    ])
    file_name = hacker_file
    print("=> name of LID score file is:{}".format(file_name))

    #! Test Lid detector --- detect hacker adversarial example
    for overlap in OVERLAP_LIST:
        check_path = os.path.join(args.outf, "lid-{}:{}.npy".format(overlap, file_name))
        if os.path.exists(check_path): continue
        else:
            LID, LID_adv, LID_noisy = get_LID(model, clean_data_test, adv_data_test, noisy_data_test, label_test, num_output)
            lid_save(LID, LID_adv, LID_noisy, file_name)
            break

    files = os.path.join(args.outf, "lid-{}:{}.npy".format(overlap_for_test, file_name))
    total_X, total_Y = load_characteristics(files)
    dataset = DetectionDataset(total_X, total_Y)
    _, auc_score, acc, _ = detect(dataset, lr_predictor, percentage=0.0)

    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(args.outf+"/readme.txt", "a") as files:
        files.write("="*13 + "Records " + "="*13 + "\n")
        files.write("=> recording time: {}\n".format(time_now))
        files.write("=> Victim model: {}\n".format(victim_file))
        files.write("=> Hacker model: {}\n".format(hacker_file))
        files.write("=> Best LID-Detector [{}]: ROC-AUC score:{:.4f} | ACC score:{:.4f}\n".format(overlap, auc_score, acc))
    files.close()
