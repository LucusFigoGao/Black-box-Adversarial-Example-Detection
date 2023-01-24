# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   md_nips2018.py
    Time:        2022/11/19 14:29:07
    Editor:      Figo
-----------------------------------
'''

"""
    To get MD score: 
    :: Stage one: Get image from CEs, AEs, NEs
    :: Stage two: Calculate MD score in feature space. (original model, specific layer)
    :: Stage three: Train a binary classifier, default LR model.
"""

import os
import time
import torch
import numpy as np
from torch.autograd import Variable
from utils.helps_load import get_dataset_path
from utils.helps_detect import DetectionDataset, detect
from lid_iclr2018 import merge_and_generate_labels, load_characteristics, \
                         DetectionDataset, load_model_and_dataset


MAHALANOBIS_LIST = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]


def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


def get_Mahalanobis_score_adv(model, test_data, test_label, num_classes, net_type, sample_mean, precision, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    batch_size = 100
    total = 0
    print("=> layer index:{}".format(layer_index))
    
    for data_index in range(int(np.floor(test_data.size(0)/batch_size))):
        target = test_label[total : total + batch_size].cuda()
        data = test_data[total : total + batch_size].cuda()
        total += batch_size
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type in ['resnet18', 'wideresnet28', 'ensemble', 'vgg16']:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        tempInputs = torch.add(data.data, -magnitude, gradient)
 
        noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        
    return Mahalanobis

# function for applying MD detector

def detection_stage(args):
    print('evaluate the MD estimator')
    score_list = ['md-{}'.format(str(idx)) for idx in MAHALANOBIS_LIST]

    print('load train data: {}'.format(args.outf))

    best_auroc_score, best_md_name, acc_score, best_lr = 0, None, 0, None
    for score in score_list:
        files = os.path.join(args.outf, "{}:{}.npy".format(score, file_name))
        print("=> name of MD score file is:{}".format(files))

        total_X, total_Y = load_characteristics(files)
        dataset = DetectionDataset(total_X, total_Y)
        lr, auc_score, acc, scaler = detect(dataset)

        if auc_score >= best_auroc_score:
            best_auroc_score, acc_score = auc_score, acc
            best_md_name, best_lr = score, lr
        
        print("Best MD-Detector [{}]: ROC-AUC score:{} | ACC score:{}".format(best_md_name, best_auroc_score, acc_score))
    
    return best_lr, best_md_name


def main(args):
    
    global file_name

    #! Load model and dataset
    get_dataset_path(args)

    #! make dirs to save md features
    if not os.path.exists(args.outf) and args.outf is not None:
        os.mkdir(args.outf)
    print("=> Make dirs ({}) to save md features".format(args.outf))

    #! Load model and dataset
    model, dataset, train_loader = load_model_and_dataset(args)
    model.cuda().eval()
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
    
    def Mahalanobis_save(adv_data, clean_data, noisy_data, label):
        for magnitude in MAHALANOBIS_LIST:
            print('\nNoise: ' + str(magnitude))
            
            for i in range(num_output):
                M_in = get_Mahalanobis_score_adv(
                    model, clean_data, label, args.num_classes,  
                    args.victim_model[1], sample_mean, precision, i, magnitude
                )
                M_in = np.asarray(M_in, dtype=np.float32)
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1)) if i == 0 else \
                                np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            print(">>> Finish calculating clean data")
            
            for i in range(num_output):
                M_out = get_Mahalanobis_score_adv(
                    model, adv_data, label, args.num_classes, 
                    args.victim_model[1], sample_mean, precision, i, magnitude
                )
                M_out = np.asarray(M_out, dtype=np.float32)
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1)) if i == 0 else \
                                np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
            print(">>> Finish calculating adversarial data")
            
            for i in range(num_output):
                M_noisy = get_Mahalanobis_score_adv(
                    model, noisy_data, label, args.num_classes, 
                    args.victim_model[1], sample_mean, precision, i, magnitude
                )
                M_noisy = np.asarray(M_noisy, dtype=np.float32)
                Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1)) if i == 0 else \
                                    np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)      
            print(">>> Finish calculating noisy data")      
            
            Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            Mahalanobis_noisy = np.asarray(Mahalanobis_noisy, dtype=np.float32)
            Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))
            Mahalanobis_data, Mahalanobis_labels = merge_and_generate_labels(Mahalanobis_out, Mahalanobis_pos)
            
            files = os.path.join(args.outf, "md-{}:{}.npy".format(magnitude, file_name))
            Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            np.save(files, Mahalanobis_data)
    
    #! Get sample mean and covariance
    print("=> Get sample mean and covariance...")
    feature_list, num_output = set_information_for_feature_extaction()
    if not os.path.exists(args.outf+'/sample_mean.pt'):
        sample_mean, precision = sample_estimator(model, args.num_classes, feature_list, train_loader)
        torch.save([sample_mean, precision], args.outf+'/sample_mean.pt')
    else:
        sample_mean, precision = torch.load(args.outf+'/sample_mean.pt')
    print("=> Got sample mean and covariance from clean {} train loader...".format(args.dataset))
    for m, c in zip(sample_mean, precision):
        print("=> The shape of mean value is:{}, the shape of covariance value is:{}".format(m.shape, c.shape))

    
    #! Train Md detector --- feature extraction
    victim_file = "-".join([
        args.victim_model[0], args.victim_model[1], args.dataset, 
        args.adv_type_train, args.adv_norm_train, str(args.adv_parameter_train)
    ])
    file_name = victim_file
    print(f"=> Get Mahalanobis scores for victim:{file_name}...")
    
    for magnitude in MAHALANOBIS_LIST:
        check_path = os.path.join(args.outf, "md-{}:{}.npy".format(magnitude, file_name))
        if os.path.exists(check_path): continue
        else:
            Mahalanobis_save(adv_data_train, clean_data_train, noisy_data_train, label_train)
            break
    
    #! Detection inference stage:
    lr_predictor, overlap = detection_stage(args)
    overlap_for_test = float(overlap.split('-')[-1])
    print(f"=> Best MD detector for {victim_file} is:{overlap_for_test}")

    #! Test Lid detector --- detect hacker adversarial example
    hacker_file = "-".join([
        args.hacker_model[0], args.hacker_model[1], args.dataset, 
        args.adv_type_test, args.adv_norm_test, str(args.adv_parameter_test)
    ])
    file_name = hacker_file
    print(f"=> Get Mahalanobis scores for victim:{file_name}...")

    for magnitude in MAHALANOBIS_LIST:
        check_path = os.path.join(args.outf, "md-{}:{}.npy".format(magnitude, file_name))
        if os.path.exists(check_path): continue
        else:
            Mahalanobis_save(adv_data_test, clean_data_test, noisy_data_test, label_test)
            break
    
    files = os.path.join(args.outf, "md-{}:{}.npy".format(overlap_for_test, file_name))
    total_X, total_Y = load_characteristics(files)
    dataset = DetectionDataset(total_X, total_Y)
    _, auc_score, acc, _ = detect(dataset, lr_predictor, percentage=0.0)

    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(args.outf+"/readme.txt", "a") as files:
        files.write("="*13 + "Records " + "="*13 + "\n")
        files.write("=> recording time: {}\n".format(time_now))
        files.write("=> Victim model: {}\n".format(victim_file))
        files.write("=> Hacker model: {}\n".format(hacker_file))
        files.write("=> Best MD-Detector [{}]: ROC-AUC score:{:.4f} | ACC score:{:.4f}\n".format(overlap, auc_score, acc))
    files.close()
