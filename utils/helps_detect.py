# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   helps_detect.py
    Time:        2022/10/27 10:43:47
    Editor:      Figo
-----------------------------------
'''

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score


def block_split(X, Y, percentange=0.7):
    """
    Split the data into 70% for training and 30% for testing in a block size of 100.
    :param X: 
    :param Y: 
    :return: 
    """
    print(f"Isolated split {percentange*100}%, {(1-percentange)*100}% for training and testing")
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]

    if partition <= 150:
        num_train = int(partition*percentange)
    else:
        num_train = int(partition*percentange/100) * 100

    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test


def train_lr(X, y):
    """
    TODO
    :param X: the data samples
    :param y: the labels
    :return:
    """
    lr = LogisticRegressionCV(n_jobs=-1, max_iter=1000).fit(X, y)
    return lr


def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score


class Container(nn.Module):
    def __init__(self):
        super(Container, self).__init__()
        self.reset()
    
    def reset(self):
        self.length = 0
        self.values = []
    
    def update(self, vals):
        if self.length == 0:
            self.values.extend(vals)
        else:
            for idx, (v, V) in enumerate(zip(vals, self.values)):
                self.values[idx] = torch.cat([V, v], dim=0)
        self.length += len(vals[-1])


class DetectionDataset(nn.Module):
    """
        :: accepted detection feature type: numpy.ndarray
    """
    def __init__(self, X_pos, X_neg):
        super(DetectionDataset, self).__init__()
        X_pos = np.asarray(X_pos, dtype=np.float32)
        print("X_pos: ", X_pos.shape)
        X_pos = X_pos.reshape((X_pos.shape[0], -1))

        X_neg = np.asarray(X_neg, dtype=np.float32)
        print("X_neg: ", X_neg.shape)
        X_neg = X_neg.reshape((X_neg.shape[0], -1))
        
        self.data = np.concatenate((X_pos, X_neg))
        label = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
        self.label = label.reshape((self.data.shape[0], 1))
        
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index], dtype=torch.float32),\
                 torch.tensor(self.label[index], dtype=torch.long)

    def __len__(self):
        return len(self.label)


def detect(dataset, lr=None, percentage=0.7):
    X, Y = dataset.data, dataset.label

    # standarization
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)

    # test attack is the same as training attack
    X_train, Y_train, X_test, Y_test = block_split(X, Y, percentage)
    if lr is None:
        lr = train_lr(X_train, Y_train)

    ## Evaluate detector
    y_pred = lr.predict_proba(X_test)[:, 1]
    y_label_pred = lr.predict(X_test)
    
    # AUC
    _, _, auc_score = compute_roc(Y_test, y_pred, plot=False)
    precision = precision_score(Y_test, y_label_pred)
    recall = recall_score(Y_test, y_label_pred)

    y_label_pred = lr.predict(X_test)
    acc = accuracy_score(Y_test, y_label_pred)
    print('Detector ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (auc_score, acc, precision, recall))

    return lr, auc_score, acc, scaler
