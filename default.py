# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   default.py
    Time:        2022/10/24 21:21:20
    Editor:      Figo
-----------------------------------
'''

DATA_SUFFIX = ['pt', 'pth', 'npy']


ADV_DEFAULT = {
    "fgsm": {"adv_type": "fgsm", "adv_parameter": 0.01, "adv_norm": "l2"}, 
    "bim": {"adv_type": "bim", "adv_parameter": 0.01, "adv_norm": "l2"}, 
    "deepfool": {"adv_type": "deepfool", "adv_parameter": 0.5, "adv_norm": "l2"}, 
    "cw": {"adv_type": "cw", "adv_parameter": 0.5, "adv_norm": "l2"}, 
    "pgd-l2": {
        "adv_type": "pgd", "adv_epsilon": 1.0, "adv_iterations": 5, "adv_step_size": 0.5, 
        "adv_constraint": "2", "adv_targeted": "False", "adv_custom_loss": "None"
        }, 
    "pgd-linf": {
        "adv_type": "pgd", "adv_epsilon": 8, "adv_iterations": 100, "adv_step_size": 2/255, 
        "adv_constraint": "inf", "adv_targeted": "False", "adv_custom_loss": "None"
        }
    }


DATASET_DEFAULT = {
    "cifar": "path/to/cifar10", 
    "cifar100": "path/to/cifar100", 
}


RESUME_DEFAULT = {
    "normal-resnet18-cifar": "./checkpoint/AT:0-cifar-resnet18-S:888/checkpoint.pt.best", 
    "normal-resnet18-cifar100": "./checkpoint/AT:0-cifar100-resnet18-S:666/checkpoint.pt.best", 
    "adver-resnet18-cifar": "./checkpoint/AT:1-cifar-resnet18-S:888/checkpoint.pt.best", 
    "adver-resnet18-cifar100": "./checkpoint/AT:1-cifar100-resnet18-S:777/checkpoint.pt.best", 
    "normal-vgg16-cifar": "./checkpoint/AT:0-cifar-vgg16-S:777/checkpoint.pt.best", 
    "normal-vgg16-cifar100": "./checkpoint/AT:0-cifar100-vgg16-S:777/checkpoint.pt.best", 
    "normal-wideresnet28-cifar": './checkpoint/AT:0-cifar-wideresnet28-S:666/checkpoint.pt.best', 
    "normal-wideresnet28-cifar100": './checkpoint/AT:0-cifar100-wideresnet28-S:777/checkpoint.pt.best', 
}


PRD_DEFAULT = {
    "cifar": "./checkpoint/PixelVAE-512/checkpoints.best.pth", 
    "cifar100": "./checkpoint/PixelVAE-512/checkpoints.best.pth", 
}


FRD_RESUME_DEFAULT = {
    "cifar": {
        "phase": "./checkpoint/phase-512/checkpoints.best.pth", 
        "amplitude": "./checkpoint/amplitude-512/checkpoints.best.pth"
    }, 
    "cifar100": {
        "phase": "./checkpoint/phase-512/checkpoints.best.pth", 
        "amplitude": "./checkpoint/amplitude-512/checkpoints.best.pth"
    }
}


SID_DUAL_RESUME = {
    "dual-resnet18-cifar": "./checkpoint/sid-dual: AT:0-cifar-resnet18-S:888/checkpoint.pt.best", 
    "dual-resnet18-cifar100": "./checkpoint/sid-dual: AT:0-cifar100-resnet18-S:777/checkpoint.pt.best"
}

"""
    :: lr, weight_decay, momentum, epoch
"""
SID_CONFIG = {
    "pgd": [5e-3, 5e-3, 0.8, 100], 
    "bim": [1e-2, 5e-3, 0.9, 100], 
    "fgsm": [1e-2, 5e-3, 0.9, 100],  
    "deepfool": [1e-2, 5e-3, 0.9, 100], 
    "cw": [1e-2, 5e-3, 0.8, 100], 
}


PRD_CONFIG = {
    "pgd": [1e-2, 5e-4, 0.9, 100], 
    "bim": [1e-2, 5e-3, 0.9, 100], 
    "fgsm": [1e-2, 5e-3, 0.9, 100], 
    "deepfool": [1e-2, 5e-4, 0.9, 100], 
    "cw": [1e-2, 5e-3, 0.9, 100], 
}


FRD_CONFIG = {
    "pgd": [5e-2, 5e-4, 0.9, 100], 
    "bim": [1e-2, 5e-3, 0.9, 100], 
    "fgsm": [1e-2, 5e-3, 0.9, 100], 
    "deepfool": [5e-2, 5e-4, 0.9, 100], 
    "cw": [1e-2, 5e-3, 0.9, 100], 
}


FRDP_CONFIG = {
    "pgd": [5e-2, 5e-4, 0.9, 100], 
    "bim": [1e-2, 5e-3, 0.9, 100], 
    "fgsm": [1e-2, 5e-3, 0.9, 100], 
    "deepfool": [5e-2, 5e-4, 0.9, 100], 
    "cw": [1e-2, 5e-3, 0.9, 100], 
}


FRDA_CONFIG = {
    "pgd": [5e-2, 5e-4, 0.9, 100], 
    "bim": [1e-2, 5e-3, 0.9, 100], 
    "fgsm": [1e-2, 5e-3, 0.9, 100], 
    "deepfool": [5e-2, 5e-4, 0.9, 100], 
    "cw": [1e-2, 5e-3, 0.9, 100],  
}