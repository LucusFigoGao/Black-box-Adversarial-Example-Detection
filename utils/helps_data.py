# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   helps_data.py
    Time:        2022/10/24 15:18:32
    Editor:      Figo
-----------------------------------
'''

import os
import torch
import numpy as np
import torch.nn as nn
import PIL.Image as Image
import torchvision as tv
import robustness.datasets as datasets
from default import DATASET_DEFAULT

DATA_SUFFIX = ['pt', 'pth', 'npy']


def load_dataset(name, types="default", transform_train=None, transform_test=None):

    data_list = ['cifar', 'mini_cifar', 'cifar10', 'cifar100', 'cifar10-c', 'cifar100-c', \
                'tiny-imagenet200', 'imagenet', 'r-imagenet']
    
    if name not in data_list:
        raise ValueError(f"No dataset {name} found! Expect dataset:{data_list}")
    
    print(f"==> Load dataset {name} from {DATASET_DEFAULT[name]}")
    
    if types == "default":
        if name in ['cifar', 'cifar10']: 
            return datasets.DATASETS['cifar'](DATASET_DEFAULT['cifar10'])
        elif name == "cifar100":
            return datasets.DATASETS['cifar100'](DATASET_DEFAULT['cifar100'])
        elif name == "mini_cifar": 
            return datasets.DATASETS['mini_cifar'](DATASET_DEFAULT['mini_cifar'])
        elif name == "imagenet": 
            return datasets.DATASETS['imagenet'](DATASET_DEFAULT['imagenet'])
        elif name == "r-imagenet": 
            return datasets.DATASETS['restricted_imagenet'](DATASET_DEFAULT['r-imagenet'])
    
    elif types == "custom": 
        if transform_train is None and transform_test is None:
            raise ValueError("Self defining transform for dataset, but got none!")
        kwargs = {
            "transform_train": transform_train, 
            "transform_test": transform_test
        }
        if name in ['cifar', 'cifar10']:
            return datasets.DATASETS['cifar'](DATASET_DEFAULT['cifar10'], **kwargs)
        elif name == "cifar100":
            return datasets.DATASETS['cifar100'](DATASET_DEFAULT['cifar100'], **kwargs)
        elif name == "mini_cifar":
            return datasets.DATASETS['mini_cifar'](DATASET_DEFAULT['mini_cifar'], **kwargs)
        elif name == "imagenet":
            return datasets.DATASETS['imagenet'](DATASET_DEFAULT['imagenet'], **kwargs)
        elif name == "r-imagenet": 
            return datasets.DATASETS['restricted_imagenet'](DATASET_DEFAULT['r-imagenet'], **kwargs)


def easy_load_model_dataset_for_test(model, dataset):    
    from robustness.model_utils import make_and_restore_model

    dataset = load_dataset(dataset, "default")
    model, _ = make_and_restore_model(arch=model, dataset=dataset, parallel=True)
    return model, dataset


def check_suffix(suffix):
    if suffix not in DATA_SUFFIX:
        raise ValueError("Unspported suffix of saving data, expect {}, but got {}".format(DATA_SUFFIX, suffix))


def get_label_root(root):
    dir = os.path.dirname(root)
    image_file = root.split('/')[-1]
    
    if not os.path.isfile(root):
        raise ValueError("No such files {} in {}".format(image_file, dir))
    
    label_file = '-'.join(['label', image_file.split('-')[-1]])
    root = os.path.join(dir, label_file)
    
    if os.path.isfile(root):
        return root
    else:
        raise ValueError("No such files {} in {}".format(label_file, dir))


def load_data_from_file(root, return_label=True):
    suffix = root.split('.')[-1]
    check_suffix(suffix)
    
    image_root = root
    
    if return_label:
        label_root = get_label_root(root)
    
    if suffix in ["pt", "pth"]:
        
        image = torch.load(image_root)
        
        if return_label:
            label = torch.load(label_root)
    
    elif suffix == "npy":
        
        image = np.load(image_root)
        if return_label:
            label = np.load(label_root)
    
    if return_label:
        return image, label
    else: 
        return image


def easy_load_puppet_dataset():
    import torchvision as tv
    dataset = tv.datasets.CIFAR10(DATASET_DEFAULT['cifar10'])
    return dataset


class FreqDataset(nn.Module):
    def __init__(self, dataset, transform):
        super(FreqDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        image, label = self.dataset.data[index], self.dataset.targets[index]
        if image.mean() <= 5.0:
            image = np.uint8(image*255)
        image = Image.fromarray(np.uint8(image))
        image = self.transform(image)
        amplitude, phase = self.get_phase_amplitude(image)
        return (image, amplitude, phase), label
    
    def __len__(self):
        return len(self.dataset)

    def get_phase_amplitude(self, data):
        if len(data.shape) == 3: dims = (1, 2)
        freq_image = torch.fft.fftn(data, dim=dims)
        amplitude = torch.abs(freq_image)
        phase = torch.angle(freq_image)
        return amplitude, phase


def check_dataset_type(dataset):
    if isinstance(dataset, tv.datasets.VisionDataset):
        return dataset


def load_freq_dataset(dataset, only_val=False, transform_train=None, transform_test=None):
    """
        :: if dataset == "str": The root of saving dataset, ['pt', 'pth', 'npy']
        :: if dataset == "dataset.DataSet": The robustness.dataset.DataSet
        :: data: numpy.ndarray, np.uint8, range(0, 255)
        :: label: numpy.ndarray
        :: e.x.1
        >>> _, test_set = load_freq_dataset("data.npy", only_val=True, transform_test=T)

        :: e.x.2
        >>> dataset = load_dataset("cifar", "custom", T1, T2)
        >>> train_set, test_set = load_freq_dataset(dataset, only_val=True, transform_test=T)
    """
    if isinstance(dataset, str):
        print("==> Load data from {}".format(dataset))
        image, label = load_data_from_file(dataset)
        test_set = easy_load_puppet_dataset()
        test_set.data, test_set.targets = np.transpose(image, (0, 2, 3, 1)), label
    
    if isinstance(dataset, datasets.DataSet):
        print("==> Load data from datasets.DataSet")
        train_loader, test_loader = dataset.make_loaders(workers=0, batch_size=100, data_aug=False)
        train_set, test_set = check_dataset_type(train_loader.dataset), check_dataset_type(test_loader.dataset)
    
    if not only_val:
        train_set = FreqDataset(train_set, transform_train)
    test_set = FreqDataset(test_set, transform_test)

    if not only_val:
        return train_set, test_set
    else:
        return None, test_set
