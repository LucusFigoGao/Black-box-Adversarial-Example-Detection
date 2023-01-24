# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   helps.py
    Time:        2022/10/24 15:19:47
    Editor:      Figo
-----------------------------------
'''

import os
import time
import math
import torch
import warnings
import numpy as np
import random as rd
import torch.nn as nn

from tqdm import tqdm
from robustness import defaults
from argparse import ArgumentParser
from utils.helps_detect import Container
from torchmetrics.functional import precision_recall, f1_score


def set_random_seed(seed):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


# compute the accuracy
def cal_accuracy(output, label):
    correct = 0
    pred = output.data.max(1)[1]
    equal_flag = pred.eq(label.data).cpu()
    correct += equal_flag.sum()
    return equal_flag.long(), correct


def cal_score(output, target):
    pred = output.data.max(1)[1]
    batch_accuracy = pred.eq(target.data).sum() / output.shape[0]
    batch_precision, batch_recall = precision_recall(
        preds=pred, target=target, average="weighted", num_classes=10
    )
    batch_f1_score = f1_score(preds=pred, target=target, average="weighted", num_classes=10)
    return batch_accuracy, batch_precision, batch_recall, batch_f1_score


def set_args():
    parser = ArgumentParser()
    parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
    return parser


def RGB2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def check_kwargs(kwargs):
    if isinstance(kwargs, dict):
        for keys, values in kwargs.items():
            if values == 'None':
                kwargs[keys] = None
            elif values == 'True':
                kwargs[keys] = True
            elif values == 'False':
                kwargs[keys] = False
    return kwargs


def get_mask_matrix(size, radii1: int, radii2: int, order='l1'):

    """ ----- Expect the activate shape is (H x W) ------ """

    height, width = size if isinstance(size, tuple) else (size, size)
    
    """get the centralization"""
    crow, ccol = int(height / 2), int(width / 2)
    centralization = [crow, ccol]
    mask_matrix1, mask_matrix2 = np.zeros((height, width), dtype=np.int64), np.zeros((height, width), dtype=np.int64)
    xaxis, yaxis = np.ogrid[:height, :width]

    if order == "l1":
        mask_area = abs(xaxis - centralization[0]) + abs(yaxis - centralization[1]) >= radii1
        mask_matrix1[mask_area] = 1
        mask_area = abs(xaxis - centralization[0]) + abs(yaxis - centralization[1]) <= radii2
        mask_matrix2[mask_area] = 1
    
    if order == "l2":
        mask_area = abs(xaxis - centralization[0])**2 + abs(yaxis - centralization[1])**2 >= radii1**2
        mask_matrix1[mask_area] = 1
        mask_area = abs(xaxis - centralization[0])**2 + abs(yaxis - centralization[1])**2 <= radii2**2
        mask_matrix2[mask_area] = 1
    
    if order == 'square':
        mask_matrix1[crow-radii1: crow+radii1, ccol-radii1: ccol+radii1] = 1
        mask_matrix2[crow-radii2: crow+radii2, ccol-radii2: ccol+radii2] = 1
        mask_area = mask_matrix2 - mask_matrix1
        return mask_area
    
    mask = mask_matrix1 + mask_matrix2 - 1
    mask[crow:, :] = 0
    mask = mask + np.flipud(mask)
    mask[:, ccol:] = 0
    mask = mask + np.fliplr(mask) 
    return mask


def charge_input_type(input: torch.Tensor):
    input_shape = len(input.shape)
    if input_shape == 4: # The input is image batch (B, C, H, W)
        dims = (1, 2, 3)
    elif input_shape == 3: # The input is RGB image (C, H, W)
        dims = (1, 2)
    elif input_shape == 2: # The input is Gray image (H, W)
        dims = (0, 1)
    return dims


def torch_mask(image_batch, mask):
    """ A new way to mask the frequency components (edit.2022.1.25) suit for pytorch
    Args:
        image_batch: The image batch, with shape (B, C, H, W) or (C, H, W), with type torch.Tensor
        mask: The mask matrix, with shape (H, W), with type torch.Tensor
    Return:
        The image batch with desired freqency bands masked
    """
    dims = charge_input_type(image_batch)
    mask = mask.expand(image_batch.shape)  # The mask matrix should match shape
    torch_freq = torch.fft.fftn(image_batch, dim=dims)
    torch_freq = torch.fft.fftshift(torch_freq, dim=dims) * mask
    freq_image = torch.fft.ifftshift(torch_freq, dim=dims)
    freq_image = torch.fft.ifftn(freq_image, dim=dims)
    freq_views = torch.clip(torch.real(freq_image), 0, 1)
    return freq_views


class MaskFreq(nn.Module):
    def __init__(self, radii1, radii2, size=32, order='l2'):
        super(MaskFreq, self).__init__()
        self.order, self.size = order, size
        self.r1, self.r2 = radii1, radii2
        self.mask = self._get_mask()
    
    def _get_mask(self):
        mask = get_mask_matrix(self.size, self.r1, self.r2, self.order)
        return torch.from_numpy(mask)
    
    def forward(self, image):
        image = torch_mask(image, self.mask)
        return image


def val_model(model, loader):
    with torch.no_grad():
        model.eval()
        total_image = len(loader.dataset)
        iterator = tqdm(enumerate(loader), total=len(loader))
        predict, score = [], 0
        for _, (image, label) in iterator:
            image, label = image.cuda(), label.cuda()
            outs = model(image, with_latent=False, with_image=False)
            preds = torch.max(outs, dim=1)[1]
            score += (preds == label).sum().item()
            predict.append(preds.cpu().data)
        print("=> The test accuracy is :{:.3f}%".format(100. * score/total_image))
        return score, torch.cat(predict).numpy()


class SpatialGaussianKernel(nn.Module):
    """ A simple convolutional layer with fixed gaussian kernels, used to smoothen the input """
    def __init__(self, kernel_size, sigma, channels, device):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "kernel_size must be an odd number (for padding), {} given".format(self.kernel_size)
        variance = sigma ** 2.
        x_cord = torch.arange(kernel_size, dtype=torch.float, device=device)  # 1, 2, 3, 4
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)  # 1, 2, 3 \ 1, 2, 3 \ 1, 2, 3
        y_grid = x_grid.t()  # 1, 1, 1 \ 2, 2, 2 \ 3, 3, 3
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        mean_xy = (kernel_size - 1) / 2.
        kernel_2d = (1. / (2. * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean_xy) ** 2., dim=-1) /
            (2 * variance)
        )
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel_3d = kernel_2d.expand(channels, 1, -1, -1)  # expand in channel dimension
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, padding=0,
                              kernel_size=kernel_size, groups=channels, bias=False)
        self.conv.weight.data.copy_(kernel_3d)
        self.conv.to(device)
        self.conv.weight.requires_grad = False
        self.pad = nn.ReflectionPad2d(int((kernel_size - 1) / 2))

    def forward(self, x):
        return self.conv(self.pad(x))


def min_max_norm(x):
    if len(x.shape) >= 3:
        warnings.warn(f"=> This function only support 2 dims matrix calculation, but got {len(x.shape)} dims, \
                        please make sure if you still use this function for min-max-normalization")
    return (x-x.min()) / (x.max()-x.min())


def mean_var_norm(x):
    if len(x.shape) >= 3:
        warnings.warn("=> This function only support 2 dims matrix calculation, but got {len(x.shape)} dims, \
                        please make sure if you still use this function for mean-var-normalization")
    return (x-x.mean()) / x.std()


def mean_var_norm3d(image):
    assert len(image.shape) == 3
    mean = torch.mean(image, dim=(1, 2))
    std = torch.std(image, dim=(1, 2))
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image


def min_max_norm3d(image):
    assert len(image.shape) == 3
    Imin = torch.min(image, dim=(1, 2))
    Imax = torch.max(image, dim=(1, 2))
    image = (image - Imin[:, None, None]) / Imax[:, None, None]
    return image


def writr_down_the_args_each_exp(args, root):
    txt_file = os.path.join(root, "record.txt")
    args_in_dicts = vars(args)
    time_now = time.strftime('%Y-%m-%d %H:%M:%S')

    with open(txt_file, "a") as file:
        file.write("="*10 + " Config " + "="*9 + "\n")
        file.write(time_now + "\n")
        for keys, values in args_in_dicts.items():
            file.write("=> {}:{}\n".format(keys, values))
        file.write("="*8 + " Finish record " + "="*8 + "\n")
    file.close()
    print("=> Finished saving the records...")


def amp_pha_recon(amp, pha):
    recon_image = torch.fft.ifft2(amp * torch.exp(1j * pha))
    recon_image = torch.abs(recon_image)
    return torch.clamp(recon_image, min=0, max=1)


def RGB2gray(rgb):
    """Turn RGB image to Gray image"""
    if rgb.shape[-1] != 3:
        raise ValueError("Expected RGB image, but only got {} channel".format(rgb.shape[-1]))
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_feature_maps(model, loader, locate=None, is_tqdm=False):
    """
        :: loader: iters (image, recon_image, label) each round
        :: return feature maps of specific layer
    """
    from tqdm import tqdm

    Fsaving = Container()
    iterator = tqdm(enumerate(loader), total=len(loader)) if is_tqdm else enumerate(loader)
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

            #! calculate the accuracy
            equal_flag, _ = cal_accuracy(logits, targets)
            recon_equal_flag, _ = cal_accuracy(recon_logits, targets)

            #! save the logits of specific layer
            Fsaving.update([features, equal_flag, recon_features, recon_equal_flag])
        
        # for i in range(4):
        #     print("=> shape of Fsaving.values[{}] is {}".format(i, Fsaving.values[i].shape))
        # print("=> got total {} example-pairs".format(Fsaving.length))

        totoal_correct = Fsaving.values[1].sum()
        recon_total_correct = Fsaving.values[-1].sum()
        total_number = Fsaving.length
        print("=> The test accuracy original image is:{:.2f}%".format(100. * totoal_correct / total_number))
        print("=> The test accuracy reconstruction image is:{:.2f}%".format(100. * recon_total_correct / total_number))
    
    return Fsaving


class FreqPSR(nn.Module):
    def __init__(self, adver_image, clean_image, label, target):
        """
        :: adver_image: torch.utils.data.Dataset
        :: clean_image: torch.utils.data.Dataset
        """
        super(FreqPSR, self).__init__()
        self.adver_image = adver_image
        self.clean_image = clean_image
        self.label, self.target = label, target
    
    def __len__(self):
        assert len(self.clean_image) == len(self.adver_image)
        return len(self.clean_image)
    
    def __getitem__(self, index):
        adv_image, target = self.adver_image[index], self.target[index]
        cle_image, label = self.clean_image[index], self.label[index]
        
        adv_patch, cle_patch = self._Img2Patch(adv_image), self._Img2Patch(cle_image) 
        AEs_amp, AEs_pha = self.get_phase_amplitude(adv_patch)
        AEs_amp, AEs_pha = AEs_amp.reshape(64, 3, 4, 4), AEs_pha.reshape(64, 3, 4, 4)
        CEs_amp, CEs_pha = self.get_phase_amplitude(cle_patch)
        CEs_amp, CEs_pha = CEs_amp.reshape(64, 3, 4, 4), CEs_pha.reshape(64, 3, 4, 4)

        # Patch-Frequency-DeAdv-Image
        PhaseAttack = self.attack_phase(CEs_amp, CEs_pha, AEs_pha)
        AmplitudeAttack = self.attack_amplitude(CEs_amp, CEs_pha, AEs_amp)
        return PhaseAttack, AmplitudeAttack, cle_image, target, label
    
    def _Img2Patch(self, image):
        image_patch = image.view(3, 8, 4, 8, 4)
        patch = image_patch.permute(1, 3, 0, 2, 4)
        return patch
    
    def _Patch2Img(self, patch):
        image_patch = patch.permute(2, 0, 3, 1, 4)
        image = image_patch.reshape(3, 32, 32)
        return image
    
    def get_phase_amplitude(self, image):
        freq_image = torch.fft.fftn(image, dim=(3, 4))
        img_amplitude = torch.abs(freq_image)  
        img_phase = torch.angle(freq_image) 
        return img_amplitude, img_phase
    
    def get_phase_amplitude2(self, image):
        """
        Input:
            :: image: rgb image, with shape of (3, 32, 32)
        Output:
            :: img_amplitude: with shape of (64, 3, 4, 4)
            :: img_phase: with shape of (64, 3, 4, 4)
        """
        freq_image = torch.fft.fftn(image, dim=(1, 2))  # (3, 32, 32)
        freq_image = self._Img2Patch(freq_image)  # (8, 8, 3, 4, 4)
        img_amplitude = torch.abs(freq_image)
        img_phase = torch.angle(freq_image)
        return img_amplitude.reshape(64, 3, 4, 4), img_phase.reshape(64, 3, 4, 4)
    
    def recon_image(self, amplitude, phase):
        re_img = amplitude * torch.exp(1j * phase)
        re_img = torch.abs(torch.fft.ifftn(re_img, dim=(2, 3)))
        return re_img
    
    def _replace(self, index, input, replace):
        locate = torch.index_select(input, 0, torch.LongTensor([index]))
        output = torch.where(input==locate, replace, input)
        return output
    
    def attack_phase(self, CEs_amp, CEs_pha, AEs_pha):
        """CEs_amp, CEs_pha, AEs_pha with the shape of (64, 3, 4, 4)
        """
        PhaseAttack = []  # collecting the phase-attacking image
        for idx in range(64):
            phase = self._replace(idx, CEs_pha, AEs_pha)
            attack_patch = self.recon_image(CEs_amp, phase)  # shape with (64, 3, 4, 4)
            attack_patch = attack_patch.reshape(8, 8, 3, 4, 4)
            attack_image = self._Patch2Img(attack_patch)  # shape with (3, 32, 32)
            PhaseAttack.append(attack_image.unsqueeze(dim=0))
        return torch.cat(PhaseAttack, dim=0)  # shape with (64, 3, 32, 32)
    
    def attack_amplitude(self, CEs_amp, CEs_pha, AEs_amp):
        """CEs_amp, CEs_pha, AEs_amp with the shape of (64, 3, 4, 4)
        """
        AmplitudeAttack = []  # collecting the phase-attacking image
        for idx in range(64):
            amplitude = self._replace(idx, CEs_amp, AEs_amp)
            attack_patch = self.recon_image(amplitude, CEs_pha)  # shape with (64, 3, 4, 4)
            attack_patch = attack_patch.reshape(8, 8, 3, 4, 4)
            attack_image = self._Patch2Img(attack_patch)  # shape with (3, 32, 32)
            AmplitudeAttack.append(attack_image.unsqueeze(dim=0))
        return torch.cat(AmplitudeAttack, dim=0)  # shape with (64, 3, 32, 32)


def cal_FreqPSR_score(dataset, model, threshold=1e-3):
    print("=> Begin calculating frequency PSR score...")
    with torch.no_grad():
        model.eval()
        PhasePSR, PhaseSen, AmplitudePSR, AmplitudeSen, index = [], [], [], [], 0
        for PhaseAttack, AmplitudeAttack, cle_image, target, label in dataset:
            print("=> Image-batch:{}|{} in processing...".format(index, len(dataset)))
            index += 1
            image_batch = torch.cat(
                (
                    PhaseAttack, 
                    AmplitudeAttack, 
                    cle_image.unsqueeze(dim=0)
                ), dim=0
            )  # with shape (128+1, 3, 32, 32)
            output = model(image_batch.cuda(), with_latent=False, with_image=False)
            phase_logit, amplitude_logit, clean_logit = torch.split(output, (64, 64, 1))  # split logit into (64, 64, 1)
            
            def suppression(logit):
                delta = clean_logit[:, label] - logit[:, label]
                replace = torch.ones_like(delta) * threshold
                delta = torch.where(delta <= threshold, replace, delta)
                return delta

            def promotion(logit):
                delta = logit[:, target] - clean_logit[:, target]
                replace = torch.ones_like(delta) * threshold
                delta = torch.where(delta <= threshold, replace, delta)
                return delta
            
            # calculate the psr score for phase and amplitude
            phase_suppression, phase_promotion = suppression(phase_logit), promotion(phase_logit)
            amplitude_suppression, amplitude_promotion = suppression(amplitude_logit), promotion(amplitude_logit)
            PhaseSen.append(phase_suppression+phase_promotion)
            AmplitudeSen.append(amplitude_promotion+amplitude_suppression)
            PhaPSR = torch.log2(phase_promotion / phase_suppression)
            AmpPSR = torch.log2(amplitude_promotion / amplitude_suppression)
            PhaPSR, AmpPSR = PhaPSR.view(8, 8), AmpPSR.view(8, 8)
            if len(dataset) == 1:
                return PhaPSR, AmpPSR
            else:
                PhasePSR.append(PhaPSR)
                AmplitudePSR.append(AmpPSR)
        print("=> Finish calculating frequency PSR score")
    return PhasePSR, PhaseSen, AmplitudePSR, AmplitudeSen


class PSR(nn.Module):
    def __init__(self, adv_image, clean_image, label, target):
        super(PSR, self).__init__()
        self.adv_image, self.target = adv_image, target
        self.cle_image, self.label = clean_image, label
    
    def __len__(self):
        assert self.adv_image.shape == self.cle_image.shape
        assert len(self.target) == len(self.label)
        return len(self.target)
    
    def __getitem__(self, index):
        adv_image, target = self.adv_image[index], self.target[index]
        cle_image, label = self.cle_image[index], self.label[index]
        adv_patch, cle_patch = self._Img2Patch(adv_image), self._Img2Patch(cle_image)
        adv_patch, cle_patch = adv_patch.reshape(64, 3, 4, 4), cle_patch.reshape(64, 3, 4, 4)
        new_adv_image = self.re_attack(adv_patch, cle_patch)
        return new_adv_image, cle_image, target, label

    def _Img2Patch(self, image):
        image_patch = image.view(3, 8, 4, 8, 4)
        patch = image_patch.permute(1, 3, 0, 2, 4)
        return patch
    
    def _Patch2Img(self, patch):
        image_patch = patch.permute(2, 0, 3, 1, 4)
        image = image_patch.reshape(3, 32, 32)
        return image
    
    def _replace(self, index, input, replace):
        locate = torch.index_select(input, 0, torch.LongTensor([index]))
        output = torch.where(input==locate, replace, input)
        return output
    
    def re_attack(self, adv_patch, cle_patch):
        recon_attack_image = []
        for idx in range(64):
            new_attack_patch = self._replace(idx, cle_patch, adv_patch)
            new_attack_patch = new_attack_patch.reshape(8, 8, 3, 4, 4)
            new_attack_image = self._Patch2Img(new_attack_patch)  # shape with (3, 32, 32)
            recon_attack_image.append(new_attack_image.unsqueeze(dim=0))
        return torch.cat(recon_attack_image, dim=0)  # shape with (64, 3, 32, 32)


def cal_PSR_score(dataset, model, threshold=1e-3):
    print("=> Begin calculating perturbation PSR score...")
    with torch.no_grad():
        model.eval()
        PSR_value, Sen_value, index = [], [], 0
        for adv_image, cle_image, target, label in dataset:
            print("=> Image-batch:{}|{} in processing...".format(index, len(dataset)))
            index += 1
            image_batch = torch.cat((adv_image, cle_image.unsqueeze(dim=0)), dim=0)  # with shape (64+1, 3, 32, 32)
            output = model(image_batch.cuda(), with_latent=False, with_image=False)
            adv_logit, cle_logit = torch.split(output, (64, 1))  # split logit into (64, 64, 1)
            
            def suppression(logit):
                delta = cle_logit[:, label] - logit[:, label]
                replace = torch.ones_like(delta) * threshold
                delta = torch.where(delta <= threshold, replace, delta)
                return delta

            def promotion(logit):
                delta = logit[:, target] - cle_logit[:, target]
                replace = torch.ones_like(delta) * threshold
                delta = torch.where(delta <= threshold, replace, delta)
                return delta
            
            # calculate the psr score for phase and amplitude
            perturb_suppression, perturb_promotion = suppression(adv_logit), promotion(adv_logit)
            Sen_value.append(perturb_suppression + perturb_promotion)
            PSR_value.append(torch.log2(perturb_promotion / perturb_suppression).view(8, 8))
        print("=> Finish calculating perturbation PSR score")
    return PSR_value, Sen_value
