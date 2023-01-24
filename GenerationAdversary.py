# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   GenerationAdversary.py
    Time:        2022/11/28 23:35:32
    Editor:      Figo
-----------------------------------
'''

import os
import time
import torch
import numpy as np
import torch.nn as nn
import lib.adversary as adversary
from tqdm import tqdm
from torch.autograd import Variable
from utils.helps import cal_accuracy



class Container(nn.Module):
    def __init__(self):
        super(Container, self).__init__()
        self.reset()

    def reset(self):
        self.image = 0
        self.label = 0
        self.total = 0
        self.correct = 0
    
    def update(self, image, label, correct):
        if self.total == 0:
            self.image = image.clone().data.cpu()
            self.label = label.clone().data.cpu()
        else:
            self.image = torch.cat((self.image, image.clone().data.cpu()), dim=0)
            self.label = torch.cat((self.label, label.clone().data.cpu()), dim=0)
        self.total += len(label)
        self.correct += correct


def image_process(image_batch):
    """
        :: Here we don't want the adversarial noise destoried by uint8 process
    """
    image = image_batch.cpu().data.numpy()
    # image = np.uint8(image * 255)
    return image


def main(args, loader, adversary, model, test_model=None):
    
    #! Make folder for AEs saving
    global folder
    folder_name = "-".join([args.net_state, args.net_type, args.dataset])
    folder = os.path.join(args.outf, folder_name)
    if not os.path.exists(folder): os.mkdir(folder)
    attack_name = "-".join([args.adv_type, args.adv_norm])
    folder = os.path.join(folder, attack_name)
    if not os.path.exists(folder): os.mkdir(folder)
    print("=> Data will be saved in {}".format(folder))

    criterion = nn.CrossEntropyLoss()
    args.num_classes, min_pixel, max_pixel, random_noise_size = Get_Parameters(args)
    clean_container, noisy_container, adversary_container = Container(), Container(), Container()
    selected_list, selected_index, generated_noise = [], 0, 0
    iterator = tqdm(enumerate(loader), total=len(loader))
    adv_parameter = args.adv_parameter if args.adv_norm == 'l2' else args.adv_parameter / 255
    print("=> Adversarial parameter is:{:.4f}".format(adv_parameter))

    test_model = model if test_model is None else test_model
    model.eval()
    test_model.eval()


    for idx, (image, label) in iterator:

        #! generate the clean & noisy image
        image, label = image.cuda(), label.cuda()
        image, label = Variable(image), Variable(label)
        noisy_data = torch.add(image.data, random_noise_size, torch.randn(image.size()).cuda())
        noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)
        noisy_data = Variable(noisy_data)
        
        #? clean calculation
        output = test_model(image, with_latent=False, with_image=False)
        equal_flag, correct = cal_accuracy(output, label)
        
        #? noisy calculation
        noisy_output = test_model(noisy_data, with_latent=False, with_image=False)
        equal_flag_noise, correct_noise = cal_accuracy(noisy_output, label)
        
        #! generate adversarial example
        model.zero_grad()
        inputs = Variable(image.data, requires_grad=True)  # To get the gradient
        output = model(inputs, with_latent=False, with_image=False)
        loss = criterion(output, label)
        loss.backward()
        
        #! attacking & measure the noise
        adv_data = adversary(args, image, model, criterion, label, inputs, adv_parameter, min_pixel, max_pixel)
        temp_noise_max = torch.abs((image.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max = torch.norm(temp_noise_max, 2)
        generated_noise += temp_noise_max
        
        #? adversarial calculation
        adv_output = test_model(Variable(adv_data), with_latent=False, with_image=False)
        equal_flag_adv, correct_adv = cal_accuracy(adv_output, label)
        
        #! clean & noisy $ adversary updatation
        clean_container.update(image, label, correct)
        noisy_container.update(noisy_data, label, correct_noise)
        adversary_container.update(adv_data, label, correct_adv)

        #! correct:(clean, noisy)  incorrect:(adversary)
        for i in range(image.size(0)):
            if equal_flag[i] == 1 and equal_flag_noise[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)  # for data loader
            selected_index += 1
    
    #! calculate the magnitude adversarial noise
    AdvNoise = generated_noise / adversary_container.total

    selected_list = torch.LongTensor(selected_list)
    for container in [clean_container, noisy_container, adversary_container]:
        container.label = torch.index_select(container.label, 0, selected_list)
        container.image = torch.index_select(container.image, 0, selected_list)
    
    #! make a record in readme.txt
    time_now = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(folder+"/readme.txt", "a") as files:
        files.write("="*10 + "Records " + "="*10 + "\n")
        files.write("=> recording time: {}\n".format(time_now))
        files.write("=> {}-{}-{}-{}\n".format(args.net_state, args.net_type, args.dataset, args.adv_type))
        files.write("=> adversary-parameter: {:.4f}\n".format(args.adv_parameter))
        files.write("=> adversary-noise: {:.4f}\n".format(AdvNoise))
        files.write("=> adversary-norm: {}\n".format(args.adv_norm))
        files.write("=> Get total {} examples\n".format(len(selected_list)))
        files.write("=> Adversarial Classify Successful rate: {:.4f}%\n".format(100. * adversary_container.correct / adversary_container.total))
        files.write("=> Noisy Classify Successful rate: {:.4f}%\n".format(100. * noisy_container.correct / noisy_container.total))
        files.write("=> Clean Classify Successful rate: {:.4f}%\n\n".format(100. * clean_container.correct / clean_container.total))
    files.close()
    print("=> Get total {} examples".format(len(selected_list)))
    print('=> Adversarial Noise:{:.2f}'.format(AdvNoise))
    print("=> Adversarial Classify Successful rate: {:.4f}%".format(100. * adversary_container.correct / adversary_container.total))
    print("=> Noisy Classify Successful rate: {:.4f}%".format(100. * noisy_container.correct / noisy_container.total))
    print("=> Clean Classify Successful rate: {:.4f}%".format(100. * clean_container.correct / clean_container.total))
    
    #! saving generated images and labels
    CEs, NEs, AEs = [image_process(container.image) for container in [clean_container, noisy_container, adversary_container]]
    np.save(folder + '/CEs-{}'.format(args.adv_parameter), CEs)
    np.save(folder + '/NEs-{}'.format(args.adv_parameter), NEs)
    np.save(folder + '/AEs-{}'.format(args.adv_parameter), AEs)
    np.save(folder + '/label-{}'.format(args.adv_parameter), adversary_container.label.cpu().data.numpy())
    print("=> Finished saving!")
     

def Get_Parameters(args):
    #### min_pixel, max_pixel
    if args.net_type == 'densenet':
        min_pixel = -1.98888885975
        max_pixel = 2.12560367584
    if args.net_type == 'resnet18':
        min_pixel = 0 # -2.72906570435
        max_pixel = 1 # 2.95373125076
    if args.net_type == 'wideresnet28':
        min_pixel = 0 # -2.72906570435
        max_pixel = 1 # 2.95373125076
    if args.net_type == 'vgg16':
        min_pixel = 0 # -2.72906570435
        max_pixel = 1 # 2.95373125076
    #### noise_size
    if args.dataset in ['cifar', 'cifar100']:
        num_class = 10 if args.dataset == 'cifar' else 100
        if args.adv_type == 'fgsm':
            random_noise_size = 0.21 / 4
        elif args.adv_type == 'bim':
            random_noise_size = 0.21 / 4
        elif args.adv_type in ['pgd', 'pgd3', "pgd5"]:
            random_noise_size = 0.13 / 4
        elif args.adv_type == 'deepfool':
            random_noise_size = 0.13 * 2 / 10
        elif args.adv_type == 'cw':
            random_noise_size = 0.03 / 2
    elif args.dataset == 'imagenet':
        num_class = 100
        if args.adv_type == 'fgsm':
            random_noise_size = 0.21 / 8
        elif args.adv_type == 'bim':
            random_noise_size = 0.21 / 8
        elif args.adv_type in ['pgd', 'pgd3', "pgd5"]:
            random_noise_size = 0.21 / 8
        elif args.adv_type == 'deepfool':
            random_noise_size = 0.13 * 2 / 8
        elif args.adv_type == 'CW':
            random_noise_size = 0.06 / 5
    elif args.dataset == 'svhn':
        num_class = 10
        if args.adv_type == 'fgsm':
            random_noise_size = 0.21 / 4
        elif args.adv_type == 'bim':
            random_noise_size = 0.21 / 4
        elif args.adv_type in ['pgd', 'pgd3', "pgd5"]:
            random_noise_size = 0.21 / 4
        elif args.adv_type == 'deepfool':
            random_noise_size = 0.16 * 2 / 5
        elif args.adv_type == 'CW':
            random_noise_size = 0.07 / 2
    return num_class, min_pixel, max_pixel, random_noise_size


def Attack(args, data, model, criterion, target, inputs, adv_noise, min_pixel, max_pixel):

    if args.adv_type == 'fgsm':
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if args.net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
        else:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

    elif args.adv_type == 'bim':
        gradient = torch.sign(inputs.grad.data)
        for k in range(5):
            inputs = torch.add(inputs.data, adv_noise, gradient)
            inputs = torch.clamp(inputs, min_pixel, max_pixel)
            inputs = Variable(inputs, requires_grad=True)
            output = model(inputs, with_latent=False, with_image=False)
            loss = criterion(output, target)
            loss.backward()
            gradient = torch.sign(inputs.grad.data)
            if args.net_type == 'densenet':
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
            else:
                gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                     gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

    if args.adv_type == 'deepfool':

        _, adv_data = adversary.deepfool(model, data.data.clone(), target.data.cpu(), \
                                         args.num_classes, step_size=adv_noise, train_mode=False)
        adv_data = adv_data.cuda()

    elif args.adv_type == 'cw':
        _, adv_data = adversary.cw(model, data.data.clone(), target.data.cpu(), 1.0, args.adv_norm, crop_frac=1.0)
    else:
        adv_data = torch.add(inputs.data, adv_noise, gradient)

    adv_data = torch.clamp(adv_data, min_pixel, max_pixel)
    return adv_data
