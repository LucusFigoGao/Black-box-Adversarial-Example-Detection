# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   GenerationPGD.py
    Time:        2022/11/28 23:35:23
    Editor:      Figo
-----------------------------------
'''

import os
import time
import torch
import numpy as np

from tqdm import tqdm
from torch.autograd import Variable
from utils.helps import cal_accuracy
from GenerationAdversary import Container, Get_Parameters


def image_process(image_batch):
    """
        :: Here we don't want the adversarial noise destoried by uint8 process
    """
    image = image_batch.cpu().data.numpy()
    return image


def main(args, loader, attack_kwargs, model, test_model=None):

    #! Make folder for AEs saving
    global folder
    folder_name = "-".join([args.net_state, args.net_type, args.dataset])
    folder = os.path.join(args.outf, folder_name)
    if not os.path.exists(folder): os.mkdir(folder)
    attack_name = "-".join([args.adv_type, args.adv_norm])
    folder = os.path.join(folder, attack_name)
    if not os.path.exists(folder): os.mkdir(folder)
    print("=> Data will be saved in {}".format(folder))

    args.num_classes, min_pixel, max_pixel, random_noise_size = Get_Parameters(args)
    clean_container, noisy_container, adversary_container = Container(), Container(), Container()
    selected_list, selected_index, generated_noise = [], 0, 0
    iterator = tqdm(enumerate(loader), total=len(loader))
    
    """
        :: adv_parameter = args.adv_epsilon, if adv_constraint is '2', e.x. 0.5, 1.0, 5.0
        :: adv_parameter = args.adv_epsilon/255, if adv_constraint is 'inf', e.x. 2, 8, 16
    """
    attack_kwargs['eps'] = args.adv_epsilon if args.adv_constraint == '2' else args.adv_epsilon / 255
    print(
        "=> Adversarial parameter is:{:.4f} for {}-{} attack".format(
            attack_kwargs['eps'], attack_kwargs['constraint'], 'targeted' if attack_kwargs['constraint'] == 1 else 'untargeted' 
        )
    )
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
        inputs = Variable(image.data, requires_grad=False)  # To get the gradient
        _, adv_data = model(inputs.cuda(), label.cuda(), make_adv=True, with_image=True, **attack_kwargs)
        
        #! attacking & measure the noise
        temp_noise_max = torch.abs((image.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max = torch.norm(temp_noise_max, 2)
        generated_noise += temp_noise_max

        #? adver calculation
        adv_out = test_model(adv_data, with_latent=False, with_image=False)
        equal_flag_adv, correct_adv = cal_accuracy(adv_out, label)
        
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
        files.write("="*13 + "Records " + "="*13 + "\n")
        files.write("=> recording time: {}\n".format(time_now))
        files.write("=> {}-{}-{}-{}\n".format(args.net_state, args.net_type, args.dataset, args.adv_type))
        files.write("=> adversary-epsilon: {:.4f}\n".format(args.adv_epsilon))
        files.write("=> adversary-iterations:{}\n".format(args.adv_iterations))
        files.write("=> adversary-noise: {:.4f}\n".format(AdvNoise))
        files.write("=> adversary-norm: {}\n".format(args.adv_constraint))
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
    
    # #! saving generated images and labels
    CEs, NEs, AEs = [image_process(container.image) for container in [clean_container, noisy_container, adversary_container]]
    np.save(folder + '/CEs-{}'.format(args.adv_epsilon), CEs)
    np.save(folder + '/NEs-{}'.format(args.adv_epsilon), NEs)
    np.save(folder + '/AEs-{}'.format(args.adv_epsilon), AEs)
    np.save(folder + '/label-{}'.format(args.adv_epsilon), adversary_container.label.cpu().data.numpy())
    print("=> Finished saving!")
