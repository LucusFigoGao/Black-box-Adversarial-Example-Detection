# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   helps_load.py
    Time:        2022/10/28 20:16:19
    Editor:      Figo
-----------------------------------
'''

import os
import torch
from model.freqvae import freqvae_cifar


def load_ckpt_freqvae_v1(args, ckpt_kwargs):
    #! Load freq vae model
    """
        :: old version (before 2022.6.18) to load model parameters from .pth file
    """
    get_params = lambda params: {k[len('module.'):]:v for k, v in params.items()}

    if args.freqvae in ["amplitude", "joint"]:
        ckpt_amplitude = ckpt_kwargs['amplitude']
        amp_file_path = torch.load(ckpt_amplitude, map_location="cpu")
        amp_model_params = amp_file_path['model']
        print("=> Best amplitude reconstruction accuracy is:{:.2f}%".format(amp_file_path['best_acc']))
        embed_dim = ckpt_amplitude.split('-')[-1].split('/')[0]
        amp_freqvae = freqvae_cifar(32, int(embed_dim))
        amp_model_params_off_parallel = get_params(amp_model_params)
        amp_freqvae.load_state_dict(amp_model_params_off_parallel)
        amp_freqvae = torch.nn.DataParallel(amp_freqvae).cuda()
    
    if args.freqvae in ["phase", "joint"]:
        ckpt_phase = ckpt_kwargs['phase']
        pha_file_path = torch.load(ckpt_phase, map_location="cpu")
        pha_model_params = pha_file_path['model']
        print("=> Best phase reconstruction accuracy is:{:.2f}%".format(pha_file_path['best_acc']))
        embed_dim = ckpt_phase.split('-')[-1].split('/')[0]
        pha_freqvae = freqvae_cifar(32, int(embed_dim))
        pha_model_params_off_parallel = get_params(pha_model_params)
        pha_freqvae.load_state_dict(pha_model_params_off_parallel)
        pha_freqvae = torch.nn.DataParallel(pha_freqvae).cuda()
    
    print("=> Well-trained FreqVAE model is OK!")

    if args.freqvae == "amplitude":
        return amp_freqvae
    elif args.freqvae == "phase":
        return pha_freqvae
    elif args.freqvae == "joint":
        return amp_freqvae, pha_freqvae


def load_ckpt_freqvae_v2(args, model_kwargs):
    #! Load freq vae model
    """
        :: Here we save the model instead of model.parameters()
    """
    print("=> Load well trained freqvae model...")

    if args.freqvae == "amplitude":
        amp_file_pt = torch.load(model_kwargs['amplitude'])
        model = torch.nn.DataParallel(amp_file_pt['model']).cuda()
        print("=> Best amplitude reconstruction accuracy is:{:.2f}%".format(amp_file_pt['best_acc']))
    
    elif args.freqvae == "phase":
        pha_file_pt = torch.load(model_kwargs['phase'])
        model = torch.nn.DataParallel(pha_file_pt['model']).cuda()
        print("=> Best phase reconstruction accuracy is:{:.2f}%".format(pha_file_pt['best_acc']))

    elif args.freqvae == "joint":
        amp_file_pt, pha_file_pt = torch.load(model_kwargs['amplitude']), torch.load(model_kwargs['phase'])
        print("=> Best phase reconstruction accuracy is:{:.2f}%".format(pha_file_pt['best_acc']))
        print("=> Best amplitude reconstruction accuracy is:{:.2f}%".format(amp_file_pt['best_acc']))
        model1 = torch.nn.DataParallel(amp_file_pt['model']).cuda()
        model2 = torch.nn.DataParallel(pha_file_pt['model']).cuda()
        model = (model1, model2)
    print("=> {} FreqVAE is OK...".format(args.freqvae))
    return model


def get_dataset_path(args):
    
    victim_model, hacker_model = args.victim_model, args.hacker_model

    victim_prefix = '-'.join([victim_model[0], victim_model[1], args.dataset])
    victim_adv = "-".join([args.adv_type_train, args.adv_norm_train])
    victim_root = os.path.join("./dataset", victim_prefix, victim_adv)

    args.fae = os.path.join(victim_root, "AEs-{}.npy".format(args.adv_parameter_train))
    args.fce = os.path.join(victim_root, "CEs-{}.npy".format(args.adv_parameter_train))
    args.fne = os.path.join(victim_root, "NEs-{}.npy".format(args.adv_parameter_train))
    args.flabel = os.path.join(victim_root, "label-{}.npy".format(args.adv_parameter_train))

    hacker_prefix = '-'.join([hacker_model[0], hacker_model[1], args.dataset])
    if hacker_model[1] == "ensemble":
        hacker_root = os.path.join("./dataset", hacker_prefix)
    else:
        hacker_adv = "-".join([args.adv_type_test, args.adv_norm_test])
        hacker_root = os.path.join("./dataset", hacker_prefix, hacker_adv)

    args.fae_test = os.path.join(hacker_root, "AEs-{}.npy".format(args.adv_parameter_test))
    args.fce_test = os.path.join(hacker_root, "CEs-{}.npy".format(args.adv_parameter_test))
    args.fne_test = os.path.join(hacker_root, "NEs-{}.npy".format(args.adv_parameter_test))
    args.flabel_test = os.path.join(hacker_root, "label-{}.npy".format(args.adv_parameter_test))


    for file in [args.fae, args.fce, args.fne, args.flabel, \
                args.fae_test, args.fce_test, args.fne_test, args.flabel_test]:
        if not os.path.exists(file):
            raise ValueError("=> no such path {} exists".format(file))
