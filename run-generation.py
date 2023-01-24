# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   run-generation.py
    Time:        2022/12/08 09:59:05
    Editor:      Figo
-----------------------------------
'''

import os
import json
import torch
import argparse

from default import ADV_DEFAULT, RESUME_DEFAULT
from utils.helps_visual import visual_aes
from utils.helps_data import load_dataset
from utils.helps import set_random_seed, check_kwargs, val_model
from robustness.main import make_and_restore_model
from robustness.data_augmentation import TEST_TRANSFORMS_DEFAULT
from GenerationAdversary import main as adversary_main, Attack
from GenerationPGD import main as pgd_main


def get_argparse():
    parser = argparse.ArgumentParser(description='Adversarial Examples Generation')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = get_argparse()
    
    #! Set default kwargs and seed
    with open("./config/config_adversary.json", "r") as file:
        kwargs = json.load(file)['kwargs']
    kwargs = check_kwargs(kwargs)
    
    CUDA = kwargs['gpuid']                      # cuda index
    SEED = kwargs['seed']                       # random seed
    VICTIM = kwargs['victim_model']             # victim model
    HACKER = kwargs['hacker_model']             # hacker model
    ATTACK = kwargs['attack']                   # attack
    DATASET = kwargs['dataset']                 # dataset

    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
    print(f"=> cuda {CUDA} device is available...")
    DEVICE_IDS = list(range(len(CUDA.split(','))))
    
    #! Load dataset for adversarial example generating
    dataset = load_dataset(DATASET, "custom", transform_test=TEST_TRANSFORMS_DEFAULT(32))
    _, test_loader = dataset.make_loaders(workers=4, batch_size=128, only_val=True)
    print("=> Got {} test images".format(len(test_loader.dataset)))

    for victim in VICTIM:
        victim_model = "-".join([victim[0], victim[1], DATASET])
        Vmodel, _ = make_and_restore_model(arch=victim[1], dataset=dataset, resume_path=RESUME_DEFAULT[victim_model])
        Vmodel = torch.nn.DataParallel(Vmodel.cuda(), device_ids=DEVICE_IDS)
        print(f"=> Victim model: {victim_model} is OK...")
        #! Check accuracy before generating AEs
        val_model(Vmodel, test_loader)

        for hacker in HACKER:
            kwargs['net_state'], kwargs['net_type'] = hacker[0], hacker[1]
            hacker_model = "-".join([hacker[0], hacker[1], DATASET])
            if hacker_model == victim_model:
                Hmodel = Vmodel
            else:
                Hmodel, _ = make_and_restore_model(arch=hacker[1], dataset=dataset, resume_path=RESUME_DEFAULT[hacker_model])
            Hmodel = torch.nn.DataParallel(Hmodel.cuda(), device_ids=DEVICE_IDS)
            print(f"=> Hacker model: {hacker_model} is OK...")
            val_model(Hmodel, test_loader)
            print("="*50)

            for attack in ATTACK:
                kwargs['adv_type'] = attack[0]
                kwargs['adv_norm'] = attack[1]
                kwargs['adv_parameter'] = attack[2]
                print(f"=> Adversary is: {'-'.join([str(_) for _ in attack])}")

                vars(args).update(kwargs)
                set_random_seed(args.seed)

                #! Generating adversarial examples
                if attack[0] in ['fgsm', 'bim', 'deepfool', 'cw']:
                    adversary_main(args, test_loader, Attack, Hmodel, Vmodel)
                elif attack[0] == 'pgd':
                    attack_kwargs = ADV_DEFAULT['-'.join([attack[0], attack[1]])]
                    vars(args).update(attack_kwargs)

                    kwargs = {
                        'constraint': attack_kwargs['adv_constraint'],         # L-inf PGD
                        'eps': attack_kwargs['adv_epsilon']  ,                 # Epsilon constraint (L-inf norm)
                        'step_size': attack_kwargs['adv_step_size'],           # Learning rate for PGD
                        'iterations': attack_kwargs['adv_iterations'],         # Number of PGD steps
                        'targeted': False,                                     # Targeted attack
                        'custom_loss': None                                    # Use default cross-entropy loss
                    }

                    args.outf = "visual"
                    pgd_main(args, test_loader, kwargs, Hmodel, Vmodel)
                
                visual_aes(args)
