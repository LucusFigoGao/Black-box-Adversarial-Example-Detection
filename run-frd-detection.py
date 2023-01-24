# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   run-frd-detection.py
    Time:        2022/12/10 11:35:35
    Editor:      Figo
-----------------------------------
'''

import os
import json
import argparse

from default import FRD_CONFIG, FRDA_CONFIG, FRDP_CONFIG
from frd_nn_ijcai2023 import main as frd_nn_main
from utils.helps import check_kwargs, set_random_seed


def get_argparse():
    parser = argparse.ArgumentParser(description='Adversarial Detection Methods Evaluation')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    #! Load args and kwargs from config file
    args = get_argparse()
    main, config = frd_nn_main, 'frd'

    with open("./config/config_{}.json".format(config), "r") as file:
        kwargs = json.load(file)['kwargs']
    kwargs = check_kwargs(kwargs)
    
    #! set gpu device
    os.environ['CUDA_VISIBLE_DEVICES'] = kwargs['gpuid']
    print("=> GPU device {} is available".format(kwargs['gpuid']))
    
    LAYER = kwargs['layer']                     # activation layer
    BRANCH = kwargs['freqvae']                  # freqvae branch
    VICTIM = kwargs['victim_model']             # victim model
    HACKER = kwargs['hacker_model']             # hacker model
    ATTACK_TRAIN = kwargs['attack_train']       # attack train
    ATTACK_TEST = kwargs['attack_test']         # attack train

    if BRANCH == "joint":
        CONFIG = FRD_CONFIG
    elif BRANCH == "phase":
        CONFIG = FRDP_CONFIG
    elif BRANCH == "amplitude":
        CONFIG = FRDA_CONFIG

    #! Set different seeds and a mean value performance
    for layer in LAYER:
        kwargs['layer'] = layer
        
        #! Set victim model and detectors
        for victim in VICTIM:
            kwargs['victim_model'] = victim
            for attack_train in ATTACK_TRAIN:
                kwargs['adv_type_train'] = attack_train[0]
                kwargs['adv_norm_train'] = attack_train[1]
                kwargs['adv_parameter_train'] = attack_train[2]
                train_stage = victim + [kwargs['dataset']] + [str(_) for _ in attack_train]
                print(f"=> [Train Stage] Train detector is: {'-'.join(train_stage)}")

                kwargs['lr'], kwargs["weight_decay"], \
                        kwargs["momentum"], kwargs["epoch"] = CONFIG[kwargs['adv_type_train']]
                print("=> lr:{}, weight_decay:{}, momentum:{}, epoch:{}".format(
                    kwargs['lr'], kwargs["weight_decay"], kwargs["momentum"], kwargs["epoch"]
                ))

                #! Set hacker model and test adv examples
                for hacker in HACKER:
                    kwargs['hacker_model'] = hacker
                    
                    #! Ensemble attack
                    if hacker[1] == "ensemble":
                        kwargs['adv_type_test'] = 'ensemble'
                        kwargs['adv_norm_test'] = 'l2'
                        kwargs['adv_parameter_test'] = 4
                        test_stage = hacker + [kwargs['dataset'], 'ensemble-l2-4']
                        print(f"=> [Test Stage] Adversary is: {'-'.join(test_stage)}")
                        print("="*70)
                        vars(args).update(kwargs)
                        set_random_seed(args.seed)
                        main(args)
                        continue
                    
                    #! Other Black-box attack
                    for attack_test in ATTACK_TEST:
                        kwargs['adv_type_test'] = attack_test[0]
                        kwargs['adv_norm_test'] = attack_test[1]
                        kwargs['adv_parameter_test'] = attack_test[2]
                        test_stage = hacker + [kwargs['dataset']] + [str(_) for _ in attack_test]
                        print(f"=> [Test Stage] Adversary is: {'-'.join(test_stage)}")
                        print("="*70)
                        vars(args).update(kwargs)
                        set_random_seed(args.seed)

                        main(args)
                        print("\n")
                        print("\n")
