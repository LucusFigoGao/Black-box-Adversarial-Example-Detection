# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   train_pipeline.py
    Time:        2022/09/18 19:04:55
    Editor:      Figo
-----------------------------------
'''
import os
import cox
import json
import cox.utils
import cox.store

from utils.helps import set_args
from utils.helps_data import load_dataset
from robustness.main import setup_args
from robustness.train import train_model
from robustness.tools.helpers import DataPrefetcher
from robustness.model_utils import make_and_restore_model


SEED = 777

if __name__ == "__main__":
    
    #! Set args
    parser = set_args()
    args = parser.parse_args()
    
    with open("config/config.json", "r") as file:
        set_kwargs = json.load(file)['kwargs']
    
    if set_kwargs['exp_name'] == "None":
        set_kwargs['exp_name'] = '-'.join(["AT:{}".format(set_kwargs['adv_train']), set_kwargs['dataset'], set_kwargs['arch'], "S:{}".format(str(SEED))])
        print(f"==> Exp name is:{set_kwargs['exp_name']}")

    #! if adversarial training
    if set_kwargs['adv_train'] == 1:
        set_kwargs["constraint"] = "inf"
        set_kwargs["eps"] = 8/255
        set_kwargs["attack_lr"] = 2/255
        set_kwargs["attack_steps"] = 10
        set_kwargs["use_best"] = True
        set_kwargs["random_restarts"] = 0
    
    vars(args).update(set_kwargs)
    args = cox.utils.Parameters(args.__dict__)
    args = setup_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_index
    print(f"=> gpu {args.cuda_index} is using...")

    #! Load dataset
    print("==> Load dataset...")
    ds = load_dataset(args.dataset)
    train_loader, val_loader = ds.make_loaders(workers=args.workers, batch_size=args.batch_size)
    print(f"==> Got {len(train_loader.dataset)} train images, {len(val_loader.dataset)} val images")
    train_loader, val_loader = DataPrefetcher(train_loader), DataPrefetcher(val_loader)

    # #! Load model from model zoo
    print("==> Load Model...")
    model, checkpoint = make_and_restore_model(arch=args.arch, dataset=ds, resume_path=None)
    print("==> Model is OK...")

    # #! Model train stage
    store = cox.store.Store(args.out_dir, args.exp_name)
    model = train_model(args, model, (train_loader, val_loader), store=store, checkpoint=checkpoint)
