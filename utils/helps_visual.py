# -*- encoding: utf-8 -*-
'''
-----------------------------------
    File name:   helps_visual.py
    Time:        2022/10/25 11:16:22
    Editor:      Figo
-----------------------------------
'''

import os
import cv2
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from utils.helps import mean_var_norm3d, min_max_norm3d, amp_pha_recon


def recon_images(args, epoch, batch_size=64, dataset=None, model=None, writer=None, normalize="min"):
    normalize = min_max_norm3d if normalize == "min" else mean_var_norm3d
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    model.eval()
    for image, _ in loader:

        ori_image, amplitude, phase = image[0].cuda(), image[1].cuda(), image[2].cuda()

        if args.freqvae == "phase":
            recon_phase, _, _, _ = model(phase)
            recon_image = amp_pha_recon(amplitude, recon_phase)
        
        elif args.freqvae == "amplitude":
            recon_amplitude, _, _, _ = model(amplitude)
            recon_image = amp_pha_recon(recon_amplitude, phase)

        # recon_image = normalize(recon_image)
        
        break
    visual_image_comparison(epoch, ori_image, recon_image, writer)

        
def visual_image_comparison(epoch, ori_image, recon_image, writer=None):
    grid_x = tv.utils.make_grid(
        tensor=ori_image.cpu().data,
        nrow=8,  # (8r8c)
        padding=2,  # padding the gap with 2
        normalize=True
    )
    grid_y = tv.utils.make_grid(
        tensor=recon_image.cpu().data,
        nrow=8,  # (8r8c)
        padding=2,  # padding the gap with 2
        normalize=True
    )

    if writer is not None:
        writer.add_image("Epoch-{} The Real image".format(epoch), grid_x)
        writer.add_image("Epoch-{} The Recon image".format(epoch), grid_y)
    else:
        plt.figure(figsize=(12, 8), dpi=500)
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(grid_x.data.numpy().transpose(1, 2, 0))
        ax1.set_title("Epoch-{} The Real image".format(epoch))
        ax2 = plt.subplot(1, 2, 1)
        ax2.imshow(grid_x.data.numpy().transpose(1, 2, 0))
        ax2.set_title("Epoch-{} The Real image".format(epoch))
        plt.savefig("/data/yifei/Research/blackbox-detection/Image/reg.png", bbox_inches='tight', pad_inches=0.0)


def visual_image(image, path, padding=2):
    if not isinstance(image, list):
        image = [image]
    if len(image) == 1:
        image = np.uint8(image[0] * 255) if np.mean(image[0]) < 1 else image[0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path, image)
    else:
        image_padding = np.ones((image[0].shape[0], padding, 3)) * 255
        for idx in range(len(image)):
            img = np.uint8(image[idx] * 255) if np.mean(image[idx]) < 1 else image[idx]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if idx == 0:
                image_batch = img
            else:
                image_batch = np.hstack([image_batch, image_padding])
                image_batch = np.hstack([image_batch, img])
        
        cv2.imwrite(path, image_batch)


def visual_aes(args):
    rnd = np.random.randint(0, 100, 1)[0]
    hacker = "-".join([args.net_state, args.net_type, args.dataset])
    path = "dataset/{}".format(hacker)
    save_dir = "Image/Image-AEs"
    adv_type, adv_norm, adv_parameter = args.adv_type, args.adv_norm, str(args.adv_parameter)
    prefix = "-".join([adv_type, adv_norm])
    root = os.path.join(path, prefix)
    image = []

    for example in ["AEs", "CEs", "NEs"]:
        name = "-".join([example, adv_parameter])+".npy"
        files = os.path.join(root, name)
        data = np.load(files)[rnd].transpose(1, 2, 0)
        image.append(np.uint8(data*255))

    image_padding = np.ones((image[0].shape[0], 2, 3)) * 255
    for i in range(3):
        if i == 0:
            image_batch = image[0]
            continue
        image_batch = np.hstack([image_batch, image_padding])
        image_batch = np.hstack([image_batch, image[i]])

    image_batch = np.uint8(image_batch)
    Image.fromarray(image_batch).save(os.path.join(save_dir, prefix+'.png'))


if __name__ == "__main__":
    
    import os

    adv_type, adv_norm, adv_parameters = "pgd", "linf", "8"
    root = "/data/yifei/Research/blackbox-detection/dataset"
    model = "-".join(['normal2normal', 'resnet18', 'cifar'])
    attack_name = '-'.join([adv_type, adv_norm])
    folder = os.path.join(root, model, attack_name)

    index = np.random.randint(0, 1000, 1).item()
    
    AEs, CEs, NEs = [os.path.join(folder, '{}Es-{}.npy'.format(idx, adv_parameters)) for idx in ['A', 'C', 'N']]
    print("=> Load adversarial examples from {}".format(AEs))
    print("=> Load clean examples from {}".format(CEs))
    print("=> Load noisy examples from {}".format(NEs))

    clean_image, noise_image, adver_image = [np.load(Ex)[index].transpose(1, 2, 0) for Ex in [CEs, NEs, AEs]]
    visual_image([clean_image, noise_image, adver_image])
