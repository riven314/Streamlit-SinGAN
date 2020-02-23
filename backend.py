"""
build backend pipeline for model computation

ISSUES:
1. why images_cur result different from original results?

NOTES:
1. how the animation algo loop?
    - outer loop: generator axis (from coarse to fine); inner loop: frame axis

REFERENCE:
1. f-string examples: http://zetcode.com/python/fstring/
2. why Z_opt changes every epoch?: https://github.com/tamarott/SinGAN/issues/63
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from SinGAN.manipulate import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions

# 1. Setup Parameters
opt = edict()
opt.not_cuda = False if torch.cuda.is_available else True
opt.nc_im = 3 # image channels no. 
opt.mode = 'animation'
opt.out = 'Output'
opt.input_name = 'lightning1'

scale_factor = 0.75 # determine no. of levels in hierarchy
input_name = opt.input_name # folder name
dir2save = Path(f'TrainedModels/{input_name}/scale_factor={scale_factor:.6f}_noise_padding')
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
ker_size = 3
num_layer = 5
nc_z = 3 # noise channel no.
alpha = 0.1
beta = 0.95
start_scale = 1


# 2. Load in Models
# may affect final result
# opt.min_size = 20
# opt.mode = 'animation_train'
# real = functions.read_image(opt)
# functions.adjust_scales2image(real, opt)
# dir2trained_model = functions.generate_dir2save(opt)

assert os.path.isdir(dir2save), f'[ERROR] dir not exist: {dir2save}'
Gs = torch.load(dir2save / 'Gs.pth') # list of generators (by scales)
Zs = torch.load(dir2save / 'Zs.pth') # list of noise (by scales)
reals = torch.load(dir2save / 'reals.pth') # list of real image patches
NoiseAmp = torch.load(dir2save / 'NoiseAmp.pth') # list of NoiseAmp


# 3. Generate GIFs (varying beta && start_scale)
in_s = torch.full(Zs[0].shape, 0, device = device)
images_cur = []
count = 0

for G, Z_opt, noise_amp, real in zip(Gs, Zs, NoiseAmp, reals):
    pad_image = int(((ker_size - 1) * num_layer) / 2) # what it means??
    nzx = Z_opt.shape[2]
    nzy = Z_opt.shape[3]
    m_image = nn.ZeroPad2d(int(pad_image))
    images_prev = images_cur
    images_cur = []
    if count == 0:
        # z_rand is gaussian noise
        z_rand = functions.generate_noise([1, nzx, nzy], device= device) 
        z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
        z_prev1 = 0.95 * Z_opt +0.05 * z_rand
        z_prev2 = Z_opt
    else:
        z_prev1 = 0.95*Z_opt +0.05*functions.generate_noise([nc_z,nzx,nzy], device = device)
        z_prev2 = Z_opt

    for i in range(0,100,1):
        if count == 0:
            z_rand = functions.generate_noise([1,nzx,nzy], device = device)
            # make z_rand same across channels
            z_rand = z_rand.expand(1,3, Z_opt.shape[2], Z_opt.shape[3])
            diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*z_rand
        else:
            diff_curr = beta*(z_prev1-z_prev2)+(1-beta)*(functions.generate_noise([nc_z,nzx,nzy], device = device))

        z_curr = alpha*Z_opt+(1-alpha)*(z_prev1+diff_curr)
        z_prev2 = z_prev1
        z_prev1 = z_curr

        if images_prev == []:
            I_prev = in_s
        else:
            I_prev = images_prev[i]
            I_prev = imresize(I_prev, 1 / scale_factor, opt) # edit
            I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
            I_prev = m_image(I_prev)
        if count < start_scale:
            z_curr = Z_opt

        z_in = noise_amp*z_curr+I_prev
        I_curr = G(z_in.detach(),I_prev)

        # convert result from GPU to CPU at last iteration
        if (count == len(Gs)-1):
            I_curr = functions.denorm(I_curr).detach()
            I_curr = I_curr[0,:,:,:].cpu().numpy()
            I_curr = I_curr.transpose(1, 2, 0)*255
            I_curr = I_curr.astype(np.uint8)

        images_cur.append(I_curr)
    # count = 10 at the end
    count += 1
    # can be kick out

def save_gif(opt, images_cur, alpha, beta):
    """
    images_cur is a list of time series images in same scale
    """
    dir2save = functions.generate_dir2save(opt)
    save_dir = os.path.join(f'{dir2save}', f'start_scale={start_scale:.2d}')
    try:
        os.makedirs(save_dir)
    except OSError:
        pass
    gif_path = os.path.join(save_dir, f'alpha={alpha:.2f}_beta={beta:.2f}__.gif')
    imageio.mimsave(gif_path, images_cur, fps = 10)