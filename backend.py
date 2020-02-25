"""
build backend pipeline for model computation

ISSUES:
1. why frames_curr result different from original results?

REFERENCE:
1. f-string examples: http://zetcode.com/python/fstring/
2. why Z_opt changes every epoch?: https://github.com/tamarott/SinGAN/issues/63
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from SinGAN.manipulate import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions

from compute import compute_z_curr, compute_z_prev, compute_z_diff
from utils import tensor_to_np


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
scale_start = 1


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
def cache_input_output(Gs, Zs, NoiseAmp, reals, scale_in = None, scale_out = None):
    """
    cache time-series input at scale i, and time-series output at scale j.
    both i and j start at 0 index

    :output:
        cache_dict -- dict, {
            'input': list of time-series np.array, 
            'output': list of time-series np.array
            }
    """
    cache_dict = defaultdict(list)
    # by default cache first scale input and final scale output
    scale_in = 0 if scale_in is None else scale_in
    scale_out = len(Gs) -1 if scale_out is None else scale_out
    # create layer for boarder padding
    pad_image = int(((ker_size - 1) * num_layer) / 2)
    m_image = nn.ZeroPad2d(int(pad_image))
    in_s = torch.full(Zs[0].shape, 0, device = device)
    frames_curr = []
    # out loop is scale iteration
    for scale_n, (G, Z_opt, noise_amp, real) in enumerate(zip(Gs, Zs, NoiseAmp, reals)):
        frames_prev = frames_curr
        frames_curr = []
        z_prev1, z_prev2 = compute_z_prev(scale_n, Z_opt, device)
        # inner loop is time iteration
        for t in range(0, 100, 1):
            z_diff = compute_z_diff(scale_n, Z_opt, z_prev1, z_prev2, beta, device)
            z_curr = compute_z_curr(Z_opt, z_prev1, z_diff, alpha)
            z_prev2 = z_prev1
            z_prev1 = z_curr
            # overwrite z_curr if init at higher scale
            if scale_n < scale_start:
                z_curr = Z_opt
            if frames_prev == []:
                I_prev = in_s
            else:
                I_prev = frames_prev[t]
                I_prev = imresize(I_prev, 1 / scale_factor, opt) # edit
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                I_prev = m_image(I_prev)
            z_in = noise_amp * z_curr + I_prev
            I_curr = G(z_in.detach(), I_prev)
            frames_curr.append(I_curr)
            # cache results
            if scale_n == scale_in:
                z_in = tensor_to_np(z_in)
                cache_dict['input'].append(z_in)
            if scale_n == scale_out:
                I_curr = tensor_to_np(I_curr)
                cache_dict['output'].append(I_curr)
    return cache_dict


if __name__ == '__main__':
    cache_dict = cache_input_output(Gs, Zs, NoiseAmp, reals, scale_in = None, scale_out = None)