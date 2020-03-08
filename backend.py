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

import streamlit as st
import time
import itertools


# 1. Setup Parameters
opt = edict()
opt.not_cuda = False if torch.cuda.is_available() else True
opt.nc_im = 3 # image channels no. 
opt.mode = 'animation'
opt.out = 'Output'
opt.input_name = 'lightning1'

scale_factor = 0.75 # determine no. of levels in hierarchy
input_name = opt.input_name # folder name
dir2save = Path(f'TrainedModels/{input_name}/scale_factor={scale_factor:.6f}_noise_padding')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ker_size = 3
num_layer = 5


# 2. Load in Models
assert os.path.isdir(dir2save), f'[ERROR] dir not exist: {dir2save}'
Gs = torch.load(dir2save / 'Gs.pth') # list of generators (by scales)
Zs = torch.load(dir2save / 'Zs.pth') # list of noise (by scales)
reals = torch.load(dir2save / 'reals.pth') # list of real image patches
NoiseAmp = torch.load(dir2save / 'NoiseAmp.pth') # list of NoiseAmp


# 3. Generate GIFs (varying beta && start_scale)
def cache_input_output(Gs, Zs, NoiseAmp, reals, alpha = 0.1, beta = 0.95, scale_start = 0, device = device):
    """
    cache time-series input at scale i, and time-series output at scale j.
    both i and j start at 0 index

    :param:
        Gs : list of generators, nn.Module
        Zs : list of input (input that map to realistic image), np.array uint8
        NoiseAmp : list of noise injection, np.array uint8
        reals : list of real image patach, np.array uint8
        scale_start : 0, scale at which we start to inject noise (for random walk)
    :return:
        cache_dict -- dict, {
            0: {
                'input': list of time-series np.array, 
                'output': list of time-series np.array
            },
            1: {
                'input': list of time-series np.array, 
                'output': list of time-series np.array
            }, ...
            }
    """
    cache_dict = defaultdict(lambda: defaultdict(list))
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
            z_in_np = tensor_to_np(z_in)
            cache_dict[scale_n]['input'].append(z_in_np)
            I_curr_np = tensor_to_np(I_curr)
            cache_dict[scale_n]['output'].append(I_curr_np)
    return cache_dict


#front-end interface
def image_display(cache_dict, input_scale = 0, output_scale = 9):
    st.title('Streamlit implementation if SinGAN')
    st.write("Here's our first attempt at implementing backend with streamlit integration for image display")
    imageLocation_input = st.empty()
    imageLocation_output = st.empty()
 
    for (i, o) in zip(cache_dict[input_scale]['input'], cache_dict[output_scale]['output']):
        imageLocation_input.image(i, channels = 'RGB')
        imageLocation_output.image(o, channels = 'RGB')
        time.sleep(0.3)


if __name__ == '__main__':
    cache_dict = cache_input_output(Gs, Zs, NoiseAmp, reals)
    #call function for front-end display
    image_display(cache_dict)
