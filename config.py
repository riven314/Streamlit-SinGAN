import os
from pathlib import Path

import torch
from easydict import EasyDict as edict


opt = edict()
opt.not_cuda = False if torch.cuda.is_available() else True
opt.nc_im = 3 # image channels no. 
opt.mode = 'animation'
opt.out = 'Output'
opt.input_name = 'lightning1'
opt.scale_factor = 0.75
opt.ker_size = 3
opt.num_layer = 5

dir2save = Path(f'TrainedModels/{opt.input_name}/scale_factor={opt.scale_factor:.6f}_noise_padding')
assert os.path.isdir(dir2save), f'[ERROR] dir not exist: {dir2save}'
opt.Gs = torch.load(dir2save / 'Gs.pth') # list of generators (by scales)
opt.Zs = torch.load(dir2save / 'Zs.pth') # list of noise (by scales)
opt.reals = torch.load(dir2save / 'reals.pth') # list of real image patches
opt.NoiseAmp = torch.load(dir2save / 'NoiseAmp.pth') # list of NoiseAmp


