import torch
from easydict import EasyDict as easydict


opt = edict()
opt.not_cuda = False if torch.cuda.is_available() else True
opt.nc_im = 3 # image channels no. 
opt.mode = 'animation'
opt.out = 'Output'
opt.input_name = 'lightning1'