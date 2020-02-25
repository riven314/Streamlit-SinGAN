import os
import imageio

import numpy as np
import SinGAN.functions as functions


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


def tensor_to_np(frame):
    """
    turn float-point tensor to uint8 np.array
    """
    frame = functions.denorm(frame).detach()
    frame = frame[0,:,:,:].cpu().numpy()
    frame = frame.transpose(1, 2, 0)*255
    frame = frame.astype(np.uint8)
    return frame