import os
import shutil
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from PIL import Image
import matplotlib.pyplot as plt


def rgb_to_lab(rgb):
    ' transform a PIL RGB image into a Lab tensor '
    img = rgb2lab(rgb).astype('float32')
    L = (img[..., 0:1] / 50.) - 1.  # [0,100] -> [-1,1]
    ab = img[..., 1:] / 128  # [-128,128] -> [-1,1]
    return {'L': L, 'ab': ab}


def lab_to_rgb(L, ab):
    ' to be transferable between keras and pytorch, L and ab are numpy arrays of form [B, H, W ,C]'
    L = (L + 1.) * 50.  # [-1,1] -> [0,100]
    ab = ab * 128.  # [-1,1] -> [-128,128]
    Lab = np.concatenate((L, ab), axis=3)
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

