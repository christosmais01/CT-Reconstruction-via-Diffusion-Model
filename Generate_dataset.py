import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import math

import os
from skimage.transform import radon, iradon, rescale
from skimage.io import imsave
from skimage import img_as_ubyte

#from skimage.data import shepp_logan_phantom
#image = shepp_logan_phantom()
#image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)

image_dir = r"C:\Users\Christos\Desktop\CT_Reconstruction\dataset\CT_128\hr_128"
sinogram_dir = r"C:\Users\Christos\Desktop\CT_Reconstruction\dataset\CT_128\lr_128"
inverse_dir = r"C:\Users\Christos\Desktop\CT_Reconstruction\dataset\CT_128\sr_128_128"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(sinogram_dir, exist_ok=True)
os.makedirs(inverse_dir, exist_ok=True)

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)

def generate_images_and_sinograms(num_samples):
    for i in range(num_samples):

        xx = np.linspace(0, 1 , 128)
        yy = np.linspace(0, 1, 128)
        x,y = np.meshgrid(xx, yy)
        r = np.sqrt((x - 0.5)**2 + (y-0.5)**2)
        f = np.zeros([128, 128])
        nb = int(np.random.uniform(1, 10, 1))

        for j in range(0, nb):
            a = np.random.uniform(0, 1, 1)
            x0 = np.random.uniform(0, 1, 1)
            y0 = np.random.uniform(0, 1, 1)
            sx = np.random.uniform(0.01, 0.3, 1)
            sy = np.random.uniform(0.01, 0.3, 1)
            f += a*np.exp(-0.5*((x- x0)**2/sx**2 + (y-y0)**2/sy**2))
        f[r>0.5] = 0
 
        image = f
        image = normalize_image(image)
        image = img_as_ubyte(image)

        image_filename = os.path.join(image_dir, f"image_{i+1:03d}.png")
        imsave(image_filename, image)

        theta = np.linspace(0.0, 360.0, max(image.shape), endpoint=False)
        sinogram = radon(image, theta=theta)

        noise_std = np.random.uniform(0, 13)
        sinogram += np.random.normal(0, noise_std, np.shape(sinogram))

        sinogram = normalize_image(sinogram)
        sinogram = img_as_ubyte(sinogram)
       
        sinogram_filename = os.path.join(sinogram_dir, f"sinogram_{i+1:03d}.png")
        imsave(sinogram_filename, sinogram)
        
        inverse = iradon(sinogram, theta=theta)

        inverse = normalize_image(inverse)
        inverse = img_as_ubyte(inverse)

        inverse_filename = os.path.join(inverse_dir, f"inverse_{i+1:03d}.png")
        imsave(inverse_filename, inverse)

generate_images_and_sinograms(2000)
