"""
Some utility functions
"""
import torch
import numpy as np
import os
from torchvision.utils import make_grid

import logging
import cv2
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import io
import pathlib

# Normalization parameters for pre-trained PyTorch models
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

# These numbers were calculated from a portion of samples
mean = np.array([6532.253599727499])
std = np.array([14615.053388025])

# Error Metrics
def rmse(x, y):
    return torch.sqrt(torch.mean(torch.pow(x-y, 2.0)))

def mae(x, y):
    return torch.mean(torch.abs(x-y))
    
def irmse(x, y):
    return torch.sqrt(torch.mean(torch.pow(1.0/x - 1.0/y, 2.0)))

def imae(x, y):
    return torch.mean(torch.abs(1.0/x - 1.0/y))


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    
    torch.round_(tensors.mul_(std[0]).add_(mean[0]))
    
    return torch.clamp(tensors, 0, 65535)

def createLogger(file_name):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO) # or any other level
    logger.addHandler(ch)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    ch.setFormatter(formatter)
    
    fh = logging.FileHandler(file_name, mode='wt')
    fh.setLevel(logging.INFO) # or any level you want
    logger.addHandler(fh)

    fh.setFormatter(formatter)
    
    return logger

#this is written after torchvision.utils.save_image
def save_my_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format: Optional[str] = None,
    ) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    
    # need to detach as grid is a tensor with grads=True
    ndarr = grid.to('cpu').detach().numpy()
    
    #convering to uint16 -> Grayscale
    im = ndarr.astype(np.uint16)
    
    # it will convert, (4, 192, 784) --> (4*192, 784) = (784, 784)
    im = np.reshape(im,(-1,im.shape[2]))
    
    #write the image with cv2
    cv2.imwrite(fp, im)

def save_sample_images(imgs_hr, imgs_lr, gen_hr, image_save_path, image_id) -> None:

    img_grid = denormalize(torch.cat((imgs_hr, imgs_lr, gen_hr), -1))
    img_grid = torch.squeeze(img_grid, 1) #this will delete the channel axis: (4, 1, 192, 256) --> (4, 192, 256)
    saved_image_file = os.path.join(image_save_path,"%04d.png"%image_id)
    save_my_image(img_grid, saved_image_file, nrow=1, normalize=False)