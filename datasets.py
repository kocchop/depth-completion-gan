"""
This is sample dataloader script for depth-completion-GAN
"""
import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from utils import *

import cv2
import torchvision.transforms as transforms

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)
    
def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                     filename_tmpl):
    """Generate paired paths from an meta information file.
    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.
    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```
    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.
    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys
    

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = os.path.splitext(os.path.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = os.path.join(input_folder, input_name)
        gt_path = os.path.join(gt_folder, gt_name)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths

def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.
    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.
    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = os.path.splitext(os.path.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = os.path.join(input_folder, input_name)
        # assert input_name in input_paths, (f'{input_name} is not in '
                                           # f'{input_key}_paths.')
        gt_path = os.path.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


class PairedImageDataset(Dataset):

    def __init__(self, root, opt, hr_shape):
    #We cannot use torch.Transforms because transforms.ToTensor() normalizes the image assuming its a 3 channel uint8 RGB image
        super(PairedImageDataset, self).__init__()
        
        self.opt = opt
        
        self.gt_folder, self.lq_folder = os.path.join(root,'image_hr'), os.path.join(root,'image_lr')
        
        self.filename_tmpl = '{}'
        
        if self.opt.meta_info_file is not None:
            self.meta_file = os.path.join(root, self.opt.meta_info_file)
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.meta_file, self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def __getitem__(self, index):

        # Load gt and lq depths. Dimension order: HW; channel: Grayscale;
        # Depth range: [0, 65535], uint16.
        gt_path = self.paths[index]['gt_path']
        # img_hi = Image.open(gt_path)
        img_hi = cv2.imread(gt_path, -1).astype(int)
        lq_path = self.paths[index]['lq_path']
        # img_lo = Image.open(lq_path)
        img_lo = cv2.imread(lq_path, -1).astype(int)
        
        # very naive way of transformation :(
        img_hr = (img_hi-mean)/std
        img_lr = (img_lo-mean)/std

        # return {"lr": img_lr, "hr": img_hr}
        return {
            'lr': img_lr,
            'hr': img_hr,
            'lr_path': lq_path,
            'hr_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

