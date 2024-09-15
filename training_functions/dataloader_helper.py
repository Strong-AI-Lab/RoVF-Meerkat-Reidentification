import sys
sys.path.append("..")

from dataloaders.ReID import AnimalClipDataset

import json

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2.functional import InterpolationMode

import numpy as np

import random
import h5py
import glob
import os
from collections import defaultdict

from augmentations.simclr_augmentations import get_meerkat_transforms

def dataloader_creation(
    transformations=["random_resized_crop", "horizontal_flip", "gaussian_blur", "color_jitter"],
    cooccurrences_filepath=None, clips_directory=None, num_frames=10, mode="positive_negative",
    K=20, total_frames=20, zfill_num=4, is_override=False, override_value=None, masks=None, 
    apply_mask_percentage=1.0, device="cpu"
):

    batch_size = 1 # This is fixed. 
    shuffle = False # dataset is random by default, so this doesn't matter; set to False.

    assert mode in ["Train", "Test", "video_only", "positive_negative"], f"Invalid mode: {mode}!"
    mode_ = None
    if mode == "Test" or mode == "video_only":
        mode_ = "video_only"
        is_override = False #
        override_value = None #
    else:
        mode_ = "positive_negative"

    with open(cooccurrences_filepath, 'r') as file:
        cooccurrences = json.load(file)

    if transformations is not None:
        #print(f"transformations: {transformations}")
        meerkat_transforms = get_meerkat_transforms(transformations)
    else:
        meerkat_transforms = None

    dataset = AnimalClipDataset(
        directory=clips_directory, animal_cooccurrences=cooccurrences, 
        transformations=meerkat_transforms, K=K, num_frames=num_frames, 
        total_frames=total_frames, mode=mode_, zfill_num=zfill_num, 
        is_override=is_override, override_value=override_value, masks=masks,
        apply_mask_percentage=apply_mask_percentage, device=device
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    # note: a non-1 batch size will result in an error. 

    return dataloader

if __name__ == "__main__":
    dataloader = dataloader_creation(
        transforms=["random_resized_crop", "horizontal_flip", "gaussian_blur", "color_jitter"],
        cooccurrences_filepath="../Dataset/polarbears_h5files/Cooccurrences.json", 
        clips_directory="/home/kkno604/data/polarbear_data/Train/", num_frames=5, mode="Train",
        K=20, total_frames=20, zfill_num=4, override_value=None
    )

    for i, (positive, negative) in enumerate(dataloader):
        print(f"i: {i}, data: {positive[0].size()}")
        if i == 1:
            break