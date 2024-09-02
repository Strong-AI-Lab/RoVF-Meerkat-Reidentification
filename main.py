import argparse
from training_functions.process_yaml import process_yaml_for_training

import torch
import torch.nn as nn

import numpy as np

import transformers
from transformers import AutoModel

import time

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

def import_recurrence_model():
    from models.perceiver_wrapper import CrossAttention, TransformerEncoder, Perceiver
    from models.recurrent_wrapper import RecurrentWrapper
    from training_functions.load_model_helper import recurrent_model_perceiver_load as model_load_helper
    from training_functions.dataloader_helper import dataloader_creation as get_dataloader
    from get_anchors.recurrence_model import get_anchor_pos_and_neg_recurrence as anchor_fn

def import_dino_model():
    from models.dinov2_wrapper import DINOv2Wrapper
    from training_functions.load_model_helper import dino_model_load as model_load_helper
    from training_functions.dataloader_helper import dataloader_creation as get_dataloader
    from get_anchors.image_model import get_anchor_pos_and_neg_dino as anchor_fn

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A script that processes command-line inputs.")

    # Add arguments
    parser.add_argument("mode", type=str, choices=["train","test"], help="The mode to run the script in. Allowed values: train, test.")
    parser.add_argument("yaml_path", type=str, help="Path to a yaml file specifying important details for this script.")
    parser.add_argument("-d", "--device", default="cpu", type=str, help="The device to run the script on. Default: cuda.")
    parser.add_argument("-hs", "--hyperparameter_search", action="store_true", help="If True, run hyperparameter search.")
    #parser.add_argument("-ce", "--current_epoch", default=0, type=int, help="If resuming training, insert the current epoch to resume training from.")
    parser.add_argument("-cp", "--ckpt_path", default="", type=str, help="If resuming training, load previous model weights from this checkpoint.")

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    mode = args.mode
    yaml_path = args.yaml_path
    #current_epoch = args.current_epoch
    ckpt_path = args.ckpt_path
    device = args.device

    # Process the yaml file
    data = process_yaml_for_training(yaml_path)

    if mode == "train":
        train(data, device, ckpt_path)

def train(yaml_dict, device, ckpt_path):

    from dataloaders.ReID import AnimalClipDataset
    from augmentations.simclr_augmentations import get_meerkat_transforms
    from training_functions.dataloader_helper import dataloader_creation as get_dataloader
    from training_functions.train import train
    from training_functions.val import val
    import json
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.v2 as transforms
    from torchvision.transforms.v2.functional import InterpolationMode
    import random
    import h5py
    import glob
    import os
    from collections import defaultdict
    from PIL import Image
    import pickle

    model = None
    if yaml_dict["model_details"]["model_type"] == "dino":
        #import_dino_model()
        from models.dinov2_wrapper import DINOv2VideoWrapper
        from training_functions.load_model_helper import dino_model_load as model_load_helper
        from training_functions.dataloader_helper import dataloader_creation as get_dataloader
        from get_anchors.image_model import get_anchor_pos_and_neg_dino as anchor_fn
        config = {
            "dino_model_name": yaml_dict["model_details"]["dino_model_name"],
            "output_dim": yaml_dict["model_details"]["output_dim"],
            "forward_strat": yaml_dict["model_details"]["strategy"],
            "sequence_length": yaml_dict["model_details"]["sequence_length"],
            "num_frames": yaml_dict["model_details"]["num_frames"],
            "dropout_rate": yaml_dict["model_details"]["dropout_rate"]
        }
        model = model_load_helper(**config)
    elif yaml_dict["model_details"]["model_type"] == "recurrent" or yaml_dict["model_details"]["model_type"] == "recurrent_perceiver":
        #import_recurrence_model()
        from models.perceiver_wrapper import CrossAttention, TransformerEncoder, Perceiver
        from models.recurrent_wrapper import RecurrentWrapper
        from training_functions.load_model_helper import recurrent_model_perceiver_load as model_load_helper
        from training_functions.dataloader_helper import dataloader_creation as get_dataloader
        from get_anchors.recurrence_model import get_anchor_pos_and_neg_recurrence as anchor_fn
        perc_config = {
            "input_dim": yaml_dict["model_details"]["input_dim"],
            "latent_dim": yaml_dict["model_details"]["latent_dim"],
            "num_heads": yaml_dict["model_details"]["num_heads"],
            "num_latents": yaml_dict["model_details"]["num_latents"],
            "num_transformer_layers": yaml_dict["model_details"]["num_tf_layers"],
            "dropout": yaml_dict["model_details"]["dropout_rate"],
            "output_dim": yaml_dict["model_details"]["output_dim"]
        }
        config = {
            "perceiver_config": perc_config,
            "dino_model_name": yaml_dict["model_details"]["dino_model_name"],
            "dropout_rate": yaml_dict["model_details"]["dropout_rate"],
            "freeze_image_model": yaml_dict["model_details"]["freeze_image_model"]
        }
        model = model_load_helper(**config)
    else:
        raise ValueError("Invalid model type.")

    start_epoch = 0
    if ckpt_path != "":
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] # assume epoch saved is +1 and train function starts at zero.

    model.to(device)

    criterion = None
    if yaml_dict["training_details"]["criterion_details"]["name"] == "triplet_margin_loss":
        criterion = nn.TripletMarginLoss(
            margin=yaml_dict["training_details"]["criterion_details"]["margin"],
            p=yaml_dict["training_details"]["criterion_details"]["p"]
        )
    else:
        raise ValueError("Invalid criterion.")

    similarity_measure = None
    if yaml_dict["training_details"]["anchor_function_details"]["similarity_measure"] == "euclidean_distance":
        similarity_measure = lambda anchor, other: torch.sqrt(torch.sum((anchor - other) ** 2, dim=-1))
    else:
        raise ValueError("Invalid similarity measure.")

    
    epochs = yaml_dict["training_details"]["epochs"]
    model_type = yaml_dict["model_details"]["model_type"]
    batch_size = yaml_dict["training_details"]["batch_size"]
    log_path = yaml_dict["training_details"]["log_directory"]
    clip_value = yaml_dict["training_details"]["clip_value"]

    # already imported above
    #if yaml_dict["training_details"]["anchor_function_details"]["type"] == "dino":
    #    from get_anchors.recurrence_model import get_anchor_pos_and_neg_dino as anchor_fn
    #elif yaml_dict["training_details"]["anchor_function_details"]["type"] == "recurrent":
    #    from get_anchors.recurrence_model import get_anchor_pos_and_neg_recurrence as anchor_fn
    #else:
    #    raise ValueError("Invalid anchor function type.")

    optimizer = None
    if yaml_dict["training_details"]["optimizer_details"]["name"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=yaml_dict["training_details"]["optimizer_details"]["lr"],
            weight_decay=yaml_dict["training_details"]["optimizer_details"]["weight_decay"]
        )
    elif yaml_dict["training_details"]["optimizer_details"]["name"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=yaml_dict["training_details"]["optimizer_details"]["lr"]
        )
    else:
        raise ValueError("Invalid optimizer.")

    scheduler = None
    if yaml_dict["training_details"]["scheduler_details"]["name"] == "WarmupCosineDecayScheduler":
        from lr_schedulers.cosine_decay import WarmupCosineDecayScheduler as LRScheduler
        scheduler = LRScheduler(
            optimizer=optimizer,
            warmup_steps=yaml_dict["training_details"]["scheduler_details"]["warmup_steps"],
            decay_steps=yaml_dict["training_details"]["scheduler_details"]["decay_steps"],
            start_lr=yaml_dict["training_details"]["scheduler_details"]["start_lr"],
            max_lr=yaml_dict["training_details"]["scheduler_details"]["max_lr"],
            end_lr=yaml_dict["training_details"]["scheduler_details"]["end_lr"]
        )
        scheduler.update_current_step(start_epoch)

    masks = None
    if yaml_dict["dataloader_details"]["mask_path"] != "" and isinstance(yaml_dict["dataloader_details"]["mask_path"], str):
        with open(yaml_dict["dataloader_details"]["mask_path"], "r") as f:
            masks = pickle.load(f)
    apply_mask_percentage = yaml_dict["dataloader_details"]["apply_mask_percentage"]

    trainloader = get_dataloader(
        transformations=yaml_dict["dataloader_details"]["transforms"], # should be a list of strings
        cooccurrences_filepath=yaml_dict["dataloader_details"]["cooccurrences_filepath"],
        clips_directory=yaml_dict["dataloader_details"]["clips_directory"]+"Train/",
        num_frames=yaml_dict["dataloader_details"]["num_frames"],
        mode=yaml_dict["dataloader_details"]["mode"],
        K=yaml_dict["dataloader_details"]["K"],
        total_frames=yaml_dict["dataloader_details"]["total_frames"],
        zfill_num=yaml_dict["dataloader_details"]["zfill_num"],
        is_override=yaml_dict["dataloader_details"]["is_override"],
        override_value=yaml_dict["dataloader_details"]["override_value"],
        masks=masks, apply_mask_percentage=apply_mask_percentage, device=device

    )
    valloader = get_dataloader(
        transformations=None,
        cooccurrences_filepath=yaml_dict["dataloader_details"]["cooccurrences_filepath"],
        clips_directory=yaml_dict["dataloader_details"]["clips_directory"]+"Val/",
        num_frames=yaml_dict["dataloader_details"]["num_frames"],
        mode=yaml_dict["dataloader_details"]["mode"],
        K=yaml_dict["dataloader_details"]["K"],
        total_frames=yaml_dict["dataloader_details"]["total_frames"],
        zfill_num=yaml_dict["dataloader_details"]["zfill_num"],
        is_override=False, override_value=None,
        masks=None, apply_mask_percentage=1.0, device=device
    )

    metadata = str(yaml_dict)

    train(
        model, model_type, epochs, trainloader, valloader, anchor_fn, similarity_measure, optimizer, scheduler,
        criterion, device, log_path, metadata, batch_size, clip_value, start_epoch
    )


# TODO: this may not be needed.
def hyperparameter_search():
    pass

def test():
    pass

if __name__ == "__main__":
    main()
