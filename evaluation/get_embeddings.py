import sys
sys.path.append("..")

import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.io as io
import matplotlib.pyplot as plt

from collections import defaultdict
from PIL import Image

from dataloaders.ReID import AnimalClipDataset
from models.dinov2_wrapper import DINOv2VideoWrapper
from models.perceiver_wrapper import CrossAttention, TransformerEncoder, Perceiver
from models.recurrent_wrapper import RecurrentWrapper
from models.recurrent_decoder import RecurrentDecoder
from training_functions.load_model_helper import dino_model_load, recurrent_model_perceiver_load
from training_functions.dataloader_helper import dataloader_creation

import pickle

import transformers
from transformers import AutoModel 

from training_functions.load_model_helper import load_model_from_checkpoint

import yaml

import argparse

def get_embeddings(
    model_ckpt, transformations, cooccurrences_filepath, clips_directory, 
    num_frames, mode, K, total_frames, zfill_num, is_override, override_value, 
    masks, apply_mask_percentage, device
):
    
     # load ckpt here  
    mdata = torch.load(model_ckpt)["metadata"] # this is a string of a dictionary, how to load it?
    # Load the string into a dictionary using YAML (use safe_load for security)
    mdata = yaml.safe_load(mdata)
    
    num_frames = mdata['dataloader_details']["num_frames"] if "num_frames" in mdata['dataloader_details'].keys() else num_frames
    print(f"num_frames: {num_frames}")

    dataloader = dataloader_creation(
        transformations=transformations, cooccurrences_filepath=cooccurrences_filepath, 
        clips_directory=clips_directory, num_frames=num_frames, mode=mode,
        K=K, total_frames=total_frames, zfill_num=zfill_num, is_override=is_override, 
        override_value=override_value, masks=masks, apply_mask_percentage=apply_mask_percentage
    )

    model = load_model_from_checkpoint(model_ckpt).to(device) # model_ckpt_is the path to the model checkpoint
    model.eval()
    
    if hasattr(model, 'set_eval_mode'):
        model.set_eval_mode(True)

    embeddings = {}
    for i, (path, data) in enumerate(dataloader):
        with torch.no_grad():
            output = model(data.to(device))
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[-1]
            if len(output.size()) == 1:
                embeddings[path[0]] = output
            elif len(output.size()) == 2: # batch processing
                assert len(path) == output.size(0)
                for j in range(output.size(0)):
                    embeddings[path[j]] = output[j]
            else:
                raise Exception(f"output size not recognized: {output.size()}")
    print(f"len(embeddings): {len(embeddings)}")
    return embeddings

def main(args):
    load_masks = args.load_masks
    masks = None
    if load_masks:
        mask_path = args.mask_path
        with open(mask_path, "rb") as f:
            masks = pickle.load(f)

    dataloader = dataloader_creation(
        transformations=None, 
        cooccurrences_filepath=args.cooccurrences_filepath,
        clips_directory=args.clips_directory, 
        num_frames=args.num_frames, 
        mode=args.mode,
        K=args.K, 
        total_frames=args.total_frames, 
        zfill_num=args.zfill_num, 
        is_override=args.is_override, 
        override_value=args.override_value, 
        masks=masks, 
        apply_mask_percentage=args.apply_mask_percentage
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = dino_model_load(
        dino_model_name=args.dino_model_name, 
        output_dim=args.output_dim, 
        forward_strat=args.forward_strat, 
        sequence_length=args.sequence_length, 
        num_frames=args.num_frames, 
        dropout_rate=args.dropout_rate
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    print(f"len(dataloader): {len(dataloader)}")

    embeddings = {}
    for i, (path, data) in enumerate(dataloader):
        with torch.no_grad():
            output = model(data.to(device))
            if isinstance(output, tuple) or isinstance(output, list):
                output = output[-1]
            if len(output.size()) == 1:
                embeddings[path[0]] = output
            elif len(output.size()) == 2:
                assert len(path) == output.size(0)
                for j in range(output.size(0)):
                    embeddings[path[j]] = output[j]
            else:
                raise Exception(f"output size not recognized: {output.size()}")

    print(f"len(embeddings): {len(embeddings)}")

    # Save the embeddings
    with open(args.output_file, "wb") as f:
        pickle.dump(embeddings, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save DINO embeddings")
    
    parser.add_argument("--load_masks", action="store_true", help="Whether to load masks")
    parser.add_argument("--mask_path", type=str, default="/path/to/mask.pkl", help="Path to mask file")
    parser.add_argument("--cooccurrences_filepath", type=str, required=True, help="Path to cooccurrences file")
    parser.add_argument("--clips_directory", type=str, required=True, help="Directory containing clips")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames")
    parser.add_argument("--mode", type=str, default="Test", help="Mode (Train/Test/Val)")
    parser.add_argument("--K", type=int, default=20, help="K value")
    parser.add_argument("--total_frames", type=int, default=20, help="Total frames")
    parser.add_argument("--zfill_num", type=int, default=4, help="Zero fill number")
    parser.add_argument("--is_override", action="store_true", help="Whether to override")
    parser.add_argument("--override_value", type=float, default=None, help="Override value")
    parser.add_argument("--apply_mask_percentage", type=float, default=1.0, help="Mask application percentage")
    
    parser.add_argument("--dino_model_name", type=str, default="facebook/dinov2-small", help="DINO model name")
    parser.add_argument("--output_dim", type=int, default=None, help="Output dimension")
    parser.add_argument("--forward_strat", type=str, default="cls", help="Forward strategy")
    parser.add_argument("--sequence_length", type=int, default=None, help="Sequence length")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output_file", type=str, required=True, help="Output file path for embeddings")

    args = parser.parse_args()
    main(args)