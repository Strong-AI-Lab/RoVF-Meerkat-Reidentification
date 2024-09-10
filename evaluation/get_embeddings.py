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

def get_embeddings(
    model_ckpt, transformations, cooccurrences_filepath, clips_directory, 
    num_frames, mode, K, total_frames, zfill_num, is_override, override_value, 
    masks, apply_mask_percentage, device
):
    
    dataloader = dataloader_creation(
        transformations=transformations, cooccurrences_filepath=cooccurrences_filepath, 
        clips_directory=clips_directory, num_frames=num_frames, mode=mode,
        K=K, total_frames=total_frames, zfill_num=zfill_num, is_override=is_override, 
        override_value=override_value, masks=masks, apply_mask_percentage=apply_mask_percentage
    )

    model = load_model_from_checkpoint(model_ckpt).to(device)
    model.eval()

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
    return embeddings

if __name__ == "__main__":

    #test_dataloader()

    load_masks = True
    masks = None
    if load_masks:
        mask_path = "/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/Dataset/meerkat_h5files/masks/meerkat_masks.pkl"
        with open(mask_path, "rb") as f:
            masks = pickle.load(f)

    dataloader = dataloader_creation(
        #num_frames=10, shuffle=True, mode_="Test", batch_size=30
        transformations=None, cooccurrences_filepath="/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/Dataset/meerkat_h5files/Cooccurrences.json", 
        clips_directory="/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/Dataset/meerkat_h5files/clips/Test", num_frames=10, mode="Test",
        K=20, total_frames=20, zfill_num=4, is_override=False, override_value=None, masks=masks, apply_mask_percentage=1.0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''
    # load model. 
    model = dino_model_load( # cat | average | max
        dino_model_name="facebook/dinov2-base", output_dim=768, forward_strat="max", sequence_length=None, num_frames=10, dropout_rate=0.1
    )
    # load checkpoint
    checkpoint = torch.load("/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/results/hyperparameter_search/dino_base_margin_0p5/checkpoint_epoch_1.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    '''

    '''
    model = recurrent_model_load(
        latent_dim=64, num_latents=64, output_dim=768, freeze_image_model=True
    )
    checkpoint = torch.load("/home/kkno604/experiments/meerkat_triplet_loss/recurrent_model_v2/recurrent_model.pth_epoch_3.pt")
    model.load_state_dict(checkpoint)
    '''
    perceiver_config = {
        "input_dim": 768,
        "latent_dim": 768,
        "num_heads": 12,
        "num_latents": 64,
        "num_transformer_layers": 2,
        "dropout": 0.1,
        "output_dim": 768
    }
    model = recurrent_model_perceiver_load(
        perceiver_config=perceiver_config, dino_model_name="facebook/dinov2-base", dropout_rate=0.1, freeze_image_model=True
    )
    checkpoint = torch.load("/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/results/hyperparameter_search/rovf_margin_0p5/checkpoint_epoch_5.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    #print(f"model: {model}")
    print(f"len(dataloader): {len(dataloader)}")

    # get the embeddings
    embeddings = {}
    for i, (path, data) in enumerate(dataloader):
        #print(f"data.size(): {data.size()}")
        with torch.no_grad():
            #output = model(data.permute(0, 1, 4, 2, 3).to(device))
            output = model(data.to(device))
            if isinstance(output, tuple) or isinstance(output, list):
                #print(f"type or list reached!")
                output = output[-1]
            if len(output.size()) == 1:
                #print(f"output.size() len of 1 reached!")
                embeddings[path[0]] = output
            elif len(output.size()) == 2: # batch processing
                #print(f"output.size() len of 2 reached!")
                #print(f"output.size(): {output.size()}")
                assert len(path) == output.size(0)
                for j in range(output.size(0)):
                    embeddings[path[j]] = output[j]
            else:
                raise Exception(f"output size not recognized: {output.size()}")
    #print(f"embeddings: {embeddings[list(embeddings.keys())[0]].size()}")

    print(f"len(embeddings): {len(embeddings)}")

    # save the embeddings
    with open("/home/kkno604/github/meerkat-repos/RoVF-meerkat-reidentification/results/hyperparameter_search/rovf_margin_0p5/checkpoint_epoch_5_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
