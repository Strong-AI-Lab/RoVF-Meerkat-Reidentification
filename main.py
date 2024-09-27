import argparse
from training_functions.process_yaml import process_yaml_for_training

import torch
import torch.nn as nn
import torch.nn.functional as F

#torch.autograd.set_detect_anomaly(True)

import numpy as np

import transformers
from transformers import AutoModel

import time
import pickle
import os
import yaml
import random

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

from training_functions.load_model_helper import load_model_from_checkpoint

def get_emb(args):
    from evaluation.get_embeddings import get_embeddings
    from dataloaders.ReID import AnimalClipDataset
    from training_functions.dataloader_helper import dataloader_creation as get_dataloader

    from models.dinov2_wrapper import DINOv2VideoWrapper
    from models.perceiver_wrapper import CrossAttention, TransformerEncoder, TransformerDecoder, Perceiver
    from models.recurrent_wrapper import RecurrentWrapper
    from models.recurrent_decoder import RecurrentDecoder
    from training_functions.load_model_helper import dino_model_load, recurrent_model_perceiver_load, load_model_from_checkpoint
    
    masks = None
    if args.mask_path != "" and args.mask_path is not None:
        with open(args.mask_path, "rb") as f:
            masks = pickle.load(f, encoding='latin1')

    embeddings = get_embeddings(
        model_ckpt=args.ckpt_path, transformations=None, cooccurrences_filepath=args.cooccurrences_filepath, 
        clips_directory=args.clips_directory, num_frames=args.num_frames, mode="Test", K=args.K, 
        total_frames=args.total_frames, zfill_num=args.zfill_num, is_override=False, 
        override_value=None, masks=masks, apply_mask_percentage=args.apply_mask_percentage, 
        device=args.device
    )

    # remove .pt from ckpt_path and add _embeddings.pkl
    embeddings_filename = args.ckpt_path[:-3] + "_embeddings"
    if masks is not None:
        embeddings_filename += "_mask"
    embeddings_filename += ".pkl"
    
    with open(embeddings_filename, "wb") as f:
        pickle.dump(embeddings, f)

def get_metrics(args):
    from evaluation.get_metrics import get_metrics, compute_distances, indices_of_smallest, open_pickle
    import pandas as pd

    models = [args.embedding_path]
    # Load dataframe of test examples
    df = pd.read_csv(args.dataframe_path)
    
    # Get metrics for all models
    metrics = get_metrics(models, df)
    
    # Print the results
    for i, (top1, top3, unique_top3) in enumerate(metrics):
        print(f"Model {i}: Top-1 Accuracy: {top1}, Top-3 Accuracy: {top3}, Unique in Top-3: {unique_top3}")

    # save results in same directory as embeddings (assume .pkl file extention)
    with open(args.embedding_path[:-4] + ("_mask" if args.mask_path is not None else "") + "_metrics.txt", "w") as f:
        f.write(f"Model {i}: Top-1 Accuracy: {top1}, Top-3 Accuracy: {top3}, Unique in Top-3: {unique_top3}")

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="A script that processes command-line inputs.")

    # Add arguments
    parser.add_argument("mode", type=str, choices=["train","test", "get_metrics", "get_embeddings"], help="The mode to run the script in. Allowed values: train, test.")
    parser.add_argument("yaml_path", type=str, help="Path to a yaml file specifying important details for this script.")
    parser.add_argument("-d", "--device", default="cpu", type=str, help="The device to run the script on. Default: cuda.")
    parser.add_argument("-cp", "--ckpt_path", default="", type=str, help="If resuming training, load previous model weights from this checkpoint.")
    parser.add_argument("-m", "--mask_path", default=None, type=str, help="Path to a pickle file containing masks for the dataset.")
    parser.add_argument("-am", "--apply_mask_percentage", default=1.0, type=float, help="The percentage of masks to apply to the dataset. Default: 1.0.")
    parser.add_argument("-o", "--override_value", default=None, type=int, help="If overriding the number of frames, use this value.")
    parser.add_argument("-is", "--is_override", default=False, type=bool, help="If overriding the number of frames, set this to True.")
    parser.add_argument("-z", "--zfill_num", default=4, type=int, help="The number of zeros to pad the frame number with. Default: 4.")
    parser.add_argument("-tf", "--total_frames", default=20, type=int, help="The total number of frames in a clip. Default: 20.")
    parser.add_argument("-K", "--K", default=20, type=int, help="The number of clips to sample. Default: 20.")
    parser.add_argument("-nf", "--num_frames", default=10, type=int, help="The number of frames to sample from a clip. Default: 10.")
    parser.add_argument("-dlm", "--dlmode", default="Test", type=str, help="The mode to run the script in. Default: Test.")
    parser.add_argument("-cd", "--clips_directory", default="Dataset/meerkat_h5files/clips/Test", type=str, help="The directory containing the clips.")
    parser.add_argument("-co", "--cooccurrences_filepath", default="Dataset/meerkat_h5files/Cooccurrences.json", type=str, help="The path to the cooccurrences file.")
    parser.add_argument("-ep", "--embedding_path", default=None, type=str, help="The path to save the embeddings to or load the embeddings.")
    parser.add_argument("-df", "--dataframe_path", default="Dataset/meerkat_h5files/Precomputed_test_examples_meerkat.csv", type=str, help="The path to the dataframe to load.")
    parser.add_argument("-lnev", "--ln_epsilon_value", default=None, type=float, help="The value to set the LayerNorm epsilon")
    parser.add_argument("-nan", "--detect_nan", default=False, type=bool, help="Detect NaN values in the model.")

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    mode = args.mode
    yaml_path = args.yaml_path
    ckpt_path = args.ckpt_path
    device = args.device


    if mode == "train":
        # Process the yaml file
        #data = process_yaml_for_training(yaml_path)
        data = None
        with open(yaml_path, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        train(data, device, ckpt_path)
    elif mode == "get_embeddings":
        assert ckpt_path != "" and ckpt_path is not None, "Need a checkpoint path to load the model."
        assert args.clips_directory != "" and args.clips_directory is not None, "Need a clips directory to load the clips."
        assert args.cooccurrences_filepath != "" and args.cooccurrences_filepath is not None, "Need a cooccurrences file to load the cooccurrences."
        
        get_emb(args)

    elif mode == "get_metrics":
        if args.embedding_path == "" or args.embedding_path is None:
            assert ckpt_path != "" and ckpt_path is not None, "Need a checkpoint path to load the model."
            print(f"Embedding path not provided. Will load embeddings from {ckpt_path[:-3] + '_embeddings.pkl'}")
            args.embedding_path = ckpt_path[:-3] + "_embeddings.pkl"
            # if mask path is provided or not add this to the filename
            if args.mask_path is not None:
                args.embedding_path = args.embedding_path[:-4] + "_mask.pkl"
            if not os.path.exists(args.embedding_path):
                raise FileNotFoundError(f"Embeddings file {args.embedding_path} does not exist.")
        assert args.dataframe_path != "" and args.dataframe_path is not None, "Need a path to the dataframe."
        
        get_metrics(args)
    elif mode == "test":
        assert ckpt_path != "" and ckpt_path is not None, "Need a checkpoint path to load the model."
        
        assert args.dataframe_path != "" and args.dataframe_path is not None, "Need a path to the dataframe."
        assert args.clips_directory != "" and args.clips_directory is not None, "Need a clips directory to load the clips."
        assert args.cooccurrences_filepath != "" and args.cooccurrences_filepath is not None, "Need a cooccurrences file to load the cooccurrences."
        
        get_emb(args)

        if args.embedding_path == "" or args.embedding_path is None:
            print(f"Embedding path not provided. Will load embeddings from {ckpt_path[:-3] + '_embeddings.pkl'}")
            args.embedding_path = ckpt_path[:-3] + "_embeddings.pkl"
            if args.mask_path is not None:
                args.embedding_path = args.embedding_path[:-4] + "_mask.pkl"
            if not os.path.exists(args.embedding_path):
                raise FileNotFoundError(f"Embeddings file {args.embedding_path} does not exist.")

        get_metrics(args) 
        
    else:
        raise ValueError(f"Invalid mode: {mode}.")

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
    from training_functions.load_model_helper import set_layernorm_eps_recursive, set_dropout_p_recursive

    model = None
    anchor_model = None
    if yaml_dict["model_details"]["model_type"] == "dino":
        #import_dino_model()
        from models.dinov2_wrapper import DINOv2VideoWrapper
        from training_functions.load_model_helper import dino_model_load as model_load_helper
        from training_functions.dataloader_helper import dataloader_creation as get_dataloader
        from get_anchors.anchor_fn import anchor_fn_hard, anchor_fn_hard_rand_anchor, anchor_fn_semi_hard
        config = {
            "dino_model_name": yaml_dict["model_details"]["dino_model_name"],
            "output_dim": yaml_dict["model_details"]["output_dim"],
            "forward_strat": yaml_dict["model_details"]["forward_strat"],
            "sequence_length": yaml_dict["model_details"]["sequence_length"],
            "num_frames": yaml_dict["model_details"]["num_frames"],
            "dropout_rate": yaml_dict["model_details"]["dropout_rate"]
        }
        model = model_load_helper(**config)

        if yaml_dict["training_details"]["anchor_dino_model"] is not None:
            config2 = {
                "dino_model_name": yaml_dict["training_details"]["anchor_dino_model"],
                "output_dim": None,
                "forward_strat": "average",
                "sequence_length": None,
                "num_frames": yaml_dict["dataloader_details"]["num_frames"],
                "dropout_rate": 0.0
            }
            anchor_model = model_load_helper(**config2)

        set_dropout_p_recursive(model, yaml_dict["model_details"]["dropout_rate"])
            
    elif yaml_dict["model_details"]["model_type"] == "recurrent" or yaml_dict["model_details"]["model_type"] == "recurrent_perceiver":
        #import_recurrence_model()
        from models.perceiver_wrapper import CrossAttention, TransformerEncoder, Perceiver
        from models.recurrent_wrapper import RecurrentWrapper
        from models.dinov2_wrapper import DINOv2VideoWrapper
        from training_functions.load_model_helper import recurrent_model_perceiver_load as model_load_helper
        from training_functions.load_model_helper import dino_model_load as anchor_model_load_helper
        from training_functions.dataloader_helper import dataloader_creation as get_dataloader
        from get_anchors.anchor_fn import anchor_fn_hard, anchor_fn_hard_rand_anchor, anchor_fn_semi_hard
        perc_config = {
            "raw_input_dim": yaml_dict["model_details"]["raw_input_dim"],
            "embedding_dim": yaml_dict["model_details"]["embedding_dim"],
            "latent_dim": yaml_dict["model_details"]["latent_dim"],
            "num_heads": yaml_dict["model_details"]["num_heads"],
            "num_latents": yaml_dict["model_details"]["num_latents"],
            "num_transformer_layers": yaml_dict["model_details"]["num_tf_layers"],
            "dropout": yaml_dict["model_details"]["dropout_rate"],
            "output_dim": yaml_dict["model_details"]["output_dim"],
            "use_raw_input": yaml_dict["model_details"]["use_raw_input"],
            "use_embeddings": yaml_dict["model_details"]["use_embeddings"],
            "flatten_channels": yaml_dict["model_details"]["flatten_channels"]
        }
        assert isinstance(yaml_dict["model_details"]["use_raw_input"], bool) and isinstance(yaml_dict["model_details"]["use_embeddings"], bool) \
            and isinstance(yaml_dict["model_details"]["flatten_channels"], bool), "use_raw_input, use_embeddings, and flatten_channels must be boolean."
        config = {
            "perceiver_config": perc_config,
            "dino_model_name": yaml_dict["model_details"]["dino_model_name"],
            "dropout_rate": yaml_dict["model_details"]["dropout_rate"],
            "freeze_image_model": yaml_dict["model_details"]["freeze_image_model"],
            "is_append_avg_emb": yaml_dict["model_details"]["is_append_avg_emb"] if "is_append_avg_emb" in yaml_dict["model_details"].keys() else False
        }
        model = model_load_helper(**config)
        
        if yaml_dict["training_details"]["anchor_dino_model"] is not None:
            config2 = {
                "dino_model_name": yaml_dict["training_details"]["anchor_dino_model"],
                "output_dim": None,
                "forward_strat": "average",
                "sequence_length": None,
                "num_frames": yaml_dict["dataloader_details"]["num_frames"],
                "dropout_rate": 0.0
            }
            anchor_model = anchor_model_load_helper(**config2)

    elif yaml_dict["model_details"]["model_type"] == "recurrent_perceiverv2":
        #import_recurrence_model()
        #print(f"RECCCC")
        from models.perceiver_wrapper import CrossAttention, TransformerEncoder
        from models.perceiver_wrapper import PerceiverV2 as Perceiver
        from models.recurrent_wrapper import RecurrentWrapper
        from models.dinov2_wrapper import DINOv2VideoWrapper
        from training_functions.load_model_helper import recurrent_model_perceiver_loadv2 as model_load_helper
        from training_functions.load_model_helper import dino_model_load as anchor_model_load_helper
        from training_functions.dataloader_helper import dataloader_creation as get_dataloader
        from get_anchors.anchor_fn import anchor_fn_hard, anchor_fn_hard_rand_anchor, anchor_fn_semi_hard
        perc_config = {
            "raw_input_dim": yaml_dict["model_details"]["raw_input_dim"],
            "embedding_dim": yaml_dict["model_details"]["embedding_dim"],
            "latent_dim": yaml_dict["model_details"]["latent_dim"],
            "num_heads": yaml_dict["model_details"]["num_heads"],
            "num_latents": yaml_dict["model_details"]["num_latents"],
            "num_transformer_layers": yaml_dict["model_details"]["num_tf_layers"],
            "dropout": yaml_dict["model_details"]["dropout_rate"],
            "output_dim": yaml_dict["model_details"]["output_dim"],
            "use_raw_input": yaml_dict["model_details"]["use_raw_input"],
            "use_embeddings": yaml_dict["model_details"]["use_embeddings"],
            "flatten_channels": yaml_dict["model_details"]["flatten_channels"]
        }
        assert isinstance(yaml_dict["model_details"]["use_raw_input"], bool) and isinstance(yaml_dict["model_details"]["use_embeddings"], bool) \
            and isinstance(yaml_dict["model_details"]["flatten_channels"], bool), "use_raw_input, use_embeddings, and flatten_channels must be boolean."
        config = {
            "perceiver_config": perc_config,
            "dino_model_name": yaml_dict["model_details"]["dino_model_name"],
            "dropout_rate": yaml_dict["model_details"]["dropout_rate"],
            "freeze_image_model": yaml_dict["model_details"]["freeze_image_model"],
            "is_append_avg_emb": yaml_dict["model_details"]["is_append_avg_emb"] if "is_append_avg_emb" in yaml_dict["model_details"].keys() else False
        }
        model = model_load_helper(**config)
        
        if yaml_dict["training_details"]["anchor_dino_model"] is not None:
            config2 = {
                "dino_model_name": yaml_dict["training_details"]["anchor_dino_model"],
                "output_dim": None,
                "forward_strat": "average",
                "sequence_length": None,
                "num_frames": yaml_dict["dataloader_details"]["num_frames"],
                "dropout_rate": 0.0
            }
            anchor_model = anchor_model_load_helper(**config2)

    elif yaml_dict["model_details"]["model_type"] == "recurrent_decoder":
        from models.recurrent_decoder import RecurrentDecoder
        from models.dinov2_wrapper import DINOv2VideoWrapper
        from training_functions.dataloader_helper import dataloader_creation as get_dataloader
        from training_functions.load_model_helper import dino_model_load as anchor_model_load_helper
        from get_anchors.anchor_fn import anchor_fn_hard, anchor_fn_hard_rand_anchor, anchor_fn_semi_hard
        model = RecurrentDecoder(
            v_size=yaml_dict["model_details"]["v_size"],
            d_model=yaml_dict["model_details"]["d_model"],
            nhead=yaml_dict["model_details"]["nhead"],
            num_layers=yaml_dict["model_details"]["num_layers"],
            dim_feedforward=yaml_dict["model_details"]["dim_feedforward"],
            dropout=yaml_dict["model_details"]["dropout_rate"], 
            activation=yaml_dict["model_details"]["activation"],
            temperature=yaml_dict["model_details"]["temperature"],
            image_model_name=yaml_dict["model_details"]["image_model_name"],
            freeze_image_model=yaml_dict["model_details"]["freeze_image_model"]
        )

        if yaml_dict["training_details"]["anchor_dino_model"] is not None:
            config2 = {
                "dino_model_name": yaml_dict["training_details"]["anchor_dino_model"],
                "output_dim": None,
                "forward_strat": "average",
                "sequence_length": None,
                "num_frames": yaml_dict["dataloader_details"]["num_frames"],
                "dropout_rate": 0.0
            }
            anchor_model = anchor_model_load_helper(**config2)
        
        set_dropout_p_recursive(model, yaml_dict["model_details"]["dropout_rate"])

    elif yaml_dict["model_details"]["model_type"] in ["ResNet152", "ResNet18", "VGG-16", "ResNet50"]:
        from get_anchors.anchor_fn import anchor_fn_hard, anchor_fn_hard_rand_anchor, anchor_fn_semi_hard
        from training_functions.dataloader_helper import dataloader_creation as get_dataloader
        from training_functions.load_model_helper import image_model_load
        model = image_model_load(yaml_dict["model_details"]["model_type"], yaml_dict["model_details"]["embedding_dim"], training=True)
    else:
        raise ValueError("Invalid model type.")

    start_epoch = 0
    checkpoint = None
    if ckpt_path != "":
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] # assume epoch saved is +1 and train function starts at zero.

    model.to(device)
    if anchor_model is not None:
        anchor_model.to(device)


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
    if optimizer is not None and checkpoint is not None and "optimizer_state_dict" in checkpoint.keys():
        print(f"Loading optimizer state dict.")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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

    masks = None
    if yaml_dict["dataloader_details"]["mask_path"] != "" and isinstance(yaml_dict["dataloader_details"]["mask_path"], str):
        with open(yaml_dict["dataloader_details"]["mask_path"], "rb") as f:
            masks = pickle.load(f, encoding='latin1')
    apply_mask_percentage = yaml_dict["dataloader_details"]["apply_mask_percentage"]

    trainloader = get_dataloader(
        transformations=yaml_dict["dataloader_details"]["transformations"], # should be a list of strings
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
        K=5, #yaml_dict["dataloader_details"]["K"],
        total_frames=yaml_dict["dataloader_details"]["total_frames"],
        zfill_num=yaml_dict["dataloader_details"]["zfill_num"],
        is_override=False, override_value=None,
        masks=masks, apply_mask_percentage=apply_mask_percentage, device=device
    )

    if scheduler is not None:
        scheduler_current_iter = start_epoch * len(trainloader)
        scheduler.update_current_step(scheduler_current_iter)

    metadata = str(yaml_dict)

    if yaml_dict["training_details"]["anchor_function_details"]["type"] == "hard":
        anchor_fn = anchor_fn_hard
    elif yaml_dict["training_details"]["anchor_function_details"]["type"] == "semi_hard":
        anchor_fn = anchor_fn_semi_hard
    elif yaml_dict["training_details"]["anchor_function_details"]["type"] == "hard_rand_anchor":
        anchor_fn = anchor_fn_hard_rand_anchor
    else:
        raise ValueError(f'Invalid anchor function {yaml_dict["training_details"]["anchor_function_details"]["type"]}.')

    train(
        model, anchor_model, epochs, trainloader, valloader, anchor_fn, similarity_measure, optimizer, scheduler,
        criterion, device, log_path, metadata, batch_size, clip_value, start_epoch,
        accumulation_steps=yaml_dict["training_details"]["accumulation_steps"] if "accumulation_steps" in yaml_dict["training_details"] else 1,
        margin=yaml_dict["training_details"]["criterion_details"]["margin"],
    )

if __name__ == "__main__":
    main()
