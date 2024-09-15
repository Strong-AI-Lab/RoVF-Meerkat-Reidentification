#!/bin/bash

# Array of checkpoint file paths
checkpoints=(
    "results/hyperparameter_search/rovf_perc/50_50/0p5_fps/aug/checkpoint_epoch_1.pt"
    "results/hyperparameter_search/rovf_perc/50_50/0p5_fps/aug/checkpoint_epoch_1.pt"
    "results/hyperparameter_search/rovf_perc/mask/0p5_fps/aug/checkpoint_epoch_1.pt"
    "results/hyperparameter_search/rovf_perc/mask/0p5_fps/no_aug/checkpoint_epoch_1.pt"
    "results/hyperparameter_search/rovf_perc/no_mask/0p5_fps/aug/checkpoint_epoch_1.pt"
    "results/hyperparameter_search/rovf_perc/no_mask/0p5_fps/no_aug/checkpoint_epoch_1.pt"
)

# Mask file
mask_file="Dataset/meerkat_h5files/masks/meerkat_masks.pkl"

# Dataset file
dataset_file="Dataset/meerkat_h5files/Precomputed_test_examples_meerkat.csv"

# Device
device="cuda"

# Loop over each checkpoint and run the command with and without the mask
for cp in "${checkpoints[@]}"; do
    echo "Running with mask for checkpoint: $cp"
    python main.py test '' -cp "$cp" -df "$dataset_file" -d "$device" -m "$mask_file"
    
    echo "Running without mask for checkpoint: $cp"
    python main.py test '' -cp "$cp" -df "$dataset_file" -d "$device"
done
