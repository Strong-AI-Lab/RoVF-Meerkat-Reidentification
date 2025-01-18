#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_name> <GPU_device_number>"
    exit 1
fi

# Read arguments
model_name=$1  # Model name (e.g., LSTM or another model directory)
GPU_device=$2  # GPU device number

# Fixed base directory
base_dir="training_scripts/exp_metadata/hyperparameter_search"

# Array of YAML file names
yaml_files=(
    "50_50_0p5_fps_aug.yml"
    "50_50_0p5_fps_no_aug.yml"
    "50_50_1_fps_aug.yml"
    "50_50_1_fps_no_aug.yml"
    "mask_0p5_fps_aug.yml"
    "mask_0p5_fps_no_aug.yml"
    "mask_1_fps_aug.yml"
    "mask_1_fps_no_aug.yml"
    "no_mask_0p5_fps_aug.yml"
    "no_mask_0p5_fps_no_aug.yml"
    "no_mask_1_fps_aug.yml"
    "no_mask_1_fps_no_aug.yml"
)

# Loop through each file and execute the command
for yaml_file in "${yaml_files[@]}"; do
    echo "Running training for $yaml_file in model directory $model_name with GPU $GPU_device..."
    CUDA_VISIBLE_DEVICES=$GPU_device python main.py train "${base_dir}/${model_name}/${yaml_file}" -d cuda
    if [ $? -ne 0 ]; then
        echo "Error occurred with $yaml_file. Exiting."
        exit 1
    fi
done

echo "All files processed successfully!"
