#!/bin/bash

# Script to run hyperparameter search for RoVF_S_st_bioclip model
# This script will train the model on all generated YAML configurations

# Set the base directory
BASE_DIR="/data/kkno604/github/RoVF-Meerkat-Reidentification"
YAML_DIR="$BASE_DIR/training_scripts/exp_metadata/hyperparameter_search/RoVF_S_st_bioclip"
MAIN_SCRIPT="$BASE_DIR/main.py"

# Check if the YAML directory exists
if [ ! -d "$YAML_DIR" ]; then
    echo "Error: YAML directory does not exist: $YAML_DIR"
    echo "Please run the generate_yml.py script first to create the YAML files."
    exit 1
fi

# Check if main.py exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: main.py not found at: $MAIN_SCRIPT"
    exit 1
fi

# Change to the base directory
cd "$BASE_DIR" || exit 1

# Counter for tracking progress
counter=0
total_files=$(find "$YAML_DIR" -name "*.yml" | wc -l)

echo "Starting RoVF_S_st_bioclip hyperparameter search..."
echo "Found $total_files YAML configuration files"
echo "================================================"

# Loop through all YAML files in the directory
for yaml_file in "$YAML_DIR"/*.yml; do
    # Check if file exists (in case no .yml files are found)
    if [ ! -f "$yaml_file" ]; then
        echo "No YAML files found in $YAML_DIR"
        break
    fi
    
    # Increment counter
    counter=$((counter + 1))
    
    # Extract filename for logging
    filename=$(basename "$yaml_file")
    
    echo "[$counter/$total_files] Training with configuration: $filename"
    echo "Started at: $(date)"
    
    # Run the training command
    CUDA_VISIBLE_DEVICES=3 python main.py train "$yaml_file" -d cuda
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed training for: $filename"
    else
        echo "✗ Training failed for: $filename"
        echo "Continuing with next configuration..."
    fi
    
    echo "Finished at: $(date)"
    echo "------------------------------------------------"
done

echo "RoVF_S_st_bioclip hyperparameter search completed!"
echo "Processed $counter configuration files"
echo "Check the results directory for outputs"
