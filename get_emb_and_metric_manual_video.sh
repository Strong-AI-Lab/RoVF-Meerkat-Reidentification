#!/bin/bash

# Function to prompt user for animal choice
choose_animal() {
    echo "Choose an animal: "
    echo "1) Meerkats"
    echo "2) Polarbears"
    read -p "Enter the number corresponding to your choice: " choice
    case $choice in
        1)
            echo "Meerkats selected."
            animal="meerkat"
            ;;
        2)
            echo "Polarbears selected."
            animal="polarbear"
            ;;
        *)
            echo "Invalid choice. Please choose either 1 (Meerkats) or 2 (Polarbears)."
            choose_animal
            ;;
    esac
}

# Function to prompt user for device choice
choose_device() {
    echo "Choose a device: "
    echo "1) CPU"
    echo "2) CUDA"
    read -p "Enter the number corresponding to your choice: " device_choice
    case $device_choice in
        1)
            echo "CPU selected."
            device="cpu"
            ;;
        2)
            echo "CUDA selected."
            device="cuda"
            ;;
        *)
            echo "Invalid choice. Please choose either 1 (CPU) or 2 (CUDA)."
            choose_device
            ;;
    esac
}

# Call the function to make the animal choice
choose_animal

# Call the function to make the device choice
choose_device

# Set paths based on the chosen animal
if [ "$animal" == "meerkat" ]; then
    mask_file="Dataset/meerkat_h5files/masks/meerkat_masks.pkl"
    dataset_file="Dataset/meerkat_h5files/Precomputed_test_examples_meerkat.csv"
    cooccurrences_file="Dataset/meerkat_h5files/Cooccurrences.json"
    clips_directory="Dataset/meerkat_h5files/clips/Test/"
elif [ "$animal" == "polarbear" ]; then
    mask_file="Dataset/polarbears_h5files/masks/PB_masks.pkl"
    dataset_file="Dataset/polarbears_h5files/Precomputed_test_examples_polarbear.csv"
    cooccurrences_file="Dataset/polarbears_h5files/Cooccurrences.json"
    clips_directory="Dataset/polarbears_h5files/clips/Test/"
fi

# Prompt user for checkpoint file paths
checkpoints=()
echo "Enter checkpoint file paths (one per line). Enter an empty line when done:"
while true; do
    read -p "Checkpoint file: " cp
    if [ -z "$cp" ]; then
        break
    fi
    if [ -f "$cp" ]; then
        checkpoints+=("$cp")
    else
        echo "Warning: File '$cp' not found. Please enter a valid path."
    fi
done

# Loop over each checkpoint and run the command with and without the mask
for cp in "${checkpoints[@]}"; do
    echo "Running with mask for checkpoint: $cp"
    python main.py test '' -cp "$cp" -df "$dataset_file" -d "$device" -m "$mask_file" -cd "$clips_directory" -co "$cooccurrences_file"
    
    echo "Running without mask for checkpoint: $cp"
    python main.py test '' -cp "$cp" -df "$dataset_file" -d "$device" -cd "$clips_directory" -co "$cooccurrences_file"
done
