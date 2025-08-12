#!/bin/bash

# Base directory - go up one level since we're in the evaluation folder
BASE_DIR=".."

# Array of MegaDescriptor model names
#megadescriptor_models=("hf-hub:BVRA/MegaDescriptor-L-224" "hf-hub:BVRA/MegaDescriptor-B-224" "hf-hub:BVRA/MegaDescriptor-S-224" "hf-hub:BVRA/MegaDescriptor-T-224")
megadescriptor_models=("hf-hub:BVRA/MegaDescriptor-S-224")

# Array of forward strategies (only average and max, not cls)
forward_strats=("average" "max")

# Array of number of frames
num_frames=(5 10)

# Array for mask options
mask_options=("with_mask" "without_mask")

# Function to generate a descriptive name
generate_name() {
    local model=$1
    local strat=$2
    local frames=$3
    local dataset=$4
    local mask_option=$5
    # Extract model size from the model name (L, B, S, or T)
    local model_size=$(echo "$model" | sed 's/.*MegaDescriptor-\([LBST]\)-.*/\1/')
    echo "megadescriptor_${model_size}_${strat}_${frames}frames_${dataset}_${mask_option}"
}

# Function to run the embedding generation
run_embedding() {
    local model=$1
    local strat=$2
    local frames=$3
    local dataset=$4
    local mask_option=$5
    local mask_path=$6
    local cooccurrences=$7
    local clips_dir=$8

    local name=$(generate_name "$model" "$strat" "$frames" "$dataset" "$mask_option")
    local output_dir="${BASE_DIR}/results/pre_trained_model/megadescriptor"
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Set load_masks flag based on mask_option
    local load_masks_flag=""
    if [ "$mask_option" == "with_mask" ]; then
        load_masks_flag="--load_masks"
    fi
    
    python ./get_embeddings.py \
        $load_masks_flag \
        --mask_path "$mask_path" \
        --cooccurrences_filepath "$cooccurrences" \
        --clips_directory "$clips_dir" \
        --num_frames "$frames" \
        --mode "Test" \
        --model_type "megadescriptor" \
        --pre_trained_model "$model" \
        --model_num_frames "$frames" \
        --forward_strat "$strat" \
        --output_file "${output_dir}/${name}.pkl"
}

# Generate embeddings for Meerkat dataset
for model in "${megadescriptor_models[@]}"; do
    for strat in "${forward_strats[@]}"; do
        for frames in "${num_frames[@]}"; do
            for mask_option in "${mask_options[@]}"; do
                mask_path="${BASE_DIR}/Dataset/meerkat_h5files/masks/meerkat_masks.pkl"
                [ "$mask_option" == "without_mask" ] && mask_path="none"
                run_embedding "$model" "$strat" "$frames" "meerkat" "$mask_option" \
                    "$mask_path" \
                    "${BASE_DIR}/Dataset/meerkat_h5files/Cooccurrences.json" \
                    "${BASE_DIR}/Dataset/meerkat_h5files/clips/Test/"
            done
        done
    done
done

# Generate embeddings for Polar Bears dataset
for model in "${megadescriptor_models[@]}"; do
    for strat in "${forward_strats[@]}"; do
        for frames in "${num_frames[@]}"; do
            for mask_option in "${mask_options[@]}"; do
                mask_path="${BASE_DIR}/Dataset/polarbears_h5files/masks/PB_masks.pkl"
                [ "$mask_option" == "without_mask" ] && mask_path="none"
                run_embedding "$model" "$strat" "$frames" "polarbears" "$mask_option" \
                    "$mask_path" \
                    "${BASE_DIR}/Dataset/polarbears_h5files/Cooccurrences.json" \
                    "${BASE_DIR}/Dataset/polarbears_h5files/clips/Test/"
            done
        done
    done
done

echo "MegaDescriptor embedding generation complete!"
