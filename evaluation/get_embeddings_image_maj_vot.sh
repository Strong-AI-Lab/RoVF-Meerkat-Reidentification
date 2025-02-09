#!/bin/bash

# Base directory
BASE_DIR="/data/kkno604/github/RoVF-meerkat-reidentification"

# Array of forward strategies
forward_strats=("cls")

# Array of number of frames (for DINO)
num_frames=(10)

# Array for mask options
mask_options=("with_mask") # without_mask

#--------------------------------------------------------------------------------
# Additional models (megadescriptor & bioclip) - use 10 frames and image_maj_vote
extra_models=("megadescriptor" "bioclip" "dino")

megadescriptor_pretrained_models=("hf-hub:BVRA/MegaDescriptor-T-224" "hf-hub:BVRA/MegaDescriptor-S-224" "hf-hub:BVRA/MegaDescriptor-B-224" "hf-hub:BVRA/MegaDescriptor-L-224")
bioclip_pretrained_models=("hf-hub:imageomics/bioclip")
dino_pretrained_models=("facebook/dinov2-small" "facebook/dinov2-base" "facebook/dinov2-large" "facebook/dinov2-giant")
#--------------------------------------------------------------------------------

# Function to generate a descriptive name
generate_name() {
    local model=$1
    local strat=$2
    local frames=$3
    local dataset=$4
    local mask_option=$5
    echo "${model##*/}_${strat}_${frames}frames_${dataset}_${mask_option}"
}

# Function to sanitize the pre-trained model name
sanitize_name() {
    local name=$1
    echo "$name" | tr -c '[:alnum:]_' '_'
}

# Function to run the embedding generation
run_embedding() {
    local model=$1
    local pre_trained_model=$2
    local strat=$3
    local frames=$4
    local dataset=$5
    local mask_option=$6
    local mask_path=$7
    local cooccurrences=$8
    local clips_dir=$9
    local image_maj_vote_flag=$10
    local extra_flags=${11:-""}  # for image_maj_vote, placeholders, etc.

    local sanitized_pre_trained_model=$(sanitize_name "$pre_trained_model")
    local name=$(generate_name "$pre_trained_model" "$strat" "$frames" "$dataset" "$mask_option")
    local output_dir="${BASE_DIR}/results/pre_trained_model/${model##*/}_${sanitized_pre_trained_model}"

    mkdir -p "$output_dir"

    local load_masks_flag=""
    [ "$mask_option" == "with_mask" ] && load_masks_flag="--load_masks"

    python ${BASE_DIR}/evaluation/get_embeddings.py \
        $load_masks_flag \
        $extra_flags \
        --mask_path "$mask_path" \
        --cooccurrences_filepath "$cooccurrences" \
        --clips_directory "$clips_dir" \
        --num_frames "$frames" \
        --forward_strat "$strat" \
        --output_file "${output_dir}/${name}.pkl" \
        --K 20 \
        --total_frames 20 \
        --zfill_num 4 \
        --apply_mask_percentage 1.0 \
        --model_type "$model" \
        --pre_trained_model "$pre_trained_model" \
        --image_maj_vote
}

#: '
# Generate embeddings for Meerkat dataset 
for model in "${extra_models[@]}"; do
    pretrained_models_var="${model}_pretrained_models[@]"
    pretrained_models=("${!pretrained_models_var}")
    for pretrained_model in "${pretrained_models[@]}"; do
        for mask_option in "${mask_options[@]}"; do
            mask_path="${BASE_DIR}/Dataset/meerkat_h5files/masks/meerkat_masks.pkl"
            [ "$mask_option" == "without_mask" ] && mask_path="none"
            run_embedding "$model" "$pretrained_model" "cls" 10 "meerkat" "$mask_option" \
                "$mask_path" \
                "${BASE_DIR}/Dataset/meerkat_h5files/Cooccurrences.json" \
                "${BASE_DIR}/Dataset/meerkat_h5files/clips/Test/" \
                "--image_maj_vote"
        done
    done
done
#'

: '
# Generate embeddings for Polar Bears dataset 

for model in "${extra_models[@]}"; do
    pretrained_models_var="${model}_pretrained_models[@]"
    pretrained_models=("${!pretrained_models_var}")
    for pretrained_model in "${pretrained_models[@]}"; do
        for mask_option in "${mask_options[@]}"; do
            mask_path="${BASE_DIR}/Dataset/polarbears_h5files/masks/PB_masks.pkl"
            [ "$mask_option" == "without_mask" ] && mask_path="none"
            run_embedding "$model" "$pretrained_model" "cls" 10 "polarbears" "$mask_option" \
                "$mask_path" \
                "${BASE_DIR}/Dataset/polarbears_h5files/Cooccurrences.json" \
                "${BASE_DIR}/Dataset/polarbears_h5files/clips/Test/" \
                "--image_maj_vote"
        done
    done
done
'

echo "Embedding generation complete!"