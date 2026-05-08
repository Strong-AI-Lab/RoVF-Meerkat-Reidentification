#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/shell_helpers.sh"

BASE_DIR="$(rovf_resolve_repo_root "$SCRIPT_DIR/..")"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage: get_embeddings_image_maj_vot.sh [options]

Generate image-majority-vote embeddings for configured pre-trained models.

Options:
  --base-dir DIR        Repository root. Defaults to the current checkout.
  --python-bin PATH     Python executable. Defaults to PYTHON_BIN or python.
  --dry-run             Print commands without executing them.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-dir)
            [[ $# -ge 2 ]] || rovf_die "--base-dir requires a value"
            BASE_DIR="$(rovf_abs_existing_dir "$2")"
            shift 2
            ;;
        --python-bin)
            [[ $# -ge 2 ]] || rovf_die "--python-bin requires a value"
            PYTHON_BIN=$2
            shift 2
            ;;
        --dry-run)
            rovf_enable_dry_run
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            rovf_die "unknown argument: $1"
            ;;
    esac
done

EVAL_DIR="$BASE_DIR/evaluation"
rovf_require_file "$EVAL_DIR/get_embeddings.py" "get_embeddings.py"

mask_options=("with_mask")
extra_models=("megadescriptor" "bioclip" "dino")
megadescriptor_pretrained_models=("hf-hub:BVRA/MegaDescriptor-T-224" "hf-hub:BVRA/MegaDescriptor-S-224" "hf-hub:BVRA/MegaDescriptor-B-224" "hf-hub:BVRA/MegaDescriptor-L-224")
bioclip_pretrained_models=("hf-hub:imageomics/bioclip")
dino_pretrained_models=("facebook/dinov2-small" "facebook/dinov2-base" "facebook/dinov2-large" "facebook/dinov2-giant")

generate_name() {
    local model=$1
    local strat=$2
    local frames=$3
    local dataset=$4
    local mask_option=$5

    printf '%s_%s_%sframes_%s_%s\n' "${model##*/}" "$strat" "$frames" "$dataset" "$mask_option"
}

sanitize_name() {
    local name=$1

    printf '%s\n' "$name" | tr -c '[:alnum:]_' '_'
}

prepare_output_dir() {
    local output_dir=$1

    if rovf_is_dry_run; then
        rovf_print_command mkdir -p "$output_dir"
    else
        mkdir -p "$output_dir"
    fi
}

validate_embedding_inputs() {
    local mask_option=$1
    local mask_path=$2
    local cooccurrences=$3
    local clips_dir=$4

    if rovf_is_dry_run; then
        return 0
    fi

    [[ "$mask_option" == "without_mask" ]] || rovf_require_file "$mask_path" "mask file"
    rovf_require_file "$cooccurrences" "cooccurrences file"
    rovf_require_dir "$clips_dir" "clips directory"
}

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
    local sanitized_pre_trained_model
    local name
    local output_dir
    local cmd

    sanitized_pre_trained_model=$(sanitize_name "$pre_trained_model")
    name=$(generate_name "$pre_trained_model" "$strat" "$frames" "$dataset" "$mask_option")
    output_dir="$BASE_DIR/results/pre_trained_model/${model##*/}_$sanitized_pre_trained_model"
    prepare_output_dir "$output_dir"
    validate_embedding_inputs "$mask_option" "$mask_path" "$cooccurrences" "$clips_dir"

    cmd=("$PYTHON_BIN" "./get_embeddings.py")
    if [[ "$mask_option" == "with_mask" ]]; then
        cmd+=(--load_masks)
    fi
    cmd+=(
        --mask_path "$mask_path"
        --cooccurrences_filepath "$cooccurrences"
        --clips_directory "$clips_dir"
        --num_frames "$frames"
        --model_num_frames 1
        --forward_strat "$strat"
        --output_file "$output_dir/$name.pkl"
        --K 20
        --total_frames 20
        --zfill_num 4
        --apply_mask_percentage 1.0
        --model_type "$model"
        --pre_trained_model "$pre_trained_model"
        --image_maj_vote
    )

    rovf_run "${cmd[@]}"
}

cd "$EVAL_DIR"

for model in "${extra_models[@]}"; do
    pretrained_models_var="${model}_pretrained_models[@]"
    pretrained_models=("${!pretrained_models_var}")
    for pretrained_model in "${pretrained_models[@]}"; do
        for mask_option in "${mask_options[@]}"; do
            mask_path="$BASE_DIR/Dataset/meerkat_h5files/masks/meerkat_masks.pkl"
            [[ "$mask_option" == "without_mask" ]] && mask_path="none"
            run_embedding "$model" "$pretrained_model" "cls" 10 "meerkat" "$mask_option" \
                "$mask_path" \
                "$BASE_DIR/Dataset/meerkat_h5files/Cooccurrences.json" \
                "$BASE_DIR/Dataset/meerkat_h5files/clips/Test"
        done
    done
done

for model in "${extra_models[@]}"; do
    pretrained_models_var="${model}_pretrained_models[@]"
    pretrained_models=("${!pretrained_models_var}")
    for pretrained_model in "${pretrained_models[@]}"; do
        for mask_option in "${mask_options[@]}"; do
            mask_path="$BASE_DIR/Dataset/polarbears_h5files/masks/PB_masks.pkl"
            [[ "$mask_option" == "without_mask" ]] && mask_path="none"
            run_embedding "$model" "$pretrained_model" "cls" 10 "polarbears" "$mask_option" \
                "$mask_path" \
                "$BASE_DIR/Dataset/polarbears_h5files/Cooccurrences.json" \
                "$BASE_DIR/Dataset/polarbears_h5files/clips/Test"
        done
    done
done

printf 'Embedding generation complete!\n'
