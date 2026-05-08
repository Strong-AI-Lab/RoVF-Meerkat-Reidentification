#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../scripts/shell_helpers.sh"

BASE_DIR="$(rovf_resolve_repo_root "$SCRIPT_DIR/..")"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage: get_bioclip_embeddings.sh [options]

Generate BioCLIP pre-trained embeddings for the configured datasets.

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

bioclip_models=("hf-hub:imageomics/bioclip")
forward_strats=("average" "max")
num_frames=(5 10)
mask_options=("with_mask" "without_mask")

generate_name() {
    local model=$1
    local strat=$2
    local frames=$3
    local dataset=$4
    local mask_option=$5
    printf 'bioclip_%s_%sframes_%s_%s\n' "$strat" "$frames" "$dataset" "$mask_option"
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
    local strat=$2
    local frames=$3
    local dataset=$4
    local mask_option=$5
    local mask_path=$6
    local cooccurrences=$7
    local clips_dir=$8
    local name
    local output_dir
    local cmd

    name=$(generate_name "$model" "$strat" "$frames" "$dataset" "$mask_option")
    output_dir="$BASE_DIR/results/pre_trained_model/bioclip"
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
        --mode "Test"
        --model_type "bioclip"
        --pre_trained_model "$model"
        --model_num_frames "$frames"
        --forward_strat "$strat"
        --output_file "$output_dir/$name.pkl"
    )

    rovf_run "${cmd[@]}"
}

cd "$EVAL_DIR"

for model in "${bioclip_models[@]}"; do
    for strat in "${forward_strats[@]}"; do
        for frames in "${num_frames[@]}"; do
            for mask_option in "${mask_options[@]}"; do
                mask_path="$BASE_DIR/Dataset/meerkat_h5files/masks/meerkat_masks.pkl"
                [[ "$mask_option" == "without_mask" ]] && mask_path="none"
                run_embedding "$model" "$strat" "$frames" "meerkat" "$mask_option" \
                    "$mask_path" \
                    "$BASE_DIR/Dataset/meerkat_h5files/Cooccurrences.json" \
                    "$BASE_DIR/Dataset/meerkat_h5files/clips/Test"
            done
        done
    done
done

for model in "${bioclip_models[@]}"; do
    for strat in "${forward_strats[@]}"; do
        for frames in "${num_frames[@]}"; do
            for mask_option in "${mask_options[@]}"; do
                mask_path="$BASE_DIR/Dataset/polarbears_h5files/masks/PB_masks.pkl"
                [[ "$mask_option" == "without_mask" ]] && mask_path="none"
                run_embedding "$model" "$strat" "$frames" "polarbears" "$mask_option" \
                    "$mask_path" \
                    "$BASE_DIR/Dataset/polarbears_h5files/Cooccurrences.json" \
                    "$BASE_DIR/Dataset/polarbears_h5files/clips/Test"
            done
        done
    done
done

printf 'BioCLIP embedding generation complete!\n'
