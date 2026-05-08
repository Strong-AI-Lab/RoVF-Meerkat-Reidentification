#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/shell_helpers.sh"

BASE_DIR="$(rovf_resolve_repo_root "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="results/hyperparameter_search/megadescriptor_video"
DEVICE="cuda"
GPU_DEVICE="${GPU_DEVICE:-0}"

usage() {
    cat <<'EOF'
Usage: run_megadescriptor_video_tests.sh [options]

Run test mode for MegaDescriptor video checkpoints, with and without masks.

Options:
  --model-dir DIR       Checkpoint directory. Defaults to results/hyperparameter_search/megadescriptor_video.
  --base-dir DIR        Repository root. Defaults to the current checkout.
  --python-bin PATH     Python executable. Defaults to PYTHON_BIN or python.
  --device DEVICE       Device passed to main.py. Defaults to cuda.
  --gpu GPU             CUDA_VISIBLE_DEVICES value. Defaults to GPU_DEVICE or 0.
  --dry-run             Print commands without executing them.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            [[ $# -ge 2 ]] || rovf_die "--model-dir requires a value"
            MODEL_DIR=$2
            shift 2
            ;;
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
        --device)
            [[ $# -ge 2 ]] || rovf_die "--device requires a value"
            DEVICE=$2
            shift 2
            ;;
        --gpu)
            [[ $# -ge 2 ]] || rovf_die "--gpu requires a value"
            GPU_DEVICE=$2
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

MODEL_DIR="$(rovf_path "$BASE_DIR" "$MODEL_DIR")"
rovf_require_file "$BASE_DIR/main.py" "main.py"
rovf_require_dir "$MODEL_DIR" "model directory"
rovf_set_dataset_paths "$BASE_DIR" meerkat
if ! rovf_is_dry_run; then
    rovf_validate_dataset_paths with-mask
fi

checkpoint_count=$(rovf_count_files "$MODEL_DIR" "checkpoint_epoch_*.pt")
if [[ "$checkpoint_count" -eq 0 ]]; then
    rovf_die "no checkpoint_epoch_*.pt files found in $MODEL_DIR"
fi

cd "$BASE_DIR"
while IFS= read -r -d '' ckpt; do
    printf 'Testing checkpoint with mask: %s\n' "$ckpt"
    rovf_run_gpu "$GPU_DEVICE" "$PYTHON_BIN" "$BASE_DIR/main.py" test "" \
        -cp "$ckpt" \
        -df "$ROVF_DATASET_FILE" \
        -d "$DEVICE" \
        -m "$ROVF_MASK_FILE" \
        -cd "$ROVF_CLIPS_DIRECTORY" \
        -co "$ROVF_COOCCURRENCES_FILE"

    printf 'Testing checkpoint without mask: %s\n' "$ckpt"
    rovf_run_gpu "$GPU_DEVICE" "$PYTHON_BIN" "$BASE_DIR/main.py" test "" \
        -cp "$ckpt" \
        -df "$ROVF_DATASET_FILE" \
        -d "$DEVICE" \
        -cd "$ROVF_CLIPS_DIRECTORY" \
        -co "$ROVF_COOCCURRENCES_FILE"
done < <(rovf_find_sorted_null "$MODEL_DIR" "checkpoint_epoch_*.pt")
