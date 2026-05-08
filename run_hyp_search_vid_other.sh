#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/shell_helpers.sh"

BASE_DIR="$(rovf_resolve_repo_root "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
GPU_DEVICE="${GPU_DEVICE:-}"
MODEL_NAME=""
POSITIONAL=()

usage() {
    cat <<'EOF'
Usage: run_hyp_search_vid_other.sh [options] <model_name> <GPU_device_number>

Run the fixed 6-file video hyperparameter search list for a model directory.

Options:
  --base-dir DIR        Repository root. Defaults to the current checkout.
  --python-bin PATH     Python executable. Defaults to PYTHON_BIN or python.
  --device DEVICE       Device passed to main.py. Defaults to cuda.
  --gpu GPU             CUDA_VISIBLE_DEVICES value. Overrides the positional GPU.
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
        --*)
            rovf_die "unknown argument: $1"
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -gt 2 ]]; then
    usage >&2
    rovf_die "too many positional arguments"
fi

MODEL_NAME="${POSITIONAL[0]:-}"
if [[ -z "$GPU_DEVICE" && ${#POSITIONAL[@]} -ge 2 ]]; then
    GPU_DEVICE="${POSITIONAL[1]}"
fi

if [[ -z "$MODEL_NAME" || ( -z "$GPU_DEVICE" && "$DEVICE" != "cpu" ) ]]; then
    usage >&2
    rovf_die "model name and GPU device are required unless --device cpu is used"
fi

base_dir="training_scripts/exp_metadata/hyperparameter_search"
yaml_dir="$BASE_DIR/$base_dir/$MODEL_NAME"
yaml_files=(
    "50_50_aug.yml"
    "50_50_no_aug.yml"
    "mask_aug.yml"
    "mask_no_aug.yml"
    "no_mask_aug.yml"
    "no_mask_no_aug.yml"
)

rovf_require_file "$BASE_DIR/main.py" "main.py"
rovf_require_dir "$yaml_dir" "YAML directory"
for yaml_file in "${yaml_files[@]}"; do
    rovf_require_file "$yaml_dir/$yaml_file" "YAML config"
done

cd "$BASE_DIR"
for yaml_file in "${yaml_files[@]}"; do
    yaml_path="$base_dir/$MODEL_NAME/$yaml_file"
    printf 'Running training for %s in model directory %s with GPU %s...\n' "$yaml_file" "$MODEL_NAME" "${GPU_DEVICE:-none}"
    if ! rovf_run_gpu "$GPU_DEVICE" "$PYTHON_BIN" "$BASE_DIR/main.py" train "$yaml_path" -d "$DEVICE"; then
        rovf_die "error occurred with $yaml_file"
    fi
done

printf 'All files processed successfully!\n'
