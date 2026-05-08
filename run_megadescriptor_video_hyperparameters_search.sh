#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/shell_helpers.sh"

BASE_DIR="$(rovf_resolve_repo_root "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="cuda"
GPU_DEVICE="${GPU_DEVICE:-4}"
YAML_SUBDIR="training_scripts/exp_metadata/hyperparameter_search/megadescriptor_video"

usage() {
    cat <<'EOF'
Usage: run_megadescriptor_video_hyperparameters_search.sh [options]

Run all generated MegaDescriptor video hyperparameter YAML files.

Options:
  --base-dir DIR        Repository root. Defaults to the current checkout.
  --yaml-dir DIR        YAML directory. Defaults to the MegaDescriptor video generated YAML directory.
  --python-bin PATH     Python executable. Defaults to PYTHON_BIN or python.
  --device DEVICE       Device passed to main.py. Defaults to cuda.
  --gpu GPU             CUDA_VISIBLE_DEVICES value. Defaults to GPU_DEVICE or 4.
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
        --yaml-dir)
            [[ $# -ge 2 ]] || rovf_die "--yaml-dir requires a value"
            YAML_SUBDIR=$2
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

YAML_DIR="$(rovf_path "$BASE_DIR" "$YAML_SUBDIR")"
rovf_run_yaml_directory_training "MegaDescriptor video" "$BASE_DIR" "$YAML_DIR" "$PYTHON_BIN" "$DEVICE" "$GPU_DEVICE"
