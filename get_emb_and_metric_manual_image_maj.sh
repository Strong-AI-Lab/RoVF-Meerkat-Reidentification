#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/shell_helpers.sh"

BASE_DIR="$(rovf_resolve_repo_root "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
ANIMAL=""
DEVICE=""
CHECKPOINTS=()

usage() {
    cat <<'EOF'
Usage: get_emb_and_metric_manual_image_maj.sh [options]

Generate image-majority-vote embeddings and metrics for explicitly listed checkpoints.

Options:
  --animal meerkat|polarbear     Dataset to evaluate. Prompts on a TTY if omitted.
  --device cpu|cuda              Device passed to main.py. Prompts on a TTY if omitted.
  --checkpoint FILE              Evaluate one checkpoint. May be repeated.
  --base-dir DIR                 Repository root. Defaults to the current checkout.
  --python-bin PATH              Python executable. Defaults to PYTHON_BIN or python.
  --dry-run                      Print commands without executing them.
  -h, --help                     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --animal)
            [[ $# -ge 2 ]] || rovf_die "--animal requires a value"
            ANIMAL=$2
            shift 2
            ;;
        --device)
            [[ $# -ge 2 ]] || rovf_die "--device requires a value"
            DEVICE=$2
            shift 2
            ;;
        --checkpoint)
            [[ $# -ge 2 ]] || rovf_die "--checkpoint requires a value"
            CHECKPOINTS+=("$2")
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

if [[ -z "$ANIMAL" ]]; then
    [[ -t 0 ]] || rovf_die "--animal is required when stdin is not a TTY"
    rovf_prompt_animal
    ANIMAL=$ROVF_PROMPT_RESULT
elif ! ANIMAL=$(rovf_normalize_animal "$ANIMAL"); then
    rovf_die "invalid animal '$ANIMAL'; expected meerkat or polarbear"
fi

if [[ -z "$DEVICE" ]]; then
    [[ -t 0 ]] || rovf_die "--device is required when stdin is not a TTY"
    rovf_prompt_device
    DEVICE=$ROVF_PROMPT_RESULT
fi
rovf_require_nonempty "$DEVICE" "device"

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    if [[ -t 0 ]]; then
        printf 'Enter checkpoint file paths (one per line). Enter an empty line when done:\n'
        while true; do
            read -r -p "Checkpoint file: " cp
            [[ -n "$cp" ]] || break
            if [[ -f "$cp" ]]; then
                CHECKPOINTS+=("$cp")
            else
                printf "Warning: File '%s' not found. Please enter a valid path.\n" "$cp"
            fi
        done
    else
        rovf_die "--checkpoint is required when stdin is not a TTY"
    fi
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    rovf_die "no checkpoints provided"
fi

rovf_require_file "$BASE_DIR/main.py" "main.py"
rovf_set_dataset_paths "$BASE_DIR" "$ANIMAL"
if ! rovf_is_dry_run; then
    rovf_validate_dataset_paths with-mask
fi

for cp in "${CHECKPOINTS[@]}"; do
    rovf_require_file "$cp" "checkpoint"

    printf 'Running with mask for checkpoint: %s\n' "$cp"
    rovf_run "$PYTHON_BIN" "$BASE_DIR/main.py" test "" \
        -cp "$cp" \
        -df "$ROVF_DATASET_FILE" \
        -d "$DEVICE" \
        -m "$ROVF_MASK_FILE" \
        -cd "$ROVF_CLIPS_DIRECTORY" \
        -co "$ROVF_COOCCURRENCES_FILE" \
        -imv True

    printf 'Running without mask for checkpoint: %s\n' "$cp"
    rovf_run "$PYTHON_BIN" "$BASE_DIR/main.py" test "" \
        -cp "$cp" \
        -df "$ROVF_DATASET_FILE" \
        -d "$DEVICE" \
        -cd "$ROVF_CLIPS_DIRECTORY" \
        -co "$ROVF_COOCCURRENCES_FILE" \
        -imv True
done
