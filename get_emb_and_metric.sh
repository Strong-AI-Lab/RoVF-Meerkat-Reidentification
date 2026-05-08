#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/shell_helpers.sh"

BASE_DIR="$(rovf_resolve_repo_root "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
ANIMAL=""
DEVICE=""
CHECKPOINT_DIR=""
CHECKPOINTS=()

usage() {
    cat <<'EOF'
Usage: get_emb_and_metric.sh [options]

Generate embeddings and metrics for checkpoints, with and without masks.

Options:
  --animal meerkat|polarbear     Dataset to evaluate. Prompts on a TTY if omitted.
  --device cpu|cuda              Device passed to main.py. Prompts on a TTY if omitted.
  --checkpoint-dir DIR           Recursively evaluate every .pt checkpoint in DIR.
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
        --checkpoint-dir)
            [[ $# -ge 2 ]] || rovf_die "--checkpoint-dir requires a value"
            CHECKPOINT_DIR=$2
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

if [[ -z "$CHECKPOINT_DIR" && ${#CHECKPOINTS[@]} -eq 0 ]]; then
    if [[ -t 0 ]]; then
        read -r -p "Enter the checkpoint directory: " CHECKPOINT_DIR
    else
        rovf_die "--checkpoint-dir or --checkpoint is required when stdin is not a TTY"
    fi
fi

rovf_require_file "$BASE_DIR/main.py" "main.py"
rovf_set_dataset_paths "$BASE_DIR" "$ANIMAL"
if ! rovf_is_dry_run; then
    rovf_validate_dataset_paths with-mask
fi

if [[ -n "$CHECKPOINT_DIR" ]]; then
    rovf_require_dir "$CHECKPOINT_DIR" "checkpoint directory"
    while IFS= read -r -d '' checkpoint; do
        CHECKPOINTS+=("$checkpoint")
    done < <(rovf_find_sorted_null "$CHECKPOINT_DIR" "*.pt")
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    rovf_die "no .pt checkpoints found"
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
        -co "$ROVF_COOCCURRENCES_FILE"

    printf 'Running without mask for checkpoint: %s\n' "$cp"
    rovf_run "$PYTHON_BIN" "$BASE_DIR/main.py" test "" \
        -cp "$cp" \
        -df "$ROVF_DATASET_FILE" \
        -d "$DEVICE" \
        -cd "$ROVF_CLIPS_DIRECTORY" \
        -co "$ROVF_COOCCURRENCES_FILE"
done
