# Shared helpers for RoVF Bash entrypoints. Source this file from scripts;
# it is not intended to be executed directly.

if [[ -n "${ROVF_SHELL_HELPERS_SOURCED:-}" ]]; then
    return 0
fi
ROVF_SHELL_HELPERS_SOURCED=1

rovf_die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

rovf_bool_enabled() {
    case "${1:-}" in
        1 | true | TRUE | yes | YES | on | ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

rovf_enable_dry_run() {
    DRY_RUN=1
    export DRY_RUN
}

rovf_is_dry_run() {
    rovf_bool_enabled "${DRY_RUN:-0}"
}

rovf_print_command() {
    local arg

    printf 'DRY_RUN:'
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

rovf_run() {
    if rovf_is_dry_run; then
        rovf_print_command "$@"
        return 0
    fi

    "$@"
}

rovf_run_gpu() {
    local gpu_device=$1
    shift

    if [[ -n "$gpu_device" ]]; then
        rovf_run env "CUDA_VISIBLE_DEVICES=$gpu_device" "$@"
    else
        rovf_run "$@"
    fi
}

rovf_abs_existing_dir() {
    local dir=$1

    [[ -d "$dir" ]] || rovf_die "directory not found: $dir"
    (cd "$dir" && pwd)
}

rovf_resolve_repo_root() {
    local start_dir=$1
    local probe

    if command -v git >/dev/null 2>&1; then
        if probe=$(git -C "$start_dir" rev-parse --show-toplevel 2>/dev/null); then
            printf '%s\n' "$probe"
            return 0
        fi
    fi

    probe=$(cd "$start_dir" && pwd)
    while [[ "$probe" != "/" ]]; do
        if [[ -f "$probe/main.py" && -d "$probe/training_functions" ]]; then
            printf '%s\n' "$probe"
            return 0
        fi
        probe=$(dirname "$probe")
    done

    rovf_die "could not resolve repository root from $start_dir"
}

rovf_path() {
    local base_dir=$1
    local path=$2

    if [[ "$path" == /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$base_dir" "$path"
    fi
}

rovf_require_file() {
    local path=$1
    local label=${2:-file}

    [[ -f "$path" ]] || rovf_die "$label not found: $path"
}

rovf_require_dir() {
    local path=$1
    local label=${2:-directory}

    [[ -d "$path" ]] || rovf_die "$label not found: $path"
}

rovf_require_nonempty() {
    local value=$1
    local label=$2

    [[ -n "$value" ]] || rovf_die "$label is required"
}

rovf_count_files() {
    local dir=$1
    local pattern=$2

    find "$dir" -type f -name "$pattern" -print | wc -l | tr -d '[:space:]'
}

rovf_find_sorted_null() {
    local dir=$1
    local pattern=$2

    find "$dir" -type f -name "$pattern" -print0 | sort -z
}

rovf_normalize_animal() {
    case "${1:-}" in
        1 | meerkat | meerkats | Meerkat | Meerkats)
            printf 'meerkat\n'
            ;;
        2 | polarbear | polarbears | polar-bear | polar-bears | Polarbear | Polarbears)
            printf 'polarbear\n'
            ;;
        *)
            return 1
            ;;
    esac
}

rovf_prompt_animal() {
    local choice

    while true; do
        printf 'Choose an animal:\n'
        printf '1) Meerkats\n'
        printf '2) Polarbears\n'
        read -r -p 'Enter the number corresponding to your choice: ' choice
        if ROVF_PROMPT_RESULT=$(rovf_normalize_animal "$choice"); then
            printf '%s selected.\n' "$ROVF_PROMPT_RESULT"
            return 0
        fi
        printf 'Invalid choice. Please choose either 1 (Meerkats) or 2 (Polarbears).\n'
    done
}

rovf_prompt_device() {
    local choice

    while true; do
        printf 'Choose a device:\n'
        printf '1) CPU\n'
        printf '2) CUDA\n'
        read -r -p 'Enter the number corresponding to your choice: ' choice
        case "$choice" in
            1 | cpu | CPU)
                ROVF_PROMPT_RESULT=cpu
                printf 'CPU selected.\n'
                return 0
                ;;
            2 | cuda | CUDA)
                ROVF_PROMPT_RESULT=cuda
                printf 'CUDA selected.\n'
                return 0
                ;;
            *)
                printf 'Invalid choice. Please choose either 1 (CPU) or 2 (CUDA).\n'
                ;;
        esac
    done
}

rovf_set_dataset_paths() {
    local base_dir=$1
    local animal=$2

    case "$animal" in
        meerkat)
            ROVF_MASK_FILE="$base_dir/Dataset/meerkat_h5files/masks/meerkat_masks.pkl"
            ROVF_DATASET_FILE="$base_dir/Dataset/meerkat_h5files/Precomputed_test_examples_meerkat.csv"
            ROVF_COOCCURRENCES_FILE="$base_dir/Dataset/meerkat_h5files/Cooccurrences.json"
            ROVF_CLIPS_DIRECTORY="$base_dir/Dataset/meerkat_h5files/clips/Test"
            ;;
        polarbear)
            ROVF_MASK_FILE="$base_dir/Dataset/polarbears_h5files/masks/PB_masks.pkl"
            ROVF_DATASET_FILE="$base_dir/Dataset/polarbears_h5files/Precomputed_test_examples_polarbear.csv"
            ROVF_COOCCURRENCES_FILE="$base_dir/Dataset/polarbears_h5files/Cooccurrences.json"
            ROVF_CLIPS_DIRECTORY="$base_dir/Dataset/polarbears_h5files/clips/Test"
            ;;
        *)
            rovf_die "invalid animal '$animal'; expected meerkat or polarbear"
            ;;
    esac
}

rovf_validate_dataset_paths() {
    rovf_require_file "$ROVF_DATASET_FILE" "dataset CSV"
    rovf_require_file "$ROVF_COOCCURRENCES_FILE" "cooccurrences file"
    rovf_require_dir "$ROVF_CLIPS_DIRECTORY" "clips directory"
    if [[ "${1:-with-mask}" == "with-mask" ]]; then
        rovf_require_file "$ROVF_MASK_FILE" "mask file"
    fi
}

rovf_run_yaml_directory_training() {
    local label=$1
    local base_dir=$2
    local yaml_dir=$3
    local python_bin=$4
    local device=$5
    local gpu_device=$6
    local counter=0
    local failures=0
    local total_files
    local yaml_file
    local filename

    rovf_require_file "$base_dir/main.py" "main.py"
    rovf_require_dir "$yaml_dir" "YAML directory"

    total_files=$(rovf_count_files "$yaml_dir" "*.yml")
    if [[ "$total_files" -eq 0 ]]; then
        rovf_die "no YAML files found in $yaml_dir"
    fi

    cd "$base_dir"

    printf 'Starting %s hyperparameter search...\n' "$label"
    printf 'Found %s YAML configuration files\n' "$total_files"
    printf '================================================\n'

    while IFS= read -r -d '' yaml_file; do
        counter=$((counter + 1))
        filename=$(basename "$yaml_file")

        printf '[%s/%s] Training with configuration: %s\n' "$counter" "$total_files" "$filename"
        printf 'Started at: %s\n' "$(date)"

        if rovf_run_gpu "$gpu_device" "$python_bin" "$base_dir/main.py" train "$yaml_file" -d "$device"; then
            printf 'Successfully completed training for: %s\n' "$filename"
        else
            printf 'Training failed for: %s\n' "$filename" >&2
            printf 'Continuing with next configuration...\n' >&2
            failures=$((failures + 1))
        fi

        printf 'Finished at: %s\n' "$(date)"
        printf '%s\n' '------------------------------------------------'
    done < <(rovf_find_sorted_null "$yaml_dir" "*.yml")

    printf '%s hyperparameter search completed!\n' "$label"
    printf 'Processed %s configuration files\n' "$counter"
    printf 'Check the results directory for outputs\n'

    if [[ "$failures" -gt 0 ]]; then
        printf '%s training runs failed.\n' "$failures" >&2
        return 1
    fi

    return 0
}
