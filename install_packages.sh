#!/bin/bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE=""
TARGET=""
EXTRAS="models,dev"
MINIMAL=false
DRY_RUN=false
REPRODUCIBLE=false
CONSTRAINTS_FILE=""

usage() {
  cat <<'USAGE'
Usage: ./install_packages.sh [--conda <env_name_or_prefix> | --venv <venv_path>] [options]

Options:
  --extras <list>   Optional groups to install: models, segmentation, dev, all.
                    Defaults to models,dev. Use comma-separated values.
  --minimal         Install core requirements only.
  --reproducible    Install with constraints-validated.txt for repeatable reruns.
  --constraints <path>
                    Install with a caller-provided pip constraints file.
  --dry-run         Print selected requirement files and packages without installing.
  -h, --help        Show this message.

PyTorch and TorchVision are intentionally not installed here because the correct
wheel depends on your platform/CUDA setup. Install them first from:
  https://pytorch.org/get-started/locally/

With no environment args, this script auto-detects in this order:
  1) active conda env
  2) active virtualenv
  3) local .venv
USAGE
}

normalize_extras() {
  local value="$1"
  value="${value// /}"
  value="${value//;/,}"
  echo "$value"
}

add_group() {
  local group="$1"
  local existing
  for existing in "${REQ_GROUPS[@]}"; do
    if [[ "$existing" == "$group" ]]; then
      return
    fi
  done
  REQ_GROUPS+=("$group")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)
      if [[ $# -lt 2 ]]; then
        echo "Error: --conda requires an environment name or prefix path."
        exit 1
      fi
      MODE="conda"
      TARGET="$2"
      shift 2
      ;;
    --venv)
      if [[ $# -lt 2 ]]; then
        echo "Error: --venv requires a path (for example, .venv)."
        exit 1
      fi
      MODE="venv"
      TARGET="$2"
      shift 2
      ;;
    --extras)
      if [[ $# -lt 2 ]]; then
        echo "Error: --extras requires a comma-separated list."
        exit 1
      fi
      EXTRAS="$(normalize_extras "$2")"
      shift 2
      ;;
    --minimal)
      MINIMAL=true
      shift
      ;;
    --reproducible)
      REPRODUCIBLE=true
      shift
      ;;
    --constraints)
      if [[ $# -lt 2 ]]; then
        echo "Error: --constraints requires a path."
        exit 1
      fi
      CONSTRAINTS_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$MINIMAL" == true && "$EXTRAS" != "models,dev" ]]; then
  echo "Error: --minimal cannot be combined with --extras."
  exit 1
fi

if [[ "$REPRODUCIBLE" == true && -n "$CONSTRAINTS_FILE" ]]; then
  echo "Error: --reproducible cannot be combined with --constraints."
  exit 1
fi

if [[ "$REPRODUCIBLE" == true ]]; then
  CONSTRAINTS_FILE="$SCRIPT_DIR/constraints-validated.txt"
fi

REQ_GROUPS=()
add_group "core"
if [[ "$MINIMAL" == false ]]; then
  if [[ -z "$EXTRAS" ]]; then
    echo "Error: --extras requires a non-empty comma-separated list."
    exit 1
  fi

  IFS=',' read -r -a EXTRA_GROUPS <<< "$EXTRAS"
  for group in "${EXTRA_GROUPS[@]}"; do
    case "$group" in
      all)
        add_group "models"
        add_group "segmentation"
        add_group "dev"
        ;;
      models|segmentation|dev)
        add_group "$group"
        ;;
      "")
        echo "Error: empty extras group in '$EXTRAS'."
        exit 1
        ;;
      *)
        echo "Error: unknown extras group '$group'."
        echo "Valid groups: models, segmentation, dev, all"
        exit 1
        ;;
    esac
  done
fi

requirements_file_for_group() {
  case "$1" in
    core) echo "$SCRIPT_DIR/requirements-core.txt" ;;
    models) echo "$SCRIPT_DIR/requirements-models.txt" ;;
    segmentation) echo "$SCRIPT_DIR/requirements-segmentation.txt" ;;
    dev) echo "$SCRIPT_DIR/requirements-dev.txt" ;;
  esac
}

echo "Selected requirement groups: ${REQ_GROUPS[*]}"

if [[ -n "$CONSTRAINTS_FILE" ]]; then
  if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
    echo "Error: constraints file not found: $CONSTRAINTS_FILE"
    exit 1
  fi
  echo "Using constraints file: $CONSTRAINTS_FILE"
fi

for group in "${REQ_GROUPS[@]}"; do
  req_file="$(requirements_file_for_group "$group")"
  if [[ ! -f "$req_file" ]]; then
    echo "Error: missing requirements file for group '$group': $req_file"
    exit 1
  fi
done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run only; no packages will be installed."
  for group in "${REQ_GROUPS[@]}"; do
    req_file="$(requirements_file_for_group "$group")"
    echo
    echo "[$group] $req_file"
    sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$req_file"
  done
  exit 0
fi

if [[ -z "$MODE" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    MODE="conda"
    TARGET="$CONDA_PREFIX"
    echo "Detected active conda environment: $TARGET"
  elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    MODE="venv"
    TARGET="$VIRTUAL_ENV"
    echo "Detected active virtual environment: $TARGET"
  elif [[ -x ".venv/bin/python" ]]; then
    MODE="venv"
    TARGET=".venv"
    echo "Detected local virtual environment: $TARGET"
  else
    echo "Error: no environment detected."
    echo "Provide one explicitly:"
    echo "  ./install_packages.sh --conda <env_name_or_prefix>"
    echo "  ./install_packages.sh --venv <venv_path>"
    exit 1
  fi
fi

PIP_CMD=()
PYTHON_CMD=()
if [[ "$MODE" == "conda" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found in PATH."
    exit 1
  fi
  if [[ -z "$TARGET" ]]; then
    echo "Error: --conda requires an environment name or prefix path."
    exit 1
  fi

  if [[ -d "$TARGET" ]]; then
    echo "Using conda env prefix: $TARGET"
    PYTHON_CMD=(conda run --prefix "$TARGET" python)
    PIP_CMD=(conda run --prefix "$TARGET" python -m pip)
  else
    echo "Using conda env name: $TARGET"
    PYTHON_CMD=(conda run -n "$TARGET" python)
    PIP_CMD=(conda run -n "$TARGET" python -m pip)
  fi
elif [[ "$MODE" == "venv" ]]; then
  if [[ -z "$TARGET" ]]; then
    echo "Error: --venv requires a path (for example, .venv)."
    exit 1
  fi
  if [[ ! -x "$TARGET/bin/python" ]]; then
    echo "Error: could not find python executable at $TARGET/bin/python"
    exit 1
  fi

  echo "Using virtual environment at: $TARGET"
  PYTHON_CMD=("$TARGET/bin/python")
  PIP_CMD=("$TARGET/bin/python" -m pip)
else
  echo "Error: unsupported mode '$MODE'"
  exit 1
fi

if ! "${PYTHON_CMD[@]}" - <<'PYTORCH_CHECK'
try:
    import torch
    import torchvision
except Exception as exc:
    raise SystemExit(str(exc))
PYTORCH_CHECK
then
  echo "Error: torch and torchvision must be installed before running this script."
  echo "Install the correct build for your platform from:"
  echo "  https://pytorch.org/get-started/locally/"
  exit 1
fi

optional_failures=()
for group in "${REQ_GROUPS[@]}"; do
  req_file="$(requirements_file_for_group "$group")"
  echo "Installing $group requirements from $req_file"
  pip_install_args=(install)
  if [[ -n "$CONSTRAINTS_FILE" ]]; then
    pip_install_args+=(-c "$CONSTRAINTS_FILE")
  fi
  pip_install_args+=(-r "$req_file")
  if ! "${PIP_CMD[@]}" "${pip_install_args[@]}"; then
    if [[ "$group" == "core" ]]; then
      echo "Error: core dependency installation failed."
      exit 1
    fi
    echo "Warning: optional dependency group '$group' failed to install."
    optional_failures+=("$group")
  fi
done

if [[ ${#optional_failures[@]} -gt 0 ]]; then
  echo
  echo "Completed with optional dependency warnings."
  echo "The following optional groups failed:"
  for group in "${optional_failures[@]}"; do
    echo "  - $group"
  done
  echo "Core dependencies were installed successfully."
else
  echo "All selected dependency groups installed successfully."
fi
