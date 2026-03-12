#!/bin/bash

set -uo pipefail

# Usage:
#   ./install_packages.sh --conda <env_name_or_prefix>
#   ./install_packages.sh --venv <venv_path>
#   ./install_packages.sh
#
# With no args, this script auto-detects in this order:
#   1) active conda env
#   2) active virtualenv
#   3) local .venv

MODE=""
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)
      MODE="conda"
      TARGET="${2:-}"
      shift 2
      ;;
    --venv)
      MODE="venv"
      TARGET="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: ./install_packages.sh [--conda <env_name_or_prefix> | --venv <venv_path>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./install_packages.sh [--conda <env_name_or_prefix> | --venv <venv_path>]"
      exit 1
      ;;
  esac
done

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
    PIP_CMD=(conda run --prefix "$TARGET" python -m pip)
  else
    echo "Using conda env name: $TARGET"
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
  PIP_CMD=("$TARGET/bin/python" -m pip)
else
  echo "Error: unsupported mode '$MODE'"
  exit 1
fi

# Define a list of packages to be installed
packages=(
    "av==12.0.0"
    "blessed==1.20.0"
    "contourpy==1.2.0"
    "cycler==0.12.1"
    "debugpy==1.6.7"
    "decorator==5.1.1"
    "exceptiongroup==1.2.2"
    "executing==2.0.1"
    "filelock==3.13.1"
    "fonttools==4.50.0"
    "fsspec==2024.3.1"
    "gmpy2==2.1.2"
    "gpustat==1.1.1"
    "h5==0.9.3"
    "h5py==3.11.0"
    "huggingface-hub==0.24.6"
    "idna==3.4"
    "importlib-metadata==8.2.0"
    "jedi==0.19.1"
    "jinja2==3.1.3"
    "joblib==1.4.2"
    "jupyter-core==5.7.2"
    "kiwisolver==1.4.5"
    "markupsafe==2.1.3"
    "matplotlib==3.8.3"
    "mkl-fft==1.3.8"
    "mkl-random==1.2.4"
    "mkl-service==2.4.0"
    "mpmath==1.3.0"
    "nest-asyncio==1.6.0"
    "networkx==3.1"
    "numpy==1.26.4"
    "nvidia-ml-py==12.535.133"
    "opencv-python==4.9.0.80"
    "packaging==24.0"
    "pandas==2.2.1"
    "parso==0.8.4"
    "pexpect==4.9.0"
    "pickleshare==0.7.5"
    "pillow==10.2.0"
    "platformdirs==4.2.2"
    "prompt-toolkit==3.0.47"
    "psutil==5.9.0"
    "pure_eval==0.2.3"
    "pygments==2.18.0"
    "pyparsing==3.1.2"
    "python-dateutil==2.9.0.post0"
    "python-graphviz==0.20.3"
    "pytz==2024.1"
    "pyyaml==6.0.1"
    "pyzmq==25.1.2"
    "regex==2023.12.25"
    "requests==2.31.0"
    "scikit-learn==1.3.2"
    "scipy==1.11.3"
    "setuptools==68.1.2"
    "six==1.16.0"
    "sqlite==3.41.2"
    "sympy==1.13"
    "tbb==2021.8.0"
    "traitlets==5.9.0"
    "tqdm==4.66.1"
    "typing-extensions==4.9.0"
    "transformers"
    "urllib3==1.27.0"
    "wcwidth==0.2.10"
    "wheel==0.41.2"
    "wrapt==1.15.0"
    "zipp==3.17.1"
    "einops==0.8.0"
    "open_clip_torch==2.30.0"
    "timm==1.0.13"
)

# Loop through the list and install each package
failed_packages=()
for package in "${packages[@]}"; do
    echo "Installing $package..."
  if ! "${PIP_CMD[@]}" install "$package"; then
    echo "Failed to install $package"
    failed_packages+=("$package")
  fi
done

if [[ ${#failed_packages[@]} -gt 0 ]]; then
  echo "Completed with failures."
  echo "The following packages failed to install:"
  for pkg in "${failed_packages[@]}"; do
    echo "  - $pkg"
  done
  exit 1
fi

echo "All packages have been installed."
