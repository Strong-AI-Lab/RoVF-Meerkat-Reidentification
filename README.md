# RoVF for Animal Re-identification

This repository contains the codebase for the paper **RoVF for Animal Re-identification**.

> **Branch notice:** This `primary` branch contains post-publication code improvements intended to make the codebase more maintainable and broadly useful. The original code associated with the published paper is preserved in the `archive/published` branch.

**Authors:** Mitchell Rogers, Kobe Knowles, Gaël Gendron, Shahrokh Heidari, Isla Duporge, David Arturo Soriano Valdez, Mihailo Azhar, Padriac O’Leary, Simon Eyre, Michael Witbrock, Patrice Delmas.<br/>
**Supported by:** *Natural, Artificial, and Organisation Intelligence Institute, The University of Auckland, New Zealand*

## Abstract

> Recent advances in deep learning have significantly improved the accuracy and scalability of animal re-identification methods by automating the extraction of subtle distinguishing features from images and videos. This enables large-scale non-invasive monitoring of animal populations. We propose a segmentation pipeline and a re-identification model to re-identify animals without ground-truth IDs. The segmentation pipeline segments animals from the background based on their bounding boxes using the DINOv2 and segment anything model 2 (SAM2) foundation models. For re-identification, we introduce a method called recurrence over video frames (RoVF), which uses a recurrent component based on the Perceiver transformer on top of a DINOv2 image model to iteratively construct embeddings from video frames. We report the performance of the proposed segmentation pipeline and re-identification model using video datasets of meerkats and polar bears (PolarBearVidID). The proposed segmentation model achieved high accuracy (94.56% and 97.37%) and IoU (73.94% and 93.08%) for meerkats and polar bears, respectively. We found that RoVF outperformed frame- and video-based baselines, achieving 46.5% and 55% top-1 accuracy on masked test sets for meerkats and polar bears, respectively. These methods show promise in reducing the annotation burden in future individual-based ecological studies. The code is available at [https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification](https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification).

## Overview
- [Installation](#installation)
- [First-time quickstart](#first-time-quickstart)
- [Downloading the datasets](#downloading-the-datasets)
- [Background masking](#background-masking)
- [Re-identification](#re-identification)
- [Folder structure](#folder-structure)
- [Models guide](#models-guide)
- [Acknowledgments](#acknowledgements)

## Installation

In your environment of choice, install PyTorch first, then use the project installer for the remaining dependency groups. Use a Python version supported by your selected PyTorch build, such as Python 3.10 or 3.11.

Create and activate an environment before installing dependencies. For a Python virtual environment:

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For conda on Windows, Linux, or macOS:

```bash
conda create -n rovf python=3.11 -y
conda activate rovf
python -m pip install --upgrade pip
```

After activating the environment, install PyTorch (2.0+) for your platform by following: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

For example (CUDA 12.6):

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

Then run the project installer to install the remaining dependencies. The canonical installer is `install_packages.py`; `install_packages.sh` and `install_packages.ps1` are thin shell wrappers around it. The installer expects the target conda or virtual environment to already exist.

The script supports conda environments, virtual environments, dependency groups, reproducible constraints, and dry-runs:

Linux/macOS Bash:

```bash
# Conda env by name
./install_packages.sh --conda your-conda-env

# Conda env by prefix/path
./install_packages.sh --conda /path/to/conda/env

# Python venv (for example, .venv)
./install_packages.sh --venv .venv

# Auto-detect active env (conda -> venv -> local .venv)
./install_packages.sh

# Preview what would be installed
./install_packages.sh --dry-run

# Core runtime dependencies only
./install_packages.sh --minimal

# Include every optional group
./install_packages.sh --extras all

# Reproduce the validated dependency set as closely as possible
./install_packages.sh --reproducible

# Use your own pip constraints file
./install_packages.sh --constraints /path/to/constraints.txt
```

Windows PowerShell:

```powershell
# Conda env by name
.\install_packages.ps1 --conda your-conda-env

# Conda env by prefix/path
.\install_packages.ps1 --conda C:\path\to\conda\env

# Python venv (for example, .venv)
.\install_packages.ps1 --venv .venv

# Auto-detect active env (conda -> venv -> local .venv)
.\install_packages.ps1

# Preview what would be installed
.\install_packages.ps1 --dry-run

# Core runtime dependencies only
.\install_packages.ps1 --minimal

# Include every optional group
.\install_packages.ps1 --extras all

# Reproduce the validated dependency set as closely as possible
.\install_packages.ps1 --reproducible

# Use your own pip constraints file
.\install_packages.ps1 --constraints C:\path\to\constraints.txt
```

If PowerShell script execution policy blocks the wrapper, call the Python installer directly:

```powershell
python install_packages.py --venv .venv
```

The default install uses the `core`, `models`, and `dev` requirement groups with flexible version ranges. Core dependency failures stop the install; optional group failures are reported as warnings so you can still use the parts of the project that installed successfully. Segmentation dependencies are optional because SAM2 is expected as a local/external dependency.

For repeatable reruns, `--reproducible` adds `constraints-validated.txt`, which was generated from a smoke-tested environment. PyTorch and TorchVision are still installed separately because the correct wheel depends on your platform and CUDA setup.

`setup_environment.py` is a legacy Linux environment snapshot retained for reference. New installs should use `install_packages.py` or one of the shell wrappers above.

## Downloading the datasets
For our experiments, we use two animal video datasets:
* [Meerkats](https://meerkat-dataset.github.io/re-id/) based on the [Meerkat behaviour recognition dataset](https://meerkat-dataset.github.io/)
* Polar bears based on the [PolarBearVidID dataset](https://doi.org/10.3390/ani13050801) (Zuerl et al. 2023)

We provide a link to download the dataset (~22 GB) in the correct format for the meerkats on [the meerkat re-identification dataset page](https://meerkat-dataset.github.io/re-id/). This zip file can be extracted to the */Dataset/* folder. For the polar bear dataset we provide a python script in the */Dataset/* folder to convert the dataset to the format we use here.

To convert the PolarBearVidID dataset, download the [PolarBearVidID.zip](https://zenodo.org/records/7564529) file and use the following script to convert the dataset into the h5 file format used by our dataloader.

```bash
cd Dataset
py convert_polar_bear_dataset.py PolarBearVidID.zip
```

Where "PolarBearVidID.zip" is the path to the zip file.

## Background masking

https://github.com/user-attachments/assets/8177ab7a-e43f-486b-aff8-f9eba1767d62

*Example video of the background masking performance, including cases where the performance is poor.*

The background masking approach we use is based on two foundation models [DINOv2](https://dinov2.metademolab.com/) and [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2). This requires SAM 2 repository to be installed, see the submodule install instructions on their GitHub repository.

The code for this approach is provided in the */segmentation/* folder and can by applied using:
```bash
py DINOv2_LDA_SAM2.py -i polarbears_h5files -o polarbears 
```

This process is resource intensive. We have optimised the batch_size and resize_factor to utilise our GPUs (RTX A6000) available memory, however, it may be possible to parallelise this more effectively. To process the **test set** of the polar bear dataset this takes ~1 hour (13s per clip), and for the meerkat dataset this is even longer.

The main arguments are:
* *-o*: Output folder name.
* *-i*: Path to input dataset folder.
* *-m*: Whether to use LDA (True) or PCA (False). Uses LDA by default.
* *-fp*: List of frame prompts to use (default [0, 10, 19])
* *-t*: Test mode, whether to apply this to only the test set or all sets, defaults to True.
* *-d*: Device to load models, default is "cuda".
* *-mb*: Whether to mask background using bounding boxes, defaults to True.
* *-s*: Whehter to apply the SAM2 model (True) or not (False). Defaults to True.
* *-b*: How many frames to process simultaneously.
* *-r*: Resize factor for images. By default we use 4x, rescaling our images from 224x224 to 896x896, which then becomes embeddings with dimensions 64x64. 

## Re-identification

https://github.com/user-attachments/assets/96c3898b-8fd6-4577-b637-33da5c7a01dd

*Example video of incorrect (red), correct (green), and correct top-3 (blue) re-identifications of a query clip (left-most column) using the best RoVF model. The embedding distance between the query and gallery clip is shown underneath each thumbnail. The embeddings are based on the masked clips and displayed unmasked.*

Most of the code for reidentification can be run through `main.py`. For training a model, choose one of `cuda` or `cpu` for the device and replace the GPU index with the appropriate value. The GPU environment variable can be omitted for CPU-only runs.

Linux/macOS Bash:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py train yml_filepath.yml -d cuda
python main.py train yml_filepath.yml -d cpu
```

Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python main.py train yml_filepath.yml -d cuda
Remove-Item Env:CUDA_VISIBLE_DEVICES

python main.py train yml_filepath.yml -d cpu
```

To get the embeddings and evaluation metrics for a model, the Bash helper `get_emb_and_metric.sh` is available on Unix-like shells (note that you have to manually edit the file with correct checkpoint paths). Windows users can run the underlying Python entrypoints directly from PowerShell.

### Loading one checkpointed model

For a trained checkpoint, use the `.pt` file and `training_functions.load_model_helper.load_model_from_checkpoint()`. This helper reads the YAML metadata saved inside the checkpoint, rebuilds the matching model wrapper, and loads `model_state_dict`. You do not need to separately pass the original YAML file for this direct Python load.

Example for a RoVF-ST no-mask checkpoint:

```python
import torch
from training_functions.load_model_helper import load_model_from_checkpoint

checkpoint_path = "path/to/full_model_training/rovf_st_no_mask_example/checkpoint_epoch_2.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model_from_checkpoint(checkpoint_path).to(device)
model.eval()
```

The `.pkl` files beside the checkpoint are saved embeddings or metrics inputs, not model weights. For `main.py get_embeddings`, pass the same `.pt` checkpoint through `-cp/--ckpt_path`; for direct model loading in your own script, prefer the helper function above.

Run `python generate_yml.py` to generate all yaml files used for training; the appropriate file structure in results/ is also created.

To replicate the pre-trained DINOv2 model results in the paper, run `evaluation/get_dino_embeddings.sh`, then run `python get_metrics.py` to get the metric results for the pre-trained DINOv2 embeddings.

The command line arguments for `main.py` are as follows:

- `mode` (`str`): Mode to run the script in. Options: `train`, `test`, `get_metrics`, `get_embeddings`.
- `yaml_path` (`str`): Path to the YAML configuration file.
- `-d, --device` (`str`, default=`"cpu"`): Device to run on (e.g., `cuda`).
- `-cp, --ckpt_path` (`str`, default=`""`): Checkpoint path for resuming training or loading a trained model in `test`/`get_embeddings` mode.
- `-m, --mask_path` (`str`, default=`None`): Path to dataset masks (pickle file).
- `-am, --apply_mask_percentage` (`float`, default=`1.0`): Percentage of masks to apply.
- `-o, --override_value` (`int`, default=`None`): Value to override the number of frames.
- `-is, --is_override` (`bool`, default=`False`): Set to `True` to override the number of frames.
- `-z, --zfill_num` (`int`, default=`4`): Number of zeros to pad the frame number with.
- `-tf, --total_frames` (`int`, default=`20`): Total frames in a clip.
- `-K, --K` (`int`, default=`20`): Number of clips to sample.
- `-nf, --num_frames` (`int`, default=`10`): Number of frames to sample from each clip.
- `-dlm, --dlmode` (`str`, default=`"Test"`): Script mode (e.g., `Test`).
- `-cd, --clips_directory` (`str`, default=`"Dataset/meerkat_h5files/clips/Test"`): Directory containing clips.
- `-co, --cooccurrences_filepath` (`str`, default=`"Dataset/meerkat_h5files/Cooccurrences.json"`): Path to cooccurrences file.
- `-ep, --embedding_path` (`str`, default=`None`): Path for saving or loading embeddings.
- `-df, --dataframe_path` (`str`, default=`"Dataset/meerkat_h5files/Precomputed_test_examples_meerkat.csv"`): Path to the dataframe.
- `-lnev, --ln_epsilon_value` (`float`, default=`None`): LayerNorm epsilon value.


## Folder Structure

Main files:
- **augmentations/:** Contains helper augmentation functions to be used by a dataset class.

- **dataloaders/:** Contains dataloaders used to process and load data for training/testing. 

- **evaluation/:** Scripts and functions for evaluating model performance.

- **figures/:** Contains figures and animations.

- **get_anchors/:** Code for obtaining "anchor" embeddings, e.g., hard sampling for triplets.

- **lr_schedulers/:** Learning rate scheduler functions.

- **models/:** Model architectures are stored here.

## Models guide

See [models/README.md](models/README.md) for a quick map of wrappers and the difference between `Perceiver` and `PerceiverV2`.

- **training_functions/:** Training, validation, and support functions related to training models.

- **training_scripts/exp_metadata/:** Contains yaml (.yml) files with model, dataloader, and other training details.

Other files:
- **Figures.ipynb:** Juypter notebook containing the code used to generate figures in the paper and supplementary material.


## Acknowledgements

This project is supported by the <a href="https://www.auckland.ac.nz/en/science/our-research/research-institutes-and-centres/nao-institute/about-naoinstitute.html">Natural, Artificial, and Organisation Intelligence Institute (NAOInstitute)</a>.

We would like to thank <a href="https://wellingtonzoo.com/">Wellington Zoo</a> for their support and expertise provided throughout the project.
