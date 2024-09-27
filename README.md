# RoVF for Animal Re-identification

This repository contains the codebase for the paper **RoVF for Animal Re-identification**.

**Authors:** Mitchell Rogers, Kobe Knowles, Gaël Gendron, Shahrokh Heidari, Isla Duporge, David Arturo Soriano Valdez, Mihailo Azhar, Padriac O’Leary, Simon Eyre, Michael Witbrock, Patrice Delmas.<br/>
**Supported by:** *Natural, Artificial, and Organisation Intelligence Institute, The University of Auckland, New Zealand*

## Abstract

> Recent advances in deep learning have significantly improved the accuracy and scalability of animal re-identification methods by automating the extraction of subtle distinguishing features from images and videos. This enables large-scale non-invasive monitoring of animal populations. We propose a segmentation pipeline and a re-identification model to re-identify animals without ground-truth IDs. The segmentation pipeline segments animals from the background based on their bounding boxes using the DINOv2 and segment anything model 2 (SAM2) foundation models. For re-identification, we introduce a method called recurrence over video frames (RoVF), which uses a recurrent component based on the Perceiver transformer on top of a DINOv2 image model to iteratively construct embeddings from video frames. We report the performance of the proposed segmentation pipeline and re-identification model using video datasets of meerkats and polar bears (PolarBearVidID). The proposed segmentation model achieved high accuracy (94.56% and 97.37%) and IoU (73.94% and 93.08%) for meerkats and polar bears, respectively. We found that RoVF outperformed frame- and video-based baselines, achieving 46.5% and 55% top-1 accuracy on masked test sets for meerkats and polar bears, respectively. These methods show promise in reducing the annotation burden in future individual-based ecological studies. The code is available at [https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification](https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification).

## Overview
- [TODO list](#todo-list)
- [Installation](#installation)
- [Downloading the datasets](#downloading-the-datasets)
- [Background masking](#background-masking)
- [Re-identification](#re-identification)
- [Folder structure](#folder-structure)
- [Acknowledgments](#acknowledgements)

## TODO list
We are still updating this repository, and in particular, we plan to make the following changes:
* Improve this README. 
* Add segmentation evaluation code.
* Add some improvements to the segmentation code and documentation.
* Update comments/documentation for all files.
* dinov2_wrapper.py bug with concat functionality needs to be fixed (results in an error currently). 
* training_scripts/val.py: getting the top-1 and top-3 accuracy has a bug that needs to be fixed (results in an error currently).

## Installation

In your environment of choice (conda is preferred) you will need to install the following packages. Most can be installed with the provided `install_packages.sh` script, but others will need to be installed manually. A Python version that mathces your PyTorch version is necessary, e.g., Python 3.11. 

Fist you will need to install PyTorch version 2.0 or greater. You can follow the instructions here: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Then run the following bash script `install_packages.sh` to install all other required packages via pip (note that this script is set up for a conda environment). 

```bash
./install_packages.sh conda-env-name-or-path
```

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

Most of the code for reidentification can be run through `main.py`. For training a model use `CUDA_VISIBLE_DEVICES=0 python main.py train yml_filepath.yml -d [cuda|cpu]` (choose one of 'cuda' or 'cpu' for device to run on, and replace the CUDA_VISIBLE_DEVICES number with the appropriate number; the latter can be ommitted if using cpu only). To get the embeddings and evaluation metrics for a model the script `get_emb_and_metrics.sh` is used (note that you have to manually edit the file with correct checkpoint paths).

Run `python generate_yml.py` to generate all yaml files used for training; the appropriate file structure in results/ is also created.

To replicate the pre-trained DINOv2 model results in the paper, run `evaluation/get_dino_embeddings.sh`, then run `python get_metrics.py` to get the metric results for the pre-trained DINOv2 embeddings.

The command line arguments for `main.py` are as follows:

- `mode` (`str`): Mode to run the script in. Options: `train`, `test`, `get_metrics`, `get_embeddings`.
- `yaml_path` (`str`): Path to the YAML configuration file.
- `-d, --device` (`str`, default=`"cpu"`): Device to run on (e.g., `cuda`).
- `-cp, --ckpt_path` (`str`, default=`""`): Checkpoint path for resuming training.
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

- **training_functions/:** Training, validation, and support functions related to training models.

- **training_scripts/exp_metadata/:** Contains yaml (.yml) files with model, dataloader, and other training details.

Other files:
- **Figures.ipynb:** Juypter notebook containing the code used to generate figures in the paper and supplementary material.


## Acknowledgements

This project is supported by the <a href="https://www.auckland.ac.nz/en/science/our-research/research-institutes-and-centres/nao-institute/about-naoinstitute.html">Natural, Artificial, and Organisation Intelligence Institute (NAOInstitute)</a>.

We would like to thank <a href="https://wellingtonzoo.com/">Wellington Zoo</a> for their support and expertise provided throughout the project.
