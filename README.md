# RoVF for Animal Re-identification

This repository contains the codebase for the paper **RoVF for Animal Re-identification**.

**Authors:** Mitchell Rogers, Kobe Knowles, Gaël Gendron, Shahrokh Heidari, Isla Duporge, David Arturo Soriano Valdez, Mihailo Azhar, Padriac O’Leary, Simon Eyre, Michael Witbrock, Patrice Delmas.<br/>
**Supported by:** *Natural, Artificial, and Organisation Intelligence Institute, The University of Auckland, New Zealand*

## Abstract

> Recent advances in deep learning have significantly improved the accuracy and scalability of animal re-identification methods by automating the extraction of subtle distinguishing features from images and videos. This enables large-scale non-invasive monitoring of animal populations. We propose a segmentation pipeline and a re-identification model to re-identify animals without ground-truth IDs. The segmentation pipeline segments animals from the background based on their bounding boxes using the DINOv2 and segment anything model 2 (SAM2) foundation models. For re-identification, we introduce a method called recurrence over video frames (RoVF), which uses a recurrent component based on the Perceiver transformer on top of a DINOv2 image model to iteratively construct embeddings from video frames. We report the performance of the proposed segmentation pipeline and re-identification model using video datasets of meerkats and polar bears (PolarBearVidID). The proposed segmentation model achieved high accuracy (94.56% and 97.37%) and IoU (73.94% and 93.08%) for meerkats and polar bears, respectively. We found that RoVF outperformed frame- and video-based baselines, achieving 46.5% and 55% top-1 accuracy on masked test sets for meerkats and polar bears, respectively. These methods show promise in reducing the annotation burden in future individual-based ecological studies. The code is available at [https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification](https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification).

## TODO list
We are still updating this repository, and in particular, we plan to make the following changes:
* Improve this README. 
* Add segmentation code.
* Fix instillation: some packages are not installed correctly.
* Upload all training yaml files.
* Update comments/documentation for all files.
* Add instructions on how to run the re-identification main.py through the command line.
* Generate_yml.py should be updated to create all yaml file documents as well as results/ folder structure.
* Instructions on how to set up Dataset/ folder (perhaps from downloadable links) and a bash script.

## Installation

To set up the environment, you can use the provided `setup_environment.py` script. This will install the necessary dependencies and configurations.

```bash
python setup_environment.py
```


## Downloading data
For our experiments, we use two animal video datasets:
* [Meerkats](https://meerkat-dataset.github.io/re-id/) based on the [Meerkat behaviour recognition dataset](https://meerkat-dataset.github.io/)
* Polar bears based on the [PolarBearVidID dataset](https://doi.org/10.3390/ani13050801) (Zuerl et al. 2023)

We provide a link to download the dataset (~22 GB) in the correct format for the meerkats on [the dataset page](https://meerkat-dataset.github.io/re-id/). The zip file can be extracted to the */Dataset/* folder. For the polar bear dataset we provide a python script in the */Dataset/* folder to convert the dataset to the format we use here.

*TODO: Add script to process the dataset*

## Background masking

https://github.com/user-attachments/assets/8177ab7a-e43f-486b-aff8-f9eba1767d62

*Example video of the background masking performance, including cases where the performance is poor.*

The background masking approach we use is based on two foundation models [DINOv2]() and [Segment Anything Model 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2).

*TODO: Add code to apply this background masking method*


## Re-identification

https://github.com/user-attachments/assets/96c3898b-8fd6-4577-b637-33da5c7a01dd

*Example video of incorrect (red), correct (green), and correct top-3 (blue) re-identifications of a query clip (left-most column) using the best RoVF model. The embedding distance between the query and gallery clip is shown underneath each thumbnail. The embeddings are based on the masked clips and displayed unmasked.

*TODO: Add instructions how to use the code




## Folder Structure

- **augmentations/:** Contains augmentation functions used for enhancing the training data. These functions improve model robustness by generating variations of the input images.

- **dataloaders/:** Data loaders for efficiently fetching and preparing data during training and inference. This directory defines how the data pipeline is structured.

- **evaluation/:** Scripts and utilities for evaluating the model performance, including metrics and visualizations. The results of the reidentification model are analyzed here.

- **figures/:** Contains generated figures and visualizations used to evaluate the performance and other aspects of the model.

- **get_anchors/:** Code for obtaining "anchor" embeddings, which serve as reference points in the reidentification task.

- **lr_schedulers/:** Learning rate scheduler configurations and implementations, designed to dynamically adjust the learning rate during training.

- **models/:** Code defining the architectures of various neural network models used in the project, including pre-trained or custom models for the reidentification task.

- **training_functions/:** Core functions required to train the models, including loss functions, optimization routines, and data pre-processing steps.

- **training_scripts/exp_metadata/:** Contains scripts and experiment metadata for training different image-based models. Useful for reproducing experiments and comparing results.

Other files:
- **Figures.ipynb:** Juypter notebook containing the code used to generate figures in the paper and supplementary material.


## Acknowledgements

This project is supported by the <a href="https://www.auckland.ac.nz/en/science/our-research/research-institutes-and-centres/nao-institute/about-naoinstitute.html">Natural, Artificial, and Organisation Intelligence Institute (NAOInstitute)</a>.

We would like to thank <a href="https://wellingtonzoo.com/">Wellington Zoo</a> for their support and expertise provided throughout the project.
