# BVM Segmentation Script

This repository contains scripts for training and validating satellite imagery segmentation models using the PyTorch framework and the dataset of Flemish sentinel-2 BVM.

## Features

- Utilizes various segmentation architectures available in segmentation_models_pytorch
- Allows dynamic choice of encoders, decoders, loss functions, and optimizers
- Customizable training and validation procedures with detailed configuration
- Includes utilities for calculating metrics such as pixel accuracy, and calibration losses
- Supports the ability to continue training from saved checkpoints

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- PyTorch 1.4 or higher
- segmentation_models_pytorch
- matplotlib
- pandas
- NumPy
- tqdm
- yaml

## Installation

Clone this repository and install the required Python packages.

git clone [https://github.com/limingshi1994/GEOInformed]  
cd [local repository]  
pip install -r requirements.txt
