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

git clone https://github.com/limingshi1994/GEOInformed  
cd local repository   
pip install -r requirements.txt


## Configuration

The script `train_calib.py` uses a command-line interface for configuration. You can specify various options such as:

- Model architecture and encoder
- Training epochs, batch sizes, and learning rates
- Loss functions and optimization strategies
- Dataset paths and input/output specifications

Use `-h` or `--help` with the script to view all available configuration options.

## Training

To start training the model with default settings, simply run:

`python train_calib.py`

with your own settings of hyperparameters.

You can customize training by specifying command-line arguments. For example:

`python main.py --epochs 50 --arch Unet --encoder resnet34`

Refer to the script's help for more detailed information on all available options.

## Validation

The script includes a validation phase after each training epoch. Validation metrics are reported and can be used for model performance tracking and early stopping.

## Output

The training process generates output files including:

- Model checkpoints saved in the specified output directory
- Training logs containing loss and accuracy metrics
- Configuration files reflecting the training setup

## Continuing Training

You can continue training from a previous checkpoint by specifying the `--continue_train` flag and providing the appropriate model checkpoint path.

## Important Notes

1. Ensure you have the right hardware (preferably GPUs) and sufficient memory for training, as BVM datasets adopt special dataloaders that randomly sample from a large surface area that can be quite large.
2. Customize the data loading, augmentation, and preprocessing steps according to your dataset's characteristics.

## License

This project is licensed under the terms of the MIT License

## Acknowledgements

This script is based on the segmentation_models_pytorch(https://github.com/qubvel/segmentation_models.pytorch) library. Special thanks to all the contributors of this library.






