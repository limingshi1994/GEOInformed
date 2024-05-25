# BVM Segmentation Example

This repository contains scripts for training and validating satellite imagery segmentation models using the PyTorch framework and the dataset of Flemish sentinel-2 BVM.

## Example Image from BVM dataset
![Example Image](https://github.com/limingshi1994/GEOInformed/main/raster_new.png)

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

## Generating Input Samples

For input data, you need it properly formatted. This involves downloading the relevant satellite images and partitioning them into kaartbladen size, as well as generating corresponding cloud masks. Follow these steps to prepare your input data:

### 1. Downloading and Partitioning Data

Before training the model, you need to download the satellite imagery and partition it into smaller segments or tiles. This can be done using the `generate_kaartbladen_allbands.py` script. This script is responsible for:

- Downloading satellite images from the specified sources
- Partitioning the images into smaller, more manageable tiles or segments
- Organizing the data into a structure suitable for training

To run the script, navigate to the directory containing either `generate_kaartbladen_allbands.py` or `generate_kaartbladen.py` and execute:

`python generate_kaartbladen_allbands.py [options]`

Replace `[options]` with any command-line arguments the script accepts for customization, such as specifying the directory for downloads, the indices of the tiles, or the specific satellite bands you wish to download.

### 2. Generating Cloud Masks

Once you have your satellite images, you may need to generate cloud masks to exclude clouds in the images. This step is crucial for maintaining the accuracy of your segmentation model, as clouds can significantly obstruct the underlying geographical features.

Use the `compute_cloud_masks.py` script to generate cloud masks corresponding to your satellite images. The script processes each image and produces a mask indicating the presence of clouds.

To generate cloud masks, navigate to the directory containing `compute_cloud_masks.py` and run:

`python compute_cloud_masks.py [options]`

Again, replace `[options]` with the necessary command-line arguments to specify the input images' directory, output directory, and any other parameters required by the script.

### Final Note

Ensure that both the segmented satellite images and the cloud masks are stored in an organized manner, as specified in the configuration of the main segmentation script. The paths to these datasets should be correctly indicated in the arguments of the main script when initiating training or validation.

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






