# Autoencoder Hyperparameter Tuning Project in TensorFlow 2

## Introduction

This repository contains code that can be used for hyperparameter tuning of autoencoders. The code is a bit dated but shared nevertheless. You may find some inefficiencies and bugs, but the core functionalities should work as expected. The dataset used for this project is the MNIST dataset.

## Features

### Hyperparameter Tuning

The code allows for hyperparameter tuning for the following parameters:

- Latent space size
- Epochs
- Learning rate
- Batch size
- Sparsity strengths
- Dropout strengths

These can be adjusted in the `utils.py` file at the `set_configurations` function.

### Models

You can run experiments with four different types of autoencoder architectures:

- AE (Basic Autoencoder)
- Dropout (Autoencoder with Dropout)
- Batch Normalization (Autoencoder with Batch Normalization)
- AE with FFT images (Autoencoder with Fast Fourier Transformed images)

Feel free to play around and combine these models by adjusting the code yourself.

### Image Sizes and Model Selection

In the `main.py` file, you can select the image sizes (only 64, 128, or 256) and the model to be used (`ae`, `batchnorm`, `dropout`, `fft`).

## Usage

To run the code, simply execute the `main.py` file after setting your desired configurations in `utils.py`.
