# DJNet-StableDiffusion

Transfer Learning for DJ Transition Generation using Stable Diffusion

## Overview

This project applies transfer learning to adapt Stable Diffusion's UNet for generating audio spectrograms representing DJ transitions. Instead of generating images, the model learns to generate spectrograms that represent smooth transitions between two audio tracks.

## Core Idea

We treat audio spectrograms as images and leverage Stable Diffusion's pre-trained UNet, which already understands textures, gradients, shapes, and smooth regions from billions of images. This knowledge transfers surprisingly well to spectrogram "textures" and "shapes".

## Architecture

- **Base Model**: UNet2DConditionModel from Stable Diffusion v1.5
- **Input Adaptation**: Modified first convolutional layer to accept 3 channels:
  - Channel 1: Preceding spectrogram
  - Channel 2: Following spectrogram  
  - Channel 3: Noisy transition spectrogram
- **Output**: Denoised transition spectrogram

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── dataset.py          # DJNetTransitionDataset
│   │   ├── preprocessing.py    # Audio to spectrogram conversion
│   │   └── augmentation.py     # Data augmentation utilities
│   ├── models/
│   │   ├── djnet_unet.py      # Modified UNet for DJ transitions
│   │   └── diffusion.py       # Diffusion pipeline
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   └── loss.py            # Loss functions
│   └── utils/
│       ├── audio.py           # Audio processing utilities
│       └── visualization.py   # Plotting and visualization
├── configs/
│   └── train_config.yaml     # Training configuration
├── scripts/
│   ├── train.py              # Main training script
│   └── inference.py          # Inference script
└── notebooks/
    └── explore_data.ipynb    # Data exploration
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your dataset**: Ensure your transition JSON files are organized properly
2. **Configure training**: Edit `configs/train_config.yaml`
3. **Start training**: Run `python scripts/train.py`
4. **Monitor progress**: Use wandb for experiment tracking