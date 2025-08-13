# DJNet-StableDiffusion

A deep learning project for generating smooth DJ transitions between music tracks using U-Net architecture.

## Overview
This project implements a neural network that learns to create professional-quality transitions between two audio segments, mimicking the skill of human DJs. The model processes mel-spectrograms and generates seamless transitions that maintain musical coherence.

## Architecture
- **U-Net Model**: Deep convolutional network with encoder-decoder structure
- **Input**: Two 12-second audio segments converted to mel-spectrograms (128×512)
- **Output**: Generated transition spectrogram converted back to audio
- **Training**: Supervised learning on real DJ transition pairs
- **Platform**: Model trained on Kaggle using GPU acceleration with Jupyter notebooks

## Project Structure
```
DJNet-StableDiffusion/
├── src/
│   ├── models/
│   │   ├── production_unet.py     # Main U-Net architecture
│   │   ├── djnet_unet.py         # Alternative U-Net implementation
│   │   └── diffusion.py          # Diffusion model components
│   ├── utils/
│   │   ├── audio_processing.py    # Mel-spectrogram processing
│   │   └── evaluation.py         # Model evaluation utilities
│   ├── training/
│   │   └── trainer.py             # Training pipeline
│   └── diffusion/
│       └── pipeline.py            # Diffusion pipeline (experimental)
├── configs/
│   ├── long_segment_config.py     # Model configuration
│   └── train_config.yaml         # Training parameters
├── app/
│   ├── app.py                     # Flask web interface
│   ├── templates/                 # HTML templates
│   └── static/                    # CSS/JS assets
├── scripts/
│   ├── train.py                   # Training script
│   └── inference.py               # Inference utilities
├── notebooks/                     # Jupyter notebooks
├── checkpoints/                   # Saved model weights
├── fad_experiments/               # FAD evaluation data
├── data/                          # Training datasets
├── outputs/                       # Generated inference transitions
├── logs/                          # Training logs
├── test/                          # Unit tests
│
├── train_model.py                 # Main training script
├── test_model.py                  # Model testing
├── evaluate_fad.py                # FAD evaluation
├── evaluate_fad_experiments.py    # Batch FAD analysis
├── rhythmic_analysis.py           # Rhythmic consistency evaluation
├── batch_rhythmic_analysis.py     # Batch rhythmic evaluation
├── combined_evaluation.py         # Combined FAD + rhythmic analysis
├── generate_djnet_transitions.py  # Batch transition generation
├── run.py                         # Universal run script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Usage
```bash
# Train model
python train_model.py

# Test web interface
cd app && python app.py
```

