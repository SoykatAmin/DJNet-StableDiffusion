# DJ Transition Generator

A clean, production-ready deep learning model for generating smooth transitions between electronic music tracks using U-Net architecture.

## Features

- **12-second segment processing** with precise spectrogram size control (128×512)
- **Production U-Net model** with 18M+ parameters for high-quality transitions
- **Smart audio cropping** to match UNet input requirements without interpolation
- **Kaggle training integration** with production-ready inference pipeline

## Quick Start

### Training
```bash
python run.py train
```

### Testing
```bash
# Test with trained model
python test_model.py
```

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── production_unet.py     # Main U-Net model for production
│   │   ├── djnet_unet.py         # Alternative U-Net implementation  
│   │   └── diffusion.py          # Diffusion model components
│   ├── data/
│   │   └── synthetic_dataset.py   # Synthetic training data generation
│   ├── training/
│   │   └── trainer.py             # Training loop and utilities
│   ├── diffusion/
│   │   └── pipeline.py            # Diffusion pipeline (experimental)
│   └── utils/
│       ├── audio_processing.py    # AudioProcessor class for mel-spectrograms
│       ├── audio.py              # Legacy audio utilities
│       ├── evaluation.py         # TransitionEvaluator for quality metrics
│       └── visualization.py      # Plotting and visualization tools
├── configs/
│   └── long_segment_config.py    # Model configuration parameters
├── notebooks/
│   └── dj-transition-training-kaggle.ipynb  # Kaggle training notebook
├── scripts/
│   ├── train.py                  # Training script
│   └── inference.py              # Inference script
├── checkpoints/                  # Saved model checkpoints
│   └── 5k/
│       └── best_model_kaggle.pt  # Best trained model
├── test/                        # Test audio files directory
├── outputs/                     # Generated transitions
├── transition_outputs/          # Additional output directory
├── data/                       # Training data storage
├── logs/                       # Training logs
├── run.py                      # Main CLI interface
├── train_model.py              # Standalone training script
├── test_model.py               # Standalone testing script
├── test.py                     # AudioProcessor testing script
├── test_reconstruction_analysis.py  # Audio reconstruction quality analysis
└── quick_test.py               # Quick functionality test
```

## How It Works

1. **Audio Processing**: Convert audio files to mel spectrograms
2. **Model Input**: Stack source A, source B, and noise as 3-channel input
3. **Transition Generation**: U-Net generates smooth transition spectrogram
4. **Audio Reconstruction**: Convert spectrogram back to audio using Griffin-Lim

## Configuration

Key parameters in `configs/long_segment_config.py`:

- `SEGMENT_DURATION = 12.0` - Length of audio segments (adjusted for UNet input size)
- `SPECTROGRAM_HEIGHT = 128` - Mel frequency bins  
- `SPECTROGRAM_WIDTH = 512` - Time frames (≈11.9 seconds effective duration)
- `MODEL_DIM = 512` - Model capacity
- `SAMPLE_RATE = 22050` - Audio sample rate
- `N_FFT = 2048` - FFT window size
- `HOP_LENGTH = 512` - STFT hop length
- `N_MELS = 128` - Number of mel frequency bins

## Model Architecture

- **Encoder**: 4 levels with max pooling
- **Bottleneck**: 512-dimensional feature space
- **Decoder**: 4 levels with skip connections
- **Output**: Tanh activation for normalized spectrograms

## Training Details

- **Dataset**: 10,000 synthetic training samples + 1,000 validation
- **Optimizer**: AdamW with cosine annealing schedule
- **Loss**: MSE between generated and target spectrograms
- **Batch Size**: 4 with gradient accumulation
- **Training Time**: ~2 hours on modern GPU

## Results

The model generates smooth, musically coherent transitions that:
- Maintain beat synchronization
- Blend frequency content naturally
- Create creative transition effects
- Preserve audio quality

## Requirements

```
torch>=1.9.0
torchaudio>=0.9.0
numpy
matplotlib
soundfile
tensorboard
```

## Audio Examples

Generated transitions demonstrate:
- Smooth crossfading between different genres
- Beat-matched transitions
- Creative frequency sweeps and effects
- Maintained audio fidelity

## License

This project is for educational and research purposes.