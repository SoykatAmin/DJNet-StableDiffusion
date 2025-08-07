# DJ Transition Generator

A clean, production-ready deep learning model for generating smooth transitions between electronic music tracks using U-Net architecture.

## ✨ Features

- **15-second segment processing** with variable transition lengths (0.1-8 seconds)
- **Production U-Net model** with 50M+ parameters for high-quality transitions
- **Clean, modular codebase** with refactored architecture
- **Simple CLI interface** for training and testing
- **Synthetic dataset generation** for easy training without large audio datasets

## Quick Start

### Training
```bash
python run.py train
```

### Testing
```bash
# Test with trained model
python run.py test --checkpoint checkpoints/best_model.pt

# Test with custom audio files
python run.py test --checkpoint checkpoints/best_model.pt --source-a track1.wav --source-b track2.wav
```

## Project Structure

```
├── src/
│ ├── models/
│ │ └── production_unet.py # Main U-Net model
│ ├── data/
│ │ └── synthetic_dataset.py # Synthetic training data
│ └── utils/
│ └── audio_processing.py # Audio conversion utilities
├── configs/
│ └── long_segment_config.py # Model configuration
├── checkpoints/ # Saved model checkpoints
├── outputs/ # Generated transitions
├── run.py # Main CLI interface
├── train_model.py # Training script
└── test_model.py # Testing script
```

## How It Works

1. **Audio Processing**: Convert audio files to mel spectrograms
2. **Model Input**: Stack source A, source B, and noise as 3-channel input
3. **Transition Generation**: U-Net generates smooth transition spectrogram
4. **Audio Reconstruction**: Convert spectrogram back to audio using Griffin-Lim

## Configuration

Key parameters in `configs/long_segment_config.py`:

- `SEGMENT_DURATION = 15.0` - Length of audio segments
- `SPECTROGRAM_HEIGHT = 128` - Frequency bins
- `SPECTROGRAM_WIDTH = 512` - Time frames
- `MODEL_DIM = 512` - Model capacity
- `SAMPLE_RATE = 22050` - Audio sample rate

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
