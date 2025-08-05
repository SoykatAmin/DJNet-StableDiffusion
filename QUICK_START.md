# Quick Start Guide for DJNet-StableDiffusion

## Project Overview

This project implements transfer learning from Stable Diffusion to generate DJ transitions between audio tracks. We treat audio spectrograms as images and adapt the pre-trained UNet to understand musical transitions.

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run setup script**:
   ```bash
   python setup.py
   ```

3. **Update configuration**:
   Edit `configs/train_config.yaml` with your dataset path:
   ```yaml
   data:
     data_dir: "path/to/your/dataset"
   ```

## Usage Examples

### 1. Data Exploration
```bash
jupyter notebook notebooks/explore_data.ipynb
```

### 2. Training
```bash
# Basic training
python scripts/train.py

# Custom configuration
python scripts/train.py --config configs/train_config.yaml --data_dir /path/to/data

# Resume training
python scripts/train.py --resume checkpoints/latest_checkpoint.pt
```

### 3. Inference
```bash
# Generate transitions
python scripts/inference.py --checkpoint checkpoints/best_checkpoint.pt --audio_a track1.mp3 --audio_b track2.mp3

# Custom parameters
python scripts/inference.py --pipeline checkpoints/pipeline_step_1000 --audio_a track1.mp3 --audio_b track2.mp3 --num_inference_steps 50 --output_dir outputs/
```

### 4. Programmatic Usage

```python
import sys
sys.path.append('src')

from models.djnet_unet import create_djnet_unet
from models.diffusion import DJNetDiffusionPipeline
from data.dataset import DJNetTransitionDataset

# Create model
model = create_djnet_unet(
    pretrained_model_name="runwayml/stable-diffusion-v1-5",
    freeze_encoder=False
)

# Create pipeline
pipeline = DJNetDiffusionPipeline(unet=model)

# Load dataset
dataset = DJNetTransitionDataset(
    data_dir="path/to/dataset",
    spectrogram_size=(128, 128)
)

# Training example
from training.trainer import create_trainer

config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'device': 'cuda'
}

trainer = create_trainer(config, dataset)
trainer.train(num_epochs=10)
```

## Project Structure

```
DJNet-StableDiffusion/
├── src/
│   ├── models/          # Model definitions
│   ├── data/            # Dataset and preprocessing
│   ├── training/        # Training loop and utilities
│   └── utils/           # Audio processing and visualization
├── scripts/             # Command-line scripts
├── configs/             # Configuration files
├── notebooks/           # Jupyter notebooks
└── requirements.txt     # Dependencies
```

## Key Features

- **Transfer Learning**: Adapts Stable Diffusion UNet for audio spectrograms
- **3-Channel Input**: [preceding_spec, following_spec, noisy_transition_spec]
- **Flexible Training**: Configurable training parameters and resuming
- **Audio Pipeline**: Complete audio-to-spectrogram-to-audio workflow
- **Evaluation**: Comprehensive metrics and visualization tools

## Dataset Format

Your dataset should contain JSON files with this structure:
```json
{
  "source_a_path": "path/to/track_a.mp3",
  "source_b_path": "path/to/track_b.mp3",
  "source_segment_length_sec": 15.0,
  "transition_length_sec": 6.778,
  "sample_rate": 16000,
  "transition_type": "exp_fade",
  "avg_tempo": 141.6,
  "start_position_a_sec": 5.3,
  "start_position_b_sec": 12.3
}
```

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU training
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB RAM, NVIDIA RTX 3080/4080 or better

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch_size in config
   - Use gradient_accumulation_steps
   - Enable mixed_precision

2. **Audio files not found**:
   - Check file paths in JSON metadata
   - Ensure audio files are accessible
   - Update data_dir in configuration

3. **Slow training**:
   - Use GPU if available
   - Increase num_workers in DataLoader
   - Enable mixed precision training

### Performance Tips

- Use SSD storage for datasets
- Enable audio caching for faster loading
- Monitor GPU memory usage
- Use wandb for experiment tracking

## Examples and Demos

See `notebooks/explore_data.ipynb` for a complete walkthrough of:
- Loading and exploring the dataset
- Audio processing pipeline
- Model architecture modification
- Training loop implementation
- Transition generation and evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes. Please respect the licenses of the underlying models and datasets.
