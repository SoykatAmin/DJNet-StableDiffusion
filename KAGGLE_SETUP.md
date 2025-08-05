# DJNet-StableDiffusion for Kaggle

This guide helps you run DJNet-StableDiffusion training on Kaggle.

## Kaggle Setup

### 1. Create a New Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Choose "GPU P100" or "GPU T4 x2" accelerator
4. Set language to Python

### 2. Clone the Repository

```python
# Clone the repository
!git clone https://github.com/SoykatAmin/DJNet-StableDiffusion.git
%cd DJNet-StableDiffusion
```

### 3. Install Dependencies

```python
# Install required packages
!pip install -q diffusers transformers accelerate
!pip install -q librosa torchaudio 
!pip install -q wandb matplotlib seaborn
!pip install -q datasets huggingface-hub
```

### 4. Setup Kaggle-Specific Configuration

```python
# Kaggle-specific configuration
import os
os.environ['TRANSFORMERS_CACHE'] = '/kaggle/tmp/transformers_cache'
os.environ['HF_HOME'] = '/kaggle/tmp/hf_cache'

# Create necessary directories
!mkdir -p /kaggle/tmp/transformers_cache
!mkdir -p /kaggle/tmp/hf_cache
!mkdir -p checkpoints
!mkdir -p outputs
```

### 5. Dataset Setup

#### Option A: Use Kaggle Dataset
If you've uploaded your dataset to Kaggle:
```python
# Add your dataset as input to the notebook
# Then copy to working directory
!cp -r /kaggle/input/your-dataset-name/* ./data/
```

#### Option B: Upload Sample Data
For testing with sample data:
```python
# Create sample dataset structure
!mkdir -p data/sample
# Upload your JSON files and audio files to data/sample/
```

### 6. Configure Training

```python
# Update configuration for Kaggle environment
import yaml

config = {
    'data': {
        'data_dir': './data',
        'sample_rate': 16000,
        'spectrogram_size': [128, 128],
        'train_split': 0.8,
        'val_split': 0.2
    },
    'training': {
        'batch_size': 4,  # Reduced for Kaggle GPU memory
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'mixed_precision': True,
        'gradient_accumulation_steps': 2,  # Effective batch size = 8
        'save_every': 100,
        'validate_every': 50,
        'device': 'cuda'
    },
    'wandb': {
        'project': 'djnet-kaggle',
        'run_name': 'kaggle-training'
    }
}

# Save configuration
with open('configs/kaggle_config.yaml', 'w') as f:
    yaml.dump(config, f)
```

### 7. Run Training

```python
# Start training
!python scripts/train.py --config configs/kaggle_config.yaml --no_wandb
```

## Kaggle-Specific Tips

### Memory Management
- Use batch_size=4 or smaller for GPU memory constraints
- Enable mixed precision training
- Use gradient accumulation for effective larger batch sizes

### Data Loading
- Use fewer num_workers (2-4) to avoid memory issues
- Enable caching for faster data loading
- Consider reducing spectrogram size if needed

### Checkpointing
- Save checkpoints frequently (every 100 steps)
- Download important checkpoints to avoid loss

### Monitoring
- Use built-in Kaggle logging instead of wandb if needed
- Print metrics regularly for monitoring

## Example Kaggle Notebook Structure

```python
# Cell 1: Setup
!git clone https://github.com/SoykatAmin/DJNet-StableDiffusion.git
%cd DJNet-StableDiffusion
!pip install -q -r requirements.txt

# Cell 2: Import and Configure
import sys
sys.path.append('src')
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")

# Cell 3: Data Setup
# Your data loading code here

# Cell 4: Model Training
!python scripts/train.py --config configs/kaggle_config.yaml --batch_size 4

# Cell 5: Inference Test
!python scripts/inference.py --checkpoint checkpoints/best_checkpoint.pt
```

## Download Results

```python
# Compress and download checkpoints
!tar -czf djnet_checkpoints.tar.gz checkpoints/
!tar -czf djnet_outputs.tar.gz outputs/

# Download files (in Kaggle notebook output)
from IPython.display import FileLink
FileLink('djnet_checkpoints.tar.gz')
FileLink('djnet_outputs.tar.gz')
```

## Troubleshooting

### Common Issues:
1. **CUDA OOM**: Reduce batch_size to 2 or 1
2. **Slow data loading**: Reduce num_workers to 2
3. **Model loading errors**: Check internet connection for Hugging Face downloads
4. **Missing audio files**: Ensure proper dataset structure

### Performance Tips:
- Use TPU if available for larger models
- Enable Kaggle's internet access for downloading pre-trained models
- Use persistent storage for large datasets
