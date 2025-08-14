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

## Pre-trained Model
Download the trained checkpoint to use the model:
- **Checkpoint**: [Download from Google Drive](https://drive.google.com/file/d/1IahkcCsRGXk6KfS8CeJIRKkhJbpOgnoZ/view?usp=sharing)
- Place the checkpoint in `checkpoints/` directory

## Usage

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/SoykatAmin/DJNet-StableDiffusion.git
cd DJNet-StableDiffusion

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Model
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download the checkpoint from Google Drive link above
# Place it in checkpoints/ directory as: checkpoints/best_model.pt
```

### 3. Run Web Interface
```bash
# Start the Flask app
cd app
python app.py

# Open browser at http://localhost:5000
# Upload two audio files and generate transitions
```

### 4. Generate Transitions (Command Line)
```bash
# Test with pre-trained model
python test_model.py

# Generate batch transitions
python generate_djnet_transitions.py
```

### 5. Evaluation
```bash
# Run FAD evaluation
python evaluate_fad_experiments.py

# Run rhythmic analysis
python batch_rhythmic_analysis.py

# Combined evaluation
python combined_evaluation.py
```

### 6. Dataset (optional)

If you want to create the dataset used to train the model, use this repository: https://github.com/SoykatAmin/DJNet-Dataset

### 6. Training
```bash
# Train your own model
python train_model.py

# Or use the universal runner
python run.py train
```

## License

This project is licensed under the MIT License - see the details below.

### MIT License

```
MIT License

Copyright (c) 2025 SoykatAmin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```