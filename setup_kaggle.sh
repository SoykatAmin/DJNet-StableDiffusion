#!/bin/bash
# Quick setup script for Kaggle environment

echo "🔧 Setting up DJNet-StableDiffusion for Kaggle..."

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs

# Install required packages (run this in Kaggle cell)
echo "📦 Installing required packages..."
echo "Run this in a Kaggle cell:"
echo ""
echo "!pip install diffusers transformers accelerate wandb"
echo "!pip install librosa torchaudio soundfile"
echo "!pip install pyyaml tensorboard"
echo ""

# Set up environment variables
echo "🌍 Setting up environment variables..."
echo "Add this to your Kaggle notebook:"
echo ""
echo "import os"
echo "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'"
echo "os.environ['TOKENIZERS_PARALLELISM'] = 'false'"
echo ""

# Quick test command
echo "🧪 Quick test command:"
echo "python kaggle_train.py --config configs/kaggle_config.yaml --fast-test"
echo ""

echo "✅ Setup complete! Check KAGGLE_SETUP.md for detailed instructions."
