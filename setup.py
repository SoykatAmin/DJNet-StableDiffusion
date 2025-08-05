#!/usr/bin/env python3
"""
Setup script for DJNet-StableDiffusion project.

This script helps you get started with the project by:
1. Installing dependencies
2. Setting up the environment
3. Testing the installation
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True


def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        "torch",
        "torchaudio", 
        "diffusers",
        "transformers",
        "librosa",
        "numpy",
        "matplotlib"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        return False
    
    print("✓ All imports successful!")
    return True


def test_gpu():
    """Test GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available! GPU: {torch.cuda.get_device_name()}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  PyTorch version: {torch.__version__}")
        else:
            print("⚠ CUDA not available. Using CPU.")
        return True
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def create_data_directories():
    """Create necessary data directories."""
    print("Creating data directories...")
    
    directories = [
        "data",
        "checkpoints", 
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✓ Created {directory}/")
    
    return True


def main():
    print("🎵 DJNet-StableDiffusion Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("✗ requirements.txt not found! Make sure you're in the project root directory.")
        sys.exit(1)
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    print()
    
    # Test imports
    if not test_imports():
        success = False
    
    print()
    
    # Test GPU
    test_gpu()
    
    print()
    
    # Create directories
    create_data_directories()
    
    print()
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update the data path in configs/train_config.yaml")
        print("2. Run the data exploration notebook: notebooks/explore_data.ipynb")
        print("3. Start training: python scripts/train.py")
    else:
        print("⚠ Setup completed with some issues. Please check the errors above.")


if __name__ == "__main__":
    main()
