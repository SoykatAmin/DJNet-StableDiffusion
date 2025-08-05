#!/usr/bin/env python3
"""
Main training script for DJNet-StableDiffusion.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/train_config.yaml --data_dir /path/to/dataset
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
import torch
from torch.utils.data import random_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import DJNetTransitionDataset
from training.trainer import create_trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def override_config_from_args(config: dict, args: argparse.Namespace) -> dict:
    """Override configuration with command line arguments."""
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        config['training']['val_batch_size'] = args.batch_size
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    if args.device:
        config['training']['device'] = args.device
    
    if args.save_dir:
        config['training']['save_dir'] = args.save_dir
    
    if args.wandb_project:
        config['wandb']['project'] = args.wandb_project
    
    if args.wandb_run_name:
        config['wandb']['run_name'] = args.wandb_run_name
    
    if args.resume:
        config['resume']['checkpoint_path'] = args.resume
    
    return config


def create_datasets(config: dict):
    """Create training and validation datasets."""
    data_config = config['data']
    
    # Create full dataset
    dataset = DJNetTransitionDataset(
        data_dir=data_config['data_dir'],
        json_files=data_config.get('json_files'),
        sample_rate=data_config['sample_rate'],
        n_fft=data_config['n_fft'],
        hop_length=data_config['hop_length'],
        n_mels=data_config['n_mels'],
        spectrogram_size=tuple(data_config['spectrogram_size']),
        normalize=data_config['normalize'],
        augment=data_config['augment'],
        cache_spectrograms=data_config['cache_spectrograms']
    )
    
    # Split into train and validation
    total_size = len(dataset)
    train_size = int(data_config['train_split'] * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducible splits
    )
    
    print(f"Created datasets:")
    print(f"  Total: {total_size} samples")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def setup_device(config: dict) -> torch.device:
    """Setup compute device."""
    device_config = config['training']['device']
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_config)
        print(f"Using device: {device}")
    
    return device


def main():
    parser = argparse.ArgumentParser(description="Train DJNet-StableDiffusion model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir", 
        type=str,
        help="Override data directory from config"
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float,
        help="Override learning rate from config"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int,
        help="Override number of epochs from config"
    )
    parser.add_argument(
        "--device", 
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        help="Override device from config"
    )
    parser.add_argument(
        "--save_dir", 
        type=str,
        help="Override save directory from config"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str,
        help="Override wandb project name from config"
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str,
        help="Override wandb run name from config"
    )
    parser.add_argument(
        "--resume", 
        type=str,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--no_wandb", 
        action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Dry run - setup everything but don't train"
    )
    
    args = parser.parse_args()
    
    # Load and process configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    config = override_config_from_args(config, args)
    
    # Disable wandb if requested
    if args.no_wandb:
        config['wandb']['project'] = None
    
    # Setup device
    device = setup_device(config)
    config['training']['device'] = str(device)
    
    # Validate data directory
    data_dir = Path(config['data']['data_dir'])
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist!")
        print("Please update the data_dir in your config or use --data_dir argument")
        sys.exit(1)
    
    # Create datasets
    print("Creating datasets...")
    try:
        train_dataset, val_dataset = create_datasets(config)
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("Please check your data directory and JSON files")
        sys.exit(1)
    
    # Create trainer
    print("Creating trainer...")
    trainer_config = {
        **config['training'],
        **config['diffusion'],
        **config['loss'],
        'wandb_project': config['wandb']['project'],
        'wandb_run_name': config['wandb']['run_name']
    }
    
    trainer = create_trainer(
        config=trainer_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    print("Trainer created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}")
    
    if args.dry_run:
        print("Dry run completed - exiting without training")
        return
    
    # Start training
    print("Starting training...")
    try:
        trainer.train(
            num_epochs=config['training']['num_epochs'],
            resume_from_checkpoint=config['resume']['checkpoint_path']
        )
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save checkpoint on interruption
        trainer.save_checkpoint()
        print("Checkpoint saved")
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Save checkpoint on error
        trainer.save_checkpoint()
        print("Emergency checkpoint saved")
        raise


if __name__ == "__main__":
    main()
