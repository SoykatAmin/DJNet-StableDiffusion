#!/usr/bin/env python3
"""
Kaggle-optimized training script for DJNet-StableDiffusion
Designed to work within Kaggle's GPU memory and time constraints
"""
import os
import sys
import gc
import torch
import yaml
import argparse
from pathlib import Path
import wandb
from datetime import datetime

# Add src to path
sys.path.append('./src')

from models.djnet_unet import DJNetUNet
from training.trainer import DJNetTrainer
from data.dataset import DJNetTransitionDataset
from utils.audio_utils import AudioProcessor
from torch.utils.data import DataLoader, random_split

def optimize_for_kaggle():
    """Optimize PyTorch settings for Kaggle environment"""
    # Memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Set environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
def check_kaggle_environment():
    """Check if running in Kaggle and setup paths accordingly"""
    is_kaggle = os.path.exists('/kaggle')
    
    if is_kaggle:
        print("✓ Running in Kaggle environment")
        data_path = "/kaggle/input"
        output_path = "/kaggle/working"
        
        # Check for common dataset patterns
        if os.path.exists("/kaggle/input/dj-transition-dataset"):
            data_path = "/kaggle/input/dj-transition-dataset"
        elif os.path.exists("/kaggle/input/audio-dataset"):
            data_path = "/kaggle/input/audio-dataset"
            
        print(f"Data path: {data_path}")
        print(f"Output path: {output_path}")
        
        return data_path, output_path
    else:
        print("ℹ Running locally")
        return "./data", "./checkpoints"

def load_config(config_path):
    """Load and modify config for Kaggle"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Kaggle-specific modifications
    data_path, output_path = check_kaggle_environment()
    
    config['data']['data_dir'] = data_path
    config['training']['save_dir'] = output_path
    
    # GPU memory check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 16:  # If less than 16GB, use smaller batch
            config['training']['batch_size'] = 2
            config['training']['gradient_accumulation_steps'] = 4
            print("⚠ Reduced batch size for GPU memory constraints")
    
    return config

def setup_wandb(config):
    """Setup Weights & Biases with Kaggle-specific settings"""
    # Check if WANDB API key is available
    api_key = os.environ.get('WANDB_API_KEY')
    
    if api_key:
        wandb.login(key=api_key)
        
        # Generate unique run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"kaggle_djnet_{timestamp}"
        
        wandb.init(
            project=config['wandb']['project'],
            name=run_name,
            config=config,
            tags=config['wandb']['tags'] + ['kaggle-run'],
            notes=config['wandb']['notes']
        )
        print("✓ Weights & Biases initialized")
        return True
    else:
        print("⚠ WANDB_API_KEY not found, running without logging")
        return False

def create_data_loaders(config):
    """Create optimized data loaders for Kaggle"""
    # Initialize audio processor
    audio_processor = AudioProcessor(
        sample_rate=config['data']['sample_rate'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length'],
        n_mels=config['data']['n_mels']
    )
    
    # Create dataset
    dataset = DJNetTransitionDataset(
        data_dir=config['data']['data_dir'],
        json_files=config['data']['json_files'],
        audio_processor=audio_processor,
        normalize=config['data']['normalize'],
        cache_spectrograms=config['data']['cache_spectrograms']
    )
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Split dataset
    train_size = int(config['data']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders with Kaggle optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        drop_last=True,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['val_batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        drop_last=False,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Kaggle training for DJNet-StableDiffusion')
    parser.add_argument('--config', type=str, default='configs/kaggle_config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--fast-test', action='store_true',
                       help='Run fast test with minimal epochs')
    
    args = parser.parse_args()
    
    print("🚀 Starting DJNet Kaggle Training")
    print("=" * 50)
    
    # Optimize for Kaggle
    optimize_for_kaggle()
    
    # Load config
    config = load_config(args.config)
    
    # Fast test mode
    if args.fast_test:
        config['training']['num_epochs'] = 2
        config['training']['log_every'] = 5
        config['training']['validate_every'] = 10
        print("🏃 Fast test mode enabled")
    
    # Setup wandb
    use_wandb = setup_wandb(config)
    
    # Create data loaders
    print("\n📁 Setting up data loaders...")
    try:
        train_loader, val_loader = create_data_loaders(config)
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        print("ℹ Trying with minimal dataset...")
        # Fallback: create dummy dataset for testing
        sys.exit(1)
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = DJNetUNet(
        pretrained_model=config['model']['pretrained_model'],
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    
    # Initialize trainer
    print("\n🎯 Initializing trainer...")
    trainer = DJNetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Print training info
    print("\n📊 Training Configuration:")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Effective batch size: {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Mixed precision: {config['training']['mixed_precision']}")
    print(f"  Device: {trainer.device}")
    
    # Start training
    print("\n🚀 Starting training...")
    print("=" * 50)
    
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
        
        if use_wandb:
            wandb.finish()
            
    except KeyboardInterrupt:
        print("\n⏸ Training interrupted by user")
        trainer.save_checkpoint(epoch=trainer.current_epoch, is_best=False, suffix='interrupted')
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save emergency checkpoint
        try:
            trainer.save_checkpoint(epoch=trainer.current_epoch, is_best=False, suffix='emergency')
            print("💾 Emergency checkpoint saved")
        except:
            pass
        
        sys.exit(1)
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
