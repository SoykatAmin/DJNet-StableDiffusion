"""
Production training script for DJ transition generation
Clean, modular implementation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np

# Add project paths
sys.path.append('src')
sys.path.append('configs')

# Optimize PyTorch performance
torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False

# Import configuration
try:
    from long_segment_config import *
    print(" Configuration loaded")
except ImportError:
    print(" Using default configuration")
SAMPLE_RATE = 22050
SPECTROGRAM_HEIGHT = 128
SPECTROGRAM_WIDTH = 512
IN_CHANNELS = 3
OUT_CHANNELS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
MODEL_DIM = 512

from src.models.production_unet import ProductionUNet
from src.data.dataset import DJNetTransitionDataset

class ProductionTrainer:
    """Main training class for DJ transition model"""

    def __init__(self, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f" Using device: {self.device}")

        # Initialize model
        self.model = ProductionUNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            model_dim=MODEL_DIM
        ).to(self.device)

        print(f" Model parameters: {self.model.count_parameters():,}")

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=NUM_EPOCHS,
            eta_min=LEARNING_RATE * 0.01
        )

        # Initialize loss function
        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Initialize tensorboard
        self.writer = SummaryWriter('logs/production_training')

        # Create checkpoint directory
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)

    def create_datasets(self):
        """Create training and validation datasets from existing audio files"""
        print(" Creating datasets from structured audio data...")

        # Check if data directory exists
        data_dir = Path('data')
        metadata_file = data_dir / 'metadata.csv'

        if not data_dir.exists():
            print(f" Data directory {data_dir} not found. Creating it...")
            data_dir.mkdir(exist_ok=True)
            print(" Please add your transition audio files to the 'data' folder")

        if not metadata_file.exists():
            print(f" Metadata file {metadata_file} not found.")
            print(" Expected structure: data/metadata.csv with transition information")
            print(" Your dataset should have folders like transition_00000/, transition_00001/, etc.")
            print(" Each folder should contain: source_a.wav, source_b.wav, target.wav")
            sys.exit(1)

        # Create structured dataset using existing DJNetTransitionDataset
        try:
            # Create full dataset first
            full_dataset = DJNetTransitionDataset(
                data_dir=str(data_dir),
                spectrogram_size=(SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH),
                sample_rate=SAMPLE_RATE,
                normalize=True
            )

            # Manual train/val split since the class doesn't support it directly
            total_size = len(full_dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size

            # Create split indices
            indices = list(range(total_size))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            # Create subset datasets
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)

        except Exception as e:
            print(f" Error creating structured dataset: {e}")
            print(" Make sure your metadata.csv and transition folders are properly formatted")
            print(" Expected: data/transition_00000/source_a.wav, source_b.wav, target.wav")
            sys.exit(1)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=False  # Disabled for CPU training
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=False  # Disabled for CPU training
        )

        print(f" Training samples: {len(train_dataset)}")
        print(f" Validation samples: {len(val_dataset)}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Handle different return formats from dataset
            if isinstance(batch_data, dict):
                # Dataset returns dictionary format with spectrograms
                if 'preceding_spec' in batch_data and 'following_spec' in batch_data and 'transition_spec' in batch_data:
                    # Concatenate preceding and following as input (along channel dimension)
                    preceding = batch_data['preceding_spec']
                    following = batch_data['following_spec']
                    targets = batch_data['transition_spec']
                    
                    # Create 3-channel input: [preceding, following, zeros] to match model expectation
                    batch_size = preceding.shape[0]
                    height, width = preceding.shape[-2:]
                    
                    # Stack spectrograms as channels
                    inputs = torch.stack([preceding, following, torch.zeros_like(preceding)], dim=1)
                    
                elif 'input' in batch_data and 'target' in batch_data:
                    inputs = batch_data['input']
                    targets = batch_data['target']
                elif 'inputs' in batch_data and 'targets' in batch_data:
                    inputs = batch_data['inputs']
                    targets = batch_data['targets']
                elif 'source' in batch_data and 'target' in batch_data:
                    inputs = batch_data['source']
                    targets = batch_data['target']
                else:
                    # Print available keys for debugging
                    print(f"Available keys in batch: {list(batch_data.keys())}")
                    continue
            elif isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                elif len(batch_data) == 3:
                    inputs, targets, _ = batch_data  # Skip metadata if present
                else:
                    print(f"Unexpected batch format: {len(batch_data)} items")
                    continue
            else:
                print(f"Unexpected batch type: {type(batch_data)}")
                continue
                
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(inputs)
            
            # Ensure output and target shapes match for loss calculation
            if outputs.dim() == 4 and targets.dim() == 3:
                # Model outputs [batch, 1, height, width], target is [batch, height, width]
                outputs = outputs.squeeze(1)  # Remove channel dimension
            elif outputs.dim() == 3 and targets.dim() == 4:
                # Model outputs [batch, height, width], target is [batch, 1, height, width]
                targets = targets.squeeze(1)  # Remove channel dimension
            elif outputs.dim() == 3 and targets.dim() == 3:
                # Both are [batch, height, width] - good
                pass
            elif outputs.dim() == 4 and targets.dim() == 4:
                # Both are [batch, channels, height, width] - good
                pass
            else:
                print(f"Shape mismatch: outputs {outputs.shape}, targets {targets.shape}")
                continue
            
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()

            # Progress logging
            if batch_idx % 100 == 0:
                progress = batch_idx / num_batches * 100
                print(f" Batch {batch_idx}/{num_batches} ({progress:.1f}%) - Loss: {loss.item():.6f}")

        return epoch_loss / num_batches

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_data in self.val_loader:
                # Handle different return formats from dataset
                if isinstance(batch_data, dict):
                    # Dataset returns dictionary format with spectrograms
                    if 'preceding_spec' in batch_data and 'following_spec' in batch_data and 'transition_spec' in batch_data:
                        # Concatenate preceding and following as input (along channel dimension)
                        preceding = batch_data['preceding_spec']
                        following = batch_data['following_spec']
                        targets = batch_data['transition_spec']
                        
                        # Create 3-channel input: [preceding, following, zeros] to match model expectation
                        batch_size = preceding.shape[0]
                        height, width = preceding.shape[-2:]
                        
                        # Stack spectrograms as channels
                        inputs = torch.stack([preceding, following, torch.zeros_like(preceding)], dim=1)
                        
                    elif 'input' in batch_data and 'target' in batch_data:
                        inputs = batch_data['input']
                        targets = batch_data['target']
                    elif 'inputs' in batch_data and 'targets' in batch_data:
                        inputs = batch_data['inputs']
                        targets = batch_data['targets']
                    elif 'source' in batch_data and 'target' in batch_data:
                        inputs = batch_data['source']
                        targets = batch_data['target']
                    else:
                        continue
                elif isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                    elif len(batch_data) == 3:
                        inputs, targets, _ = batch_data  # Skip metadata if present
                    else:
                        continue
                else:
                    continue
                    
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                
                # Ensure output and target shapes match for loss calculation
                if outputs.dim() == 4 and targets.dim() == 3:
                    # Model outputs [batch, 1, height, width], target is [batch, height, width]
                    outputs = outputs.squeeze(1)  # Remove channel dimension
                elif outputs.dim() == 3 and targets.dim() == 4:
                    # Model outputs [batch, height, width], target is [batch, 1, height, width]
                    targets = targets.squeeze(1)  # Remove channel dimension
                elif outputs.dim() == 3 and targets.dim() == 3:
                    # Both are [batch, height, width] - good
                    pass
                elif outputs.dim() == 4 and targets.dim() == 4:
                    # Both are [batch, channels, height, width] - good
                    pass
                else:
                    continue
                
                loss = self.criterion(outputs, targets)
                epoch_loss += loss.item()

        return epoch_loss / num_batches

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'in_channels': IN_CHANNELS,
                'out_channels': OUT_CHANNELS,
                'model_dim': MODEL_DIM
            }
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'production_model_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f" New best model saved: {best_path}")

    def train(self):
        """Main training loop"""
        print(" Starting training...")
        print(f" Epochs: {NUM_EPOCHS} | Batch size: {BATCH_SIZE}")

        # Create datasets
        self.create_datasets()

        start_time = time.time()

        for epoch in range(NUM_EPOCHS):
            self.current_epoch = epoch + 1

            print(f"\n Epoch {self.current_epoch}/{NUM_EPOCHS}")
            print("-" * 50)

            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, self.current_epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, self.current_epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, self.current_epoch)

            # Print epoch summary
            print(f" Train Loss: {train_loss:.6f}")
            print(f" Val Loss: {val_loss:.6f}")
            print(f"📚 Learning Rate: {current_lr:.2e}")

            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint
            if self.current_epoch % 5 == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Training completed
        total_time = time.time() - start_time
        print(f"\n Training completed!")
        print(f" Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f" Best validation loss: {self.best_val_loss:.6f}")

        # Save final checkpoint
        self.save_checkpoint()

        # Close tensorboard writer
        self.writer.close()

def main():
    """Main execution function"""
    print(" DJ Transition Model - Production Training")
    print("=" * 60)

    try:
        # Initialize and start training
        trainer = ProductionTrainer()
        trainer.train()

        print("\n Training completed successfully!")

    except KeyboardInterrupt:
        print("\n Training interrupted by user")
    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
