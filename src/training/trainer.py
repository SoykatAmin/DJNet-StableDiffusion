import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import time
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

from models.djnet_unet import DJNetUNet
from models.diffusion import DJNetDiffusionPipeline, DJNetLoss
from data.dataset import DJNetTransitionDataset


class DJNetTrainer:
    """
    Trainer class for DJNet diffusion model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: DJNetUNet,
        pipeline: DJNetDiffusionPipeline,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[DJNetLoss] = None,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        save_dir: str = "./checkpoints",
        log_every: int = 100,
        save_every: int = 1000,
        validate_every: int = 500,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: DJNet UNet model
            pipeline: Diffusion pipeline
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            loss_fn: Loss function (default: DJNetLoss)
            device: Device for training
            mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            save_dir: Directory to save checkpoints
            log_every: Log every N steps
            save_every: Save checkpoint every N steps
            validate_every: Validate every N steps
            wandb_project: Weights & Biases project name
            wandb_run_name: Weights & Biases run name
        """
        self.model = model.to(device)
        self.pipeline = pipeline
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = Path(save_dir)
        self.log_every = log_every
        self.save_every = save_every
        self.validate_every = validate_every
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.get_trainable_parameters(),
                lr=1e-4,
                weight_decay=1e-2,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer
        
        # Initialize scheduler
        self.scheduler = scheduler
        
        # Initialize loss function
        if loss_fn is None:
            self.loss_fn = DJNetLoss()
        else:
            self.loss_fn = loss_fn
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Initialize wandb
        if wandb_project:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    'model_name': 'DJNet-StableDiffusion',
                    'batch_size': train_dataloader.batch_size,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'mixed_precision': mixed_precision,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
                    'max_grad_norm': max_grad_norm
                }
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of metrics
        """
        if self.mixed_precision:
            with autocast():
                result = self.pipeline.training_step(batch, self.device)
                loss = result['loss']
        else:
            result = self.pipeline.training_step(batch, self.device)
            loss = result['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'timesteps_mean': result['timesteps'].float().mean().item()
        }
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Validation batch
            
        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            if self.mixed_precision:
                with autocast():
                    result = self.pipeline.training_step(batch, self.device)
            else:
                result = self.pipeline.training_step(batch, self.device)
        
        return {
            'val_loss': result['loss'].item(),
            'val_timesteps_mean': result['timesteps'].float().mean().item()
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Run validation on the entire validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_metrics = []
        
        for batch in tqdm(self.val_dataloader, desc="Validating"):
            metrics = self.validation_step(batch)
            val_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in val_metrics])
        
        self.model.train()
        return avg_metrics
    
    def generate_samples(self, num_samples: int = 4) -> List[torch.Tensor]:
        """
        Generate sample transitions for visualization.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of generated transition spectrograms
        """
        self.model.eval()
        
        samples = []
        with torch.no_grad():
            # Get a validation batch for context
            if self.val_dataloader:
                val_batch = next(iter(self.val_dataloader))
                preceding = val_batch['preceding_spec'][:num_samples].to(self.device)
                following = val_batch['following_spec'][:num_samples].to(self.device)
            else:
                # Use random spectrograms as context
                height, width = 128, 128
                preceding = torch.randn(num_samples, height, width, device=self.device)
                following = torch.randn(num_samples, height, width, device=self.device)
            
            # Generate transitions
            generated = self.pipeline.generate(
                preceding,
                following,
                num_inference_steps=20  # Fast generation for validation
            )
            
            for i in range(num_samples):
                samples.append(generated[i].cpu())
        
        self.model.train()
        return samples
    
    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None):
        """
        Save model checkpoint.
        
        Args:
            metrics: Current metrics to save with checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pt')
        
        # Save best checkpoint if validation improved
        if metrics and 'val_loss' in metrics:
            if metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = metrics['val_loss']
                torch.save(checkpoint, self.save_dir / 'best_checkpoint.pt')
                print(f"New best validation loss: {self.best_val_loss:.4f}")
        
        # Save periodic checkpoint
        torch.save(checkpoint, self.save_dir / f'checkpoint_step_{self.global_step}.pt')
        
        # Also save the pipeline
        self.pipeline.save_pipeline(str(self.save_dir / f'pipeline_step_{self.global_step}'))
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from step {self.global_step}")
    
    def train(self, num_epochs: int, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        self.model.train()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            epoch_metrics = []
            
            # Training loop
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            
            for step, batch in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                # Update model parameters
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.global_step += 1
                
                # Update progress bar
                current_loss = np.mean([m['loss'] for m in epoch_metrics[-10:]])
                progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
                
                # Logging
                if self.global_step % self.log_every == 0:
                    avg_metrics = {}
                    for key in metrics.keys():
                        avg_metrics[f'train_{key}'] = np.mean([m[key] for m in epoch_metrics[-self.log_every:]])
                    
                    if wandb.run:
                        wandb.log(avg_metrics, step=self.global_step)
                    
                    print(f"Step {self.global_step}: {avg_metrics}")
                
                # Validation
                if self.global_step % self.validate_every == 0 and self.val_dataloader:
                    val_metrics = self.validate()
                    
                    if wandb.run:
                        wandb.log(val_metrics, step=self.global_step)
                    
                    print(f"Validation at step {self.global_step}: {val_metrics}")
                    
                    # Generate samples for visualization
                    if wandb.run:
                        samples = self.generate_samples(num_samples=4)
                        # Log samples to wandb (you might want to convert to images first)
                
                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    val_metrics = self.validate() if self.val_dataloader else {}
                    self.save_checkpoint(val_metrics)
            
            # End of epoch
            epoch_avg = {}
            for key in epoch_metrics[0].keys():
                epoch_avg[f'epoch_{key}'] = np.mean([m[key] for m in epoch_metrics])
            
            print(f"Epoch {epoch+1} completed: {epoch_avg}")
            
            if wandb.run:
                wandb.log(epoch_avg, step=self.global_step)
        
        # Final save
        final_metrics = self.validate() if self.val_dataloader else {}
        self.save_checkpoint(final_metrics)
        print("Training completed!")


def create_trainer(
    config: Dict,
    train_dataset: DJNetTransitionDataset,
    val_dataset: Optional[DJNetTransitionDataset] = None
) -> DJNetTrainer:
    """
    Factory function to create a trainer from configuration.
    
    Args:
        config: Training configuration dictionary
        train_dataset: Training dataset
        val_dataset: Validation dataset
        
    Returns:
        Configured trainer instance
    """
    from models.djnet_unet import create_djnet_unet
    from data.dataset import create_dataloader
    
    # Create model
    model = create_djnet_unet(
        pretrained_model_name=config.get('pretrained_model', 'runwayml/stable-diffusion-v1-5'),
        freeze_encoder=config.get('freeze_encoder', False)
    )
    
    # Create pipeline
    pipeline = DJNetDiffusionPipeline(
        unet=model,
        num_train_timesteps=config.get('num_train_timesteps', 1000)
    )
    
    # Create data loaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=config.get('val_batch_size', config.get('batch_size', 8)),
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-2)
    )
    
    # Create scheduler
    scheduler = None
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 100) * len(train_dataloader),
            eta_min=config.get('min_lr', 1e-6)
        )
    
    # Create trainer
    trainer = DJNetTrainer(
        model=model,
        pipeline=pipeline,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')),
        mixed_precision=config.get('mixed_precision', True),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        save_dir=config.get('save_dir', './checkpoints'),
        log_every=config.get('log_every', 100),
        save_every=config.get('save_every', 1000),
        validate_every=config.get('validate_every', 500),
        wandb_project=config.get('wandb_project'),
        wandb_run_name=config.get('wandb_run_name')
    )
    
    return trainer


if __name__ == "__main__":
    # Test trainer creation with dummy config
    config = {
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 1,
        'device': 'cpu',  # Use CPU for testing
        'mixed_precision': False,  # Disable for CPU
        'log_every': 10,
        'save_every': 50,
        'validate_every': 25
    }
    
    print("Trainer module ready for testing with real dataset!")
