import torch
import torch.nn as nn
from diffusers import DDPMScheduler, DDIMScheduler
from typing import Optional, Union, Dict, Any
import numpy as np
from .djnet_unet import DJNetUNet


class DJNetDiffusionPipeline:
    """
    Diffusion pipeline for DJ transition generation.
    
    This pipeline handles the forward and reverse diffusion processes
    for generating smooth DJ transitions between audio spectrograms.
    """
    
    def __init__(
        self,
        unet: DJNetUNet,
        scheduler: Optional[Union[DDPMScheduler, DDIMScheduler]] = None,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True
    ):
        """
        Initialize the diffusion pipeline.
        
        Args:
            unet: The modified UNet model for DJ transitions
            scheduler: Noise scheduler (DDPM or DDIM)
            num_train_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value for noise schedule
            beta_end: Ending beta value for noise schedule
            beta_schedule: Type of beta schedule ("linear", "scaled_linear", "squaredcos_cap_v2")
            clip_sample: Whether to clip samples to [-1, 1]
        """
        self.unet = unet
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        
        # Initialize scheduler if not provided
        if scheduler is None:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                clip_sample=clip_sample
            )
        else:
            self.scheduler = scheduler
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to the original samples according to the noise schedule.
        
        Args:
            original_samples: Original clean spectrograms
            noise: Gaussian noise to add
            timesteps: Diffusion timesteps
            
        Returns:
            Noisy samples
        """
        return self.scheduler.add_noise(original_samples, noise, timesteps)
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch containing preceding_spec, following_spec, transition_spec
            device: Device to run computation on
            
        Returns:
            Dictionary containing loss and other metrics
        """
        # Move batch to device
        preceding_spec = batch['preceding_spec'].to(device)
        following_spec = batch['following_spec'].to(device)
        transition_spec = batch['transition_spec'].to(device)
        
        batch_size = transition_spec.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,), device=device
        ).long()
        
        # Sample noise to add to the transition spectrograms
        noise = torch.randn_like(transition_spec)
        
        # Add noise to the transition spectrograms
        noisy_transition = self.add_noise(transition_spec, noise, timesteps)
        
        # Create 3-channel input: [preceding, following, noisy_transition]
        model_input = torch.cat([
            preceding_spec.unsqueeze(1),      # Add channel dim
            following_spec.unsqueeze(1),      # Add channel dim  
            noisy_transition.unsqueeze(1)     # Add channel dim
        ], dim=1)  # Shape: (batch_size, 3, height, width)
        
        # Predict the noise
        model_output = self.unet(model_input, timesteps)
        
        # Calculate loss (MSE between predicted and actual noise)
        loss = nn.functional.mse_loss(model_output.sample, noise)
        
        return {
            'loss': loss,
            'predicted_noise': model_output.sample,
            'actual_noise': noise,
            'noisy_transition': noisy_transition,
            'timesteps': timesteps
        }
    
    @torch.no_grad()
    def generate(
        self,
        preceding_spec: torch.Tensor,
        following_spec: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate a transition spectrogram given preceding and following spectrograms.
        
        Args:
            preceding_spec: Preceding track spectrogram (batch_size, height, width)
            following_spec: Following track spectrogram (batch_size, height, width)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for conditional generation
            generator: Random number generator for reproducible results
            
        Returns:
            Generated transition spectrogram
        """
        device = preceding_spec.device
        batch_size = preceding_spec.shape[0]
        height, width = preceding_spec.shape[-2:]
        
        # Set scheduler for inference
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Start with random noise
        transition_spec = torch.randn(
            (batch_size, height, width),
            device=device,
            generator=generator
        )
        
        # Denoising loop
        for timestep in self.scheduler.timesteps:
            # Create model input
            model_input = torch.cat([
                preceding_spec.unsqueeze(1),
                following_spec.unsqueeze(1),
                transition_spec.unsqueeze(1)
            ], dim=1)
            
            # Predict noise
            timestep_tensor = timestep.unsqueeze(0).repeat(batch_size).to(device)
            noise_pred = self.unet(model_input, timestep_tensor).sample
            
            # Compute the previous noisy sample
            transition_spec = self.scheduler.step(
                noise_pred, timestep, transition_spec
            ).prev_sample
        
        # Clip samples if requested
        if self.clip_sample:
            transition_spec = torch.clamp(transition_spec, -1.0, 1.0)
        
        return transition_spec
    
    @torch.no_grad()
    def interpolate_transitions(
        self,
        spec_a: torch.Tensor,
        spec_b: torch.Tensor,
        num_steps: int = 10,
        num_inference_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate a sequence of transitions that interpolate between two spectrograms.
        
        Args:
            spec_a: Starting spectrogram
            spec_b: Ending spectrogram
            num_steps: Number of interpolation steps
            num_inference_steps: Number of denoising steps per generation
            
        Returns:
            Tensor of interpolated transitions
        """
        device = spec_a.device
        transitions = []
        
        for i in range(num_steps):
            # Linear interpolation between specs
            alpha = i / (num_steps - 1)
            interpolated_preceding = (1 - alpha) * spec_a + alpha * spec_b
            interpolated_following = spec_b  # Always transition to spec_b
            
            # Generate transition
            transition = self.generate(
                interpolated_preceding.unsqueeze(0),
                interpolated_following.unsqueeze(0),
                num_inference_steps=num_inference_steps
            )
            
            transitions.append(transition.squeeze(0))
        
        return torch.stack(transitions, dim=0)
    
    def save_pipeline(self, save_directory: str):
        """Save the entire pipeline."""
        self.unet.save_pretrained(f"{save_directory}/unet")
        self.scheduler.save_config(f"{save_directory}/scheduler")
    
    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        """Load a pre-trained pipeline."""
        unet = DJNetUNet.from_pretrained(f"{pretrained_path}/unet")
        scheduler = DDPMScheduler.from_config(f"{pretrained_path}/scheduler")
        return cls(unet=unet, scheduler=scheduler, **kwargs)


class DJNetLoss(nn.Module):
    """
    Custom loss function for DJ transition generation.
    
    Combines multiple loss terms for better transition quality.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        temporal_weight: float = 0.05
    ):
        """
        Initialize the loss function.
        
        Args:
            mse_weight: Weight for MSE loss
            perceptual_weight: Weight for perceptual loss
            temporal_weight: Weight for temporal consistency loss
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
    
    def forward(
        self,
        predicted_noise: torch.Tensor,
        actual_noise: torch.Tensor,
        predicted_transition: Optional[torch.Tensor] = None,
        target_transition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss.
        
        Args:
            predicted_noise: Predicted noise from the model
            actual_noise: Ground truth noise
            predicted_transition: Predicted transition (if available)
            target_transition: Target transition (if available)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # MSE loss on noise prediction
        mse_loss = nn.functional.mse_loss(predicted_noise, actual_noise)
        losses['mse_loss'] = mse_loss
        
        total_loss = self.mse_weight * mse_loss
        
        # Additional losses if transition spectrograms are provided
        if predicted_transition is not None and target_transition is not None:
            # Perceptual loss (L1 loss as proxy)
            perceptual_loss = nn.functional.l1_loss(predicted_transition, target_transition)
            losses['perceptual_loss'] = perceptual_loss
            total_loss += self.perceptual_weight * perceptual_loss
            
            # Temporal consistency loss (difference in adjacent time frames)
            if predicted_transition.shape[-1] > 1:
                pred_diff = torch.diff(predicted_transition, dim=-1)
                target_diff = torch.diff(target_transition, dim=-1)
                temporal_loss = nn.functional.mse_loss(pred_diff, target_diff)
                losses['temporal_loss'] = temporal_loss
                total_loss += self.temporal_weight * temporal_loss
        
        losses['total_loss'] = total_loss
        return losses


if __name__ == "__main__":
    # Test pipeline creation
    from .djnet_unet import create_djnet_unet
    
    # Create model and pipeline
    unet = create_djnet_unet()
    pipeline = DJNetDiffusionPipeline(unet)
    
    print("Pipeline created successfully!")
    
    # Test training step with dummy data
    batch_size = 2
    height, width = 128, 128
    
    dummy_batch = {
        'preceding_spec': torch.randn(batch_size, height, width),
        'following_spec': torch.randn(batch_size, height, width),
        'transition_spec': torch.randn(batch_size, height, width)
    }
    
    device = torch.device('cpu')
    result = pipeline.training_step(dummy_batch, device)
    print(f"Training step completed. Loss: {result['loss'].item():.4f}")
    
    # Test generation
    preceding = torch.randn(1, height, width)
    following = torch.randn(1, height, width)
    
    generated = pipeline.generate(
        preceding, 
        following, 
        num_inference_steps=10  # Fast test
    )
    print(f"Generated transition shape: {generated.shape}")
