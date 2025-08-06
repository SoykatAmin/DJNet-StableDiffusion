"""
DJNet Diffusion Pipeline for generating DJ transitions
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List
from diffusers import DDPMScheduler


class DJNetDiffusionPipeline:
    """
    Pipeline for generating DJ transitions using DJNet UNet model
    """
    
    def __init__(
        self,
        unet,
        scheduler_config: Optional[dict] = None
    ):
        """
        Initialize the DJNet diffusion pipeline
        
        Args:
            unet: The DJNet UNet model
            scheduler_config: Configuration for the noise scheduler
        """
        self.unet = unet
        
        # Initialize scheduler
        if scheduler_config is None:
            scheduler_config = {
                'num_train_timesteps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'linear'
            }
        
        self.scheduler = DDPMScheduler(**scheduler_config)
        
    def __call__(
        self,
        preceding_spectrogram: torch.Tensor,
        following_spectrogram: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate a transition spectrogram between two input spectrograms
        
        Args:
            preceding_spectrogram: Spectrogram of the preceding track [B, 1, H, W]
            following_spectrogram: Spectrogram of the following track [B, 1, H, W]
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            generator: Random generator for reproducibility
            
        Returns:
            Generated transition spectrogram [B, 1, H, W]
        """
        device = preceding_spectrogram.device
        batch_size = preceding_spectrogram.shape[0]
        
        # Set inference timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Initialize random noise
        shape = (batch_size, 1, *preceding_spectrogram.shape[2:])
        latents = torch.randn(shape, generator=generator, device=device)
        
        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        # Prepare condition input by concatenating spectrograms
        condition = torch.cat([preceding_spectrogram, following_spectrogram], dim=1)
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # Expand the latents if we are doing classifier-free guidance
            latent_model_input = latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Concatenate condition with noisy latents
            model_input = torch.cat([latent_model_input, condition], dim=1)
            
            # Predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    model_input,
                    t,
                    return_dict=False
                )[0]
            
            # Compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        return latents
    
    def to(self, device):
        """Move pipeline to device"""
        self.unet = self.unet.to(device)
        return self
    
    def eval(self):
        """Set pipeline to evaluation mode"""
        self.unet.eval()
        return self
    
    def train(self):
        """Set pipeline to training mode"""
        self.unet.train()
        return self
