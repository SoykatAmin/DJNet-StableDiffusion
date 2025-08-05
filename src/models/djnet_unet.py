import torch
import torch.nn as nn
import numpy as np
from diffusers import UNet2DConditionModel
from typing import Optional, Tuple, Union


class DJNetUNet(nn.Module):
    """
    Modified UNet2DConditionModel for DJ transition generation.
    
    This class adapts Stable Diffusion's UNet to work with 3-channel spectrogram input:
    - Channel 1: Preceding spectrogram
    - Channel 2: Following spectrogram  
    - Channel 3: Noisy transition spectrogram
    """
    
    def __init__(
        self,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        in_channels: int = 3,
        out_channels: int = 1,
        freeze_encoder: bool = False,
        **kwargs
    ):
        """
        Initialize DJNet UNet with pre-trained weights.
        
        Args:
            pretrained_model_name: Hugging Face model name for pre-trained weights
            in_channels: Number of input channels (3 for our case)
            out_channels: Number of output channels (1 for transition spec)
            freeze_encoder: Whether to freeze encoder layers during fine-tuning
        """
        super().__init__()
        
        # Load pre-trained UNet from Stable Diffusion
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            **kwargs
        )
        
        # Store original configuration
        self.original_in_channels = self.unet.config.in_channels
        self.target_in_channels = in_channels
        self.target_out_channels = out_channels
        
        # Modify input layer to accept 3 channels
        self._modify_input_layer()
        
        # Modify output layer if needed
        if out_channels != self.unet.config.out_channels:
            self._modify_output_layer()
        
        # Optionally freeze encoder layers
        if freeze_encoder:
            self._freeze_encoder()
    
    def _modify_input_layer(self):
        """
        Replace the first convolutional layer to accept 3 channels instead of 4.
        Initializes new layer weights using the first 3 channels of original weights.
        """
        original_conv = self.unet.conv_in
        
        # Create new conv layer with 3 input channels
        new_conv = nn.Conv2d(
            in_channels=self.target_in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights: copy first 3 channels from original weights
        with torch.no_grad():
            if self.target_in_channels <= self.original_in_channels:
                # Take first 3 channels from original 4-channel weights
                new_conv.weight.copy_(
                    original_conv.weight[:, :self.target_in_channels, :, :]
                )
            else:
                # If we need more channels, repeat the pattern
                original_weight = original_conv.weight
                new_weight = new_conv.weight
                
                for i in range(self.target_in_channels):
                    source_channel = i % self.original_in_channels
                    new_weight[:, i:i+1, :, :] = original_weight[:, source_channel:source_channel+1, :, :]
            
            # Copy bias if it exists
            if original_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)
        
        # Replace the layer
        self.unet.conv_in = new_conv
        
        print(f"Modified input layer: {self.original_in_channels} -> {self.target_in_channels} channels")
    
    def _modify_output_layer(self):
        """
        Modify the output layer if we need different number of output channels.
        """
        original_conv = self.unet.conv_out
        
        new_conv = nn.Conv2d(
            in_channels=original_conv.in_channels,
            out_channels=self.target_out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize with truncated or repeated weights
        with torch.no_grad():
            original_out_channels = original_conv.out_channels
            if self.target_out_channels <= original_out_channels:
                new_conv.weight.copy_(
                    original_conv.weight[:self.target_out_channels, :, :, :]
                )
                if original_conv.bias is not None:
                    new_conv.bias.copy_(original_conv.bias[:self.target_out_channels])
            else:
                # Repeat pattern if we need more output channels
                for i in range(self.target_out_channels):
                    source_channel = i % original_out_channels
                    new_conv.weight[i:i+1, :, :, :] = original_conv.weight[source_channel:source_channel+1, :, :, :]
                    if original_conv.bias is not None:
                        new_conv.bias[i] = original_conv.bias[source_channel]
        
        self.unet.conv_out = new_conv
        print(f"Modified output layer: {original_conv.out_channels} -> {self.target_out_channels} channels")
    
    def _freeze_encoder(self):
        """
        Freeze encoder layers to preserve pre-trained features.
        Only the decoder will be fine-tuned.
        """
        # Freeze down blocks (encoder)
        for param in self.unet.down_blocks.parameters():
            param.requires_grad = False
        
        # Freeze middle block
        for param in self.unet.mid_block.parameters():
            param.requires_grad = False
        
        print("Encoder layers frozen. Only decoder will be fine-tuned.")
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Forward pass through the modified UNet.
        
        Args:
            sample: Input tensor of shape (batch_size, 3, height, width)
                   [preceding_spec, following_spec, noisy_transition_spec]
            timestep: Diffusion timestep
            encoder_hidden_states: Optional conditioning (not used in our case)
            class_labels: Optional class labels
            return_dict: Whether to return dict or tuple
        
        Returns:
            Predicted noise to be removed from the transition spectrogram
        """
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            return_dict=return_dict,
            **kwargs
        )
    
    def get_trainable_parameters(self):
        """Get only the trainable parameters for optimization."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def save_pretrained(self, save_directory: str):
        """Save the modified model."""
        self.unet.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, **kwargs):
        """Load a pre-trained DJNet model."""
        instance = cls(**kwargs)
        instance.unet = UNet2DConditionModel.from_pretrained(pretrained_model_path)
        return instance


def create_djnet_unet(
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
    freeze_encoder: bool = False,
    **kwargs
) -> DJNetUNet:
    """
    Factory function to create a DJNet UNet model.
    
    Args:
        pretrained_model_name: Name of the pre-trained Stable Diffusion model
        freeze_encoder: Whether to freeze encoder during fine-tuning
        **kwargs: Additional arguments for UNet configuration
    
    Returns:
        Configured DJNet UNet model
    """
    return DJNetUNet(
        pretrained_model_name=pretrained_model_name,
        freeze_encoder=freeze_encoder,
        **kwargs
    )


if __name__ == "__main__":
    # Test model creation
    model = create_djnet_unet()
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 2
    height, width = 128, 128
    
    # Create dummy input: [preceding_spec, following_spec, noisy_transition_spec]
    dummy_input = torch.randn(batch_size, 3, height, width)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(dummy_input, timesteps)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.sample.shape}")
