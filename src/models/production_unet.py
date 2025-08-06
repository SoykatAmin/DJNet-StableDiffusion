"""
Production U-Net model for DJ transition generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProductionUNet(nn.Module):
    """
    Production U-Net model for generating DJ transitions
    """
    
    def __init__(self, in_channels=3, out_channels=1, model_dim=512):
        super().__init__()
        
        # Encoder path
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128) 
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, model_dim, 3, padding=1),
            nn.BatchNorm2d(model_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_dim, model_dim, 3, padding=1),
            nn.BatchNorm2d(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(model_dim, 512, 2, stride=2)
        self.dec4 = self._make_decoder_block(512 + 256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._make_decoder_block(256 + 128, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._make_decoder_block(128 + 64, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._make_decoder_block(64, 64)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
            nn.Tanh()
        )
        
    def _make_encoder_block(self, in_channels, out_channels):
        """Create encoder block with convolution, normalization, and pooling"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create decoder block with convolution, normalization, and dropout"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        """Forward pass through the U-Net"""
        # Encoder path with skip connections
        e1 = self.enc1(x)      # [B, 64, H/2, W/2]
        e2 = self.enc2(e1)     # [B, 128, H/4, W/4]
        e3 = self.enc3(e2)     # [B, 256, H/8, W/8]
        e4 = self.enc4(e3)     # [B, 512, H/16, W/16]
        
        # Bottleneck
        bottleneck = self.bottleneck(e4)  # [B, model_dim, H/16, W/16]
        
        # Decoder path with skip connections
        d4 = self.upconv4(bottleneck)     # [B, 512, H/8, W/8]
        d4 = torch.cat([d4, e3], dim=1)   # [B, 512+256, H/8, W/8]
        d4 = self.dec4(d4)                # [B, 512, H/8, W/8]
        
        d3 = self.upconv3(d4)             # [B, 256, H/4, W/4]
        d3 = torch.cat([d3, e2], dim=1)   # [B, 256+128, H/4, W/4]
        d3 = self.dec3(d3)                # [B, 256, H/4, W/4]
        
        d2 = self.upconv2(d3)             # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, e1], dim=1)   # [B, 128+64, H/2, W/2]
        d2 = self.dec2(d2)                # [B, 128, H/2, W/2]
        
        d1 = self.upconv1(d2)             # [B, 64, H, W]
        d1 = self.dec1(d1)                # [B, 64, H, W]
        
        # Final output
        output = self.final(d1)           # [B, out_channels, H, W]
        
        return output
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
