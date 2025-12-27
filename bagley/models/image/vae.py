"""
🎨 BagleyVAE - Variational Autoencoder for Image Generation
Custom VAE for encoding/decoding images to latent space
"""

import math
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ResnetBlock(nn.Module):
    """Residual block with group normalization"""
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class AttentionBlock(nn.Module):
    """Self-attention block for VAE"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.view(batch, channels, -1).transpose(1, 2)  # [B, H*W, C]
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2).view(batch, channels, height, width)
        
        return x + residual


class Downsample(nn.Module):
    """2x downsampling"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x upsampling"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class VAEEncoder(nn.Module):
    """VAE Encoder - maps images to latent distributions"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        block_channels: Tuple[int, ...] = (128, 256, 512, 512),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, block_channels[0], 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = block_channels[0]
        
        for i, out_ch in enumerate(block_channels):
            block = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            
            if i == len(block_channels) - 1:
                block.append(AttentionBlock(out_ch))
            
            if i < len(block_channels) - 1:
                block.append(Downsample(out_ch))
            
            self.down_blocks.append(block)
        
        # Middle
        self.mid_block = nn.ModuleList([
            ResnetBlock(block_channels[-1], block_channels[-1]),
            AttentionBlock(block_channels[-1]),
            ResnetBlock(block_channels[-1], block_channels[-1]),
        ])
        
        # Output
        self.norm_out = nn.GroupNorm(32, block_channels[-1])
        self.conv_out = nn.Conv2d(block_channels[-1], latent_channels * 2, 3, padding=1)  # Mean and logvar
        self.act = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv_in(x)
        
        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
        
        for layer in self.mid_block:
            x = layer(x)
        
        x = self.act(self.norm_out(x))
        x = self.conv_out(x)
        
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar


class VAEDecoder(nn.Module):
    """VAE Decoder - maps latents back to images"""
    
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_channels: Tuple[int, ...] = (512, 512, 256, 128),
        num_res_blocks: int = 2,
    ):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(latent_channels, block_channels[0], 3, padding=1)
        
        # Middle
        self.mid_block = nn.ModuleList([
            ResnetBlock(block_channels[0], block_channels[0]),
            AttentionBlock(block_channels[0]),
            ResnetBlock(block_channels[0], block_channels[0]),
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_ch = block_channels[0]
        
        for i, out_ch in enumerate(block_channels):
            block = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch))
                in_ch = out_ch
            
            if i == 0:
                block.append(AttentionBlock(out_ch))
            
            if i < len(block_channels) - 1:
                block.append(Upsample(out_ch))
            
            self.up_blocks.append(block)
        
        # Output
        self.norm_out = nn.GroupNorm(32, block_channels[-1])
        self.conv_out = nn.Conv2d(block_channels[-1], out_channels, 3, padding=1)
        self.act = nn.SiLU()
    
    def forward(self, z: Tensor) -> Tensor:
        x = self.conv_in(z)
        
        for layer in self.mid_block:
            x = layer(x)
        
        for block in self.up_blocks:
            for layer in block:
                x = layer(x)
        
        x = self.act(self.norm_out(x))
        x = self.conv_out(x)
        
        return x


class BagleyVAE(nn.Module):
    """
    🎨 BagleyVAE - Custom Variational Autoencoder
    
    Features:
    - High compression ratio (8x downsampling)
    - 16-channel latent space for quality
    - Attention in bottleneck for global coherence
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
        )
        
        self.decoder = VAEDecoder(
            out_channels=in_channels,
            latent_channels=latent_channels,
        )
        
        logger.info(f"Initialized BagleyVAE with {latent_channels} latent channels")
    
    def encode(self, x: Tensor, sample: bool = True) -> Tensor:
        """Encode images to latent space"""
        mean, logvar = self.encoder(x)
        
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        else:
            z = mean
        
        return z * self.scaling_factor
    
    def decode(self, z: Tensor) -> Tensor:
        """Decode latents to images"""
        z = z / self.scaling_factor
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Full forward pass with reconstruction"""
        mean, logvar = self.encoder(x)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        recon = self.decoder(z)
        
        return recon, mean, logvar
    
    def loss(
        self, 
        x: Tensor, 
        recon: Tensor, 
        mean: Tensor, 
        logvar: Tensor,
        kl_weight: float = 1e-6,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute VAE loss"""
        # Reconstruction loss (L1 + perceptual would be added in training)
        recon_loss = F.l1_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "BagleyVAE":
        """Load from pretrained weights"""
        import os
        from safetensors.torch import load_file
        
        model = cls()
        
        weights_path = os.path.join(path, "vae.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict)
        
        return model.to(device)
    
    def save_pretrained(self, path: str):
        """Save to disk"""
        import os
        from safetensors.torch import save_file
        
        os.makedirs(path, exist_ok=True)
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        save_file(state_dict, os.path.join(path, "vae.safetensors"))
