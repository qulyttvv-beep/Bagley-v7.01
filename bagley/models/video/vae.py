"""
🎬 BagleyVideo3DVAE - 3D VAE for Video Generation
Compresses video spatially and temporally for efficient generation
"""

from typing import Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ResnetBlock3D(nn.Module):
    """3D Residual block"""
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class TemporalDownsample(nn.Module):
    """Temporal 2x downsampling"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class SpatialDownsample(nn.Module):
    """Spatial 2x downsampling"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """Temporal 2x upsampling"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0))
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=(2, 1, 1), mode="nearest")
        return self.conv(x)


class SpatialUpsample(nn.Module):
    """Spatial 2x upsampling"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1))
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)


class Video3DEncoder(nn.Module):
    """3D VAE Encoder for video"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        block_channels: Tuple[int, ...] = (128, 256, 512, 512),
    ):
        super().__init__()
        
        self.conv_in = nn.Conv3d(in_channels, block_channels[0], 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        in_ch = block_channels[0]
        
        for i, out_ch in enumerate(block_channels):
            block = nn.ModuleList([
                ResnetBlock3D(in_ch, out_ch),
                ResnetBlock3D(out_ch, out_ch),
            ])
            in_ch = out_ch
            
            # Downsample: temporal at layer 1, spatial at all except last
            if i == 1:
                block.append(TemporalDownsample(out_ch))
            if i < len(block_channels) - 1:
                block.append(SpatialDownsample(out_ch))
            
            self.down_blocks.append(block)
        
        self.mid = nn.ModuleList([
            ResnetBlock3D(block_channels[-1], block_channels[-1]),
            ResnetBlock3D(block_channels[-1], block_channels[-1]),
        ])
        
        self.norm_out = nn.GroupNorm(32, block_channels[-1])
        self.conv_out = nn.Conv3d(block_channels[-1], latent_channels * 2, 3, padding=1)
        self.act = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv_in(x)
        
        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
        
        for layer in self.mid:
            x = layer(x)
        
        x = self.act(self.norm_out(x))
        x = self.conv_out(x)
        
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar


class Video3DDecoder(nn.Module):
    """3D VAE Decoder for video"""
    
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_channels: Tuple[int, ...] = (512, 512, 256, 128),
    ):
        super().__init__()
        
        self.conv_in = nn.Conv3d(latent_channels, block_channels[0], 3, padding=1)
        
        self.mid = nn.ModuleList([
            ResnetBlock3D(block_channels[0], block_channels[0]),
            ResnetBlock3D(block_channels[0], block_channels[0]),
        ])
        
        self.up_blocks = nn.ModuleList()
        in_ch = block_channels[0]
        
        for i, out_ch in enumerate(block_channels):
            block = nn.ModuleList([
                ResnetBlock3D(in_ch, out_ch),
                ResnetBlock3D(out_ch, out_ch),
            ])
            in_ch = out_ch
            
            # Upsample: temporal at layer 2, spatial at all except last
            if i == 2:
                block.append(TemporalUpsample(out_ch))
            if i < len(block_channels) - 1:
                block.append(SpatialUpsample(out_ch))
            
            self.up_blocks.append(block)
        
        self.norm_out = nn.GroupNorm(32, block_channels[-1])
        self.conv_out = nn.Conv3d(block_channels[-1], out_channels, 3, padding=1)
        self.act = nn.SiLU()
    
    def forward(self, z: Tensor) -> Tensor:
        x = self.conv_in(z)
        
        for layer in self.mid:
            x = layer(x)
        
        for block in self.up_blocks:
            for layer in block:
                x = layer(x)
        
        x = self.act(self.norm_out(x))
        x = self.conv_out(x)
        
        return x


class BagleyVideo3DVAE(nn.Module):
    """
    🎬 BagleyVideo3DVAE - 3D VAE for Video
    
    Features:
    - 8x spatial compression
    - 4x temporal compression
    - 16-channel latent space
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 16,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        
        self.encoder = Video3DEncoder(in_channels, latent_channels)
        self.decoder = Video3DDecoder(in_channels, latent_channels)
        
        logger.info(f"Initialized BagleyVideo3DVAE with {latent_channels} latent channels")
    
    def encode(self, x: Tensor, sample: bool = True) -> Tensor:
        """Encode video to latent space. Input: [B, C, T, H, W]"""
        mean, logvar = self.encoder(x)
        
        if sample:
            std = torch.exp(0.5 * logvar)
            z = mean + torch.randn_like(std) * std
        else:
            z = mean
        
        return z * self.scaling_factor
    
    def decode(self, z: Tensor) -> Tensor:
        """Decode latents to video"""
        z = z / self.scaling_factor
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mean + torch.randn_like(std) * std
        recon = self.decoder(z)
        return recon, mean, logvar
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "BagleyVideo3DVAE":
        import os
        from safetensors.torch import load_file
        
        model = cls()
        weights_path = os.path.join(path, "vae3d.safetensors")
        if os.path.exists(weights_path):
            model.load_state_dict(load_file(weights_path))
        return model.to(device)
