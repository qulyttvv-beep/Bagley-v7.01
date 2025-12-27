"""
🔊 BagleyVocoder - Neural Vocoder for Audio Synthesis
HiFi-GAN v2 inspired architecture for high-quality waveform generation
"""

from typing import Optional, Tuple, List
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Residual block with dilated convolutions"""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d,
                     padding=(kernel_size * d - d) // 2)
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=1,
                     padding=(kernel_size - 1) // 2)
            for _ in dilations
        ])
        
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x: Tensor) -> Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = self.act(x)
            x = conv1(x)
            x = self.act(x)
            x = conv2(x)
            x = x + residual
        return x


class MRF(nn.Module):
    """Multi-receptive field fusion module"""
    
    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(channels, k, d) for k, d in zip(kernel_sizes, dilations)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        out = None
        for block in self.resblocks:
            if out is None:
                out = block(x)
            else:
                out = out + block(x)
        return out / len(self.resblocks)


class BagleyVocoder(nn.Module):
    """
    🔊 BagleyVocoder - HiFi-GAN v2 Inspired Neural Vocoder
    
    Converts acoustic tokens/mel spectrograms to high-quality audio waveforms.
    
    Features:
    - Multi-receptive field fusion
    - Multi-period discriminator compatible
    - Fast inference with optimized kernels
    """
    
    def __init__(
        self,
        in_channels: int = 256,  # From acoustic decoder
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        sample_rate: int = 44100,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.num_upsamples = len(upsample_rates)
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k - u) // 2)
            )
            ch = ch // 2
            self.mrfs.append(MRF(ch, resblock_kernel_sizes, resblock_dilations))
        
        # Final convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)
        
        self.act = nn.LeakyReLU(0.1)
        
        logger.info(f"Initialized BagleyVocoder at {sample_rate}Hz")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Convert features to waveform.
        
        Args:
            x: Input features [B, C, T]
            
        Returns:
            Waveform [B, 1, T']
        """
        x = self.conv_pre(x)
        
        for up, mrf in zip(self.ups, self.mrfs):
            x = self.act(x)
            x = up(x)
            x = mrf(x)
        
        x = self.act(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    @torch.no_grad()
    def generate(self, features: Tensor) -> Tensor:
        """Generate audio from features (inference mode)"""
        return self(features)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "BagleyVocoder":
        import os
        from safetensors.torch import load_file
        
        model = cls()
        
        weights_path = os.path.join(path, "vocoder.safetensors")
        if os.path.exists(weights_path):
            model.load_state_dict(load_file(weights_path))
        
        return model.to(device)


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator for GAN training"""
    
    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
    
    def forward(self, y: Tensor, y_hat: Tensor) -> Tuple[List[Tensor], List[Tensor], List[List[Tensor]], List[List[Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class PeriodDiscriminator(nn.Module):
    """Single period discriminator"""
    
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        fmap = []
        
        # Reshape to period
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        
        x = x.view(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = self.act(x)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap
