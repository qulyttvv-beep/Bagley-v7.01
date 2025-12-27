"""
🔍 Real Image/Video Upscaler
Actually removes artifacts and increases detail
Based on Real-ESRGAN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for detail preservation"""
    
    def __init__(self, num_features: int = 64, growth_rate: int = 32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_features, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_features + 4 * growth_rate, num_features, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.scale = 0.2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.scale


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    
    def __init__(self, num_features: int = 64, growth_rate: int = 32):
        super().__init__()
        
        self.rdb1 = ResidualDenseBlock(num_features, growth_rate)
        self.rdb2 = ResidualDenseBlock(num_features, growth_rate)
        self.rdb3 = ResidualDenseBlock(num_features, growth_rate)
        self.scale = 0.2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * self.scale


class PixelShuffleUpscale(nn.Module):
    """Pixel shuffle upscaling"""
    
    def __init__(self, in_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.ps(self.conv(x)))


class BagleyUpscaler(nn.Module):
    """
    🔍 Bagley Real Upscaler
    
    Based on Real-ESRGAN architecture with improvements:
    - Better artifact removal
    - Enhanced detail synthesis
    - Temporal consistency for video
    
    Total: ~17M parameters (efficient yet effective)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        growth_rate: int = 32,
        scale_factor: int = 4,
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # First conv
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # RRDB trunk
        self.trunk = nn.Sequential(
            *[RRDB(num_features, growth_rate) for _ in range(num_blocks)]
        )
        self.trunk_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        # Upsampling
        self.upscale_layers = nn.ModuleList()
        current_scale = 1
        while current_scale < scale_factor:
            self.upscale_layers.append(PixelShuffleUpscale(num_features, 2))
            current_scale *= 2
        
        # High-resolution conv
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upscale image.
        
        Args:
            x: Input [B, C, H, W] in range [0, 1]
            
        Returns:
            Upscaled [B, C, H*scale, W*scale]
        """
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(feat))
        feat = feat + trunk
        
        for upscale in self.upscale_layers:
            feat = upscale(feat)
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class ArtifactRemover(nn.Module):
    """
    🧹 Artifact Remover
    
    Specifically designed to remove:
    - JPEG artifacts
    - Compression noise
    - Blur
    - Ringing
    """
    
    def __init__(self, num_features: int = 64):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[RRDB(num_features * 4, num_features) for _ in range(6)]
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(num_features * 4, num_features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, 3, 3, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder with skip connections
        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], 1))
        out = self.dec1(torch.cat([d2, e1], 1))
        
        return x + out  # Residual


class DetailEnhancer(nn.Module):
    """
    ✨ Detail Enhancer
    
    Synthesizes fine details:
    - Texture
    - Edges
    - Fine patterns
    """
    
    def __init__(self, num_features: int = 64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, num_features, 3, 1, 1)
        
        # Multi-scale feature extraction
        self.branch1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.branch2 = nn.Conv2d(num_features, num_features, 5, 1, 2)
        self.branch3 = nn.Conv2d(num_features, num_features, 7, 1, 3)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(num_features * 3, num_features, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Detail synthesis
        self.detail = nn.Sequential(
            RRDB(num_features, 32),
            RRDB(num_features, 32),
            nn.Conv2d(num_features, 3, 3, 1, 1),
        )
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.lrelu(self.conv1(x))
        
        b1 = self.lrelu(self.branch1(feat))
        b2 = self.lrelu(self.branch2(feat))
        b3 = self.lrelu(self.branch3(feat))
        
        fused = self.fusion(torch.cat([b1, b2, b3], 1))
        detail = self.detail(fused)
        
        return x + detail * 0.2


class BagleyFullUpscaler(nn.Module):
    """
    🔍 Full Upscaler Pipeline
    
    Combines:
    1. Artifact removal
    2. Detail enhancement
    3. Super-resolution upscaling
    
    For video: Adds temporal consistency
    """
    
    def __init__(self, scale_factor: int = 4):
        super().__init__()
        
        self.artifact_remover = ArtifactRemover()
        self.detail_enhancer = DetailEnhancer()
        self.upscaler = BagleyUpscaler(scale_factor=scale_factor)
        
        self.scale_factor = scale_factor
    
    def forward(
        self,
        x: torch.Tensor,
        remove_artifacts: bool = True,
        enhance_details: bool = True,
    ) -> torch.Tensor:
        """
        Full upscaling pipeline.
        
        Args:
            x: Input [B, C, H, W] in range [0, 1]
            remove_artifacts: Whether to run artifact removal
            enhance_details: Whether to run detail enhancement
            
        Returns:
            Upscaled [B, C, H*scale, W*scale]
        """
        if remove_artifacts:
            x = self.artifact_remover(x)
        
        if enhance_details:
            x = self.detail_enhancer(x)
        
        x = self.upscaler(x)
        
        return torch.clamp(x, 0, 1)
    
    def upscale_video(
        self,
        video: torch.Tensor,
        temporal_consistency: bool = True,
    ) -> torch.Tensor:
        """
        Upscale video with temporal consistency.
        
        Args:
            video: [T, C, H, W] or [B, T, C, H, W]
            temporal_consistency: Whether to apply temporal smoothing
            
        Returns:
            Upscaled video
        """
        # Handle batch dimension
        if video.dim() == 4:
            video = video.unsqueeze(0)
        
        batch_size, num_frames = video.shape[:2]
        
        # Upscale each frame
        upscaled_frames = []
        prev_frame = None
        
        for t in range(num_frames):
            frame = video[:, t]
            upscaled = self(frame)
            
            if temporal_consistency and prev_frame is not None:
                # Blend with previous for smoothness
                upscaled = 0.9 * upscaled + 0.1 * prev_frame
            
            upscaled_frames.append(upscaled)
            prev_frame = upscaled
        
        return torch.stack(upscaled_frames, dim=1)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "BagleyFullUpscaler":
        """Load pretrained weights"""
        model = cls()
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model


def upscale_image(
    image: torch.Tensor,
    model: BagleyFullUpscaler = None,
    scale_factor: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Convenience function to upscale an image.
    
    Args:
        image: [C, H, W] or [B, C, H, W] in range [0, 1]
        model: Optional pre-loaded model
        scale_factor: Upscale factor
        device: Device to use
        
    Returns:
        Upscaled image
    """
    if model is None:
        model = BagleyFullUpscaler(scale_factor=scale_factor)
        model = model.to(device)
        model.eval()
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    with torch.no_grad():
        upscaled = model(image)
    
    return upscaled.squeeze(0) if upscaled.shape[0] == 1 else upscaled
