"""
🎬 BagleyVideo - Custom Video Generation Model
Wan2.2/Mochi inspired Asymmetric Diffusion Transformer
"""

from bagley.models.video.config import BagleyVideoConfig
from bagley.models.video.model import BagleyVideoMoE
from bagley.models.video.vae import BagleyVideo3DVAE
from bagley.models.video.pipeline import BagleyVideoPipeline
from bagley.models.video.infinite_video import (
    InfiniteVideoGenerator,
    VideoStitcher,
    VideoSegment,
)

__all__ = [
    "BagleyVideoConfig",
    "BagleyVideoMoE",
    "BagleyVideo3DVAE",
    "BagleyVideoPipeline",
    # Infinite video
    "InfiniteVideoGenerator",
    "VideoStitcher",
    "VideoSegment",
]
