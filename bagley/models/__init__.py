"""
🤖 Bagley Models - The AI Arsenal
All the models that make Bagley the BEST AI in the world
"""

# Chat model
from bagley.models.chat import (
    BagleyChatModel,
    BagleyTokenizer,
    ChatConfig,
)

# Import upscaler
from bagley.models.upscaler import (
    BagleyUpscaler,
    BagleyFullUpscaler,
    ArtifactRemover,
    DetailEnhancer,
)

__all__ = [
    # Chat
    "BagleyChatModel",
    "BagleyTokenizer",
    "ChatConfig",
    # Upscaler
    "BagleyUpscaler",
    "BagleyFullUpscaler",
    "ArtifactRemover",
    "DetailEnhancer",
]

# Lazy imports for heavy modules
def get_video_generator():
    """Lazy load video generator to avoid heavy imports"""
    from bagley.models.video.infinite_video import InfiniteVideoGenerator
    return InfiniteVideoGenerator

def get_video_stitcher():
    """Lazy load video stitcher"""
    from bagley.models.video.infinite_video import VideoStitcher
    return VideoStitcher
