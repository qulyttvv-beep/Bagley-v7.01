"""
📁 Multimodal File Handling System
Processes all file types: images, audio, video, documents, code
"""

from bagley.models.multimodal.processor import MultimodalProcessor
from bagley.models.multimodal.encoder import MultimodalEncoder

__all__ = [
    "MultimodalProcessor",
    "MultimodalEncoder",
]
