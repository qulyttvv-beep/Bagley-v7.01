"""
🗣️ BagleyVoice - Custom TTS/Voice System
Fish Speech/Chatterbox inspired DualAR architecture
"""

from bagley.models.tts.config import BagleyVoiceConfig
from bagley.models.tts.model import BagleyVoice
from bagley.models.tts.vocoder import BagleyVocoder
from bagley.models.tts.pipeline import BagleyTTSPipeline

__all__ = [
    "BagleyVoiceConfig",
    "BagleyVoice",
    "BagleyVocoder",
    "BagleyTTSPipeline",
]
