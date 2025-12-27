"""
🔥 BAGLEY v7 - Ultimate All-in-One Local AI System
Main package initialization
"""

__version__ = "7.0.0"
__author__ = "Bagley Engineering Team"
__description__ = "Your chaotic, hilarious, infinitely capable AI assistant"

from bagley.core.orchestrator import BagleyOrchestrator
from bagley.core.personality import BagleyPersonality

__all__ = [
    "BagleyOrchestrator",
    "BagleyPersonality",
    "__version__",
]
