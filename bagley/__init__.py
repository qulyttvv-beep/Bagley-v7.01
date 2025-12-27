"""
🤖 BAGLEY v7.01 "Genesis" - Ultimate AI Assistant
==================================================

Inspired by Watch Dogs: Legion
Built with love and Python

FEATURES:
- 🧠 Advanced Reasoning (Tree-of-Thought, Self-Reflection)
- 🔮 Long-term Memory with Forgetting Curves
- 🎭 Dynamic Personality (Big Five, Communication Styles)
- 💖 Human-like Emotions (Plutchik's Wheel, PAD Model)
- 🛡️ Anti-Hallucination (Self-Consistency, Fact Verification)
- ♾️ Infinite Context Window
- 🎨 Image Generation (Stable Diffusion)
- 🎬 Video Generation (CogVideo)
- 🔊 Text-to-Speech (Bark)

USAGE:
    from bagley import Bagley, create_bagley
    
    # Quick start
    bagley = create_bagley()
    response = bagley.chat("Hello Bagley!")
    print(response.content)
    
    # Or with config
    from bagley import BagleyConfig
    config = BagleyConfig(
        enable_reasoning=True,
        enable_memory=True,
        enable_emotion=True,
    )
    bagley = Bagley(config)
"""

__version__ = "7.01"
__codename__ = "Genesis"
__author__ = "Bagley Engineering Team"
__description__ = "The chaotic, brilliant, infinitely capable AI assistant"

# Import availability flags first
from bagley.core import (
    UNIFIED_AVAILABLE,
    REASONING_AVAILABLE,
    MEMORY_SYSTEM_AVAILABLE,
    EMOTION_AVAILABLE,
    ANTI_HALLUCINATION_AVAILABLE,
)

# Main unified class
if UNIFIED_AVAILABLE:
    from bagley.core import (
        Bagley,
        BagleyConfig,
        create_bagley,
    )
else:
    Bagley = None
    BagleyConfig = None
    create_bagley = None

# Reasoning
if REASONING_AVAILABLE:
    from bagley.core import (
        AdvancedReasoningEngine,
        ReasoningStrategy,
        create_reasoning_engine,
    )

# Memory
if MEMORY_SYSTEM_AVAILABLE:
    from bagley.core import (
        MemorySystem,
        MemoryType,
        create_memory_system,
    )

# Emotion
if EMOTION_AVAILABLE:
    from bagley.core import (
        BagleyEmotionSystem,
        EmotionState,
        PrimaryEmotion,
        create_emotion_system,
    )

# Anti-Hallucination
if ANTI_HALLUCINATION_AVAILABLE:
    from bagley.core import (
        AntiHallucinationSystem,
        ConfidenceLevel,
        create_anti_hallucination_system,
    )

# Legacy imports for backwards compatibility
try:
    from bagley.core.orchestrator import BagleyOrchestrator
    from bagley.core.personality import BagleyPersonality
except ImportError:
    BagleyOrchestrator = None
    BagleyPersonality = None

__all__ = [
    # Version
    "__version__",
    "__codename__",
    
    # Main
    "Bagley",
    "BagleyConfig",
    "create_bagley",
    
    # Subsystems
    "AdvancedReasoningEngine",
    "ReasoningStrategy",
    "create_reasoning_engine",
    "MemorySystem",
    "MemoryType",
    "create_memory_system",
    "BagleyEmotionSystem",
    "EmotionState",
    "PrimaryEmotion",
    "create_emotion_system",
    "AntiHallucinationSystem",
    "ConfidenceLevel",
    "create_anti_hallucination_system",
    
    # Legacy
    "BagleyOrchestrator",
    "BagleyPersonality",
]
