"""
� Bagley Core Module v7.01 "Genesis"
======================================

The brain of Bagley - Advanced AI Assistant
Inspired by Watch Dogs: Legion

SUBSYSTEMS:
- Bagley: Main unified class
- ReasoningEngine: Tree-of-thought, self-consistency, meta-cognition
- MemorySystem: Long-term memory with forgetting curves
- PersonalityEngine: Big Five traits, communication styles
- EmotionSystem: Plutchik's wheel, PAD model, emotional contagion
- AntiHallucination: Self-consistency, fact verification, confidence calibration
"""

__version__ = "7.01"
__codename__ = "Genesis"

# === NEW UNIFIED SYSTEM ===
try:
    from bagley.core.bagley import (
        Bagley,
        BagleyConfig,
        BagleyResponse as UnifiedBagleyResponse,
        create_bagley,
    )
    UNIFIED_AVAILABLE = True
except ImportError:
    UNIFIED_AVAILABLE = False
    Bagley = None
    BagleyConfig = None
    create_bagley = None
    UnifiedBagleyResponse = None

# === REASONING ENGINE ===
try:
    from bagley.core.reasoning_engine import (
        AdvancedReasoningEngine,
        ReasoningStrategy,
        ReasoningResult,
        ReasoningStep,
        ThoughtNode,
        TreeOfThought,
        SelfReflection,
        MetaCognition,
        create_reasoning_engine,
    )
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False
    AdvancedReasoningEngine = None
    ReasoningStrategy = None
    ReasoningResult = None
    create_reasoning_engine = None

# === LONG-TERM MEMORY ===
try:
    from bagley.core.long_term_memory import (
        MemorySystem,
        LongTermMemory,
        WorkingMemory,
        Memory,
        MemoryType,
        MemoryStrength,
        create_memory_system,
    )
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    MEMORY_SYSTEM_AVAILABLE = False
    MemorySystem = None
    LongTermMemory = None
    create_memory_system = None

# === PERSONALITY ENGINE ===
try:
    from bagley.core.personality_engine import (
        BagleyPersonality as NewBagleyPersonality,
        PersonalityTrait,
        CommunicationStyle,
        ContextType,
        PersonalityProfile,
        create_bagley_personality,
    )
    PERSONALITY_ENGINE_AVAILABLE = True
except ImportError:
    PERSONALITY_ENGINE_AVAILABLE = False
    NewBagleyPersonality = None
    create_bagley_personality = None

# === EMOTION SYSTEM ===
try:
    from bagley.core.emotion_system import (
        BagleyEmotionSystem,
        EmotionState,
        EmotionDetector,
        SituationAppraiser,
        PrimaryEmotion,
        ComplexEmotion,
        EmotionalMemory,
        create_emotion_system,
    )
    EMOTION_AVAILABLE = True
except ImportError:
    EMOTION_AVAILABLE = False
    BagleyEmotionSystem = None
    EmotionState = None
    create_emotion_system = None

# === ANTI-HALLUCINATION ===
try:
    from bagley.core.anti_hallucination import (
        AntiHallucinationSystem,
        GroundedResponse,
        VerificationResult,
        ConfidenceLevel,
        SelfConsistencyChecker,
        FactVerifier,
        ChainOfThoughtVerifier,
        create_anti_hallucination_system,
    )
    ANTI_HALLUCINATION_AVAILABLE = True
except ImportError:
    ANTI_HALLUCINATION_AVAILABLE = False
    AntiHallucinationSystem = None
    ConfidenceLevel = None
    create_anti_hallucination_system = None

# === LEGACY SYSTEMS (backwards compatibility) ===
try:
    from bagley.core.orchestrator import BagleyOrchestrator
    from bagley.core.memory import BagleyMemory
    from bagley.core.personality import BagleyPersonality
    from bagley.core.router import IntentRouter
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False
    BagleyOrchestrator = None
    BagleyMemory = None
    BagleyPersonality = None
    IntentRouter = None

try:
    from bagley.core.brain import (
        UnifiedBrain,
        TaskRouter,
        TaskType,
        BagleyRequest,
        BagleyResponse,
        BagleyAdvantages,
    )
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    UnifiedBrain = None
    TaskRouter = None

# === LAZY LOADERS ===
def get_infinite_context():
    """Lazy load infinite context (requires torch)"""
    from bagley.core.infinite_context import (
        StreamingKVCache,
        InfiniteContextAttention,
        InfiniteContextProcessor,
        ContextMemoryBank,
    )
    return StreamingKVCache, InfiniteContextAttention, InfiniteContextProcessor, ContextMemoryBank

# === EXPORTS ===
__all__ = [
    # Version
    "__version__",
    "__codename__",
    
    # NEW: Unified Bagley
    "Bagley",
    "BagleyConfig",
    "create_bagley",
    
    # NEW: Reasoning Engine
    "AdvancedReasoningEngine",
    "ReasoningStrategy",
    "ReasoningResult",
    "TreeOfThought",
    "SelfReflection",
    "MetaCognition",
    "create_reasoning_engine",
    
    # NEW: Memory System  
    "MemorySystem",
    "LongTermMemory",
    "WorkingMemory",
    "Memory",
    "MemoryType",
    "MemoryStrength",
    "create_memory_system",
    
    # NEW: Personality Engine
    "NewBagleyPersonality",
    "PersonalityTrait",
    "CommunicationStyle",
    "ContextType",
    "create_bagley_personality",
    
    # Emotion System
    "BagleyEmotionSystem",
    "EmotionState",
    "EmotionDetector",
    "SituationAppraiser",
    "PrimaryEmotion",
    "ComplexEmotion",
    "EmotionalMemory",
    "create_emotion_system",
    
    # Anti-Hallucination
    "AntiHallucinationSystem",
    "ConfidenceLevel",
    "VerificationResult",
    "SelfConsistencyChecker",
    "FactVerifier",
    "ChainOfThoughtVerifier",
    "create_anti_hallucination_system",
    
    # Legacy (backwards compatibility)
    "BagleyOrchestrator",
    "BagleyMemory",
    "BagleyPersonality",
    "IntentRouter",
    "UnifiedBrain",
    "TaskRouter",
    "TaskType",
    "BagleyRequest",
    "BagleyResponse",
    "BagleyAdvantages",
    
    # Lazy loaders
    "get_infinite_context",
    
    # Availability flags
    "UNIFIED_AVAILABLE",
    "REASONING_AVAILABLE",
    "MEMORY_SYSTEM_AVAILABLE",
    "PERSONALITY_ENGINE_AVAILABLE",
    "EMOTION_AVAILABLE",
    "ANTI_HALLUCINATION_AVAILABLE",
]
