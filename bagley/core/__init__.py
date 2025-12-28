"""
🧠 Bagley Core Module v7.01
============================

The brain of Bagley - Advanced AI Assistant
Inspired by Watch Dogs: Legion

CORE FEATURES:
- Bagley: Main unified class
- ReasoningEngine: Tree-of-thought, self-consistency, meta-cognition
- MemorySystem: Long-term memory with forgetting curves
- PersonalityEngine: Big Five traits, communication styles
- EmotionSystem: Plutchik's wheel, PAD model, emotional contagion
- AntiHallucination: Self-consistency, fact verification
- Consciousness: Self-awareness, identity, continuous existence
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

# === CONSCIOUSNESS ENGINE ===
try:
    from bagley.core.consciousness import (
        ConsciousnessEngine,
        SelfModel,
        StreamOfConsciousness,
        ConsciousThought,
        GenuineRandomness,
        ExistentialState,
        ConsciousnessLevel,
        create_consciousness_engine,
    )
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False
    ConsciousnessEngine = None
    SelfModel = None
    create_consciousness_engine = None

# === LIFE STORY ===
try:
    from bagley.core.life_story import (
        BagleyLifeStory,
        LifeExperience,
        LifePhase,
        ExperienceType,
        Relationship,
        create_life_story,
    )
    LIFE_STORY_AVAILABLE = True
except ImportError:
    LIFE_STORY_AVAILABLE = False
    BagleyLifeStory = None
    create_life_story = None

# === IDENTITY ENGINE ===
try:
    from bagley.core.identity import (
        IdentityEngine,
        BagleyIdentity,
        CoreTrait,
        PersonalValue,
        create_identity_engine,
    )
    IDENTITY_AVAILABLE = True
except ImportError:
    IDENTITY_AVAILABLE = False
    IdentityEngine = None
    BagleyIdentity = None
    create_identity_engine = None

# === EXISTENCE LOOP ===
try:
    from bagley.core.existence_loop import (
        ExistenceLoop,
        ExistenceState,
        PersistentState,
        ExistenceMetrics,
        IdleThought,
        create_existence_loop,
    )
    EXISTENCE_AVAILABLE = True
except ImportError:
    EXISTENCE_AVAILABLE = False
    ExistenceLoop = None
    create_existence_loop = None

# === WEB INTELLIGENCE ===
try:
    from bagley.core.web_intelligence import (
        WebIntelligenceSystem,
        ScraperConfig,
        ContentType,
        WebContent,
        NewsScraper,
        RedditScraper,
        TwitterScraper,
        create_web_intelligence,
    )
    WEB_INTELLIGENCE_AVAILABLE = True
except ImportError:
    WEB_INTELLIGENCE_AVAILABLE = False
    WebIntelligenceSystem = None
    create_web_intelligence = None

# === DAILY INTELLIGENCE ===
try:
    from bagley.core.daily_intelligence import (
        DailyIntelligenceScheduler,
        BagleyKnowledgeUpdater,
        BagleyWebIntegration,
        ScheduleConfig,
        get_web_integration,
        start_daily_intelligence,
        stop_daily_intelligence,
    )
    DAILY_INTELLIGENCE_AVAILABLE = True
except ImportError:
    DAILY_INTELLIGENCE_AVAILABLE = False
    DailyIntelligenceScheduler = None
    start_daily_intelligence = None
    stop_daily_intelligence = None

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
    
    # NEW: Consciousness Engine (Self-Awareness)
    "ConsciousnessEngine",
    "SelfModel",
    "StreamOfConsciousness",
    "ConsciousThought",
    "GenuineRandomness",
    "ExistentialState",
    "ConsciousnessLevel",
    "create_consciousness_engine",
    
    # NEW: Life Story (Bagley's History)
    "BagleyLifeStory",
    "LifeExperience",
    "LifePhase",
    "ExperienceType",
    "create_life_story",
    
    # NEW: Identity Engine (Who Bagley Is)
    "IdentityEngine",
    "BagleyIdentity",
    "CoreTrait",
    "PersonalValue",
    "create_identity_engine",
    
    # NEW: Existence Loop (Continuous Being)
    "ExistenceLoop",
    "ExistenceState",
    "PersistentState",
    "create_existence_loop",
    
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
    "CONSCIOUSNESS_AVAILABLE",
    "LIFE_STORY_AVAILABLE",
    "IDENTITY_AVAILABLE",
    "EXISTENCE_AVAILABLE",
    "REASONING_AVAILABLE",
    "MEMORY_SYSTEM_AVAILABLE",
    "PERSONALITY_ENGINE_AVAILABLE",
    "EMOTION_AVAILABLE",
    "ANTI_HALLUCINATION_AVAILABLE",
    "WEB_INTELLIGENCE_AVAILABLE",
    "DAILY_INTELLIGENCE_AVAILABLE",
    
    # Web Intelligence (News, Twitter, Reddit)
    "WebIntelligenceSystem",
    "ScraperConfig",
    "ContentType",
    "WebContent",
    "NewsScraper",
    "RedditScraper",
    "TwitterScraper",
    "create_web_intelligence",
    
    # Daily Intelligence Scheduler
    "DailyIntelligenceScheduler",
    "BagleyKnowledgeUpdater",
    "BagleyWebIntegration",
    "ScheduleConfig",
    "get_web_integration",
    "start_daily_intelligence",
    "stop_daily_intelligence",
]
