"""
🧠 Bagley Core - Central Orchestration System
The brain that coordinates all Bagley subsystems
"""

from bagley.core.orchestrator import BagleyOrchestrator
from bagley.core.memory import BagleyMemory
from bagley.core.personality import BagleyPersonality
from bagley.core.router import IntentRouter
from bagley.core.brain import (
    UnifiedBrain,
    TaskRouter,
    TaskType,
    BagleyRequest,
    BagleyResponse,
    BagleyAdvantages,
)

# Lazy imports for torch-dependent modules
def get_infinite_context():
    """Lazy load infinite context (requires torch)"""
    from bagley.core.infinite_context import (
        StreamingKVCache,
        InfiniteContextAttention,
        InfiniteContextProcessor,
        ContextMemoryBank,
    )
    return StreamingKVCache, InfiniteContextAttention, InfiniteContextProcessor, ContextMemoryBank

__all__ = [
    # Orchestration
    "BagleyOrchestrator",
    "BagleyMemory", 
    "BagleyPersonality",
    "IntentRouter",
    # Unified Brain
    "UnifiedBrain",
    "TaskRouter",
    "TaskType",
    "BagleyRequest",
    "BagleyResponse",
    "BagleyAdvantages",
    # Lazy loaders
    "get_infinite_context",
]
