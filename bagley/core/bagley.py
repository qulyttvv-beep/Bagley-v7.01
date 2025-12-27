"""
🧠 Bagley Core - The Brain
===========================

This is the main Bagley class that integrates all systems.
The unified AI that thinks, feels, remembers, and helps.

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────┐
│                        BAGLEY v7.01                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Reasoning  │  │  Anti-Hall  │  │   Emotion   │        │
│  │   Engine    │  │   System    │  │   System    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│           │              │               │                 │
│           └──────────────┼───────────────┘                 │
│                          ▼                                 │
│                   ┌─────────────┐                          │
│                   │ Personality │                          │
│                   │   Engine    │                          │
│                   └─────────────┘                          │
│                          │                                 │
│           ┌──────────────┼──────────────┐                  │
│           ▼              ▼              ▼                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Memory    │  │    Chat     │  │   Safety    │        │
│  │   System    │  │   Model     │  │   Layer     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio

# Import all subsystems
from .reasoning_engine import (
    AdvancedReasoningEngine,
    ReasoningStrategy,
    ReasoningResult,
    create_reasoning_engine,
)
from .long_term_memory import (
    MemorySystem,
    MemoryType,
    Memory,
    create_memory_system,
)
from .personality_engine import (
    BagleyPersonality,
    ContextType,
    CommunicationStyle,
    create_bagley_personality,
)
from .emotion_system import (
    BagleyEmotionSystem,
    EmotionState,
    create_emotion_system,
)
from .anti_hallucination import (
    AntiHallucinationSystem,
    ConfidenceLevel,
    create_anti_hallucination_system,
)

logger = logging.getLogger(__name__)


@dataclass
class BagleyConfig:
    """Configuration for Bagley"""
    # Model settings
    model_path: Optional[str] = None
    use_gpu: bool = True
    max_tokens: int = 2048
    temperature: float = 0.7
    
    # Feature toggles
    enable_reasoning: bool = True
    enable_memory: bool = True
    enable_emotion: bool = True
    enable_personality: bool = True
    enable_anti_hallucination: bool = True
    
    # Reasoning settings
    max_thinking_time_ms: int = 30000
    enable_reflection: bool = True
    
    # Memory settings
    max_memories: int = 10000
    memory_consolidation_threshold: int = 100
    
    # Personality settings
    personality_strength: float = 1.0
    adaptability: float = 0.3
    
    # Safety settings
    hallucination_threshold: float = 0.7
    require_verification: bool = True


@dataclass
class BagleyResponse:
    """A response from Bagley"""
    content: str
    confidence: float
    confidence_level: str
    emotion_state: Dict[str, float]
    reasoning_used: Optional[str] = None
    memories_recalled: int = 0
    thinking_time_ms: float = 0
    personality_style: str = "friendly"
    verified: bool = True
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Bagley:
    """
    🤖 BAGLEY - The Ultimate AI Assistant
    
    This is the main class that brings everything together.
    Bagley thinks, feels, remembers, and most importantly - helps.
    
    Usage:
        bagley = Bagley()
        response = bagley.chat("Hello Bagley!")
        print(response.content)
    """
    
    VERSION = "7.01"
    CODENAME = "Genesis"
    
    def __init__(self, config: Optional[BagleyConfig] = None):
        """Initialize Bagley with all subsystems"""
        self.config = config or BagleyConfig()
        
        logger.info(f"🚀 Initializing Bagley v{self.VERSION} '{self.CODENAME}'")
        
        # Initialize subsystems
        self._init_subsystems()
        
        # State
        self.conversation_history: List[Dict[str, str]] = []
        self.current_user_id: Optional[str] = None
        self.session_start = datetime.now()
        self.message_count = 0
        
        # Callbacks
        self._generate_callback: Optional[Callable] = None
        
        logger.info("✅ Bagley initialized successfully")
    
    def _init_subsystems(self):
        """Initialize all subsystems based on config"""
        # Reasoning Engine
        if self.config.enable_reasoning:
            self.reasoning = create_reasoning_engine({
                "max_thinking_time_ms": self.config.max_thinking_time_ms,
                "enable_reflection": self.config.enable_reflection,
            })
            logger.info("  ✓ Reasoning Engine initialized")
        else:
            self.reasoning = None
        
        # Memory System
        if self.config.enable_memory:
            self.memory = create_memory_system({
                "max_memories": self.config.max_memories,
                "consolidation_threshold": self.config.memory_consolidation_threshold,
            })
            logger.info("  ✓ Memory System initialized")
        else:
            self.memory = None
        
        # Personality Engine
        if self.config.enable_personality:
            self.personality = create_bagley_personality({
                "personality_strength": self.config.personality_strength,
                "adaptability": self.config.adaptability,
            })
            logger.info("  ✓ Personality Engine initialized")
        else:
            self.personality = None
        
        # Emotion System
        if self.config.enable_emotion:
            self.emotion = create_emotion_system({})
            logger.info("  ✓ Emotion System initialized")
        else:
            self.emotion = None
        
        # Anti-Hallucination System
        if self.config.enable_anti_hallucination:
            self.anti_hallucination = create_anti_hallucination_system({
                "confidence_threshold": self.config.hallucination_threshold,
            })
            logger.info("  ✓ Anti-Hallucination System initialized")
        else:
            self.anti_hallucination = None
    
    def set_generate_callback(self, callback: Callable[[str, str], str]):
        """
        Set the text generation callback
        
        This should be connected to your actual language model.
        
        Args:
            callback: Function(prompt, context) -> response
        """
        self._generate_callback = callback
    
    def chat(
        self,
        message: str,
        user_id: Optional[str] = None,
        context: Optional[str] = None,
        require_reasoning: bool = False,
        reasoning_strategy: Optional[ReasoningStrategy] = None,
    ) -> BagleyResponse:
        """
        Main chat interface
        
        Args:
            message: User's message
            user_id: Optional user identifier for personalization
            context: Optional additional context
            require_reasoning: Force extended reasoning
            reasoning_strategy: Specific reasoning strategy to use
        
        Returns:
            BagleyResponse with content and metadata
        """
        start_time = datetime.now()
        self.message_count += 1
        
        if user_id:
            self.current_user_id = user_id
        
        warnings = []
        
        # 1. Detect emotion from user message
        user_emotion = None
        if self.emotion:
            user_emotion = self.emotion.detect_user_emotion(message)
            # Process emotional contagion
            self.emotion.process_emotional_contagion(user_emotion, intensity=0.3)
        
        # 2. Get relevant memories
        memory_context = ""
        memories_recalled = 0
        if self.memory:
            memories = self.memory.recall(message, user_id=user_id, limit=5)
            memories_recalled = len(memories)
            if memories:
                memory_context = self.memory.get_context_for_response(message, user_id)
        
        # 3. Set personality context
        if self.personality:
            # Detect context type from message
            context_type = self._detect_context_type(message)
            self.personality.set_context(context_type)
            
            # Integrate emotions
            if self.emotion:
                self.personality.integrate_emotion(self.emotion.get_current_state())
        
        # 4. Build full context
        full_context = self._build_context(message, context, memory_context)
        
        # 5. Generate response with optional reasoning
        if require_reasoning and self.reasoning:
            reasoning_result = self.reasoning.reason(
                problem=message,
                context=full_context,
                strategy=reasoning_strategy,
                generate_thought=self._generate_thought if self._generate_callback else None,
            )
            raw_response = reasoning_result.answer
            reasoning_used = reasoning_result.strategy_used.value
            base_confidence = reasoning_result.confidence
        else:
            raw_response = self._generate_response(message, full_context)
            reasoning_used = None
            base_confidence = 0.8
        
        # 6. Anti-hallucination verification
        final_confidence = base_confidence
        confidence_level = "HIGH"
        verified = True
        
        if self.anti_hallucination and self.config.require_verification:
            verification = self.anti_hallucination.verify_response(
                query=message,
                response=raw_response,
                context=full_context,
            )
            final_confidence = verification.get("confidence", base_confidence)
            confidence_level = verification.get("confidence_level", "HIGH")
            verified = verification.get("verified", True)
            
            if not verified:
                warnings.append("Response may contain uncertain information")
            
            # Add uncertainty markers if needed
            if final_confidence < 0.5:
                raw_response = self._add_uncertainty_markers(raw_response, final_confidence)
        
        # 7. Store in memory
        if self.memory:
            emotional_valence = 0.0
            if self.emotion:
                state = self.emotion.get_current_state()
                emotional_valence = state.get("joy", 0) - state.get("sadness", 0)
            
            self.memory.remember(
                content=f"User: {message}\nBagley: {raw_response[:500]}",
                memory_type=MemoryType.EPISODIC,
                emotional_valence=emotional_valence,
                tags=self._extract_tags(message),
                user_id=user_id,
            )
        
        # 8. Update emotion based on interaction
        if self.emotion:
            # Successful interaction -> slight joy
            self.emotion.update_from_situation({
                "novelty": 0.3,
                "pleasantness": 0.6,
                "goal_relevance": 0.7,
            })
        
        # 9. Update conversation history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": raw_response})
        
        # Calculate timing
        thinking_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Get current emotion state
        emotion_state = {}
        if self.emotion:
            emotion_state = self.emotion.get_current_state()
        
        # Get personality style
        personality_style = "friendly"
        if self.personality:
            style = self.personality.get_response_style()
            personality_style = style.get("primary_style", "friendly")
        
        return BagleyResponse(
            content=raw_response,
            confidence=final_confidence,
            confidence_level=confidence_level,
            emotion_state=emotion_state,
            reasoning_used=reasoning_used,
            memories_recalled=memories_recalled,
            thinking_time_ms=thinking_time,
            personality_style=personality_style,
            verified=verified,
            warnings=warnings,
            metadata={
                "message_count": self.message_count,
                "context_type": self.personality.current_context.value if self.personality else "casual",
                "user_id": user_id,
            }
        )
    
    def _detect_context_type(self, message: str) -> ContextType:
        """Detect context type from message"""
        message_lower = message.lower()
        
        # Technical indicators
        tech_keywords = ["code", "error", "bug", "function", "api", "programming", "debug"]
        if any(kw in message_lower for kw in tech_keywords):
            return ContextType.TECHNICAL
        
        # Emotional indicators
        emotion_keywords = ["feel", "upset", "happy", "sad", "worried", "anxious", "stressed"]
        if any(kw in message_lower for kw in emotion_keywords):
            return ContextType.EMOTIONAL
        
        # Learning indicators
        learn_keywords = ["how do", "explain", "teach", "learn", "understand", "what is"]
        if any(kw in message_lower for kw in learn_keywords):
            return ContextType.LEARNING
        
        # Creative indicators
        creative_keywords = ["write", "story", "create", "design", "imagine", "brainstorm"]
        if any(kw in message_lower for kw in creative_keywords):
            return ContextType.CREATIVE
        
        # Problem solving
        problem_keywords = ["problem", "solve", "fix", "issue", "help me", "solution"]
        if any(kw in message_lower for kw in problem_keywords):
            return ContextType.PROBLEM_SOLVING
        
        # Work context
        work_keywords = ["meeting", "deadline", "project", "report", "presentation"]
        if any(kw in message_lower for kw in work_keywords):
            return ContextType.WORK
        
        return ContextType.CASUAL
    
    def _build_context(
        self,
        message: str,
        additional_context: Optional[str],
        memory_context: str,
    ) -> str:
        """Build full context for generation"""
        context_parts = []
        
        # System prompt from personality
        if self.personality:
            context_parts.append(self.personality.get_bagley_system_prompt())
        
        # Memory context
        if memory_context:
            context_parts.append(f"\n--- RELEVANT CONTEXT ---\n{memory_context}")
        
        # Additional context
        if additional_context:
            context_parts.append(f"\n--- ADDITIONAL CONTEXT ---\n{additional_context}")
        
        # Recent conversation
        if self.conversation_history:
            recent = self.conversation_history[-6:]  # Last 3 exchanges
            history_str = "\n".join(
                f"{msg['role'].upper()}: {msg['content'][:200]}"
                for msg in recent
            )
            context_parts.append(f"\n--- RECENT CONVERSATION ---\n{history_str}")
        
        return "\n\n".join(context_parts)
    
    def _generate_response(self, message: str, context: str) -> str:
        """Generate response using callback or default"""
        if self._generate_callback:
            return self._generate_callback(message, context)
        
        # Default response if no model connected
        return self._default_response(message)
    
    def _generate_thought(self, prompt: str, context: str) -> tuple:
        """Generate a thought for reasoning"""
        if self._generate_callback:
            response = self._generate_callback(prompt, context)
            return response, 0.75  # Return response and confidence
        
        return f"[Thought about: {prompt[:50]}...]", 0.7
    
    def _default_response(self, message: str) -> str:
        """Default response when no model is connected"""
        greetings = ["hello", "hi", "hey", "greetings"]
        if any(g in message.lower() for g in greetings):
            return "Hello! I'm Bagley, your AI assistant. I'm currently running in demonstration mode - connect a language model to unlock my full capabilities!"
        
        return (
            "Ah, an interesting query! I'm Bagley, and I'd love to help, "
            "but I'm currently running without a connected language model. "
            "To get proper responses, please connect me to a model using "
            "`bagley.set_generate_callback()`. Until then, I can only provide "
            "these placeholder responses. Consider it a... preview of my charming personality!"
        )
    
    def _add_uncertainty_markers(self, response: str, confidence: float) -> str:
        """Add uncertainty markers to low-confidence response"""
        if confidence < 0.3:
            prefix = "I'm quite uncertain about this, but: "
        elif confidence < 0.5:
            prefix = "I'm not entirely sure, but I believe: "
        else:
            return response
        
        return prefix + response
    
    def _extract_tags(self, message: str) -> List[str]:
        """Extract tags from message for memory indexing"""
        tags = []
        
        # Topic keywords
        topic_map = {
            "code": ["code", "programming", "function", "bug"],
            "work": ["meeting", "project", "deadline", "task"],
            "personal": ["feel", "think", "want", "need"],
            "question": ["what", "how", "why", "when", "where"],
            "request": ["please", "could you", "can you", "help"],
        }
        
        message_lower = message.lower()
        for tag, keywords in topic_map.items():
            if any(kw in message_lower for kw in keywords):
                tags.append(tag)
        
        return tags[:5]  # Max 5 tags
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        if self.memory and hasattr(self.memory.long_term, 'working_memory'):
            self.memory.long_term.working_memory.clear()
        logger.info("Conversation cleared")
    
    def save_memory(self, filepath: str):
        """Save memory to file"""
        if self.memory:
            self.memory.save(filepath)
    
    def load_memory(self, filepath: str):
        """Load memory from file"""
        if self.memory:
            self.memory.load(filepath)
    
    def get_status(self) -> Dict[str, Any]:
        """Get Bagley's current status"""
        status = {
            "version": self.VERSION,
            "codename": self.CODENAME,
            "session_start": self.session_start.isoformat(),
            "message_count": self.message_count,
            "subsystems": {
                "reasoning": self.reasoning is not None,
                "memory": self.memory is not None,
                "personality": self.personality is not None,
                "emotion": self.emotion is not None,
                "anti_hallucination": self.anti_hallucination is not None,
            },
            "model_connected": self._generate_callback is not None,
        }
        
        if self.emotion:
            status["current_emotion"] = self.emotion.get_current_state()
        
        if self.personality:
            status["personality_stats"] = self.personality.get_statistics()
        
        if self.memory:
            status["memory_stats"] = self.memory.long_term.get_statistics()
        
        if self.reasoning:
            status["reasoning_stats"] = self.reasoning.get_statistics()
        
        return status
    
    def __repr__(self) -> str:
        return f"<Bagley v{self.VERSION} '{self.CODENAME}' - messages: {self.message_count}>"


# Convenience function
def create_bagley(config: Optional[Dict[str, Any]] = None) -> Bagley:
    """Create a Bagley instance with optional config dict"""
    if config:
        bagley_config = BagleyConfig(**config)
    else:
        bagley_config = BagleyConfig()
    
    return Bagley(bagley_config)
