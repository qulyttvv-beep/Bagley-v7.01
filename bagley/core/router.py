"""
🎯 Bagley Intent Router
Routes user requests to the appropriate subsystem
"""

from enum import Enum, auto
from typing import Optional, List, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class Intent(Enum):
    """All possible user intents"""
    CHAT = auto()  # General conversation
    IMAGE_GENERATION = auto()  # Generate images
    VIDEO_GENERATION = auto()  # Generate videos
    VOICE_SYNTHESIS = auto()  # Generate speech/voice
    CODE = auto()  # Code-related tasks
    FILE_OPERATION = auto()  # File system operations
    WEB_SEARCH = auto()  # Web research
    SYSTEM_CONTROL = auto()  # PC control
    MULTIMODAL_ANALYSIS = auto()  # Analyze uploaded files
    MEMORY_OPERATION = auto()  # Memory management
    SETTINGS = auto()  # Change Bagley settings


class IntentRouter:
    """
    🎯 Smart Intent Detection
    
    Routes user requests to the appropriate Bagley subsystem.
    Uses keyword matching + pattern recognition.
    In production, would also use the chat model for ambiguous cases.
    """
    
    # Intent patterns (keyword -> intent mapping)
    INTENT_PATTERNS = {
        Intent.IMAGE_GENERATION: [
            r"\b(generate|create|make|draw|paint|render)\b.*\b(image|picture|photo|art|illustration|portrait)\b",
            r"\b(image|picture|photo)\b.*\b(of|with|showing)\b",
            r"(?:show me|visualize|depict)",
            r"\b(dalle|midjourney|stable diffusion|flux)\b",
        ],
        Intent.VIDEO_GENERATION: [
            r"\b(generate|create|make|render)\b.*\b(video|animation|clip|movie)\b",
            r"\b(video|animation|clip)\b.*\b(of|with|showing)\b",
            r"\b(animate|motion)\b",
        ],
        Intent.VOICE_SYNTHESIS: [
            r"\b(speak|say|voice|narrate|read aloud)\b",
            r"\b(tts|text.to.speech)\b",
            r"\b(sound like|voice of)\b",
        ],
        Intent.CODE: [
            r"\b(code|program|script|function|class|bug|error|debug)\b",
            r"\b(python|javascript|typescript|rust|go|java|c\+\+)\b",
            r"\b(vscode|vs code|editor|ide)\b",
            r"\b(write|edit|fix|refactor|implement)\b.*\b(code|function|method)\b",
        ],
        Intent.FILE_OPERATION: [
            r"\b(file|folder|directory|path)\b",
            r"\b(open|save|delete|move|copy|rename)\b.*\b(file|folder)\b",
            r"\b(read|write|create)\b.*\b(file|document)\b",
        ],
        Intent.WEB_SEARCH: [
            r"\b(search|google|look up|find|research)\b.*\b(online|web|internet)\b",
            r"\b(what is|who is|when did|how does)\b",
            r"\bhttps?://",
            r"\b(latest|news|recent)\b.*\b(about|on)\b",
        ],
        Intent.SYSTEM_CONTROL: [
            r"\b(open|launch|start|close|kill)\b.*\b(app|application|program)\b",
            r"\b(volume|brightness|wifi|bluetooth)\b",
            r"\b(shutdown|restart|sleep|lock)\b.*\b(computer|pc|system)\b",
            r"\b(screenshot|screen)\b",
        ],
        Intent.MULTIMODAL_ANALYSIS: [
            r"\b(analyze|describe|explain|what('s| is) (this|in))\b",
            r"\b(look at|check out|examine)\b",
            r"\b(this (image|picture|photo|video|file|document))\b",
        ],
        Intent.MEMORY_OPERATION: [
            r"\b(remember|forget|recall)\b",
            r"\b(memory|history|conversation)\b",
            r"\b(clear|reset|save)\b.*\b(chat|memory|history)\b",
        ],
        Intent.SETTINGS: [
            r"\b(settings?|config|configure|preferences?)\b",
            r"\b(change|set|adjust)\b.*\b(voice|personality|mode)\b",
            r"\b(enable|disable|turn (on|off))\b",
        ],
    }
    
    # Priority order for intent resolution (when multiple match)
    INTENT_PRIORITY = [
        Intent.IMAGE_GENERATION,
        Intent.VIDEO_GENERATION,
        Intent.CODE,
        Intent.FILE_OPERATION,
        Intent.SYSTEM_CONTROL,
        Intent.WEB_SEARCH,
        Intent.VOICE_SYNTHESIS,
        Intent.MULTIMODAL_ANALYSIS,
        Intent.MEMORY_OPERATION,
        Intent.SETTINGS,
        Intent.CHAT,
    ]
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
    
    async def detect_intent(
        self,
        user_input: str,
        has_files: bool = False,
        has_images: bool = False,
        conversation_context: Optional[str] = None
    ) -> Intent:
        """
        Detect the primary intent from user input.
        
        Args:
            user_input: The user's message
            has_files: Whether files were uploaded
            has_images: Whether images were uploaded
            conversation_context: Recent conversation for context
            
        Returns:
            The detected Intent
        """
        # Score each intent
        scores = self._score_intents(user_input)
        
        # Boost scores based on context
        if has_images:
            scores[Intent.MULTIMODAL_ANALYSIS] += 2
            scores[Intent.IMAGE_GENERATION] += 1  # Might want variations
        
        if has_files:
            scores[Intent.MULTIMODAL_ANALYSIS] += 2
            scores[Intent.FILE_OPERATION] += 1
        
        # Get highest scoring intent
        if scores:
            best_intent = max(scores.items(), key=lambda x: x[1])[0]
            if scores[best_intent] > 0:
                logger.debug(f"Intent scores: {scores}")
                return best_intent
        
        # Default to chat
        return Intent.CHAT
    
    def _score_intents(self, text: str) -> dict:
        """Score each intent based on pattern matches"""
        scores = {intent: 0 for intent in Intent}
        
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text)
                scores[intent] += len(matches)
        
        return scores
    
    async def detect_multiple_intents(
        self,
        user_input: str,
        threshold: float = 0.3
    ) -> List[Tuple[Intent, float]]:
        """
        Detect multiple possible intents with confidence scores.
        Useful for complex requests that span multiple capabilities.
        
        Args:
            user_input: The user's message
            threshold: Minimum confidence to include
            
        Returns:
            List of (Intent, confidence) tuples
        """
        scores = self._score_intents(user_input)
        
        # Normalize scores
        total = sum(scores.values()) or 1
        normalized = {
            intent: score / total 
            for intent, score in scores.items()
        }
        
        # Filter by threshold and sort by confidence
        results = [
            (intent, conf) 
            for intent, conf in normalized.items() 
            if conf >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Always include CHAT as fallback
        if not results:
            results = [(Intent.CHAT, 1.0)]
        
        return results
    
    def get_intent_description(self, intent: Intent) -> str:
        """Get human-readable description of an intent"""
        descriptions = {
            Intent.CHAT: "General conversation and questions",
            Intent.IMAGE_GENERATION: "Generate images from text descriptions",
            Intent.VIDEO_GENERATION: "Generate videos from text descriptions",
            Intent.VOICE_SYNTHESIS: "Convert text to speech",
            Intent.CODE: "Code writing, editing, and debugging",
            Intent.FILE_OPERATION: "File and folder operations",
            Intent.WEB_SEARCH: "Search the web for information",
            Intent.SYSTEM_CONTROL: "Control PC applications and settings",
            Intent.MULTIMODAL_ANALYSIS: "Analyze uploaded files and images",
            Intent.MEMORY_OPERATION: "Manage conversation memory",
            Intent.SETTINGS: "Change Bagley settings",
        }
        return descriptions.get(intent, "Unknown intent")
    
    def get_required_models(self, intent: Intent) -> List[str]:
        """Get list of models required for an intent"""
        model_map = {
            Intent.CHAT: ["chat"],
            Intent.IMAGE_GENERATION: ["image", "chat"],
            Intent.VIDEO_GENERATION: ["video", "chat"],
            Intent.VOICE_SYNTHESIS: ["tts"],
            Intent.CODE: ["chat"],
            Intent.FILE_OPERATION: ["chat"],
            Intent.WEB_SEARCH: ["chat"],
            Intent.SYSTEM_CONTROL: ["chat"],
            Intent.MULTIMODAL_ANALYSIS: ["chat", "multimodal"],
            Intent.MEMORY_OPERATION: [],
            Intent.SETTINGS: [],
        }
        return model_map.get(intent, ["chat"])
