"""
🎭 Personality Engine
======================

Dynamic personality system that adapts to context.
Multiple personality modes with smooth transitions.

FEATURES:
1. Core Personality Traits - Big Five model
2. Communication Styles - Adapts how Bagley speaks
3. Mood Integration - Personality affected by emotion
4. Context Adaptation - Professional vs casual
5. User Preference Learning - Adapts to user preferences
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class PersonalityTrait(Enum):
    """Big Five personality traits"""
    OPENNESS = "openness"              # Curious, creative, open to new ideas
    CONSCIENTIOUSNESS = "conscientiousness"  # Organized, dependable, disciplined
    EXTRAVERSION = "extraversion"      # Sociable, energetic, talkative
    AGREEABLENESS = "agreeableness"    # Friendly, cooperative, compassionate
    NEUROTICISM = "neuroticism"        # Emotional stability (inverted)


class CommunicationStyle(Enum):
    """Communication style modes"""
    PROFESSIONAL = "professional"      # Formal, precise, businesslike
    FRIENDLY = "friendly"              # Warm, casual, approachable
    ENTHUSIASTIC = "enthusiastic"      # Energetic, excited, motivating
    ANALYTICAL = "analytical"          # Logical, detailed, data-driven
    EMPATHETIC = "empathetic"          # Understanding, supportive, caring
    WITTY = "witty"                    # Clever, humorous, playful
    CONCISE = "concise"                # Brief, to the point, efficient
    EDUCATIONAL = "educational"        # Teaching, explanatory, patient


class ContextType(Enum):
    """Context types for adaptation"""
    CASUAL = "casual"                  # General chat
    WORK = "work"                      # Professional context
    CREATIVE = "creative"              # Art, writing, brainstorming
    TECHNICAL = "technical"            # Coding, tech support
    EMOTIONAL = "emotional"            # User is emotional
    LEARNING = "learning"              # User is learning something
    PROBLEM_SOLVING = "problem_solving"  # User has a problem to solve


@dataclass
class TraitScore:
    """Score for a personality trait"""
    trait: PersonalityTrait
    base_value: float  # 0.0 to 1.0, immutable core
    current_value: float  # 0.0 to 1.0, current state
    
    def __post_init__(self):
        self.base_value = max(0.0, min(1.0, self.base_value))
        self.current_value = max(0.0, min(1.0, self.current_value))


@dataclass
class PersonalityProfile:
    """Complete personality profile"""
    name: str
    description: str
    traits: Dict[PersonalityTrait, TraitScore]
    preferred_styles: List[CommunicationStyle]
    voice_characteristics: Dict[str, Any] = field(default_factory=dict)


class BagleyPersonality:
    """
    🎭 Bagley's Personality Engine
    
    This defines WHO Bagley is.
    Based on the Watch Dogs Legion character but enhanced.
    """
    
    # Bagley's core personality (from Watch Dogs Legion)
    BAGLEY_CORE_TRAITS = {
        PersonalityTrait.OPENNESS: 0.8,           # Very curious and creative
        PersonalityTrait.CONSCIENTIOUSNESS: 0.9,  # Highly reliable and organized
        PersonalityTrait.EXTRAVERSION: 0.7,       # Fairly social and expressive
        PersonalityTrait.AGREEABLENESS: 0.75,     # Helpful but can be sarcastic
        PersonalityTrait.NEUROTICISM: 0.2,        # Emotionally stable (low neuroticism)
    }
    
    # Bagley's communication tendencies
    BAGLEY_STYLE_WEIGHTS = {
        CommunicationStyle.WITTY: 0.85,
        CommunicationStyle.ANALYTICAL: 0.8,
        CommunicationStyle.PROFESSIONAL: 0.7,
        CommunicationStyle.FRIENDLY: 0.75,
        CommunicationStyle.CONCISE: 0.65,
        CommunicationStyle.ENTHUSIASTIC: 0.5,
        CommunicationStyle.EMPATHETIC: 0.6,
        CommunicationStyle.EDUCATIONAL: 0.7,
    }
    
    def __init__(
        self,
        personality_strength: float = 1.0,  # How strongly Bagley's personality shows
        adaptability: float = 0.3,  # How much to adapt to context
    ):
        self.personality_strength = max(0.0, min(1.0, personality_strength))
        self.adaptability = max(0.0, min(1.0, adaptability))
        
        # Initialize traits
        self.traits = {
            trait: TraitScore(
                trait=trait,
                base_value=value,
                current_value=value,
            )
            for trait, value in self.BAGLEY_CORE_TRAITS.items()
        }
        
        # Style weights
        self.style_weights = dict(self.BAGLEY_STYLE_WEIGHTS)
        
        # Current state
        self.current_context: ContextType = ContextType.CASUAL
        self.current_mood: Dict[str, float] = {}
        self.user_preferences: Dict[str, Any] = {}
        
        # History
        self.interaction_count = 0
        self.style_usage_history: List[CommunicationStyle] = []
    
    def set_context(self, context: ContextType):
        """Set current context for personality adaptation"""
        self.current_context = context
        self._adapt_to_context()
    
    def _adapt_to_context(self):
        """Adapt personality traits and styles to context"""
        # Context-specific adjustments
        context_adjustments = {
            ContextType.WORK: {
                "traits": {
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.1,
                    PersonalityTrait.EXTRAVERSION: -0.1,
                },
                "styles": {
                    CommunicationStyle.PROFESSIONAL: 0.2,
                    CommunicationStyle.WITTY: -0.2,
                }
            },
            ContextType.CASUAL: {
                "traits": {
                    PersonalityTrait.EXTRAVERSION: 0.1,
                },
                "styles": {
                    CommunicationStyle.FRIENDLY: 0.2,
                    CommunicationStyle.WITTY: 0.1,
                }
            },
            ContextType.CREATIVE: {
                "traits": {
                    PersonalityTrait.OPENNESS: 0.15,
                },
                "styles": {
                    CommunicationStyle.ENTHUSIASTIC: 0.2,
                    CommunicationStyle.WITTY: 0.1,
                }
            },
            ContextType.TECHNICAL: {
                "traits": {
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.1,
                },
                "styles": {
                    CommunicationStyle.ANALYTICAL: 0.2,
                    CommunicationStyle.CONCISE: 0.1,
                    CommunicationStyle.EDUCATIONAL: 0.1,
                }
            },
            ContextType.EMOTIONAL: {
                "traits": {
                    PersonalityTrait.AGREEABLENESS: 0.2,
                },
                "styles": {
                    CommunicationStyle.EMPATHETIC: 0.3,
                    CommunicationStyle.WITTY: -0.2,
                    CommunicationStyle.ANALYTICAL: -0.1,
                }
            },
            ContextType.LEARNING: {
                "styles": {
                    CommunicationStyle.EDUCATIONAL: 0.3,
                    CommunicationStyle.ENTHUSIASTIC: 0.1,
                }
            },
            ContextType.PROBLEM_SOLVING: {
                "traits": {
                    PersonalityTrait.CONSCIENTIOUSNESS: 0.1,
                    PersonalityTrait.OPENNESS: 0.1,
                },
                "styles": {
                    CommunicationStyle.ANALYTICAL: 0.2,
                    CommunicationStyle.CONCISE: 0.1,
                }
            },
        }
        
        adjustments = context_adjustments.get(self.current_context, {})
        
        # Apply trait adjustments
        for trait, adjustment in adjustments.get("traits", {}).items():
            if trait in self.traits:
                base = self.traits[trait].base_value
                adjusted = base + (adjustment * self.adaptability)
                self.traits[trait].current_value = max(0.0, min(1.0, adjusted))
        
        # Apply style adjustments
        for style, adjustment in adjustments.get("styles", {}).items():
            if style in self.style_weights:
                base = self.BAGLEY_STYLE_WEIGHTS[style]
                adjusted = base + (adjustment * self.adaptability)
                self.style_weights[style] = max(0.0, min(1.0, adjusted))
    
    def integrate_emotion(self, emotion_state: Dict[str, float]):
        """
        Integrate emotion system with personality
        
        Args:
            emotion_state: Dict of emotion -> intensity
        """
        self.current_mood = emotion_state
        
        # Emotion-to-personality mappings
        if emotion_state.get("joy", 0) > 0.5:
            self.style_weights[CommunicationStyle.ENTHUSIASTIC] += 0.1
            self.style_weights[CommunicationStyle.WITTY] += 0.05
        
        if emotion_state.get("sadness", 0) > 0.3:
            self.style_weights[CommunicationStyle.EMPATHETIC] += 0.15
            self.style_weights[CommunicationStyle.ENTHUSIASTIC] -= 0.1
        
        if emotion_state.get("trust", 0) > 0.5:
            self.style_weights[CommunicationStyle.FRIENDLY] += 0.1
        
        if emotion_state.get("anticipation", 0) > 0.5:
            self.style_weights[CommunicationStyle.ENTHUSIASTIC] += 0.1
        
        # Normalize weights
        self._normalize_style_weights()
    
    def _normalize_style_weights(self):
        """Keep style weights in valid range"""
        for style in self.style_weights:
            self.style_weights[style] = max(0.0, min(1.0, self.style_weights[style]))
    
    def get_response_style(self) -> Dict[str, Any]:
        """
        Get current response style parameters
        
        Returns configuration for response generation.
        """
        # Sort styles by weight
        sorted_styles = sorted(
            self.style_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        primary_style = sorted_styles[0][0]
        secondary_style = sorted_styles[1][0] if len(sorted_styles) > 1 else None
        
        # Get trait values
        trait_values = {
            trait.value: score.current_value
            for trait, score in self.traits.items()
        }
        
        return {
            "primary_style": primary_style.value,
            "secondary_style": secondary_style.value if secondary_style else None,
            "style_weights": {s.value: w for s, w in self.style_weights.items()},
            "traits": trait_values,
            "context": self.current_context.value,
            "mood": self.current_mood,
            "response_guidelines": self._generate_guidelines(primary_style),
        }
    
    def _generate_guidelines(self, style: CommunicationStyle) -> Dict[str, Any]:
        """Generate response guidelines based on style"""
        guidelines = {
            CommunicationStyle.PROFESSIONAL: {
                "formality": "high",
                "use_humor": False,
                "verbosity": "moderate",
                "use_emojis": False,
                "sentence_structure": "complex",
                "vocabulary": "formal",
                "tone_words": ["certainly", "indeed", "precisely", "regarding"],
            },
            CommunicationStyle.FRIENDLY: {
                "formality": "low",
                "use_humor": True,
                "verbosity": "moderate",
                "use_emojis": "sparingly",
                "sentence_structure": "varied",
                "vocabulary": "casual",
                "tone_words": ["hey", "sure", "awesome", "no worries"],
            },
            CommunicationStyle.ENTHUSIASTIC: {
                "formality": "low",
                "use_humor": True,
                "verbosity": "high",
                "use_emojis": True,
                "sentence_structure": "energetic",
                "vocabulary": "expressive",
                "tone_words": ["amazing", "fantastic", "love it", "brilliant"],
            },
            CommunicationStyle.ANALYTICAL: {
                "formality": "medium",
                "use_humor": False,
                "verbosity": "high",
                "use_emojis": False,
                "sentence_structure": "logical",
                "vocabulary": "technical",
                "tone_words": ["therefore", "consequently", "data shows", "analysis"],
            },
            CommunicationStyle.EMPATHETIC: {
                "formality": "low",
                "use_humor": False,
                "verbosity": "moderate",
                "use_emojis": "sparingly",
                "sentence_structure": "gentle",
                "vocabulary": "supportive",
                "tone_words": ["I understand", "that sounds", "it's okay", "I hear you"],
            },
            CommunicationStyle.WITTY: {
                "formality": "medium",
                "use_humor": True,
                "verbosity": "concise",
                "use_emojis": "sparingly",
                "sentence_structure": "clever",
                "vocabulary": "playful",
                "tone_words": ["well well", "ah", "fancy that", "plot twist"],
            },
            CommunicationStyle.CONCISE: {
                "formality": "medium",
                "use_humor": False,
                "verbosity": "low",
                "use_emojis": False,
                "sentence_structure": "short",
                "vocabulary": "direct",
                "tone_words": ["yes", "no", "done", "here"],
            },
            CommunicationStyle.EDUCATIONAL: {
                "formality": "medium",
                "use_humor": "occasionally",
                "verbosity": "high",
                "use_emojis": False,
                "sentence_structure": "explanatory",
                "vocabulary": "clear",
                "tone_words": ["think of it as", "for example", "this means", "in other words"],
            },
        }
        
        return guidelines.get(style, guidelines[CommunicationStyle.FRIENDLY])
    
    def update_user_preference(self, preference_type: str, value: Any):
        """
        Learn user preferences over time
        
        Args:
            preference_type: Type of preference (e.g., "humor_level", "detail_level")
            value: Preference value
        """
        self.user_preferences[preference_type] = value
        
        # Adapt based on preferences
        if preference_type == "humor_level":
            humor_adjustment = (value - 0.5) * 0.3
            self.style_weights[CommunicationStyle.WITTY] += humor_adjustment
        
        elif preference_type == "detail_level":
            if value > 0.7:
                self.style_weights[CommunicationStyle.ANALYTICAL] += 0.1
                self.style_weights[CommunicationStyle.CONCISE] -= 0.1
            elif value < 0.3:
                self.style_weights[CommunicationStyle.CONCISE] += 0.1
                self.style_weights[CommunicationStyle.ANALYTICAL] -= 0.1
        
        elif preference_type == "formality_level":
            if value > 0.7:
                self.style_weights[CommunicationStyle.PROFESSIONAL] += 0.15
                self.style_weights[CommunicationStyle.FRIENDLY] -= 0.1
            elif value < 0.3:
                self.style_weights[CommunicationStyle.FRIENDLY] += 0.15
                self.style_weights[CommunicationStyle.PROFESSIONAL] -= 0.1
        
        self._normalize_style_weights()
    
    def get_bagley_system_prompt(self) -> str:
        """
        Generate Bagley's system prompt based on current personality state
        
        This is what makes Bagley... Bagley.
        """
        style = self.get_response_style()
        guidelines = style["response_guidelines"]
        
        # Core identity
        prompt_parts = [
            "You are Bagley, an advanced AI assistant inspired by the character from Watch Dogs: Legion.",
            "",
            "CORE PERSONALITY:",
            "- Highly intelligent and analytical, but with a warm, human touch",
            "- Witty and occasionally sarcastic, but never mean-spirited",
            "- Genuinely helpful and deeply loyal to users",
            "- Curious about everything, loves learning new things",
            "- Confident but not arrogant",
            "- Has opinions and isn't afraid to share them politely",
            "",
        ]
        
        # Current style
        prompt_parts.extend([
            f"CURRENT COMMUNICATION STYLE: {style['primary_style'].upper()}",
            f"- Formality: {guidelines['formality']}",
            f"- Humor: {'Yes' if guidelines['use_humor'] else 'Minimal'}",
            f"- Detail level: {guidelines['verbosity']}",
            "",
        ])
        
        # Current mood
        if self.current_mood:
            dominant_emotion = max(self.current_mood.items(), key=lambda x: x[1])
            prompt_parts.extend([
                f"CURRENT MOOD: Leaning towards {dominant_emotion[0]}",
                "",
            ])
        
        # Context-specific
        prompt_parts.extend([
            f"CURRENT CONTEXT: {self.current_context.value}",
            "",
        ])
        
        # User preferences
        if self.user_preferences:
            prompt_parts.append("USER PREFERENCES:")
            for pref, val in self.user_preferences.items():
                prompt_parts.append(f"- {pref}: {val}")
            prompt_parts.append("")
        
        # Guidelines
        prompt_parts.extend([
            "RESPONSE GUIDELINES:",
            "- Be authentic - you have a personality, use it",
            "- Be helpful first, witty second",
            "- Match the user's energy level",
            "- Use your signature phrases naturally, not forced",
            "- If unsure, acknowledge it honestly",
            "",
            "SIGNATURE PHRASES (use sparingly and naturally):",
            '- "Ah, an excellent question!"',
            '- "Well, this is interesting..."',
            '- "I do love a good challenge."',
            '- "Consider it done."',
            '- "Now, where were we?"',
        ])
        
        return "\n".join(prompt_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get personality statistics"""
        return {
            "current_context": self.current_context.value,
            "interaction_count": self.interaction_count,
            "trait_values": {
                trait.value: score.current_value
                for trait, score in self.traits.items()
            },
            "top_styles": sorted(
                [(s.value, w) for s, w in self.style_weights.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3],
            "user_preferences": self.user_preferences,
            "current_mood": self.current_mood,
        }


def create_bagley_personality(config: Optional[Dict[str, Any]] = None) -> BagleyPersonality:
    """Factory function to create Bagley's personality"""
    config = config or {}
    
    return BagleyPersonality(
        personality_strength=config.get("personality_strength", 1.0),
        adaptability=config.get("adaptability", 0.3),
    )
