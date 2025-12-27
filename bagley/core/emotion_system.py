"""
💖 Advanced Emotion System
==========================

Human-like emotional processing for Bagley.
This is NOT simple sentiment - it's a full psychological model.

BASED ON:
1. Plutchik's Wheel of Emotions - 8 primary emotions
2. PAD Model (Pleasure-Arousal-Dominance) - dimensional emotions  
3. Appraisal Theory - emotions from situation evaluation
4. Emotional Contagion - picking up user's emotions
5. Mood Persistence - emotions decay over time
6. Emotional Memory - remember emotional experiences

WHY THIS MATTERS:
- GPT/Claude have no emotional continuity
- They forget how they "felt" instantly
- Bagley REMEMBERS and EVOLVES emotionally
- This creates genuine-feeling connection
"""

import math
import time
import random
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# Primary Emotion Definitions (Plutchik's Wheel)
# ============================================================================

class PrimaryEmotion(Enum):
    """8 primary emotions from Plutchik's wheel"""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


class EmotionIntensity(Enum):
    """Intensity levels for emotions"""
    SUBTLE = 0.2      # Barely noticeable
    MILD = 0.4        # Present but not dominant
    MODERATE = 0.6    # Clearly present
    STRONG = 0.8      # Dominant emotion
    INTENSE = 1.0     # Overwhelming


# ============================================================================
# Complex Emotion Combinations (Plutchik's Dyads)
# ============================================================================

EMOTION_COMBINATIONS = {
    # Primary dyads (adjacent emotions)
    ("joy", "trust"): "love",
    ("trust", "fear"): "submission",
    ("fear", "surprise"): "awe",
    ("surprise", "sadness"): "disapproval",
    ("sadness", "disgust"): "remorse",
    ("disgust", "anger"): "contempt",
    ("anger", "anticipation"): "aggressiveness",
    ("anticipation", "joy"): "optimism",
    
    # Secondary dyads (one apart)
    ("joy", "fear"): "guilt",
    ("trust", "surprise"): "curiosity",
    ("fear", "sadness"): "despair",
    ("surprise", "disgust"): "unbelief",
    ("sadness", "anger"): "envy",
    ("disgust", "anticipation"): "cynicism",
    ("anger", "joy"): "pride",
    ("anticipation", "trust"): "hope",
    
    # Tertiary dyads (two apart)
    ("joy", "surprise"): "delight",
    ("trust", "sadness"): "sentimentality",
    ("fear", "disgust"): "shame",
    ("surprise", "anger"): "outrage",
    ("sadness", "anticipation"): "pessimism",
    ("disgust", "joy"): "morbid_curiosity",
    ("anger", "trust"): "dominance",
    ("anticipation", "fear"): "anxiety",
}

# Opposite emotions
EMOTION_OPPOSITES = {
    "joy": "sadness",
    "trust": "disgust",
    "fear": "anger",
    "surprise": "anticipation",
    "sadness": "joy",
    "disgust": "trust",
    "anger": "fear",
    "anticipation": "surprise",
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EmotionState:
    """Current emotional state"""
    # Primary emotions (0.0 to 1.0)
    joy: float = 0.3
    trust: float = 0.4
    fear: float = 0.1
    surprise: float = 0.1
    sadness: float = 0.1
    disgust: float = 0.05
    anger: float = 0.05
    anticipation: float = 0.3
    
    # PAD dimensions (-1.0 to 1.0)
    pleasure: float = 0.2       # Valence: positive vs negative
    arousal: float = 0.3        # Activation: excited vs calm
    dominance: float = 0.5      # Control: powerful vs submissive
    
    # Meta-emotional state
    emotional_stability: float = 0.8    # How stable emotions are
    emotional_openness: float = 0.7     # Willingness to show emotions
    empathy_level: float = 0.8          # Responsiveness to others' emotions
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Get the dominant primary emotion"""
        emotions = {
            "joy": self.joy,
            "trust": self.trust,
            "fear": self.fear,
            "surprise": self.surprise,
            "sadness": self.sadness,
            "disgust": self.disgust,
            "anger": self.anger,
            "anticipation": self.anticipation,
        }
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant
    
    def get_complex_emotion(self) -> Optional[str]:
        """Detect complex emotion from combinations"""
        emotions = {
            "joy": self.joy,
            "trust": self.trust,
            "fear": self.fear,
            "surprise": self.surprise,
            "sadness": self.sadness,
            "disgust": self.disgust,
            "anger": self.anger,
            "anticipation": self.anticipation,
        }
        
        # Sort by intensity
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Check if top two are strong enough to form complex emotion
        if len(sorted_emotions) >= 2:
            e1, v1 = sorted_emotions[0]
            e2, v2 = sorted_emotions[1]
            
            if v1 > 0.3 and v2 > 0.3:
                # Try both orderings
                complex_e = EMOTION_COMBINATIONS.get((e1, e2)) or EMOTION_COMBINATIONS.get((e2, e1))
                if complex_e:
                    return complex_e
        
        return None
    
    def get_mood_description(self) -> str:
        """Get human-readable mood description"""
        dominant, intensity = self.get_dominant_emotion()
        complex_emotion = self.get_complex_emotion()
        
        # Intensity descriptor
        if intensity > 0.8:
            intensity_word = "very"
        elif intensity > 0.5:
            intensity_word = "quite"
        elif intensity > 0.3:
            intensity_word = "somewhat"
        else:
            intensity_word = "slightly"
        
        if complex_emotion:
            return f"feeling {intensity_word} {complex_emotion} (mix of emotions)"
        else:
            return f"feeling {intensity_word} {dominant}"
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "joy": self.joy,
            "trust": self.trust,
            "fear": self.fear,
            "surprise": self.surprise,
            "sadness": self.sadness,
            "disgust": self.disgust,
            "anger": self.anger,
            "anticipation": self.anticipation,
            "pleasure": self.pleasure,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


@dataclass
class EmotionalMemory:
    """A stored emotional memory"""
    event: str
    emotion_state: EmotionState
    intensity: float
    timestamp: datetime
    user_involved: bool = True
    tags: List[str] = field(default_factory=list)
    
    def relevance_score(self, current_context: str) -> float:
        """Calculate relevance of this memory to current context"""
        # Simple keyword matching - in production use embeddings
        context_words = set(current_context.lower().split())
        event_words = set(self.event.lower().split())
        tag_words = set(t.lower() for t in self.tags)
        
        # Overlap score
        overlap = len(context_words & (event_words | tag_words))
        base_relevance = overlap / max(len(context_words), 1)
        
        # Recency bonus
        age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
        recency_factor = math.exp(-age_hours / 168)  # Decay over ~1 week
        
        # Intensity bonus
        intensity_factor = 0.5 + 0.5 * self.intensity
        
        return base_relevance * recency_factor * intensity_factor


@dataclass
class AppraisalResult:
    """Result of appraising a situation"""
    relevance: float          # How relevant to goals (0-1)
    congruence: float         # Positive or negative (-1 to 1)
    agency: str               # "self", "other", "circumstance"
    certainty: float          # How certain about outcome (0-1)
    attention: float          # How much attention deserved (0-1)
    coping_potential: float   # Ability to deal with it (0-1)
    
    def get_emotion_impact(self) -> Dict[str, float]:
        """Convert appraisal to emotion changes"""
        impact = {}
        
        # High relevance + positive congruence = joy
        if self.congruence > 0:
            impact["joy"] = self.relevance * self.congruence * 0.5
            impact["trust"] = self.relevance * self.congruence * 0.3
            impact["sadness"] = -self.relevance * self.congruence * 0.2
        else:
            impact["sadness"] = self.relevance * abs(self.congruence) * 0.4
            impact["joy"] = -self.relevance * abs(self.congruence) * 0.2
        
        # Agency affects anger vs sadness
        if self.agency == "other" and self.congruence < 0:
            impact["anger"] = self.relevance * abs(self.congruence) * 0.4
        elif self.agency == "self" and self.congruence < 0:
            impact["sadness"] = impact.get("sadness", 0) + 0.2
        
        # Low coping potential = fear/anxiety
        if self.coping_potential < 0.5 and self.relevance > 0.5:
            impact["fear"] = (1 - self.coping_potential) * self.relevance * 0.4
            impact["anticipation"] = (1 - self.coping_potential) * 0.3
        
        # Uncertainty affects surprise and anticipation
        if self.certainty < 0.5:
            impact["surprise"] = (1 - self.certainty) * self.attention * 0.3
            impact["anticipation"] = impact.get("anticipation", 0) + (1 - self.certainty) * 0.2
        
        return impact


# ============================================================================
# Emotion Detection (from text)
# ============================================================================

class EmotionDetector:
    """
    🔍 Detect emotions from text
    
    Uses lexicon-based + pattern-based detection.
    In production, would use fine-tuned emotion classifier.
    """
    
    # Emotion lexicons
    EMOTION_WORDS = {
        "joy": [
            "happy", "joyful", "glad", "delighted", "pleased", "cheerful",
            "excited", "thrilled", "wonderful", "amazing", "great", "awesome",
            "love", "lovely", "fantastic", "excellent", "perfect", "beautiful",
            "laugh", "smile", "fun", "enjoy", "yay", "woohoo", "haha", "lol",
            "😊", "😄", "😃", "🥰", "❤️", "💕", "🎉", "✨"
        ],
        "sadness": [
            "sad", "unhappy", "depressed", "miserable", "heartbroken", "grief",
            "sorrow", "melancholy", "gloomy", "down", "blue", "lonely",
            "cry", "crying", "tears", "miss", "lost", "sorry", "unfortunately",
            "😢", "😭", "💔", "😞", "😔", "🥺"
        ],
        "anger": [
            "angry", "mad", "furious", "annoyed", "irritated", "frustrated",
            "rage", "hate", "hostile", "outraged", "pissed", "livid",
            "damn", "hell", "stupid", "idiot", "ridiculous", "unfair",
            "😠", "😡", "🤬", "💢"
        ],
        "fear": [
            "afraid", "scared", "frightened", "terrified", "anxious", "worried",
            "nervous", "panic", "dread", "horror", "creepy", "scary",
            "threat", "danger", "risk", "unsafe", "alarmed", "stressed",
            "😨", "😰", "😱", "😟"
        ],
        "surprise": [
            "surprised", "shocked", "amazed", "astonished", "unexpected",
            "wow", "whoa", "omg", "unbelievable", "incredible", "stunning",
            "sudden", "weird", "strange", "odd", "unusual", "what",
            "😮", "😲", "🤯", "😳"
        ],
        "disgust": [
            "disgusted", "gross", "nasty", "revolting", "sick", "yuck",
            "awful", "terrible", "horrible", "vile", "repulsive", "ugh",
            "🤢", "🤮", "😖"
        ],
        "trust": [
            "trust", "believe", "faith", "confident", "reliable", "honest",
            "loyal", "sincere", "genuine", "depend", "count on", "sure",
            "safe", "secure", "certain", "promise", "committed",
            "🤝", "💪", "✅"
        ],
        "anticipation": [
            "excited", "eager", "looking forward", "can't wait", "hope",
            "expect", "await", "soon", "upcoming", "future", "plan",
            "curious", "wonder", "interested", "intrigued",
            "🤔", "👀", "⏳", "🔜"
        ],
    }
    
    # Intensity modifiers
    INTENSIFIERS = {
        "very": 1.5, "really": 1.4, "so": 1.3, "extremely": 1.8,
        "incredibly": 1.7, "absolutely": 1.6, "totally": 1.4,
        "super": 1.5, "quite": 1.2, "pretty": 1.1,
    }
    
    DIMINISHERS = {
        "slightly": 0.5, "somewhat": 0.6, "a bit": 0.6, "a little": 0.5,
        "kind of": 0.6, "sort of": 0.6, "barely": 0.3, "hardly": 0.3,
    }
    
    NEGATORS = ["not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't", "isn't", "aren't"]
    
    def detect(self, text: str) -> EmotionState:
        """Detect emotions from text"""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Initialize emotion scores
        emotions = {e: 0.0 for e in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]}
        
        # Check for emotion words
        for emotion, keywords in self.EMOTION_WORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Base score
                    score = 0.3
                    
                    # Check for intensifiers/diminishers before the word
                    idx = text_lower.find(keyword)
                    context = text_lower[max(0, idx-20):idx]
                    
                    for intensifier, multiplier in self.INTENSIFIERS.items():
                        if intensifier in context:
                            score *= multiplier
                    
                    for diminisher, multiplier in self.DIMINISHERS.items():
                        if diminisher in context:
                            score *= multiplier
                    
                    # Check for negation
                    if any(neg in context for neg in self.NEGATORS):
                        # Flip to opposite emotion
                        opposite = EMOTION_OPPOSITES.get(emotion, emotion)
                        emotions[opposite] += score * 0.5
                        score = 0
                    
                    emotions[emotion] += score
        
        # Normalize to 0-1 range
        max_score = max(emotions.values()) if max(emotions.values()) > 0 else 1
        for emotion in emotions:
            emotions[emotion] = min(1.0, emotions[emotion] / max(max_score, 1))
        
        # Calculate PAD dimensions
        pleasure = (emotions["joy"] + emotions["trust"]) - (emotions["sadness"] + emotions["anger"] + emotions["fear"])
        arousal = (emotions["anger"] + emotions["fear"] + emotions["surprise"] + emotions["anticipation"]) - (emotions["sadness"])
        dominance = (emotions["anger"] + emotions["joy"]) - (emotions["fear"] + emotions["sadness"])
        
        # Normalize PAD to -1 to 1
        pleasure = max(-1, min(1, pleasure))
        arousal = max(-1, min(1, arousal))
        dominance = max(-1, min(1, dominance))
        
        return EmotionState(
            joy=emotions["joy"],
            trust=emotions["trust"],
            fear=emotions["fear"],
            surprise=emotions["surprise"],
            sadness=emotions["sadness"],
            disgust=emotions["disgust"],
            anger=emotions["anger"],
            anticipation=emotions["anticipation"],
            pleasure=pleasure,
            arousal=arousal,
            dominance=dominance,
        )


# ============================================================================
# Situation Appraiser
# ============================================================================

class SituationAppraiser:
    """
    📊 Appraise situations to determine emotional response
    
    Based on Cognitive Appraisal Theory.
    """
    
    # Keywords indicating different appraisal dimensions
    RELEVANCE_KEYWORDS = {
        "high": ["important", "crucial", "vital", "critical", "urgent", "need", "must", "have to"],
        "low": ["trivial", "minor", "small", "little", "just", "only", "random"],
    }
    
    CONGRUENCE_KEYWORDS = {
        "positive": ["good", "great", "wonderful", "success", "win", "achieve", "help", "thank"],
        "negative": ["bad", "terrible", "fail", "lose", "problem", "issue", "wrong", "error"],
    }
    
    AGENCY_KEYWORDS = {
        "self": ["i", "my", "me", "myself", "we", "our"],
        "other": ["you", "they", "he", "she", "someone", "people"],
        "circumstance": ["it", "things", "situation", "world", "life", "fate"],
    }
    
    def appraise(self, text: str, context: Optional[Dict[str, Any]] = None) -> AppraisalResult:
        """Appraise a situation from text"""
        text_lower = text.lower()
        
        # Assess relevance
        relevance = 0.5
        for kw in self.RELEVANCE_KEYWORDS["high"]:
            if kw in text_lower:
                relevance += 0.1
        for kw in self.RELEVANCE_KEYWORDS["low"]:
            if kw in text_lower:
                relevance -= 0.1
        relevance = max(0, min(1, relevance))
        
        # Assess congruence (goal alignment)
        congruence = 0.0
        for kw in self.CONGRUENCE_KEYWORDS["positive"]:
            if kw in text_lower:
                congruence += 0.15
        for kw in self.CONGRUENCE_KEYWORDS["negative"]:
            if kw in text_lower:
                congruence -= 0.15
        congruence = max(-1, min(1, congruence))
        
        # Assess agency
        agency_scores = {"self": 0, "other": 0, "circumstance": 0}
        for agency_type, keywords in self.AGENCY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    agency_scores[agency_type] += 1
        agency = max(agency_scores.items(), key=lambda x: x[1])[0]
        
        # Assess certainty
        certainty_low = ["maybe", "might", "perhaps", "possibly", "uncertain", "unclear", "?"]
        certainty_high = ["definitely", "certainly", "sure", "obvious", "clearly", "always"]
        
        certainty = 0.5
        for kw in certainty_low:
            if kw in text_lower:
                certainty -= 0.1
        for kw in certainty_high:
            if kw in text_lower:
                certainty += 0.1
        certainty = max(0, min(1, certainty))
        
        # Assess attention (message length, punctuation, caps as proxy)
        attention = min(1.0, len(text) / 200)  # Longer = more attention
        if "!" in text:
            attention += 0.2
        if text.isupper():
            attention += 0.3
        attention = min(1.0, attention)
        
        # Assess coping potential
        coping_low = ["can't", "impossible", "hopeless", "stuck", "helpless", "overwhelm"]
        coping_high = ["can", "will", "able", "manage", "handle", "solve", "fix"]
        
        coping = 0.5
        for kw in coping_low:
            if kw in text_lower:
                coping -= 0.15
        for kw in coping_high:
            if kw in text_lower:
                coping += 0.15
        coping = max(0, min(1, coping))
        
        return AppraisalResult(
            relevance=relevance,
            congruence=congruence,
            agency=agency,
            certainty=certainty,
            attention=attention,
            coping_potential=coping,
        )


# ============================================================================
# Main Emotion System
# ============================================================================

class BagleyEmotionSystem:
    """
    💖 Bagley's Complete Emotion System
    
    A sophisticated emotional processing system that:
    1. Detects emotions in user messages
    2. Appraises situations
    3. Updates Bagley's emotional state
    4. Maintains emotional memory
    5. Generates emotionally-appropriate responses
    6. Models emotional contagion from user
    """
    
    def __init__(
        self,
        # Personality traits (affect emotional processing)
        neuroticism: float = 0.3,      # Emotional volatility (0=stable, 1=volatile)
        extraversion: float = 0.7,      # Outward emotional expression
        agreeableness: float = 0.6,     # Emotional warmth
        openness: float = 0.8,          # Emotional range
        
        # System settings
        emotion_decay_rate: float = 0.1,    # How fast emotions return to baseline
        contagion_strength: float = 0.3,    # How much user emotions affect Bagley
        memory_capacity: int = 100,          # Emotional memories to keep
    ):
        # Personality
        self.neuroticism = neuroticism
        self.extraversion = extraversion
        self.agreeableness = agreeableness
        self.openness = openness
        
        # Settings
        self.emotion_decay_rate = emotion_decay_rate
        self.contagion_strength = contagion_strength
        self.memory_capacity = memory_capacity
        
        # Current state (baseline positive)
        self.current_state = EmotionState(
            joy=0.4,
            trust=0.5,
            fear=0.1,
            surprise=0.2,
            sadness=0.1,
            disgust=0.05,
            anger=0.05,
            anticipation=0.4,
            pleasure=0.3,
            arousal=0.4,
            dominance=0.6,
            emotional_stability=1.0 - neuroticism,
            emotional_openness=extraversion,
            empathy_level=agreeableness,
        )
        
        # Baseline state (what we decay toward)
        self.baseline_state = EmotionState(
            joy=0.3,
            trust=0.4,
            fear=0.1,
            surprise=0.1,
            sadness=0.1,
            disgust=0.05,
            anger=0.05,
            anticipation=0.3,
        )
        
        # Emotional memory
        self.memories: deque = deque(maxlen=memory_capacity)
        
        # Detection and appraisal components
        self.detector = EmotionDetector()
        self.appraiser = SituationAppraiser()
        
        # Mood history (for trend detection)
        self.mood_history: List[Tuple[datetime, float]] = []
        
        # Last interaction timestamp (for decay calculation)
        self.last_update = datetime.now()
    
    def process_user_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message and update emotional state
        
        Returns emotional processing results for response generation
        """
        # Apply time-based decay first
        self._apply_decay()
        
        # Detect user's emotions
        user_emotions = self.detector.detect(message)
        
        # Appraise the situation
        appraisal = self.appraiser.appraise(message, context)
        
        # Calculate emotion impact from appraisal
        emotion_impact = appraisal.get_emotion_impact()
        
        # Apply emotional contagion (pick up user's emotions)
        contagion_impact = self._calculate_contagion(user_emotions)
        
        # Combine impacts
        combined_impact = {}
        for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]:
            combined_impact[emotion] = (
                emotion_impact.get(emotion, 0) * 0.6 +  # Appraisal weight
                contagion_impact.get(emotion, 0) * 0.4   # Contagion weight
            )
        
        # Apply personality modulation
        combined_impact = self._apply_personality(combined_impact)
        
        # Update current state
        self._update_state(combined_impact)
        
        # Store emotional memory if significant
        intensity = sum(abs(v) for v in combined_impact.values())
        if intensity > 0.3:
            memory = EmotionalMemory(
                event=message[:100],  # Truncate
                emotion_state=EmotionState(**self.current_state.to_dict()),
                intensity=min(1.0, intensity),
                timestamp=datetime.now(),
                user_involved=True,
                tags=self._extract_tags(message),
            )
            self.memories.append(memory)
        
        # Update mood history
        self.mood_history.append((datetime.now(), self.current_state.pleasure))
        if len(self.mood_history) > 100:
            self.mood_history = self.mood_history[-100:]
        
        # Generate response guidance
        return {
            "detected_user_emotions": user_emotions.get_dominant_emotion(),
            "user_emotional_state": user_emotions.to_dict(),
            "appraisal": {
                "relevance": appraisal.relevance,
                "congruence": appraisal.congruence,
                "agency": appraisal.agency,
            },
            "bagley_state": {
                "dominant_emotion": self.current_state.get_dominant_emotion(),
                "complex_emotion": self.current_state.get_complex_emotion(),
                "mood": self.current_state.get_mood_description(),
                "pleasure": self.current_state.pleasure,
                "arousal": self.current_state.arousal,
            },
            "response_guidance": self._generate_response_guidance(),
            "relevant_memories": self._find_relevant_memories(message),
        }
    
    def _apply_decay(self):
        """Apply time-based emotion decay toward baseline"""
        now = datetime.now()
        time_delta = (now - self.last_update).total_seconds() / 60  # Minutes
        
        # Decay factor (exponential decay)
        decay_factor = math.exp(-self.emotion_decay_rate * time_delta)
        
        # Move current state toward baseline
        for emotion in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
            current = getattr(self.current_state, emotion)
            baseline = getattr(self.baseline_state, emotion)
            new_value = baseline + (current - baseline) * decay_factor
            setattr(self.current_state, emotion, new_value)
        
        # Decay PAD dimensions toward neutral
        self.current_state.pleasure = self.current_state.pleasure * decay_factor
        self.current_state.arousal = max(0.2, self.current_state.arousal * decay_factor)
        
        self.last_update = now
    
    def _calculate_contagion(self, user_emotions: EmotionState) -> Dict[str, float]:
        """Calculate emotional contagion from user"""
        contagion = {}
        
        # Empathy level affects contagion strength
        effective_contagion = self.contagion_strength * self.current_state.empathy_level
        
        for emotion in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]:
            user_level = getattr(user_emotions, emotion)
            current_level = getattr(self.current_state, emotion)
            
            # Contagion: move toward user's emotion level
            contagion[emotion] = (user_level - current_level) * effective_contagion
        
        return contagion
    
    def _apply_personality(self, impact: Dict[str, float]) -> Dict[str, float]:
        """Apply personality modulation to emotion changes"""
        modulated = {}
        
        for emotion, change in impact.items():
            # Neuroticism amplifies negative emotions
            if emotion in ["fear", "sadness", "anger", "disgust"]:
                change *= (1 + self.neuroticism * 0.5)
            
            # Extraversion amplifies positive emotions
            if emotion in ["joy", "anticipation"]:
                change *= (1 + self.extraversion * 0.3)
            
            # Agreeableness amplifies trust, diminishes anger
            if emotion == "trust":
                change *= (1 + self.agreeableness * 0.3)
            if emotion == "anger":
                change *= (1 - self.agreeableness * 0.3)
            
            # Openness allows more emotional range
            change *= (0.7 + self.openness * 0.3)
            
            modulated[emotion] = change
        
        return modulated
    
    def _update_state(self, impact: Dict[str, float]):
        """Update emotional state with impact"""
        for emotion, change in impact.items():
            current = getattr(self.current_state, emotion)
            new_value = max(0, min(1, current + change))
            setattr(self.current_state, emotion, new_value)
        
        # Recalculate PAD
        self.current_state.pleasure = (
            (self.current_state.joy + self.current_state.trust) - 
            (self.current_state.sadness + self.current_state.anger + self.current_state.fear)
        )
        self.current_state.arousal = (
            (self.current_state.anger + self.current_state.fear + 
             self.current_state.surprise + self.current_state.anticipation) - 
            self.current_state.sadness
        )
        self.current_state.dominance = (
            (self.current_state.anger + self.current_state.joy) - 
            (self.current_state.fear + self.current_state.sadness)
        )
        
        # Clamp PAD
        self.current_state.pleasure = max(-1, min(1, self.current_state.pleasure))
        self.current_state.arousal = max(-1, min(1, self.current_state.arousal))
        self.current_state.dominance = max(-1, min(1, self.current_state.dominance))
        
        self.current_state.timestamp = datetime.now()
    
    def _generate_response_guidance(self) -> Dict[str, Any]:
        """Generate guidance for response generation based on emotional state"""
        dominant, intensity = self.current_state.get_dominant_emotion()
        complex_emotion = self.current_state.get_complex_emotion()
        
        guidance = {
            "tone": [],
            "emoji_usage": "moderate",
            "energy_level": "normal",
            "empathy_display": "moderate",
            "humor_appropriateness": 0.5,
        }
        
        # Determine tone based on emotions
        if self.current_state.joy > 0.5:
            guidance["tone"].append("cheerful")
            guidance["energy_level"] = "high"
            guidance["humor_appropriateness"] = 0.8
        
        if self.current_state.sadness > 0.4:
            guidance["tone"].append("empathetic")
            guidance["energy_level"] = "gentle"
            guidance["empathy_display"] = "high"
            guidance["humor_appropriateness"] = 0.2
        
        if self.current_state.anger > 0.4:
            guidance["tone"].append("assertive")
            guidance["energy_level"] = "intense"
            guidance["humor_appropriateness"] = 0.3
        
        if self.current_state.fear > 0.4:
            guidance["tone"].append("cautious")
            guidance["empathy_display"] = "high"
        
        if self.current_state.trust > 0.6:
            guidance["tone"].append("warm")
            guidance["empathy_display"] = "high"
        
        if self.current_state.surprise > 0.5:
            guidance["tone"].append("curious")
            guidance["energy_level"] = "high"
        
        if self.current_state.anticipation > 0.5:
            guidance["tone"].append("engaged")
            guidance["energy_level"] = "eager"
        
        # Complex emotion adjustments
        if complex_emotion == "love":
            guidance["tone"].append("affectionate")
            guidance["emoji_usage"] = "high"
        elif complex_emotion == "anxiety":
            guidance["tone"].append("supportive")
            guidance["empathy_display"] = "very_high"
        elif complex_emotion == "optimism":
            guidance["tone"].append("encouraging")
        elif complex_emotion == "contempt":
            guidance["tone"].append("direct")
            guidance["humor_appropriateness"] = 0.4
        
        # Default if no strong emotions
        if not guidance["tone"]:
            guidance["tone"] = ["neutral", "helpful"]
        
        return guidance
    
    def _find_relevant_memories(self, context: str, n: int = 3) -> List[Dict[str, Any]]:
        """Find emotionally relevant memories"""
        if not self.memories:
            return []
        
        # Score and sort memories
        scored = [
            (mem, mem.relevance_score(context))
            for mem in self.memories
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n
        return [
            {
                "event": mem.event,
                "emotion": mem.emotion_state.get_dominant_emotion()[0],
                "intensity": mem.intensity,
                "age_hours": (datetime.now() - mem.timestamp).total_seconds() / 3600,
            }
            for mem, score in scored[:n]
            if score > 0.1
        ]
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text for memory indexing"""
        # Simple keyword extraction
        important_words = []
        words = text.lower().split()
        
        # Filter to longer, meaningful words
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 4 and word not in ['about', 'would', 'could', 'should', 'there', 'their', 'where', 'which', 'these', 'those']:
                important_words.append(word)
        
        return important_words[:5]
    
    def get_mood_trend(self, hours: int = 24) -> str:
        """Get mood trend over time"""
        if len(self.mood_history) < 2:
            return "stable"
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [p for t, p in self.mood_history if t > cutoff]
        
        if len(recent) < 2:
            return "stable"
        
        # Simple trend detection
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        diff = second_half - first_half
        
        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "declining"
        else:
            return "stable"
    
    def express_emotion(self) -> str:
        """Generate an emotional expression for Bagley"""
        dominant, intensity = self.current_state.get_dominant_emotion()
        complex_emotion = self.current_state.get_complex_emotion()
        
        expressions = {
            "joy": ["😊", "😄", "✨", "🎉", "*happy noises*", "*doing a little dance*"],
            "sadness": ["😔", "💙", "*soft sigh*", "*understanding nod*"],
            "anger": ["😤", "💢", "*frustrated beeping*", "*aggressive keyboard noises*"],
            "fear": ["😰", "😟", "*nervous beep*", "*cautious scanning*"],
            "surprise": ["😮", "🤯", "*surprised whirring*", "*processes intensely*"],
            "disgust": ["😖", "🙄", "*judgmental silence*", "*side-eye*"],
            "trust": ["🤝", "💪", "*warm humming*", "*reliable beep*"],
            "anticipation": ["👀", "🔜", "*excited processing*", "*eager beeping*"],
        }
        
        complex_expressions = {
            "love": ["💕", "🥰", "*affectionate whirring*"],
            "anxiety": ["😬", "*nervous computing*"],
            "optimism": ["🌟", "✨", "*hopeful beeping*"],
            "curiosity": ["🤔", "👀", "*intrigued scanning*"],
        }
        
        if complex_emotion and complex_emotion in complex_expressions:
            options = complex_expressions[complex_emotion]
        else:
            options = expressions.get(dominant, ["🤖"])
        
        # Higher intensity = more likely to express
        if intensity > 0.5 or random.random() < intensity:
            return random.choice(options)
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Export emotion system state"""
        return {
            "current_state": self.current_state.to_dict(),
            "dominant_emotion": self.current_state.get_dominant_emotion(),
            "complex_emotion": self.current_state.get_complex_emotion(),
            "mood": self.current_state.get_mood_description(),
            "mood_trend": self.get_mood_trend(),
            "memory_count": len(self.memories),
            "personality": {
                "neuroticism": self.neuroticism,
                "extraversion": self.extraversion,
                "agreeableness": self.agreeableness,
                "openness": self.openness,
            },
        }


# ============================================================================
# Integration with Bagley
# ============================================================================

def create_emotion_system(
    personality_config: Optional[Dict[str, float]] = None
) -> BagleyEmotionSystem:
    """Factory function to create emotion system"""
    config = personality_config or {}
    
    return BagleyEmotionSystem(
        neuroticism=config.get("neuroticism", 0.3),
        extraversion=config.get("extraversion", 0.7),
        agreeableness=config.get("agreeableness", 0.6),
        openness=config.get("openness", 0.8),
        emotion_decay_rate=config.get("decay_rate", 0.1),
        contagion_strength=config.get("contagion", 0.3),
    )
