"""
🔮 Long-term Memory System
==========================

Persistent memory across conversations.
Semantic search, importance weighting, forgetting curves.

FEATURES:
1. Episodic Memory - Remember specific events/conversations
2. Semantic Memory - General knowledge and facts
3. Working Memory - Active context
4. Importance Scoring - What's worth remembering
5. Memory Consolidation - Sleep-like processing
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory"""
    EPISODIC = "episodic"      # Specific events/conversations
    SEMANTIC = "semantic"       # General facts/knowledge
    PROCEDURAL = "procedural"   # How to do things
    EMOTIONAL = "emotional"     # Emotional associations


class MemoryStrength(Enum):
    """Memory strength levels"""
    VIVID = "vivid"            # Very recent or important
    STRONG = "strong"          # Well-remembered
    MODERATE = "moderate"      # Normal memory
    FADING = "fading"          # Starting to forget
    FAINT = "faint"            # Almost forgotten


@dataclass
class Memory:
    """A single memory unit"""
    id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime
    importance: float  # 0.0 to 1.0
    emotional_valence: float  # -1.0 to 1.0 (negative to positive)
    emotional_arousal: float  # 0.0 to 1.0 (calm to excited)
    tags: List[str] = field(default_factory=list)
    associations: Dict[str, float] = field(default_factory=dict)  # memory_id -> strength
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    context: Optional[str] = None
    user_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def get_strength(self) -> MemoryStrength:
        """Calculate current memory strength"""
        now = datetime.now()
        age_hours = (now - self.timestamp).total_seconds() / 3600
        
        # Ebbinghaus forgetting curve with importance modifier
        # R = e^(-t/S) where S is stability (affected by importance, access, emotion)
        stability = 24 * (1 + self.importance) * (1 + self.access_count * 0.1) * (1 + abs(self.emotional_valence) * 0.5)
        retention = math.exp(-age_hours / stability)
        
        if retention > 0.8:
            return MemoryStrength.VIVID
        elif retention > 0.6:
            return MemoryStrength.STRONG
        elif retention > 0.4:
            return MemoryStrength.MODERATE
        elif retention > 0.2:
            return MemoryStrength.FADING
        else:
            return MemoryStrength.FAINT
    
    def get_retrieval_score(self, query_embedding: Optional[List[float]] = None) -> float:
        """Calculate retrieval score for ranking"""
        strength = self.get_strength()
        strength_scores = {
            MemoryStrength.VIVID: 1.0,
            MemoryStrength.STRONG: 0.8,
            MemoryStrength.MODERATE: 0.6,
            MemoryStrength.FADING: 0.4,
            MemoryStrength.FAINT: 0.2,
        }
        
        base_score = strength_scores[strength]
        
        # Boost by importance
        score = base_score * (0.5 + self.importance * 0.5)
        
        # Boost by emotional intensity
        emotional_intensity = abs(self.emotional_valence) * self.emotional_arousal
        score *= (1 + emotional_intensity * 0.3)
        
        # Semantic similarity if embeddings available
        if query_embedding and self.embedding:
            similarity = self._cosine_similarity(query_embedding, self.embedding)
            score *= (1 + similarity)
        
        return min(1.0, score)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


@dataclass
class WorkingMemory:
    """
    Working memory - active context
    
    Limited capacity (like human working memory 7±2 items).
    Most relevant items for current task.
    """
    capacity: int = 7
    items: List[Memory] = field(default_factory=list)
    focus: Optional[str] = None  # Current focus/topic
    
    def add(self, memory: Memory) -> Optional[Memory]:
        """Add item, returns evicted item if at capacity"""
        evicted = None
        
        if len(self.items) >= self.capacity:
            # Evict least relevant item
            self.items.sort(key=lambda m: m.get_retrieval_score(), reverse=True)
            evicted = self.items.pop()
        
        self.items.insert(0, memory)
        return evicted
    
    def get_context(self) -> str:
        """Get working memory as context string"""
        if not self.items:
            return ""
        
        context_parts = []
        if self.focus:
            context_parts.append(f"Current focus: {self.focus}")
        
        for i, mem in enumerate(self.items[:5]):  # Top 5 most relevant
            context_parts.append(f"- {mem.content[:200]}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear working memory"""
        self.items.clear()
        self.focus = None


class LongTermMemory:
    """
    🧠 Long-term Memory Store
    
    Persistent storage for memories with:
    - Semantic search
    - Importance-based retention
    - Forgetting curves
    - Memory consolidation
    """
    
    def __init__(
        self,
        max_memories: int = 10000,
        consolidation_threshold: int = 100,  # Memories before consolidation
        min_importance_to_keep: float = 0.1,
    ):
        self.max_memories = max_memories
        self.consolidation_threshold = consolidation_threshold
        self.min_importance_to_keep = min_importance_to_keep
        
        # Memory stores by type
        self.memories: Dict[str, Memory] = {}
        self.by_type: Dict[MemoryType, List[str]] = defaultdict(list)
        self.by_tag: Dict[str, List[str]] = defaultdict(list)
        self.by_user: Dict[str, List[str]] = defaultdict(list)
        
        # Working memory
        self.working_memory = WorkingMemory()
        
        # Memory counter
        self.memory_count = 0
        self.memories_since_consolidation = 0
    
    def _generate_id(self, content: str) -> str:
        """Generate unique memory ID"""
        self.memory_count += 1
        hash_input = f"{content}:{datetime.now().isoformat()}:{self.memory_count}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        emotional_arousal: float = 0.5,
        tags: Optional[List[str]] = None,
        context: Optional[str] = None,
        user_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> Memory:
        """Store a new memory"""
        memory_id = self._generate_id(content)
        
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            tags=tags or [],
            context=context,
            user_id=user_id,
            embedding=embedding,
        )
        
        # Store in main dict
        self.memories[memory_id] = memory
        
        # Index by type
        self.by_type[memory_type].append(memory_id)
        
        # Index by tags
        for tag in memory.tags:
            self.by_tag[tag.lower()].append(memory_id)
        
        # Index by user
        if user_id:
            self.by_user[user_id].append(memory_id)
        
        # Check for consolidation
        self.memories_since_consolidation += 1
        if self.memories_since_consolidation >= self.consolidation_threshold:
            self._consolidate()
        
        # Add to working memory
        self.working_memory.add(memory)
        
        logger.debug(f"Stored memory {memory_id}: {content[:50]}...")
        
        return memory
    
    def recall(
        self,
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 10,
        min_strength: MemoryStrength = MemoryStrength.FAINT,
    ) -> List[Memory]:
        """
        Recall memories matching criteria
        
        Returns memories sorted by relevance.
        """
        # Start with all memories or filtered set
        candidates = set(self.memories.keys())
        
        # Filter by type
        if memory_type:
            type_memories = set(self.by_type.get(memory_type, []))
            candidates &= type_memories
        
        # Filter by tags
        if tags:
            for tag in tags:
                tag_memories = set(self.by_tag.get(tag.lower(), []))
                candidates &= tag_memories
        
        # Filter by user
        if user_id:
            user_memories = set(self.by_user.get(user_id, []))
            candidates &= user_memories
        
        # Filter by time
        if time_range:
            start, end = time_range
            candidates = {
                mid for mid in candidates
                if start <= self.memories[mid].timestamp <= end
            }
        
        # Filter by strength
        strength_order = [
            MemoryStrength.FAINT, MemoryStrength.FADING,
            MemoryStrength.MODERATE, MemoryStrength.STRONG, MemoryStrength.VIVID
        ]
        min_strength_idx = strength_order.index(min_strength)
        
        filtered = []
        for mid in candidates:
            memory = self.memories[mid]
            strength = memory.get_strength()
            if strength_order.index(strength) >= min_strength_idx:
                filtered.append(memory)
        
        # Score and rank
        scored = []
        for memory in filtered:
            score = memory.get_retrieval_score(query_embedding)
            
            # Keyword matching if query provided
            if query:
                query_words = set(query.lower().split())
                content_words = set(memory.content.lower().split())
                overlap = len(query_words & content_words)
                score *= (1 + overlap * 0.1)
            
            scored.append((memory, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Update access counts and times
        results = []
        for memory, _ in scored[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            results.append(memory)
            
            # Add to working memory
            self.working_memory.add(memory)
        
        return results
    
    def associate(
        self,
        memory_id_1: str,
        memory_id_2: str,
        strength: float = 0.5,
    ):
        """Create association between memories"""
        if memory_id_1 in self.memories and memory_id_2 in self.memories:
            self.memories[memory_id_1].associations[memory_id_2] = strength
            self.memories[memory_id_2].associations[memory_id_1] = strength
    
    def get_associated(
        self,
        memory_id: str,
        min_strength: float = 0.3,
        limit: int = 5,
    ) -> List[Memory]:
        """Get memories associated with given memory"""
        if memory_id not in self.memories:
            return []
        
        memory = self.memories[memory_id]
        associated = []
        
        for assoc_id, strength in memory.associations.items():
            if strength >= min_strength and assoc_id in self.memories:
                associated.append((self.memories[assoc_id], strength))
        
        associated.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in associated[:limit]]
    
    def _consolidate(self):
        """
        Memory consolidation (like sleep processing)
        
        - Strengthens important memories
        - Weakens unimportant ones
        - Creates associations between related memories
        - Removes very faint memories
        """
        logger.info("Running memory consolidation...")
        self.memories_since_consolidation = 0
        
        # Remove faint, unimportant memories
        to_remove = []
        for memory_id, memory in self.memories.items():
            strength = memory.get_strength()
            if strength == MemoryStrength.FAINT and memory.importance < self.min_importance_to_keep:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            self._remove_memory(memory_id)
        
        # Enforce max capacity
        if len(self.memories) > self.max_memories:
            # Sort by retrieval score
            sorted_memories = sorted(
                self.memories.values(),
                key=lambda m: m.get_retrieval_score(),
                reverse=True
            )
            
            # Keep top N
            to_keep = {m.id for m in sorted_memories[:self.max_memories]}
            to_remove = [mid for mid in self.memories if mid not in to_keep]
            
            for memory_id in to_remove:
                self._remove_memory(memory_id)
        
        # Strengthen frequently accessed memories
        for memory in self.memories.values():
            if memory.access_count > 3:
                memory.importance = min(1.0, memory.importance + 0.1)
        
        logger.info(f"Consolidation complete. {len(to_remove)} memories removed. {len(self.memories)} remaining.")
    
    def _remove_memory(self, memory_id: str):
        """Remove a memory and its indices"""
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # Remove from type index
        if memory_id in self.by_type[memory.memory_type]:
            self.by_type[memory.memory_type].remove(memory_id)
        
        # Remove from tag indices
        for tag in memory.tags:
            if memory_id in self.by_tag[tag.lower()]:
                self.by_tag[tag.lower()].remove(memory_id)
        
        # Remove from user index
        if memory.user_id and memory_id in self.by_user[memory.user_id]:
            self.by_user[memory.user_id].remove(memory_id)
        
        # Remove associations
        for assoc_id in memory.associations:
            if assoc_id in self.memories:
                self.memories[assoc_id].associations.pop(memory_id, None)
        
        # Remove memory
        del self.memories[memory_id]
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Build user profile from memories"""
        user_memories = self.recall(user_id=user_id, limit=100)
        
        if not user_memories:
            return {"user_id": user_id, "memory_count": 0}
        
        # Analyze memories
        topics = defaultdict(int)
        sentiments = []
        
        for mem in user_memories:
            for tag in mem.tags:
                topics[tag] += 1
            sentiments.append(mem.emotional_valence)
        
        # Top topics
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        return {
            "user_id": user_id,
            "memory_count": len(user_memories),
            "top_topics": [t[0] for t in top_topics],
            "average_sentiment": avg_sentiment,
            "first_interaction": min(m.timestamp for m in user_memories),
            "last_interaction": max(m.timestamp for m in user_memories),
        }
    
    def save_to_file(self, filepath: str):
        """Save memories to file"""
        data = {
            "metadata": {
                "memory_count": len(self.memories),
                "saved_at": datetime.now().isoformat(),
            },
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "memory_type": m.memory_type.value,
                    "timestamp": m.timestamp.isoformat(),
                    "importance": m.importance,
                    "emotional_valence": m.emotional_valence,
                    "emotional_arousal": m.emotional_arousal,
                    "tags": m.tags,
                    "associations": m.associations,
                    "access_count": m.access_count,
                    "context": m.context,
                    "user_id": m.user_id,
                }
                for m in self.memories.values()
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.memories)} memories to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load memories from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for mem_data in data.get("memories", []):
            memory = Memory(
                id=mem_data["id"],
                content=mem_data["content"],
                memory_type=MemoryType(mem_data["memory_type"]),
                timestamp=datetime.fromisoformat(mem_data["timestamp"]),
                importance=mem_data["importance"],
                emotional_valence=mem_data["emotional_valence"],
                emotional_arousal=mem_data["emotional_arousal"],
                tags=mem_data.get("tags", []),
                associations=mem_data.get("associations", {}),
                access_count=mem_data.get("access_count", 0),
                context=mem_data.get("context"),
                user_id=mem_data.get("user_id"),
            )
            
            self.memories[memory.id] = memory
            self.by_type[memory.memory_type].append(memory.id)
            
            for tag in memory.tags:
                self.by_tag[tag.lower()].append(memory.id)
            
            if memory.user_id:
                self.by_user[memory.user_id].append(memory.id)
        
        logger.info(f"Loaded {len(self.memories)} memories from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        type_counts = {t.value: len(ids) for t, ids in self.by_type.items()}
        
        strength_counts = defaultdict(int)
        for mem in self.memories.values():
            strength_counts[mem.get_strength().value] += 1
        
        return {
            "total_memories": len(self.memories),
            "by_type": type_counts,
            "by_strength": dict(strength_counts),
            "unique_tags": len(self.by_tag),
            "unique_users": len(self.by_user),
            "working_memory_size": len(self.working_memory.items),
        }


class MemorySystem:
    """
    🔮 Unified Memory System
    
    Combines working memory and long-term memory
    with automatic importance scoring and consolidation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.long_term = LongTermMemory(
            max_memories=config.get("max_memories", 10000),
            consolidation_threshold=config.get("consolidation_threshold", 100),
            min_importance_to_keep=config.get("min_importance", 0.1),
        )
        
        # Importance scoring parameters
        self.base_importance = config.get("base_importance", 0.5)
        self.emotional_weight = config.get("emotional_weight", 0.3)
        self.novelty_weight = config.get("novelty_weight", 0.2)
    
    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        emotional_valence: float = 0.0,
        emotional_arousal: float = 0.5,
        tags: Optional[List[str]] = None,
        context: Optional[str] = None,
        user_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> Memory:
        """
        Remember something with automatic importance scoring
        """
        # Calculate importance
        importance = self._calculate_importance(
            content=content,
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            tags=tags,
        )
        
        return self.long_term.store(
            content=content,
            memory_type=memory_type,
            importance=importance,
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            tags=tags,
            context=context,
            user_id=user_id,
            embedding=embedding,
        )
    
    def recall(self, query: str, **kwargs) -> List[Memory]:
        """Recall memories relevant to query"""
        return self.long_term.recall(query=query, **kwargs)
    
    def _calculate_importance(
        self,
        content: str,
        emotional_valence: float,
        emotional_arousal: float,
        tags: Optional[List[str]],
    ) -> float:
        """Calculate importance score for memory"""
        importance = self.base_importance
        
        # Emotional intensity increases importance
        emotional_intensity = abs(emotional_valence) * emotional_arousal
        importance += emotional_intensity * self.emotional_weight
        
        # More tags = potentially more important
        if tags:
            importance += len(tags) * 0.02
        
        # Longer content might be more important (up to a point)
        word_count = len(content.split())
        if 20 <= word_count <= 200:
            importance += 0.1
        
        return min(1.0, importance)
    
    def get_context_for_response(
        self,
        current_input: str,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """Get relevant context for generating a response"""
        # Get working memory context
        working_context = self.long_term.working_memory.get_context()
        
        # Get relevant long-term memories
        relevant = self.long_term.recall(
            query=current_input,
            user_id=user_id,
            limit=limit,
            min_strength=MemoryStrength.FADING,
        )
        
        memory_context = "\n".join(
            f"[{m.memory_type.value}] {m.content[:150]}"
            for m in relevant
        )
        
        if working_context and memory_context:
            return f"Working Memory:\n{working_context}\n\nRelevant Memories:\n{memory_context}"
        elif working_context:
            return f"Working Memory:\n{working_context}"
        elif memory_context:
            return f"Relevant Memories:\n{memory_context}"
        else:
            return ""
    
    def save(self, filepath: str):
        """Save memory to file"""
        self.long_term.save_to_file(filepath)
    
    def load(self, filepath: str):
        """Load memory from file"""
        self.long_term.load_from_file(filepath)


def create_memory_system(config: Optional[Dict[str, Any]] = None) -> MemorySystem:
    """Factory function to create memory system"""
    return MemorySystem(config)
