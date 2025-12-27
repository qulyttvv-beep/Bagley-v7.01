"""
🧠 Bagley Memory System
Infinite context with smart summarization and callback tracking
"""

import asyncio
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in conversation history"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    
    # For callback humor tracking
    topics: List[str] = field(default_factory=list)
    jokes: List[str] = field(default_factory=list)
    memorable: bool = False  # Flag for particularly memorable moments


@dataclass
class ConversationSummary:
    """Summary of a conversation segment"""
    content: str
    original_messages: int
    original_tokens: int
    topics: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class BagleyMemory:
    """
    🧠 Smart Memory System
    
    Handles:
    - Infinite context via intelligent summarization
    - Callback humor tracking (remembering jokes/topics)
    - Cross-session persistent memory
    - Efficient token management
    """
    
    def __init__(
        self,
        max_context: int = 131072,  # 128K default
        summarization_threshold: int = 50000,  # Summarize when exceeding this
        persistent_path: Optional[str] = None
    ):
        self.max_context = max_context
        self.summarization_threshold = summarization_threshold
        self.persistent_path = Path(persistent_path) if persistent_path else None
        
        # Message storage
        self.messages: List[Message] = []
        self.summaries: List[ConversationSummary] = []
        
        # Callback tracking
        self.topics_mentioned: Dict[str, int] = {}  # topic -> count
        self.jokes_made: List[str] = []
        self.memorable_moments: List[Message] = []
        
        # Token tracking
        self.current_token_count = 0
        self._tokenizer = None  # Lazy loaded
        
        # Load persistent memory if exists
        if self.persistent_path and self.persistent_path.exists():
            self._load_persistent()
    
    async def add_message(
        self,
        role: str,
        content: str,
        files: Optional[List[str]] = None,
        images: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to memory"""
        # Count tokens
        token_count = await self._count_tokens(content)
        
        # Extract topics and jokes for callback tracking
        topics = self._extract_topics(content)
        jokes = self._extract_jokes(content)
        
        message = Message(
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
            files=files or [],
            images=images or [],
            topics=topics,
            jokes=jokes,
            memorable=self._is_memorable(content)
        )
        
        self.messages.append(message)
        self.current_token_count += token_count
        
        # Track topics
        for topic in topics:
            self.topics_mentioned[topic] = self.topics_mentioned.get(topic, 0) + 1
        
        # Track jokes
        self.jokes_made.extend(jokes)
        
        # Track memorable moments
        if message.memorable:
            self.memorable_moments.append(message)
        
        # Summarize if exceeding threshold
        if self.current_token_count > self.summarization_threshold:
            await self._summarize_old_messages()
        
        logger.debug(f"Added message: {role}, {token_count} tokens, total: {self.current_token_count}")
    
    async def get_context(self) -> str:
        """Get the full context for model input"""
        context_parts = []
        
        # Include summaries first (oldest context)
        for summary in self.summaries:
            context_parts.append(f"[Previous conversation summary]\n{summary.content}")
        
        # Include callback context
        callback_context = self._get_callback_context()
        if callback_context:
            context_parts.append(f"[Context for callbacks]\n{callback_context}")
        
        # Include recent messages
        for msg in self.messages:
            if msg.role == "user":
                context_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                context_parts.append(f"Bagley: {msg.content}")
            elif msg.role == "system":
                context_parts.append(f"System: {msg.content}")
        
        return "\n\n".join(context_parts)
    
    async def get_context_messages(self) -> List[Dict[str, str]]:
        """Get context as message list format (for chat APIs)"""
        messages = []
        
        # Include summary as system message if exists
        if self.summaries:
            summary_text = "\n\n".join(s.content for s in self.summaries)
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{summary_text}"
            })
        
        # Include recent messages
        for msg in self.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages
    
    def _get_callback_context(self) -> str:
        """Generate context for callback humor"""
        parts = []
        
        # Top mentioned topics
        top_topics = sorted(
            self.topics_mentioned.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        if top_topics:
            parts.append(f"Recurring topics: {', '.join(t[0] for t in top_topics)}")
        
        # Recent jokes
        if self.jokes_made:
            recent_jokes = self.jokes_made[-5:]
            parts.append(f"Recent jokes made: {'; '.join(recent_jokes)}")
        
        # Memorable moments
        if self.memorable_moments:
            moments = [m.content[:100] for m in self.memorable_moments[-3:]]
            parts.append(f"Memorable moments: {'; '.join(moments)}")
        
        return "\n".join(parts)
    
    async def _summarize_old_messages(self) -> None:
        """Summarize old messages to save context space"""
        # Keep the most recent messages, summarize the rest
        messages_to_summarize = []
        messages_to_keep = []
        
        tokens_kept = 0
        target_keep_tokens = self.summarization_threshold // 2
        
        # Work backwards, keeping recent messages
        for msg in reversed(self.messages):
            if tokens_kept < target_keep_tokens:
                messages_to_keep.insert(0, msg)
                tokens_kept += msg.token_count
            else:
                messages_to_summarize.insert(0, msg)
        
        if not messages_to_summarize:
            return
        
        # Create summary
        # In production, this would use the chat model to generate a summary
        summary_content = self._create_basic_summary(messages_to_summarize)
        
        summary = ConversationSummary(
            content=summary_content,
            original_messages=len(messages_to_summarize),
            original_tokens=sum(m.token_count for m in messages_to_summarize),
            topics=list(set(t for m in messages_to_summarize for t in m.topics))
        )
        
        self.summaries.append(summary)
        self.messages = messages_to_keep
        self.current_token_count = tokens_kept
        
        logger.info(f"Summarized {len(messages_to_summarize)} messages into summary")
    
    def _create_basic_summary(self, messages: List[Message]) -> str:
        """Create a basic summary of messages (fallback when model unavailable)"""
        # Extract key points
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]
        
        summary_parts = [
            f"Conversation segment with {len(messages)} messages:",
            f"- User asked about: {', '.join(set(t for m in user_messages for t in m.topics[:3]))}",
            f"- Topics discussed: {', '.join(list(set(t for m in messages for t in m.topics))[:10])}"
        ]
        
        if self.jokes_made:
            summary_parts.append(f"- Jokes/humor involved: Yes")
        
        return "\n".join(summary_parts)
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content for callback tracking"""
        # Simple keyword-based extraction
        # In production, would use NER or the chat model
        topics = []
        
        # Common discussion topics
        topic_keywords = {
            "code": ["code", "programming", "function", "bug", "error"],
            "image": ["image", "picture", "photo", "generate", "art"],
            "video": ["video", "movie", "animation", "clip"],
            "life": ["life", "advice", "help", "rough", "day"],
            "tech": ["computer", "gpu", "ai", "model", "server"],
            "humor": ["joke", "funny", "laugh", "roast"],
            "existential": ["existence", "meaning", "why", "purpose"],
        }
        
        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_jokes(self, content: str) -> List[str]:
        """Extract jokes/humor for callback tracking"""
        jokes = []
        
        # Simple heuristic: look for punchline patterns
        indicators = ["😂", "💀", "🤣", "lmao", "lol", "joke:", "why did"]
        
        content_lower = content.lower()
        if any(ind in content_lower for ind in indicators):
            # Extract a short version as the "joke"
            jokes.append(content[:100] if len(content) > 100 else content)
        
        return jokes
    
    def _is_memorable(self, content: str) -> bool:
        """Determine if a message is particularly memorable"""
        # Heuristics for memorable content
        memorable_indicators = [
            len(content) > 500,  # Long responses
            "🔥" in content and "😈" in content,  # High energy
            "advice" in content.lower(),  # Life advice
            any(word in content.lower() for word in ["remember", "important", "key"]),
        ]
        return any(memorable_indicators)
    
    async def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        # Rough estimate: ~4 chars per token
        # In production, use actual tokenizer
        return len(text) // 4
    
    async def clear(self, keep_personality: bool = True) -> None:
        """Clear conversation memory"""
        self.messages = []
        self.current_token_count = 0
        
        if not keep_personality:
            self.summaries = []
            self.topics_mentioned = {}
            self.jokes_made = []
            self.memorable_moments = []
        
        logger.info("Memory cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "message_count": len(self.messages),
            "summary_count": len(self.summaries),
            "current_tokens": self.current_token_count,
            "max_tokens": self.max_context,
            "topics_tracked": len(self.topics_mentioned),
            "jokes_tracked": len(self.jokes_made),
            "memorable_moments": len(self.memorable_moments),
            "utilization": self.current_token_count / self.max_context
        }
    
    async def save(self, path: Optional[str] = None) -> None:
        """Save memory state to disk"""
        save_path = Path(path) if path else self.persistent_path
        if not save_path:
            return
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "token_count": m.token_count,
                    "topics": m.topics,
                    "memorable": m.memorable
                }
                for m in self.messages
            ],
            "summaries": [
                {
                    "content": s.content,
                    "original_messages": s.original_messages,
                    "original_tokens": s.original_tokens,
                    "topics": s.topics,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in self.summaries
            ],
            "topics_mentioned": self.topics_mentioned,
            "jokes_made": self.jokes_made[-100:],  # Keep last 100
        }
        
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Memory saved to {save_path}")
    
    async def load(self, path: Optional[str] = None) -> None:
        """Load memory state from disk"""
        load_path = Path(path) if path else self.persistent_path
        if not load_path or not load_path.exists():
            return
        
        with open(load_path, "r") as f:
            data = json.load(f)
        
        self.messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
                token_count=m["token_count"],
                topics=m.get("topics", []),
                memorable=m.get("memorable", False)
            )
            for m in data.get("messages", [])
        ]
        
        self.summaries = [
            ConversationSummary(
                content=s["content"],
                original_messages=s["original_messages"],
                original_tokens=s["original_tokens"],
                topics=s.get("topics", []),
                timestamp=datetime.fromisoformat(s["timestamp"])
            )
            for s in data.get("summaries", [])
        ]
        
        self.topics_mentioned = data.get("topics_mentioned", {})
        self.jokes_made = data.get("jokes_made", [])
        
        self.current_token_count = sum(m.token_count for m in self.messages)
        
        logger.info(f"Memory loaded from {load_path}")
    
    def _load_persistent(self) -> None:
        """Load persistent memory on init"""
        asyncio.create_task(self.load())
