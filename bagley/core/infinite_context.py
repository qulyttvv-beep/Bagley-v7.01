"""
♾️ Infinite Context System
Handles unlimited context length for all models
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextChunk:
    """A chunk of context"""
    tokens: torch.Tensor
    position_start: int
    position_end: int
    summary: Optional[torch.Tensor] = None  # Compressed representation
    importance: float = 1.0


class StreamingKVCache:
    """
    🔄 Streaming KV Cache for Infinite Context
    
    Uses sliding window + compression for unlimited length.
    Based on StreamingLLM / Infinite-LLM techniques.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_cache_size: int = 8192,  # Active window
        sink_size: int = 4,  # Attention sinks (important initial tokens)
        compress_ratio: int = 4,  # Compression for old context
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_cache_size = max_cache_size
        self.sink_size = sink_size
        self.compress_ratio = compress_ratio
        self.device = device
        
        # Initialize caches
        self.k_cache: List[torch.Tensor] = [None] * num_layers
        self.v_cache: List[torch.Tensor] = [None] * num_layers
        
        # Compressed memory for old context
        self.compressed_k: List[torch.Tensor] = [None] * num_layers
        self.compressed_v: List[torch.Tensor] = [None] * num_layers
        
        # Position tracking
        self.position = 0
        self.total_tokens_seen = 0
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value and return full cache.
        
        Implements:
        1. Keep attention sinks (first few tokens)
        2. Sliding window for recent tokens
        3. Compress old tokens periodically
        """
        batch_size = key.shape[0]
        seq_len = key.shape[2]
        
        if self.k_cache[layer_idx] is None:
            # Initialize
            self.k_cache[layer_idx] = key
            self.v_cache[layer_idx] = value
        else:
            # Append new tokens
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], key], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], value], dim=2)
            
            # Check if we need to compress
            current_len = self.k_cache[layer_idx].shape[2]
            
            if current_len > self.max_cache_size:
                self._compress_cache(layer_idx)
        
        # Return full cache (compressed + active)
        full_k = self._get_full_cache(layer_idx, 'k')
        full_v = self._get_full_cache(layer_idx, 'v')
        
        return full_k, full_v
    
    def _compress_cache(self, layer_idx: int):
        """Compress old tokens in cache"""
        k_cache = self.k_cache[layer_idx]
        v_cache = self.v_cache[layer_idx]
        
        current_len = k_cache.shape[2]
        
        # Keep sinks and recent window
        keep_recent = self.max_cache_size // 2
        to_compress = current_len - self.sink_size - keep_recent
        
        if to_compress <= 0:
            return
        
        # Extract parts
        sink_k = k_cache[:, :, :self.sink_size, :]
        sink_v = v_cache[:, :, :self.sink_size, :]
        
        compress_k = k_cache[:, :, self.sink_size:self.sink_size + to_compress, :]
        compress_v = v_cache[:, :, self.sink_size:self.sink_size + to_compress, :]
        
        recent_k = k_cache[:, :, -keep_recent:, :]
        recent_v = v_cache[:, :, -keep_recent:, :]
        
        # Compress middle section (average pooling)
        compressed_len = to_compress // self.compress_ratio
        if compressed_len > 0:
            compress_k = compress_k.view(
                compress_k.shape[0], compress_k.shape[1], 
                compressed_len, self.compress_ratio, compress_k.shape[3]
            ).mean(dim=3)
            compress_v = compress_v.view(
                compress_v.shape[0], compress_v.shape[1],
                compressed_len, self.compress_ratio, compress_v.shape[3]
            ).mean(dim=3)
            
            # Add to compressed storage
            if self.compressed_k[layer_idx] is None:
                self.compressed_k[layer_idx] = compress_k
                self.compressed_v[layer_idx] = compress_v
            else:
                self.compressed_k[layer_idx] = torch.cat([self.compressed_k[layer_idx], compress_k], dim=2)
                self.compressed_v[layer_idx] = torch.cat([self.compressed_v[layer_idx], compress_v], dim=2)
        
        # Update active cache
        self.k_cache[layer_idx] = torch.cat([sink_k, recent_k], dim=2)
        self.v_cache[layer_idx] = torch.cat([sink_v, recent_v], dim=2)
    
    def _get_full_cache(self, layer_idx: int, cache_type: str) -> torch.Tensor:
        """Get full cache including compressed history"""
        if cache_type == 'k':
            active = self.k_cache[layer_idx]
            compressed = self.compressed_k[layer_idx]
        else:
            active = self.v_cache[layer_idx]
            compressed = self.compressed_v[layer_idx]
        
        if compressed is not None:
            return torch.cat([compressed, active], dim=2)
        return active
    
    def clear(self):
        """Clear all caches"""
        for i in range(self.num_layers):
            self.k_cache[i] = None
            self.v_cache[i] = None
            self.compressed_k[i] = None
            self.compressed_v[i] = None
        
        self.position = 0
        self.total_tokens_seen = 0


class InfiniteContextAttention(nn.Module):
    """
    ♾️ Attention with Infinite Context Support
    
    Combines:
    - Sliding window attention (for efficiency)
    - Attention sinks (for stability)
    - Memory compression (for long-term recall)
    - Flash attention (when available)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 4096,
        use_flash: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.use_flash = use_flash
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # RoPE for position encoding
        self.rotary_emb = None  # Would be initialized with RoPE
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[StreamingKVCache] = None,
        layer_idx: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Update cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
        
        # Attention
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's SDPA
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
            )
        else:
            # Manual attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # Sliding window (mask out beyond window)
            if k.shape[2] > self.window_size:
                window_mask = torch.ones(seq_len, k.shape[2], device=scores.device)
                window_mask = torch.triu(window_mask, diagonal=k.shape[2] - self.window_size)
                window_mask = window_mask.bool()
                scores = scores.masked_fill(window_mask, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class ContextMemoryBank:
    """
    🧠 Context Memory Bank
    
    Long-term memory storage for extremely long contexts.
    Uses:
    - Chunked storage
    - Importance scoring
    - Retrieval-augmented attention
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_memories: int = 1000,
        chunk_size: int = 512,
    ):
        self.hidden_size = hidden_size
        self.max_memories = max_memories
        self.chunk_size = chunk_size
        
        self.memories: List[ContextChunk] = []
        self.memory_index = None  # Would use FAISS or similar for retrieval
    
    def add_chunk(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        position_start: int,
    ):
        """Add a context chunk to memory"""
        # Compress to summary
        summary = hidden_states.mean(dim=1)  # Simple pooling
        
        chunk = ContextChunk(
            tokens=tokens.cpu(),
            position_start=position_start,
            position_end=position_start + tokens.shape[1],
            summary=summary.cpu(),
            importance=1.0,
        )
        
        self.memories.append(chunk)
        
        # Prune if needed
        if len(self.memories) > self.max_memories:
            self._prune_memories()
    
    def _prune_memories(self):
        """Remove least important memories"""
        # Sort by importance and keep top N
        self.memories.sort(key=lambda x: x.importance, reverse=True)
        self.memories = self.memories[:self.max_memories]
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> List[ContextChunk]:
        """Retrieve relevant memories"""
        if not self.memories:
            return []
        
        # Compute similarities
        query_flat = query.mean(dim=1)  # [batch, hidden]
        
        similarities = []
        for mem in self.memories:
            sim = F.cosine_similarity(
                query_flat, 
                mem.summary.to(query.device),
                dim=-1
            ).mean().item()
            similarities.append((sim, mem))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in similarities[:top_k]]
    
    def update_importance(self, position: int, importance_delta: float):
        """Update importance of memories near a position"""
        for mem in self.memories:
            if mem.position_start <= position <= mem.position_end:
                mem.importance += importance_delta


class InfiniteContextProcessor:
    """
    ♾️ Main Infinite Context Processor
    
    Handles arbitrarily long inputs by:
    1. Chunking input
    2. Processing with streaming KV cache
    3. Storing summaries in memory bank
    4. Retrieving relevant context on demand
    """
    
    def __init__(
        self,
        model,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        chunk_size: int = 4096,
        max_active_context: int = 8192,
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.max_active_context = max_active_context
        
        # Initialize streaming cache
        self.kv_cache = StreamingKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=hidden_size // num_heads,
            max_cache_size=max_active_context,
        )
        
        # Memory bank for old context
        self.memory_bank = ContextMemoryBank(
            hidden_size=hidden_size,
            max_memories=1000,
            chunk_size=chunk_size,
        )
        
        self.total_tokens_processed = 0
    
    def process_long_input(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """
        Process arbitrarily long input.
        
        Chunks it and processes with streaming cache.
        """
        total_len = input_ids.shape[1]
        outputs = []
        all_hidden_states = [] if return_hidden_states else None
        
        # Process in chunks
        for start in range(0, total_len, self.chunk_size):
            end = min(start + self.chunk_size, total_len)
            chunk = input_ids[:, start:end]
            
            # Process chunk
            with torch.no_grad():
                chunk_output = self._process_chunk(chunk, start)
            
            outputs.append(chunk_output['logits'])
            
            if return_hidden_states:
                all_hidden_states.append(chunk_output['hidden_states'])
            
            # Store in memory bank periodically
            if start > 0 and start % (self.chunk_size * 4) == 0:
                self.memory_bank.add_chunk(
                    tokens=chunk,
                    hidden_states=chunk_output['hidden_states'],
                    position_start=start,
                )
            
            self.total_tokens_processed = end
        
        return {
            'logits': torch.cat(outputs, dim=1),
            'hidden_states': torch.cat(all_hidden_states, dim=1) if all_hidden_states else None,
            'total_tokens': self.total_tokens_processed,
        }
    
    def _process_chunk(
        self,
        chunk: torch.Tensor,
        position: int,
    ) -> Dict[str, torch.Tensor]:
        """Process a single chunk"""
        # This would call the actual model
        # For now, placeholder
        batch_size, seq_len = chunk.shape
        
        # Placeholder output
        return {
            'logits': torch.randn(batch_size, seq_len, 32000, device=chunk.device),
            'hidden_states': torch.randn(batch_size, seq_len, 4096, device=chunk.device),
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate with infinite context support"""
        # Process input
        self.process_long_input(input_ids)
        
        # Generate tokens one at a time
        generated = []
        current_token = input_ids[:, -1:]
        
        for _ in range(max_new_tokens):
            # Get next token logits
            output = self._process_chunk(current_token, self.total_tokens_processed)
            logits = output['logits'][:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            
            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices.gather(-1, next_token)
            
            generated.append(next_token)
            current_token = next_token
            self.total_tokens_processed += 1
            
            # Check for EOS
            if next_token.item() == 2:  # EOS token
                break
        
        return torch.cat(generated, dim=1)
    
    def reset(self):
        """Reset context"""
        self.kv_cache.clear()
        self.memory_bank.memories.clear()
        self.total_tokens_processed = 0
