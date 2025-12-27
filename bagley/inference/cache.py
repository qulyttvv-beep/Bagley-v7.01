"""
🗄️ KV Cache Implementation
Efficient key-value caching for autoregressive generation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class CacheConfig:
    """KV Cache configuration"""
    max_seq_len: int = 8192
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


class KVCache:
    """
    Simple KV Cache for transformer inference.
    
    Stores key/value tensors for each layer to avoid recomputation.
    """
    
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Current position in cache
        self.seq_len = 0
        
        # Pre-allocate cache tensors
        # Shape: [batch, num_heads, max_seq_len, head_dim]
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        
        self.k_cache = [
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value and return full cached tensors.
        
        Args:
            layer_idx: Which layer's cache to update
            key: New key tensor [batch, num_heads, seq_len, head_dim]
            value: New value tensor [batch, num_heads, seq_len, head_dim]
            
        Returns:
            Full cached key and value tensors
        """
        new_seq_len = key.shape[2]
        
        # Update cache
        self.k_cache[layer_idx][:, :, self.seq_len:self.seq_len + new_seq_len, :] = key
        self.v_cache[layer_idx][:, :, self.seq_len:self.seq_len + new_seq_len, :] = value
        
        # Return full cached tensors up to current position
        end_pos = self.seq_len + new_seq_len
        
        return (
            self.k_cache[layer_idx][:, :, :end_pos, :],
            self.v_cache[layer_idx][:, :, :end_pos, :],
        )
    
    def advance(self, num_tokens: int = 1):
        """Advance the sequence position"""
        self.seq_len += num_tokens
    
    def reset(self):
        """Reset cache to empty state"""
        self.seq_len = 0
        # Optionally zero out tensors (not strictly necessary)
        for k, v in zip(self.k_cache, self.v_cache):
            k.zero_()
            v.zero_()
    
    def get_seq_len(self) -> int:
        """Get current sequence length in cache"""
        return self.seq_len
    
    @property
    def memory_usage_mb(self) -> float:
        """Calculate memory usage in MB"""
        bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        total_elements = (
            2 *  # k and v
            self.num_layers *
            self.batch_size *
            self.num_heads *
            self.max_seq_len *
            self.head_dim
        )
        return (total_elements * bytes_per_element) / (1024 * 1024)


class PagedKVCache:
    """
    Paged KV Cache for efficient memory management.
    
    Uses paging to handle variable-length sequences efficiently.
    Inspired by vLLM's PagedAttention.
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        page_size: int = 16,  # Tokens per page
        num_pages: int = 512,  # Total pages in pool
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.num_pages = num_pages
        self.dtype = dtype
        self.device = device
        
        # Page pool: [num_pages, 2, num_layers, num_heads, page_size, head_dim]
        # 2 is for key and value
        self.page_pool = torch.zeros(
            (num_pages, 2, num_layers, num_heads, page_size, head_dim),
            dtype=dtype,
            device=device,
        )
        
        # Track which pages are allocated
        self.free_pages = list(range(num_pages))
        self.allocated_pages: dict = {}  # sequence_id -> list of page indices
    
    def allocate_sequence(self, sequence_id: int, num_tokens: int) -> List[int]:
        """Allocate pages for a new sequence"""
        num_pages_needed = math.ceil(num_tokens / self.page_size)
        
        if len(self.free_pages) < num_pages_needed:
            raise RuntimeError(f"Not enough free pages. Need {num_pages_needed}, have {len(self.free_pages)}")
        
        pages = [self.free_pages.pop() for _ in range(num_pages_needed)]
        self.allocated_pages[sequence_id] = pages
        
        return pages
    
    def free_sequence(self, sequence_id: int):
        """Free pages for a completed sequence"""
        if sequence_id in self.allocated_pages:
            pages = self.allocated_pages.pop(sequence_id)
            self.free_pages.extend(pages)
    
    def get_kv(
        self,
        sequence_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached key/value for a sequence and layer"""
        pages = self.allocated_pages.get(sequence_id, [])
        
        if not pages:
            return None, None
        
        # Gather from pages
        keys = []
        values = []
        
        for page_idx in pages:
            keys.append(self.page_pool[page_idx, 0, layer_idx])
            values.append(self.page_pool[page_idx, 1, layer_idx])
        
        return torch.cat(keys, dim=1), torch.cat(values, dim=1)
    
    def update_kv(
        self,
        sequence_id: int,
        layer_idx: int,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Update cache at a specific position"""
        pages = self.allocated_pages[sequence_id]
        page_idx = position // self.page_size
        offset = position % self.page_size
        
        if page_idx >= len(pages):
            # Need to allocate more pages
            new_page = self.free_pages.pop()
            pages.append(new_page)
        
        actual_page = pages[page_idx]
        self.page_pool[actual_page, 0, layer_idx, :, offset, :] = key
        self.page_pool[actual_page, 1, layer_idx, :, offset, :] = value


class SlidingWindowCache(KVCache):
    """
    Sliding window KV cache for models with local attention.
    Only keeps the last `window_size` tokens.
    """
    
    def __init__(
        self,
        batch_size: int,
        window_size: int,  # Instead of max_seq_len
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__(
            batch_size=batch_size,
            max_seq_len=window_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        self.window_size = window_size
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update with sliding window behavior"""
        new_seq_len = key.shape[2]
        
        if self.seq_len + new_seq_len > self.window_size:
            # Need to shift cache
            shift = self.seq_len + new_seq_len - self.window_size
            
            # Shift existing cache
            self.k_cache[layer_idx] = torch.roll(self.k_cache[layer_idx], -shift, dims=2)
            self.v_cache[layer_idx] = torch.roll(self.v_cache[layer_idx], -shift, dims=2)
            
            self.seq_len = max(0, self.seq_len - shift)
        
        return super().update(layer_idx, key, value)
