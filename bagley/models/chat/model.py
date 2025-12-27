"""
🧠 BagleyMoE Model - Custom Mixture-of-Experts Language Model
Full implementation with DeepSeek-R1/Qwen3 inspired architecture
"""

import math
from typing import Optional, Tuple, List, Union, AsyncGenerator
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bagley.models.chat.config import BagleyMoEConfig

logger = logging.getLogger(__name__)


# ==================== Custom Activation ====================

class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(xW) * xV"""
    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        return F.silu(gate) * x


# ==================== RMSNorm ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ==================== Rotary Position Embedding ====================

class YaRNRotaryEmbedding(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) for extended context
    Allows 4x context extension with minimal quality loss
    """
    def __init__(self, config: BagleyMoEConfig):
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.scaling_factor = config.rope_scaling.get("factor", 1.0)
        
        # YaRN parameters
        self.beta_fast = config.rope_scaling.get("beta_fast", 32)
        self.beta_slow = config.rope_scaling.get("beta_slow", 1)
        self.original_max_pos = config.rope_scaling.get(
            "original_max_position_embeddings", 32768
        )
        
        self._build_cache()
    
    def _build_cache(self):
        """Build the sin/cos cache for rotary embeddings"""
        # Compute inverse frequencies with YaRN scaling
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        
        # Apply YaRN frequency scaling
        if self.scaling_factor != 1.0:
            inv_freq = self._yarn_scale_frequencies(inv_freq)
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build position cache
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def _yarn_scale_frequencies(self, inv_freq: Tensor) -> Tensor:
        """Apply YaRN frequency scaling"""
        # Compute frequency bands
        low_freq_factor = self.original_max_pos / (2 * math.pi * self.beta_slow)
        high_freq_factor = self.original_max_pos / (2 * math.pi * self.beta_fast)
        
        freq = 1.0 / inv_freq
        
        # Interpolate between low and high frequency scaling
        smooth = (freq - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth = torch.clamp(smooth, 0, 1)
        
        # Apply scaling
        scaled_inv_freq = inv_freq / self.scaling_factor
        interpolated = (1 - smooth) * inv_freq + smooth * scaled_inv_freq
        
        return interpolated
    
    def forward(self, x: Tensor, position_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Get rotary embeddings for given positions"""
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims of the input"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==================== Attention ====================

class BagleyAttention(nn.Module):
    """
    Grouped-Query Attention with YaRN RoPE
    Supports Flash Attention 2 and sliding window
    """
    def __init__(self, config: BagleyMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rotary_emb = YaRNRotaryEmbedding(config)
        
        # Sliding window
        self.sliding_window = config.sliding_window if config.use_sliding_window else None
        
        # Attention dropout
        self.attention_dropout = config.attention_dropout
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.unsqueeze(1), sin.unsqueeze(1))
        
        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        if use_cache:
            past_key_value = (key_states, value_states)
        
        # Repeat KV for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Compute attention
        # Try Flash Attention if available
        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                query_states, 
                key_states, 
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )
        else:
            # Standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Apply sliding window mask if configured
            if self.sliding_window is not None:
                window_mask = self._create_sliding_window_mask(seq_len, key_states.shape[2])
                attn_weights = attn_weights + window_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value
    
    def _create_sliding_window_mask(self, q_len: int, kv_len: int) -> Tensor:
        """Create sliding window attention mask"""
        mask = torch.full((q_len, kv_len), float('-inf'))
        for i in range(q_len):
            start = max(0, kv_len - q_len + i - self.sliding_window + 1)
            end = kv_len - q_len + i + 1
            mask[i, start:end] = 0
        return mask.unsqueeze(0).unsqueeze(0)


# ==================== MoE Layer ====================

class MoERouter(nn.Module):
    """
    Expert router with load balancing
    Routes tokens to top-k experts based on learned gating
    """
    def __init__(self, config: BagleyMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_coef = config.router_aux_loss_coef
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Route tokens to experts
        
        Returns:
            router_probs: Softmax probabilities for each expert
            expert_indices: Top-k expert indices for each token
            router_loss: Auxiliary load balancing loss
        """
        # Compute router logits
        router_logits = self.gate(hidden_states)  # [batch, seq, num_experts]
        
        # Softmax for probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, expert_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)
        
        # Normalize top-k probs
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss
        router_loss = self._compute_aux_loss(router_probs, expert_indices)
        
        return top_k_probs, expert_indices, router_loss
    
    def _compute_aux_loss(self, router_probs: Tensor, expert_indices: Tensor) -> Tensor:
        """Compute auxiliary load balancing loss"""
        # Fraction of tokens routed to each expert
        num_tokens = router_probs.shape[0] * router_probs.shape[1]
        
        expert_mask = F.one_hot(expert_indices, self.num_experts).sum(dim=-2)
        tokens_per_expert = expert_mask.sum(dim=(0, 1))
        
        # Average probability for each expert
        prob_per_expert = router_probs.mean(dim=(0, 1))
        
        # Load balancing loss
        aux_loss = self.num_experts * (tokens_per_expert * prob_per_expert).sum() / num_tokens
        
        return aux_loss * self.aux_loss_coef


class Expert(nn.Module):
    """Single expert FFN with SwiGLU"""
    def __init__(self, config: BagleyMoEConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.expert_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.expert_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.expert_intermediate_size, config.hidden_size, bias=False)
        self.act = SwiGLU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.up_proj(x), self.gate_proj(x)))


class BagleyMoELayer(nn.Module):
    """
    Mixture of Experts FFN layer
    Includes shared experts (always active) + routed experts
    """
    def __init__(self, config: BagleyMoEConfig):
        super().__init__()
        self.config = config
        
        # Router
        self.router = MoERouter(config)
        
        # Routed experts
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])
        
        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_shared_experts)
        ])
    
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Route tokens
        router_probs, expert_indices, router_loss = self.router(hidden_states)
        
        # Flatten for expert processing
        hidden_flat = hidden_states.view(-1, hidden_dim)
        router_probs_flat = router_probs.view(-1, self.config.num_experts_per_tok)
        expert_indices_flat = expert_indices.view(-1, self.config.num_experts_per_tok)
        
        # Process through routed experts
        expert_outputs = torch.zeros_like(hidden_flat)
        
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            mask = (expert_indices_flat == i).any(dim=-1)
            if mask.any():
                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = hidden_flat[token_indices]
                expert_output = expert(expert_input)
                
                # Weight by router probability
                expert_mask_per_token = (expert_indices_flat[token_indices] == i)
                weights = (router_probs_flat[token_indices] * expert_mask_per_token).sum(dim=-1, keepdim=True)
                
                expert_outputs[token_indices] += weights * expert_output
        
        # Add shared expert contributions
        for shared_expert in self.shared_experts:
            expert_outputs += shared_expert(hidden_flat) / len(self.shared_experts)
        
        # Reshape back
        output = expert_outputs.view(batch_size, seq_len, hidden_dim)
        
        return output, router_loss


# ==================== Transformer Block ====================

class BagleyDecoderLayer(nn.Module):
    """Single transformer decoder layer with MoE"""
    def __init__(self, config: BagleyMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Attention
        self.self_attn = BagleyAttention(config, layer_idx)
        
        # Post-attention norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # MoE FFN
        self.moe = BagleyMoELayer(config)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]], Tensor]:
        
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MoE FFN with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, router_loss


# ==================== Main Model ====================

@dataclass
class BagleyMoEOutput:
    """Output container for BagleyMoE"""
    logits: Tensor
    past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None
    hidden_states: Optional[Tensor] = None
    router_loss: Optional[Tensor] = None
    loss: Optional[Tensor] = None


class BagleyMoEForCausalLM(nn.Module):
    """
    🧠 BagleyMoE - The Complete Custom MoE Language Model
    
    A fully custom Mixture-of-Experts architecture combining:
    - DeepSeek-R1 hybrid thinking modes
    - Qwen3 efficient expert routing
    - Extended context via YaRN RoPE
    - Personality injection capabilities
    """
    
    def __init__(self, config: BagleyMoEConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            BagleyDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Personality embedding (optional)
        if config.personality_embedding_dim > 0:
            self.personality_embed = nn.Embedding(
                config.num_personality_modes, 
                config.personality_embedding_dim
            )
            self.personality_proj = nn.Linear(
                config.personality_embedding_dim, 
                config.hidden_size
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        logger.info(f"Initialized BagleyMoE with {config.total_parameters:,} total parameters")
        logger.info(f"Active parameters per forward: {config.active_parameters:,}")
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        personality_mode: Optional[int] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> BagleyMoEOutput:
        
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Add personality embedding if provided
        if personality_mode is not None and hasattr(self, 'personality_embed'):
            personality_emb = self.personality_embed(
                torch.tensor([personality_mode], device=input_ids.device)
            )
            personality_emb = self.personality_proj(personality_emb)
            hidden_states = hidden_states + personality_emb.unsqueeze(1)
        
        # Build position ids if not provided
        if position_ids is None:
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = 0
            position_ids = torch.arange(
                past_length, past_length + seq_len,
                device=input_ids.device
            ).unsqueeze(0)
        
        # Build causal mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Process through layers
        all_hidden_states = () if output_hidden_states else None
        all_router_losses = []
        new_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_kv, router_loss = layer(
                hidden_states,
                attention_mask=None,  # Causal masking in attention
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            
            if use_cache:
                new_past_key_values.append(present_kv)
            
            all_router_losses.append(router_loss)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Add router loss
            total_router_loss = sum(all_router_losses)
            loss = loss + total_router_loss
        
        return BagleyMoEOutput(
            logits=logits,
            past_key_values=new_past_key_values,
            hidden_states=all_hidden_states,
            router_loss=sum(all_router_losses) if all_router_losses else None,
            loss=loss,
        )
    
    @torch.no_grad()
    async def generate(
        self,
        context: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text completion
        
        Args:
            context: The conversation context
            system_prompt: System prompt for personality
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Penalty for repetition
            stop_sequences: Sequences that stop generation
            
        Returns:
            Generated text
        """
        # This is a placeholder - actual implementation would use the tokenizer
        # and proper generation loop
        raise NotImplementedError("Generation requires tokenizer - use stream_generate")
    
    @torch.no_grad()
    async def stream_generate(
        self,
        context: str,
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation token by token
        
        Yields tokens as they're generated for real-time streaming.
        """
        # Placeholder for streaming generation
        # Actual implementation would:
        # 1. Tokenize input
        # 2. Run forward pass
        # 3. Sample next token
        # 4. Yield decoded token
        # 5. Repeat until done
        
        yield "🔥 "  # Placeholder
        yield "Streaming generation coming soon!"
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
    ) -> "BagleyMoEForCausalLM":
        """Load model from pretrained weights"""
        import os
        from safetensors.torch import load_file
        
        # Load config
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                config_dict = json.load(f)
            config = BagleyMoEConfig(**config_dict)
        else:
            config = BagleyMoEConfig()
        
        # Initialize model
        model = cls(config)
        
        # Load weights
        weights_path = os.path.join(path, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict)
        
        # Move to device
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        model = model.to(device=device, dtype=dtype_map.get(dtype, torch.bfloat16))
        
        # Apply quantization if requested
        if quantization:
            model = cls._apply_quantization(model, quantization)
        
        return model
    
    @staticmethod
    def _apply_quantization(model: nn.Module, method: str) -> nn.Module:
        """Apply quantization to model"""
        if method == "int4":
            # Would use GPTQ or AWQ
            logger.info("Applying INT4 quantization...")
        elif method == "int8":
            # Would use bitsandbytes
            logger.info("Applying INT8 quantization...")
        elif method == "fp8":
            # Would use FP8 quantization
            logger.info("Applying FP8 quantization...")
        
        return model
    
    def save_pretrained(self, path: str):
        """Save model to disk"""
        import os
        import json
        from safetensors.torch import save_file
        
        os.makedirs(path, exist_ok=True)
        
        # Save config
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Save weights
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        save_file(state_dict, os.path.join(path, "model.safetensors"))
        
        logger.info(f"Model saved to {path}")
