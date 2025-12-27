"""
🎨 BagleyDiT Model - Custom Diffusion Transformer for Image Generation
Full implementation with FLUX.1/HiDream inspired architecture
"""

import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bagley.models.image.config import BagleyDiTConfig

logger = logging.getLogger(__name__)


# ==================== Position Embeddings ====================

class Rotary2DPositionEmbedding(nn.Module):
    """2D Rotary Position Embeddings for spatial attention"""
    
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
        # Pre-compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, height: int, width: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Generate 2D RoPE embeddings for given spatial dimensions"""
        # Create position grids
        y_pos = torch.arange(height, device=device, dtype=torch.float32)
        x_pos = torch.arange(width, device=device, dtype=torch.float32)
        
        # Compute frequencies
        y_freqs = torch.outer(y_pos, self.inv_freq.to(device))
        x_freqs = torch.outer(x_pos, self.inv_freq.to(device))
        
        # Combine into 2D grid
        y_freqs = y_freqs.unsqueeze(1).expand(-1, width, -1)  # [H, W, D/2]
        x_freqs = x_freqs.unsqueeze(0).expand(height, -1, -1)  # [H, W, D/2]
        
        freqs = torch.cat([y_freqs, x_freqs], dim=-1)  # [H, W, D]
        freqs = freqs.reshape(-1, freqs.shape[-1])  # [H*W, D]
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        return cos, sin


# ==================== Normalization ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization Zero (AdaLN-Zero)
    Used for conditioning DiT on timestep and text embeddings
    """
    
    def __init__(self, hidden_size: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        # Project condition to scale, shift, gate for both norm layers
        # Output: [shift1, scale1, gate1, shift2, scale2, gate2]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_size * 6)
        )
        
        # Initialize gate to zero for residual learning
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x: Tensor, condition: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Returns modulation parameters for two norm layers"""
        modulation = self.adaLN_modulation(condition)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
        return shift1, scale1, gate1, shift2, scale2, gate2
    
    def modulate(self, x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        """Apply modulation: x * (1 + scale) + shift"""
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ==================== Attention ====================

class DiTAttention(nn.Module):
    """Multi-head self-attention for DiT with RoPE"""
    
    def __init__(self, config: BagleyDiTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden_states: Tensor,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            q, k = self._apply_rope(q, k, rope_cos, rope_sin)
        
        # Compute attention
        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def _apply_rope(
        self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embeddings"""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        # Expand cos/sin for batch and heads
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed


class DiTCrossAttention(nn.Module):
    """Cross-attention for text conditioning"""
    
    def __init__(self, config: BagleyDiTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_attention_dim = config.cross_attention_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.cross_attention_dim, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


# ==================== MoE Feed-Forward ====================

class DiTExpert(nn.Module):
    """Single expert FFN for DiT"""
    
    def __init__(self, hidden_size: int, expert_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, expert_dim)
        self.fc2 = nn.Linear(expert_dim, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTMoERouter(nn.Module):
    """Expert router for DiT MoE layers"""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Load balancing loss
        tokens_per_expert = F.one_hot(expert_indices, self.num_experts).sum(dim=-2).float().mean(0)
        avg_prob_per_expert = router_probs.mean(dim=(0, 1))
        aux_loss = self.num_experts * (tokens_per_expert * avg_prob_per_expert).sum()
        
        return top_k_probs, expert_indices, aux_loss


class DiTMoELayer(nn.Module):
    """Mixture of Experts FFN layer for DiT"""
    
    def __init__(self, config: BagleyDiTConfig):
        super().__init__()
        self.config = config
        self.router = DiTMoERouter(config.hidden_size, config.num_experts, config.num_experts_per_tok)
        self.experts = nn.ModuleList([
            DiTExpert(config.hidden_size, config.expert_hidden_dim)
            for _ in range(config.num_experts)
        ])
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, hidden_dim = x.shape
        
        router_probs, expert_indices, aux_loss = self.router(x)
        
        x_flat = x.view(-1, hidden_dim)
        router_probs_flat = router_probs.view(-1, self.config.num_experts_per_tok)
        expert_indices_flat = expert_indices.view(-1, self.config.num_experts_per_tok)
        
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            mask = (expert_indices_flat == i).any(dim=-1)
            if mask.any():
                token_indices = mask.nonzero(as_tuple=True)[0]
                expert_input = x_flat[token_indices]
                expert_output = expert(expert_input)
                
                expert_mask = (expert_indices_flat[token_indices] == i)
                weights = (router_probs_flat[token_indices] * expert_mask).sum(dim=-1, keepdim=True)
                output[token_indices] += weights * expert_output
        
        return output.view(batch_size, seq_len, hidden_dim), aux_loss


# ==================== DiT Block ====================

class BagleyDiTBlock(nn.Module):
    """Single DiT transformer block with MoE"""
    
    def __init__(self, config: BagleyDiTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # AdaLN conditioning
        self.adaln = AdaLNZero(config.hidden_size, config.hidden_size)
        
        # Self-attention
        self.self_attn = DiTAttention(config)
        
        # Cross-attention for text
        self.cross_attn = DiTCrossAttention(config)
        self.cross_norm = nn.LayerNorm(config.hidden_size)
        
        # Feed-forward (MoE or standard)
        if config.use_moe:
            self.ffn = DiTMoELayer(config)
        else:
            hidden_dim = int(config.hidden_size * config.mlp_ratio)
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, config.hidden_size),
            )
    
    def forward(
        self,
        hidden_states: Tensor,
        condition: Tensor,
        encoder_hidden_states: Tensor,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        
        # Get modulation parameters
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(hidden_states, condition)
        
        # Self-attention block
        residual = hidden_states
        hidden_states = self.adaln.modulate(hidden_states, shift1, scale1)
        hidden_states = self.self_attn(hidden_states, rope_cos, rope_sin)
        hidden_states = residual + gate1.unsqueeze(1) * hidden_states
        
        # Cross-attention block
        residual = hidden_states
        hidden_states = self.cross_norm(hidden_states)
        hidden_states = self.cross_attn(hidden_states, encoder_hidden_states)
        hidden_states = residual + hidden_states
        
        # FFN block
        residual = hidden_states
        hidden_states = self.adaln.modulate(hidden_states, shift2, scale2)
        
        if self.config.use_moe:
            hidden_states, aux_loss = self.ffn(hidden_states)
        else:
            hidden_states = self.ffn(hidden_states)
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        hidden_states = residual + gate2.unsqueeze(1) * hidden_states
        
        return hidden_states, aux_loss


# ==================== Main Model ====================

@dataclass
class BagleyDiTOutput:
    """Output container for BagleyDiT"""
    sample: Tensor
    aux_loss: Optional[Tensor] = None


class BagleyDiT(nn.Module):
    """
    🎨 BagleyDiT - Custom Diffusion Transformer for Image Generation
    
    A fully custom rectified flow DiT with:
    - FLUX.1 inspired rectified flow formulation
    - HiDream-style Sparse MoE for efficient scaling
    - T5-XXL text encoder for superior prompt understanding
    - Multi-resolution support up to 4096x4096
    - 2D RoPE for spatial awareness
    """
    
    def __init__(self, config: BagleyDiTConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
        )
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        
        # Text pooling projection
        self.text_pool_proj = nn.Linear(config.text_hidden_size, config.hidden_size)
        
        # Position embedding
        self.pos_embed = Rotary2DPositionEmbedding(config.head_dim, config.rope_theta)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BagleyDiTBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Final layers
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.final_linear = nn.Linear(config.hidden_size, config.patch_size ** 2 * config.in_channels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Zero-initialize final layer
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        
        logger.info(f"Initialized BagleyDiT with ~{config.total_parameters:,} parameters")
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _get_timestep_embedding(self, timesteps: Tensor, dim: int = 256) -> Tensor:
        """Sinusoidal timestep embeddings"""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(
        self,
        latents: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        pooled_text_embeds: Tensor,
        return_dict: bool = True,
    ) -> BagleyDiTOutput:
        """
        Forward pass for training/denoising.
        
        Args:
            latents: Noisy latents [B, C, H, W]
            timestep: Diffusion timestep [B]
            encoder_hidden_states: Text encoder output [B, L, D]
            pooled_text_embeds: Pooled text embedding [B, D]
        """
        batch_size, channels, height, width = latents.shape
        
        # Patch embedding
        hidden_states = self.patch_embed(latents)  # [B, D, H/p, W/p]
        h_patches = height // self.config.patch_size
        w_patches = width // self.config.patch_size
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B, H*W/p^2, D]
        
        # Timestep conditioning
        t_emb = self._get_timestep_embedding(timestep)
        t_emb = self.time_embed(t_emb)  # [B, D]
        
        # Add pooled text to condition
        text_pool = self.text_pool_proj(pooled_text_embeds)  # [B, D]
        condition = t_emb + text_pool  # [B, D]
        
        # Get 2D RoPE
        rope_cos, rope_sin = self.pos_embed(h_patches, w_patches, hidden_states.device)
        
        # Process through transformer blocks
        total_aux_loss = 0.0
        for block in self.blocks:
            hidden_states, aux_loss = block(
                hidden_states,
                condition,
                encoder_hidden_states,
                rope_cos,
                rope_sin,
            )
            total_aux_loss = total_aux_loss + aux_loss
        
        # Final processing
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.final_linear(hidden_states)  # [B, H*W/p^2, p^2*C]
        
        # Unpatchify
        hidden_states = hidden_states.view(
            batch_size, h_patches, w_patches,
            self.config.patch_size, self.config.patch_size, channels
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        hidden_states = hidden_states.reshape(batch_size, channels, height, width)
        
        if return_dict:
            return BagleyDiTOutput(sample=hidden_states, aux_loss=total_aux_loss)
        return hidden_states
    
    @torch.no_grad()
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Generate images from text prompt.
        
        This is a placeholder - actual implementation requires:
        - Text encoder (T5-XXL)
        - VAE decoder
        - Scheduler for rectified flow sampling
        """
        logger.info(f"Generating image: {prompt[:50]}...")
        # Placeholder - returns empty list
        # Full implementation in pipeline.py
        return []
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> "BagleyDiT":
        """Load model from pretrained weights"""
        import os
        import json
        from safetensors.torch import load_file
        
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
            config = BagleyDiTConfig(**config_dict)
        else:
            config = BagleyDiTConfig()
        
        model = cls(config)
        
        weights_path = os.path.join(path, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict)
        
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        model = model.to(device=device, dtype=dtype_map.get(dtype, torch.bfloat16))
        
        return model
    
    def save_pretrained(self, path: str):
        """Save model to disk"""
        import os
        import json
        from safetensors.torch import save_file
        
        os.makedirs(path, exist_ok=True)
        
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        save_file(state_dict, os.path.join(path, "model.safetensors"))
        
        logger.info(f"BagleyDiT saved to {path}")
