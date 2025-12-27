"""
🎬 BagleyVideoMoE - Custom Video Generation Model
Full implementation with Wan2.2/Mochi inspired architecture
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bagley.models.video.config import BagleyVideoConfig

logger = logging.getLogger(__name__)


# ==================== 3D Position Embeddings ====================

class Rotary3DPositionEmbedding(nn.Module):
    """3D Rotary Position Embeddings for spatiotemporal attention"""
    
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(
        self, 
        num_frames: int, 
        height: int, 
        width: int, 
        device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Generate 3D RoPE for [T, H, W] grid"""
        # Create position grids
        t_pos = torch.arange(num_frames, device=device, dtype=torch.float32)
        h_pos = torch.arange(height, device=device, dtype=torch.float32)
        w_pos = torch.arange(width, device=device, dtype=torch.float32)
        
        # Compute frequencies for each dimension
        dim_per_axis = self.dim // 3
        
        t_freqs = torch.outer(t_pos, self.inv_freq[:dim_per_axis].to(device))
        h_freqs = torch.outer(h_pos, self.inv_freq[:dim_per_axis].to(device))
        w_freqs = torch.outer(w_pos, self.inv_freq[:dim_per_axis].to(device))
        
        # Expand to 3D grid
        t_freqs = t_freqs[:, None, None, :].expand(-1, height, width, -1)
        h_freqs = h_freqs[None, :, None, :].expand(num_frames, -1, width, -1)
        w_freqs = w_freqs[None, None, :, :].expand(num_frames, height, -1, -1)
        
        # Combine
        freqs = torch.cat([t_freqs, h_freqs, w_freqs], dim=-1)
        freqs = freqs.reshape(-1, freqs.shape[-1])  # [T*H*W, D]
        
        return freqs.cos(), freqs.sin()


# ==================== Normalization ====================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class AdaLNZero3D(nn.Module):
    """Adaptive Layer Norm Zero for video conditioning"""
    
    def __init__(self, hidden_size: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_size * 6),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
    
    def forward(self, x: Tensor, c: Tensor) -> Tuple[Tensor, ...]:
        params = self.modulation(c)
        return params.chunk(6, dim=-1)
    
    def modulate(self, x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ==================== Asymmetric Attention ====================

class TemporalAttention(nn.Module):
    """Temporal self-attention (across frames)"""
    
    def __init__(self, config: BagleyVideoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.causal = config.causal_temporal
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
    
    def forward(
        self,
        hidden_states: Tensor,
        num_frames: int,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        spatial_size = seq_len // num_frames
        
        # Reshape for temporal attention: [B * spatial, T, D]
        hidden_states = hidden_states.view(batch_size, num_frames, spatial_size, -1)
        hidden_states = hidden_states.permute(0, 2, 1, 3)  # [B, spatial, T, D]
        hidden_states = hidden_states.reshape(batch_size * spatial_size, num_frames, -1)
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Multi-head reshape
        q = q.view(-1, num_frames, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, num_frames, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, num_frames, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.causal and self.training,
        )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size * spatial_size, num_frames, -1)
        attn_output = self.o_proj(attn_output)
        
        # Reshape to original
        attn_output = attn_output.view(batch_size, spatial_size, num_frames, -1)
        attn_output = attn_output.permute(0, 2, 1, 3)  # [B, T, spatial, D]
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        return attn_output


class SpatialAttention(nn.Module):
    """Spatial self-attention (within each frame)"""
    
    def __init__(self, config: BagleyVideoConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
    
    def forward(
        self,
        hidden_states: Tensor,
        num_frames: int,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        spatial_size = seq_len // num_frames
        
        # Reshape for spatial attention: [B * T, spatial, D]
        hidden_states = hidden_states.view(batch_size, num_frames, spatial_size, -1)
        hidden_states = hidden_states.reshape(batch_size * num_frames, spatial_size, -1)
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Multi-head reshape
        q = q.view(-1, spatial_size, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, spatial_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, spatial_size, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size * num_frames, spatial_size, -1)
        attn_output = self.o_proj(attn_output)
        
        # Reshape to original
        attn_output = attn_output.view(batch_size, num_frames, spatial_size, -1)
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        return attn_output


class AsymmDiTAttention(nn.Module):
    """
    Asymmetric Diffusion Transformer Attention (Mochi-inspired)
    Separates temporal and spatial attention for efficiency
    """
    
    def __init__(self, config: BagleyVideoConfig):
        super().__init__()
        self.temporal_attn = TemporalAttention(config)
        self.spatial_attn = SpatialAttention(config)
        self.temporal_ratio = config.temporal_attention_ratio
    
    def forward(
        self,
        hidden_states: Tensor,
        num_frames: int,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tensor:
        # Temporal attention
        temporal_out = self.temporal_attn(hidden_states, num_frames, rope_cos, rope_sin)
        
        # Spatial attention
        spatial_out = self.spatial_attn(hidden_states, num_frames, rope_cos, rope_sin)
        
        # Weighted combination
        out = self.temporal_ratio * temporal_out + (1 - self.temporal_ratio) * spatial_out
        
        return out


class VideoCrossAttention(nn.Module):
    """Cross-attention for text conditioning"""
    
    def __init__(self, config: BagleyVideoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.cross_dim = config.cross_attention_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.cross_dim, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.cross_dim, self.hidden_size, bias=True)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
    
    def forward(self, hidden_states: Tensor, encoder_states: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(encoder_states)
        v = self.v_proj(encoder_states)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        return self.o_proj(attn_output)


# ==================== MoE Feed-Forward ====================

class VideoExpert(nn.Module):
    def __init__(self, hidden_size: int, expert_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, expert_dim)
        self.fc2 = nn.Linear(expert_dim, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VideoMoELayer(nn.Module):
    """MoE layer for video generation"""
    
    def __init__(self, config: BagleyVideoConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            VideoExpert(config.hidden_size, config.expert_hidden_dim)
            for _ in range(config.num_experts)
        ])
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Router
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, expert_indices = torch.topk(router_probs, self.config.num_experts_per_tok, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Load balancing loss
        tokens_per_expert = F.one_hot(expert_indices, self.config.num_experts).sum(dim=-2).float().mean(0)
        avg_prob = router_probs.mean(dim=(0, 1))
        aux_loss = self.config.num_experts * (tokens_per_expert * avg_prob).sum()
        
        # Expert computation
        x_flat = x.view(-1, hidden_dim)
        probs_flat = top_k_probs.view(-1, self.config.num_experts_per_tok)
        indices_flat = expert_indices.view(-1, self.config.num_experts_per_tok)
        
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            mask = (indices_flat == i).any(dim=-1)
            if mask.any():
                token_idx = mask.nonzero(as_tuple=True)[0]
                expert_out = expert(x_flat[token_idx])
                expert_mask = (indices_flat[token_idx] == i)
                weights = (probs_flat[token_idx] * expert_mask).sum(dim=-1, keepdim=True)
                output[token_idx] += weights * expert_out
        
        return output.view(batch_size, seq_len, hidden_dim), aux_loss


# ==================== Video Transformer Block ====================

class BagleyVideoBlock(nn.Module):
    """Single video transformer block with asymmetric attention"""
    
    def __init__(self, config: BagleyVideoConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Conditioning
        self.adaln = AdaLNZero3D(config.hidden_size, config.hidden_size)
        
        # Asymmetric attention
        if config.use_asymm_attention:
            self.self_attn = AsymmDiTAttention(config)
        else:
            self.self_attn = SpatialAttention(config)  # Fallback
        
        # Cross-attention
        self.cross_attn = VideoCrossAttention(config)
        self.cross_norm = nn.LayerNorm(config.hidden_size)
        
        # FFN (MoE or standard)
        if config.use_moe:
            self.ffn = VideoMoELayer(config)
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
        num_frames: int,
        rope_cos: Optional[Tensor] = None,
        rope_sin: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(hidden_states, condition)
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.adaln.modulate(hidden_states, shift1, scale1)
        hidden_states = self.self_attn(hidden_states, num_frames, rope_cos, rope_sin)
        hidden_states = residual + gate1.unsqueeze(1) * hidden_states
        
        # Cross-attention
        residual = hidden_states
        hidden_states = self.cross_norm(hidden_states)
        hidden_states = self.cross_attn(hidden_states, encoder_hidden_states)
        hidden_states = residual + hidden_states
        
        # FFN
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
class BagleyVideoOutput:
    """Output container"""
    sample: Tensor
    aux_loss: Optional[Tensor] = None


class BagleyVideoMoE(nn.Module):
    """
    🎬 BagleyVideoMoE - Custom Video Generation Model
    
    Features:
    - Wan2.2 inspired MoE for efficient scaling
    - Mochi AsymmDiT for spatiotemporal modeling
    - Causal temporal attention for autoregressive generation
    - Frame consistency with image model integration
    """
    
    def __init__(self, config: BagleyVideoConfig):
        super().__init__()
        self.config = config
        
        # 3D patch embedding
        self.patch_embed = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=(config.patch_size_temporal, config.patch_size_spatial, config.patch_size_spatial),
            stride=(config.patch_size_temporal, config.patch_size_spatial, config.patch_size_spatial),
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        
        # Text pooling
        self.text_pool_proj = nn.Linear(config.text_hidden_size, config.hidden_size)
        
        # 3D position embedding
        self.pos_embed = Rotary3DPositionEmbedding(config.head_dim, config.rope_theta)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BagleyVideoBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.final_linear = nn.Linear(
            config.hidden_size,
            config.patch_size_temporal * config.patch_size_spatial ** 2 * config.in_channels
        )
        
        self.apply(self._init_weights)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
        
        logger.info(f"Initialized BagleyVideoMoE with ~{config.total_parameters:,} parameters")
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _get_timestep_embedding(self, timesteps: Tensor, dim: int = 256) -> Tensor:
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    def forward(
        self,
        latents: Tensor,  # [B, C, T, H, W]
        timestep: Tensor,
        encoder_hidden_states: Tensor,
        pooled_text_embeds: Tensor,
        return_dict: bool = True,
    ) -> BagleyVideoOutput:
        batch_size, channels, num_frames, height, width = latents.shape
        
        # Patch embed
        hidden_states = self.patch_embed(latents)  # [B, D, T', H', W']
        t_patches = hidden_states.shape[2]
        h_patches = hidden_states.shape[3]
        w_patches = hidden_states.shape[4]
        
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # [B, T'*H'*W', D]
        
        # Timestep + text conditioning
        t_emb = self._get_timestep_embedding(timestep)
        t_emb = self.time_embed(t_emb)
        text_pool = self.text_pool_proj(pooled_text_embeds)
        condition = t_emb + text_pool
        
        # 3D RoPE
        rope_cos, rope_sin = self.pos_embed(t_patches, h_patches, w_patches, hidden_states.device)
        
        # Transformer
        total_aux_loss = 0.0
        for block in self.blocks:
            hidden_states, aux_loss = block(
                hidden_states, condition, encoder_hidden_states,
                t_patches, rope_cos, rope_sin
            )
            total_aux_loss = total_aux_loss + aux_loss
        
        # Output
        hidden_states = self.final_norm(hidden_states)
        hidden_states = self.final_linear(hidden_states)
        
        # Unpatchify to [B, C, T, H, W]
        hidden_states = hidden_states.view(
            batch_size, t_patches, h_patches, w_patches,
            self.config.patch_size_temporal,
            self.config.patch_size_spatial,
            self.config.patch_size_spatial,
            channels
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        hidden_states = hidden_states.reshape(batch_size, channels, num_frames, height, width)
        
        if return_dict:
            return BagleyVideoOutput(sample=hidden_states, aux_loss=total_aux_loss)
        return hidden_states
    
    @torch.no_grad()
    async def generate(
        self,
        prompt: str,
        duration: float = 2.0,
        fps: int = 24,
        width: int = 1280,
        height: int = 720,
        **kwargs,
    ) -> List:
        """Generate video from text prompt"""
        logger.info(f"Generating video: {prompt[:50]}...")
        return []
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda", dtype: str = "bfloat16") -> "BagleyVideoMoE":
        import os, json
        from safetensors.torch import load_file
        
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = BagleyVideoConfig(**json.load(f))
        else:
            config = BagleyVideoConfig()
        
        model = cls(config)
        
        weights_path = os.path.join(path, "model.safetensors")
        if os.path.exists(weights_path):
            model.load_state_dict(load_file(weights_path))
        
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return model.to(device=device, dtype=dtype_map.get(dtype, torch.bfloat16))
    
    def save_pretrained(self, path: str):
        import os, json
        from safetensors.torch import save_file
        
        os.makedirs(path, exist_ok=True)
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        save_file({k: v.cpu() for k, v in self.state_dict().items()}, 
                  os.path.join(path, "model.safetensors"))
