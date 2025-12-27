"""
⚙️ BagleyDiT Configuration
Hyperparameters for the custom Image Generation model
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Tuple


@dataclass
class BagleyDiTConfig:
    """
    Configuration for BagleyDiT - Custom Diffusion Transformer for Image Generation
    
    Architecture based on:
    - FLUX.1: Rectified flow transformers for faster, higher quality generation
    - HiDream-I1: Sparse MoE DiT for efficient computation
    - SD3/PixArt: Multi-scale DiT architecture innovations
    """
    
    # ==================== Model Architecture ====================
    
    # Patch embedding
    patch_size: int = 2  # Patch size for VAE latents
    in_channels: int = 16  # VAE latent channels
    
    # Hidden dimensions
    hidden_size: int = 3072
    num_attention_heads: int = 24
    head_dim: int = 128  # hidden_size // num_attention_heads
    
    # Transformer blocks
    num_layers: int = 38  # Deep for quality
    
    # MoE configuration (HiDream-inspired)
    use_moe: bool = True
    num_experts: int = 16
    num_experts_per_tok: int = 4
    expert_hidden_dim: int = 8192
    
    # Expert specializations for image generation
    expert_specializations: List[str] = field(default_factory=lambda: [
        "photorealism",     # Experts 0-3
        "anatomy",          # Experts 4-7
        "composition",      # Experts 8-11
        "style_transfer",   # Experts 12-15
    ])
    
    # MLP configuration
    mlp_ratio: float = 4.0
    
    # ==================== Text Conditioning ====================
    
    # T5 encoder for text understanding
    text_encoder_type: str = "t5-xxl"
    text_hidden_size: int = 4096
    max_text_length: int = 512
    
    # CLIP for additional conditioning
    use_clip: bool = True
    clip_hidden_size: int = 1280
    
    # Cross-attention configuration
    cross_attention_dim: int = 4096
    
    # ==================== Position Encoding ====================
    
    # 2D RoPE for spatial positions
    use_2d_rope: bool = True
    rope_theta: float = 10000.0
    
    # ==================== Diffusion Configuration ====================
    
    # Rectified flow (FLUX-style)
    flow_type: Literal["rectified", "edm", "ddpm"] = "rectified"
    
    # Noise schedule
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "scaled_linear"
    
    # Sampling
    default_num_inference_steps: int = 28  # Rectified flow needs fewer steps
    guidance_scale: float = 7.5
    
    # ==================== Resolution Support ====================
    
    # Multi-resolution support
    supported_resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1280, 1280),
        (1536, 1536),
        (2048, 2048),
        (4096, 4096),  # Max resolution
        # Aspect ratios
        (1024, 768),
        (768, 1024),
        (1280, 720),
        (720, 1280),
        (1920, 1080),
        (1080, 1920),
    ])
    
    default_resolution: Tuple[int, int] = (1024, 1024)
    max_resolution: Tuple[int, int] = (4096, 4096)
    
    # ==================== VAE Configuration ====================
    
    vae_scale_factor: int = 8
    vae_latent_channels: int = 16
    
    # ==================== Normalization ====================
    
    norm_type: str = "ada_norm_zero"  # AdaLN-Zero for conditioning
    norm_eps: float = 1e-6
    
    # ==================== Training Configuration ====================
    
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Classifier-free guidance dropout
    cfg_dropout: float = 0.1
    
    # ==================== Performance ====================
    
    use_flash_attention: bool = True
    use_xformers: bool = True
    enable_gradient_checkpointing: bool = True
    
    # ==================== Computed Properties ====================
    
    @property
    def total_parameters(self) -> int:
        """Estimate total parameters"""
        # Patch embedding
        patch_embed = self.in_channels * self.hidden_size * (self.patch_size ** 2)
        
        # Per transformer block
        # Self-attention
        attn_params = self.hidden_size * self.hidden_size * 4  # Q, K, V, O
        
        # Cross-attention
        cross_attn_params = (
            self.hidden_size * self.cross_attention_dim +  # Q
            self.cross_attention_dim * self.hidden_size * 2 +  # K, V
            self.hidden_size * self.hidden_size  # O
        )
        
        # MoE FFN
        if self.use_moe:
            ffn_params = self.num_experts * (
                self.hidden_size * self.expert_hidden_dim * 2 +
                self.hidden_size * self.expert_hidden_dim
            )
            router_params = self.hidden_size * self.num_experts
        else:
            ffn_params = self.hidden_size * int(self.hidden_size * self.mlp_ratio) * 2
            router_params = 0
        
        # AdaLN parameters
        adaln_params = self.hidden_size * 6 * 2  # Scale and shift for 2 norms
        
        per_layer = attn_params + cross_attn_params + ffn_params + router_params + adaln_params
        
        total = patch_embed + per_layer * self.num_layers
        
        return total
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
