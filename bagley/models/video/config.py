"""
⚙️ BagleyVideo Configuration
Hyperparameters for the custom Video Generation model
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal


@dataclass
class BagleyVideoConfig:
    """
    Configuration for BagleyVideoMoE - Custom Video Generation Model
    
    Architecture based on:
    - Wan2.2: MoE diffusion for video
    - Mochi 1: Asymmetric Diffusion Transformer (AsymmDiT)
    - CogVideoX: Efficient spatiotemporal modeling
    """
    
    # ==================== Model Architecture ====================
    
    # Patch embedding for 3D (temporal + spatial)
    patch_size_spatial: int = 2
    patch_size_temporal: int = 1
    in_channels: int = 16  # 3D VAE latent channels
    
    # Hidden dimensions
    hidden_size: int = 2048
    num_attention_heads: int = 16
    head_dim: int = 128
    
    # Transformer blocks
    num_layers: int = 28
    
    # MoE configuration (Wan2.2-inspired)
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    expert_hidden_dim: int = 5504
    
    # Expert specializations for video
    expert_specializations: List[str] = field(default_factory=lambda: [
        "motion_dynamics",      # Experts 0-1
        "scene_understanding",  # Experts 2-3
        "character_consistency",# Experts 4-5
        "physics_simulation",   # Experts 6-7
    ])
    
    # MLP ratio
    mlp_ratio: float = 4.0
    
    # ==================== Asymmetric Attention ====================
    
    # AsymmDiT configuration (Mochi-inspired)
    use_asymm_attention: bool = True
    temporal_attention_ratio: float = 0.5  # Ratio of temporal to spatial attention
    causal_temporal: bool = True  # Causal attention for autoregressive generation
    
    # ==================== Text Conditioning ====================
    
    text_encoder_type: str = "t5-xxl"
    text_hidden_size: int = 4096
    max_text_length: int = 256
    cross_attention_dim: int = 4096
    
    # ==================== Position Encoding ====================
    
    # 3D RoPE for spatiotemporal positions
    use_3d_rope: bool = True
    rope_theta: float = 10000.0
    
    # ==================== Diffusion Configuration ====================
    
    flow_type: Literal["rectified", "edm", "ddpm"] = "rectified"
    num_train_timesteps: int = 1000
    default_num_inference_steps: int = 50
    guidance_scale: float = 6.0
    
    # ==================== Video Configuration ====================
    
    # Frame settings
    max_frames: int = 129  # ~5 seconds at 24fps
    default_frames: int = 49  # ~2 seconds at 24fps
    fps: int = 24
    
    # Resolution settings
    supported_resolutions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (512, 512),
        (720, 480),   # 480p
        (1280, 720),  # 720p
        (1920, 1080), # 1080p
    ])
    default_resolution: Tuple[int, int] = (1280, 720)
    
    # ==================== VAE Configuration ====================
    
    vae_spatial_scale: int = 8
    vae_temporal_scale: int = 4
    vae_latent_channels: int = 16
    
    # ==================== Normalization ====================
    
    norm_type: str = "ada_norm_zero"
    norm_eps: float = 1e-6
    
    # ==================== Training ====================
    
    initializer_range: float = 0.02
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    cfg_dropout: float = 0.1
    
    # ==================== Performance ====================
    
    use_flash_attention: bool = True
    enable_gradient_checkpointing: bool = True
    
    # Frame consistency (uses image model for refinement)
    use_frame_consistency: bool = True
    frame_consistency_strength: float = 0.3
    
    # ==================== Computed Properties ====================
    
    @property
    def total_parameters(self) -> int:
        """Estimate total parameters"""
        # Patch embedding
        patch_embed = self.in_channels * self.hidden_size * (
            self.patch_size_spatial ** 2 * self.patch_size_temporal
        )
        
        # Per transformer block
        attn_params = self.hidden_size * self.hidden_size * 4 * 2  # Spatial + temporal
        cross_attn_params = self.hidden_size * self.cross_attention_dim * 3
        
        if self.use_moe:
            ffn_params = self.num_experts * self.hidden_size * self.expert_hidden_dim * 3
            router_params = self.hidden_size * self.num_experts
        else:
            ffn_params = self.hidden_size * int(self.hidden_size * self.mlp_ratio) * 2
            router_params = 0
        
        per_layer = attn_params + cross_attn_params + ffn_params + router_params
        
        return patch_embed + per_layer * self.num_layers
    
    @property
    def max_video_length_seconds(self) -> float:
        """Maximum video length in seconds"""
        return self.max_frames / self.fps
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
