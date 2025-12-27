"""
⚙️ BagleyMoE Configuration
Hyperparameters for the custom MoE architecture
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class BagleyMoEConfig:
    """
    Configuration for BagleyMoE - Custom Mixture-of-Experts Language Model
    
    Architecture based on:
    - DeepSeek-R1: Hybrid thinking/non-thinking modes
    - Qwen3-235B-A22B: Efficient MoE with massive parameter counts
    - Custom innovations for personality injection and infinite context
    """
    
    # ==================== Model Architecture ====================
    
    # Vocabulary
    vocab_size: int = 151936  # Extended for multilingual support
    
    # Hidden dimensions
    hidden_size: int = 8192  # Main hidden dimension
    intermediate_size: int = 24576  # FFN intermediate (3x hidden)
    
    # Attention configuration
    num_attention_heads: int = 64
    num_key_value_heads: int = 8  # Grouped Query Attention (GQA)
    head_dim: int = 128  # hidden_size // num_attention_heads
    
    # Layers
    num_hidden_layers: int = 80  # Deep for reasoning capability
    
    # MoE configuration
    num_experts: int = 64  # Total number of experts
    num_experts_per_tok: int = 8  # Top-K experts activated per token
    num_shared_experts: int = 2  # Experts that are always active
    expert_intermediate_size: int = 3072  # Per-expert FFN size
    
    # Specialized expert assignment (semantic routing hints)
    expert_specializations: List[str] = field(default_factory=lambda: [
        "general_knowledge",  # Experts 0-7
        "code_technical",     # Experts 8-15
        "creative_humor",     # Experts 16-23
        "reasoning_logic",    # Experts 24-31
        "multilingual",       # Experts 32-39
        "emotional_personality",  # Experts 40-47
        "visual_understanding",   # Experts 48-55
        "task_planning",      # Experts 56-63
    ])
    
    # Router configuration
    router_aux_loss_coef: float = 0.001  # Load balancing loss coefficient
    router_z_loss_coef: float = 0.001  # Router z-loss coefficient
    router_type: Literal["top_k", "expert_choice", "soft_moe"] = "top_k"
    
    # ==================== Position Encoding ====================
    
    # RoPE configuration (YaRN extended for 128K+ context)
    rope_theta: float = 10000000.0  # Extended base for long context
    rope_scaling: dict = field(default_factory=lambda: {
        "type": "yarn",
        "factor": 4.0,  # 4x context extension
        "original_max_position_embeddings": 32768,
        "attention_factor": 1.0,
        "beta_fast": 32,
        "beta_slow": 1,
    })
    max_position_embeddings: int = 131072  # 128K context
    
    # ==================== Normalization & Activation ====================
    
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"  # SwiGLU activation
    
    # ==================== Hybrid Thinking Mode ====================
    # Inspired by DeepSeek-R1's thinking/non-thinking toggle
    
    enable_thinking_mode: bool = True
    thinking_budget_tokens: int = 32768  # Max tokens for thinking
    soft_thinking_threshold: float = 0.5  # Complexity threshold to trigger thinking
    
    # ==================== Training Configuration ====================
    
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    
    # Dropout (set to 0 for inference)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # ==================== Memory & Performance ====================
    
    # Sliding window attention (for efficiency on very long contexts)
    sliding_window: Optional[int] = 4096
    use_sliding_window: bool = True
    
    # Flash Attention configuration
    use_flash_attention: bool = True
    flash_attention_version: int = 2
    
    # KV-cache optimization
    use_paged_attention: bool = True
    page_size: int = 16
    
    # ==================== Personality Injection ====================
    
    # Dedicated personality embedding layer
    personality_embedding_dim: int = 1024
    num_personality_modes: int = 4  # chaos, chill, focus, custom
    
    # ==================== Inference Optimization ====================
    
    # Speculative decoding
    use_speculative_decoding: bool = False
    draft_model_layers: int = 8  # Shallow draft model
    speculation_lookahead: int = 5
    
    # Quantization hints
    quantization_config: dict = field(default_factory=lambda: {
        "bits": 4,
        "group_size": 128,
        "desc_act": True,
        "sym": True,
    })
    
    # ==================== Computed Properties ====================
    
    @property
    def total_parameters(self) -> int:
        """Estimate total parameters"""
        # Embedding
        embed_params = self.vocab_size * self.hidden_size
        
        # Per layer
        # Attention
        attn_params = (
            self.hidden_size * self.head_dim * self.num_attention_heads +  # Q
            self.hidden_size * self.head_dim * self.num_key_value_heads * 2 +  # K, V
            self.head_dim * self.num_attention_heads * self.hidden_size  # O
        )
        
        # MoE FFN
        expert_params = self.num_experts * (
            self.hidden_size * self.expert_intermediate_size * 2 +  # Up, Down
            self.hidden_size * self.expert_intermediate_size  # Gate
        )
        
        # Router
        router_params = self.hidden_size * self.num_experts
        
        # Layer norm
        norm_params = self.hidden_size * 2
        
        per_layer = attn_params + expert_params + router_params + norm_params
        
        # Total
        total = embed_params + per_layer * self.num_hidden_layers + embed_params
        
        return total
    
    @property
    def active_parameters(self) -> int:
        """Estimate active parameters per forward pass"""
        # Embedding
        embed_params = self.vocab_size * self.hidden_size
        
        # Per layer (only top-k experts active)
        attn_params = (
            self.hidden_size * self.head_dim * self.num_attention_heads +
            self.hidden_size * self.head_dim * self.num_key_value_heads * 2 +
            self.head_dim * self.num_attention_heads * self.hidden_size
        )
        
        # Only top-k + shared experts
        active_experts = self.num_experts_per_tok + self.num_shared_experts
        active_expert_params = active_experts * (
            self.hidden_size * self.expert_intermediate_size * 3
        )
        
        per_layer = attn_params + active_expert_params
        
        total = embed_params + per_layer * self.num_hidden_layers + embed_params
        
        return total
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_size % self.num_attention_heads == 0
        assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.num_experts_per_tok <= self.num_experts
        
        # Update head_dim if needed
        self.head_dim = self.hidden_size // self.num_attention_heads


# Pre-configured model sizes
class BagleyMoEConfigSmall(BagleyMoEConfig):
    """Small model for testing (~7B total, ~2B active)"""
    def __init__(self):
        super().__init__(
            hidden_size=4096,
            intermediate_size=11264,
            num_attention_heads=32,
            num_key_value_heads=4,
            num_hidden_layers=32,
            num_experts=32,
            num_experts_per_tok=4,
            expert_intermediate_size=1408,
            max_position_embeddings=32768,
        )


class BagleyMoEConfigMedium(BagleyMoEConfig):
    """Medium model (~30B total, ~8B active)"""
    def __init__(self):
        super().__init__(
            hidden_size=6144,
            intermediate_size=16384,
            num_attention_heads=48,
            num_key_value_heads=6,
            num_hidden_layers=48,
            num_experts=48,
            num_experts_per_tok=6,
            expert_intermediate_size=2048,
            max_position_embeddings=65536,
        )


class BagleyMoEConfigLarge(BagleyMoEConfig):
    """Large model - default (~70B total, ~8B active)"""
    pass  # Uses default values


class BagleyMoEConfigXL(BagleyMoEConfig):
    """XL model for maximum capability (~200B total, ~20B active)"""
    def __init__(self):
        super().__init__(
            hidden_size=12288,
            intermediate_size=32768,
            num_attention_heads=96,
            num_key_value_heads=12,
            num_hidden_layers=96,
            num_experts=128,
            num_experts_per_tok=16,
            num_shared_experts=4,
            expert_intermediate_size=4096,
            max_position_embeddings=262144,  # 256K context
        )
