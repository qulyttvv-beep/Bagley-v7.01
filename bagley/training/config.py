"""
⚙️ Training Configuration
Configuration for distributed training on GPU clusters
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ParallelismType(Enum):
    """Types of parallelism for distributed training"""
    DATA = "data"           # Data parallelism (DDP)
    TENSOR = "tensor"       # Tensor parallelism (Megatron-LM style)
    PIPELINE = "pipeline"   # Pipeline parallelism
    SEQUENCE = "sequence"   # Sequence parallelism
    EXPERT = "expert"       # Expert parallelism (MoE)
    FSDP = "fsdp"          # Fully Sharded Data Parallel


class OptimizerType(Enum):
    """Supported optimizers"""
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    LAMB = "lamb"
    ADAFACTOR = "adafactor"
    LION = "lion"


class SchedulerType(Enum):
    """Learning rate schedulers"""
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_with_warmup"
    LINEAR = "linear"
    CONSTANT = "constant"
    POLYNOMIAL = "polynomial"
    ONE_CYCLE = "one_cycle"
    WSD = "warmup_stable_decay"


class PrecisionType(Enum):
    """Training precision"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


@dataclass
class TrainingConfig:
    """
    🎯 Main Training Configuration
    
    Comprehensive configuration for training Bagley models
    on GPU clusters with distributed strategies.
    """
    
    # ==================== Basic Training ====================
    model_type: str = "chat"  # chat, image, video, tts
    output_dir: str = "./outputs"
    run_name: str = "bagley-training"
    
    # Training steps
    num_epochs: int = 3
    max_steps: int = -1  # -1 = use epochs
    gradient_accumulation_steps: int = 8
    
    # Batch sizes
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    global_batch_size: Optional[int] = None  # Computed if not set
    
    # ==================== Optimizer ====================
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler: SchedulerType = SchedulerType.COSINE_WARMUP
    warmup_steps: int = 1000
    warmup_ratio: float = 0.0  # Alternative to warmup_steps
    min_lr_ratio: float = 0.1  # Min LR as fraction of max
    
    # ==================== Precision & Memory ====================
    precision: PrecisionType = PrecisionType.BF16
    gradient_checkpointing: bool = True
    
    # Memory optimization
    cpu_offload: bool = False
    activation_offload: bool = False
    optimizer_offload: bool = False
    
    # ==================== Logging & Checkpoints ====================
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    log_to_wandb: bool = True
    wandb_project: str = "bagley-training"
    wandb_entity: Optional[str] = None
    
    # ==================== Sequence Lengths ====================
    max_seq_length: int = 8192
    
    # ==================== Advanced ====================
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # Loss configuration
    label_smoothing: float = 0.0
    auxiliary_loss_weight: float = 0.01  # For MoE load balancing
    
    def __post_init__(self):
        """Validate and compute derived values"""
        if self.global_batch_size is None:
            # Will be computed when distributed config is known
            self.global_batch_size = self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass  
class DistributedConfig:
    """
    🌐 Distributed Training Configuration
    
    Supports:
    - DeepSpeed ZeRO stages 1/2/3
    - PyTorch FSDP
    - Megatron-LM tensor/pipeline parallelism
    - Expert parallelism for MoE
    """
    
    # ==================== Cluster Setup ====================
    num_nodes: int = 1
    gpus_per_node: int = 8
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # ==================== Parallelism Strategy ====================
    parallelism_types: List[ParallelismType] = field(default_factory=lambda: [ParallelismType.DATA])
    
    # Data parallelism
    data_parallel_size: int = 8  # Number of data parallel replicas
    
    # Tensor parallelism (split layers across GPUs)
    tensor_parallel_size: int = 1
    
    # Pipeline parallelism (split model layers across GPUs)
    pipeline_parallel_size: int = 1
    num_pipeline_stages: int = 1
    
    # Sequence parallelism (for very long sequences)
    sequence_parallel: bool = False
    
    # Expert parallelism (for MoE models)
    expert_parallel_size: int = 1
    
    # ==================== DeepSpeed ====================
    use_deepspeed: bool = True
    deepspeed_stage: int = 3  # ZeRO stage (1, 2, or 3)
    
    # ZeRO optimization
    zero_optimization: Dict[str, Any] = field(default_factory=lambda: {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    })
    
    # ==================== FSDP Alternative ====================
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_cpu_offload: bool = False
    fsdp_auto_wrap_policy: str = "transformer_auto_wrap"
    
    # ==================== Communication ====================
    backend: str = "nccl"  # nccl, gloo
    init_method: str = "env://"
    
    # Gradient communication
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    
    # ==================== Flash Attention ====================
    use_flash_attention: bool = True
    flash_attention_version: int = 2
    
    def __post_init__(self):
        """Compute derived values"""
        self.world_size = self.num_nodes * self.gpus_per_node
        
        # Validate parallelism configuration
        total_parallelism = (
            self.data_parallel_size * 
            self.tensor_parallel_size * 
            self.pipeline_parallel_size
        )
        
        if total_parallelism > self.world_size:
            raise ValueError(
                f"Total parallelism ({total_parallelism}) exceeds world size ({self.world_size})"
            )
    
    def get_deepspeed_config(self) -> Dict[str, Any]:
        """Generate DeepSpeed configuration dict"""
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": 1.0,
            "zero_optimization": self.zero_optimization,
            "fp16": {"enabled": False},
            "bf16": {"enabled": True},
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False,
            "tensorboard": {
                "enabled": True,
                "output_path": "./tensorboard",
            },
        }
    
    def get_fsdp_config(self) -> Dict[str, Any]:
        """Generate FSDP configuration"""
        return {
            "sharding_strategy": self.fsdp_sharding_strategy,
            "cpu_offload": self.fsdp_cpu_offload,
            "auto_wrap_policy": self.fsdp_auto_wrap_policy,
            "backward_prefetch": "BACKWARD_PRE",
            "forward_prefetch": True,
            "use_orig_params": True,
            "sync_module_states": True,
        }


@dataclass
class MoETrainingConfig:
    """
    🧠 MoE-Specific Training Configuration
    
    Special settings for training Mixture of Experts models.
    """
    
    # Expert configuration
    num_experts: int = 64
    num_experts_per_token: int = 8
    
    # Load balancing
    load_balance_loss_weight: float = 0.01
    router_z_loss_weight: float = 0.001
    
    # Expert dropping
    expert_dropout: float = 0.0
    capacity_factor: float = 1.25  # Training
    eval_capacity_factor: float = 2.0  # Inference
    
    # Expert parallelism
    expert_parallel: bool = True
    experts_per_rank: int = 8  # How many experts per GPU
    
    # Communication
    all2all_communication: bool = True
    communication_overlap: bool = True


@dataclass  
class ImageTrainingConfig:
    """
    🎨 Image Model Training Configuration
    
    Settings specific to DiT/FLUX-style image models.
    """
    
    # Image settings
    image_size: int = 512
    patch_size: int = 2
    latent_channels: int = 16
    
    # Training
    num_timesteps: int = 1000
    prediction_type: str = "velocity"  # epsilon, velocity, sample
    
    # Loss weighting
    loss_type: str = "mse"
    snr_gamma: Optional[float] = 5.0  # Min-SNR weighting
    
    # Conditioning
    text_encoder_training: bool = False  # Usually frozen
    cfg_scale: float = 7.5
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999


@dataclass
class VideoTrainingConfig:
    """
    🎬 Video Model Training Configuration
    """
    
    # Video settings
    num_frames: int = 16
    frame_rate: int = 24
    image_size: int = 512
    
    # Temporal settings
    temporal_compression: int = 4
    
    # Training
    num_timesteps: int = 1000
    
    # Frame prediction
    frame_prediction_type: str = "all"  # all, keyframe


@dataclass
class TTSTrainingConfig:
    """
    🎵 TTS Training Configuration
    """
    
    # Audio settings
    sample_rate: int = 24000
    hop_length: int = 256
    n_mels: int = 80
    
    # Training
    semantic_weight: float = 1.0
    acoustic_weight: float = 1.0
    duration_weight: float = 0.1
    
    # Voice cloning
    speaker_embedding_dim: int = 256
    num_speakers: Optional[int] = None  # None for multi-speaker


def get_training_config(
    model_type: str,
    cluster_size: int = 8,
    use_deepspeed: bool = True,
) -> tuple[TrainingConfig, DistributedConfig]:
    """
    Get recommended training configuration.
    
    Args:
        model_type: Type of model (chat, image, video, tts)
        cluster_size: Number of GPUs
        use_deepspeed: Whether to use DeepSpeed
        
    Returns:
        Tuple of (TrainingConfig, DistributedConfig)
    """
    
    # Base training config
    train_cfg = TrainingConfig(
        model_type=model_type,
        precision=PrecisionType.BF16,
        gradient_checkpointing=True,
    )
    
    # Distributed config
    dist_cfg = DistributedConfig(
        gpus_per_node=min(8, cluster_size),
        num_nodes=max(1, cluster_size // 8),
        use_deepspeed=use_deepspeed,
        deepspeed_stage=3 if cluster_size > 4 else 2,
    )
    
    # Model-specific adjustments
    if model_type == "chat":
        train_cfg.learning_rate = 1e-4
        train_cfg.max_seq_length = 8192
        train_cfg.per_device_train_batch_size = 1
        train_cfg.gradient_accumulation_steps = 16
        
    elif model_type == "image":
        train_cfg.learning_rate = 1e-4
        train_cfg.per_device_train_batch_size = 1
        train_cfg.gradient_accumulation_steps = 8
        
    elif model_type == "video":
        train_cfg.learning_rate = 5e-5
        train_cfg.per_device_train_batch_size = 1
        train_cfg.gradient_accumulation_steps = 32
        train_cfg.cpu_offload = True
        
    elif model_type == "tts":
        train_cfg.learning_rate = 2e-4
        train_cfg.per_device_train_batch_size = 4
        train_cfg.gradient_accumulation_steps = 4
    
    return train_cfg, dist_cfg
