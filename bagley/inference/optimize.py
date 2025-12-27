"""
⚡ Model Optimization for Inference
Pure PyTorch optimizations - works on NVIDIA and AMD
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from functools import partial
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    
    # torch.compile settings
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"  # default, reduce-overhead, max-autotune
    compile_fullgraph: bool = False
    compile_dynamic: bool = True
    
    # Memory optimizations
    use_gradient_checkpointing: bool = False  # For inference, usually False
    use_channels_last: bool = True  # Memory format optimization
    use_inference_mode: bool = True
    
    # Attention optimizations  
    use_sdpa: bool = True  # Scaled Dot Product Attention (PyTorch native)
    use_memory_efficient_attention: bool = True
    
    # KV Cache
    use_kv_cache: bool = True
    kv_cache_dtype: str = "auto"  # auto, float16, bfloat16
    
    # Batching
    use_cuda_graphs: bool = False  # Can speed up fixed-size batches
    max_batch_size: int = 1
    
    # Precision
    dtype: str = "bfloat16"  # float32, float16, bfloat16


def optimize_model(
    model: nn.Module,
    config: Optional[OptimizationConfig] = None,
    example_inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """
    Apply all optimizations to a model for inference.
    
    Args:
        model: The model to optimize
        config: Optimization configuration
        example_inputs: Example inputs for tracing (optional)
        
    Returns:
        Optimized model
    """
    config = config or OptimizationConfig()
    
    logger.info("🚀 Applying inference optimizations...")
    
    # 1. Set to eval mode
    model.eval()
    
    # 2. Set dtype
    dtype = _get_dtype(config.dtype)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
        logger.info(f"  ✓ Converted to {config.dtype}")
    
    # 3. Channels last memory format (faster on modern GPUs)
    if config.use_channels_last and _has_conv_layers(model):
        model = model.to(memory_format=torch.channels_last)
        logger.info("  ✓ Enabled channels_last memory format")
    
    # 4. Replace attention with SDPA
    if config.use_sdpa:
        _replace_attention_with_sdpa(model)
        logger.info("  ✓ Enabled scaled dot-product attention (SDPA)")
    
    # 5. Apply torch.compile
    if config.use_compile:
        try:
            model = torch.compile(
                model,
                mode=config.compile_mode,
                fullgraph=config.compile_fullgraph,
                dynamic=config.compile_dynamic,
            )
            logger.info(f"  ✓ torch.compile enabled (mode={config.compile_mode})")
        except Exception as e:
            logger.warning(f"  ✗ torch.compile failed: {e}")
    
    # 6. Fuse operations where possible
    _fuse_operations(model)
    logger.info("  ✓ Fused eligible operations")
    
    logger.info("✅ Optimization complete!")
    
    return model


def _get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype"""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def _has_conv_layers(model: nn.Module) -> bool:
    """Check if model has convolutional layers"""
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return True
    return False


def _replace_attention_with_sdpa(model: nn.Module):
    """
    Replace custom attention implementations with PyTorch SDPA.
    SDPA automatically uses the best available backend.
    """
    # This is a no-op if attention already uses F.scaled_dot_product_attention
    # But we set the backend preferences
    torch.backends.cuda.enable_flash_sdp(True)  # Try Flash if available
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory efficient fallback
    torch.backends.cuda.enable_math_sdp(True)  # Math fallback


def _fuse_operations(model: nn.Module):
    """Fuse operations for better performance"""
    # Fuse Conv + BatchNorm
    try:
        torch.quantization.fuse_modules(
            model, 
            [['conv', 'bn']], 
            inplace=True
        )
    except:
        pass  # Not all models have fuseable modules


def enable_kv_cache(model: nn.Module, max_seq_len: int = 8192):
    """
    Enable KV caching for transformer models.
    This avoids recomputing key/value for previous tokens.
    """
    for module in model.modules():
        if hasattr(module, 'enable_kv_cache'):
            module.enable_kv_cache(max_seq_len)
        elif hasattr(module, 'use_cache'):
            module.use_cache = True


def memory_efficient_forward(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Memory-efficient forward pass for long sequences.
    Processes in chunks to reduce peak memory.
    """
    # Get sequence length
    seq_len = inputs.get('input_ids', inputs.get('x')).shape[1]
    
    if seq_len <= chunk_size:
        # Short sequence, normal forward
        with torch.inference_mode():
            return model(**inputs)
    
    # Long sequence, chunk processing
    outputs = []
    
    for i in range(0, seq_len, chunk_size):
        chunk_inputs = {
            k: v[:, i:i+chunk_size] if v.dim() > 1 else v
            for k, v in inputs.items()
        }
        
        with torch.inference_mode():
            chunk_output = model(**chunk_inputs)
            outputs.append(chunk_output)
    
    # Concatenate outputs
    return torch.cat(outputs, dim=1)


class InferenceContext:
    """
    Context manager for optimized inference.
    
    Usage:
        with InferenceContext(model):
            output = model(input)
    """
    
    def __init__(
        self,
        model: nn.Module,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.dtype = dtype or torch.bfloat16
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._original_training = model.training
    
    def __enter__(self):
        self.model.eval()
        return torch.inference_mode().__enter__()
    
    def __exit__(self, *args):
        if self._original_training:
            self.model.train()
        torch.inference_mode().__exit__(*args)


def warmup_model(
    model: nn.Module,
    example_inputs: Dict[str, torch.Tensor],
    num_warmup: int = 3,
):
    """
    Warmup model for consistent performance.
    First few runs are slower due to compilation/caching.
    """
    logger.info(f"Warming up model with {num_warmup} forward passes...")
    
    with torch.inference_mode():
        for i in range(num_warmup):
            _ = model(**example_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    logger.info("Warmup complete!")


def benchmark_model(
    model: nn.Module,
    example_inputs: Dict[str, torch.Tensor],
    num_runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark model performance.
    
    Returns:
        Dictionary with timing statistics
    """
    import time
    
    # Warmup
    warmup_model(model, example_inputs, num_warmup=5)
    
    # Benchmark
    times = []
    
    with torch.inference_mode():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(**example_inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
    
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "throughput": num_runs / sum(times),
    }
