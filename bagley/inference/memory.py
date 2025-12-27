"""
🧠 Memory Management for Inference
Efficient GPU/CPU memory handling
"""

import torch
import gc
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics"""
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_total_mb: float
    cpu_used_mb: float
    
    @property
    def gpu_free_mb(self) -> float:
        return self.gpu_total_mb - self.gpu_allocated_mb


class MemoryManager:
    """
    Memory manager for efficient GPU/CPU utilization.
    
    Features:
    - Memory tracking
    - Automatic garbage collection
    - Memory-efficient model loading
    - Layer offloading to CPU
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        
        # Offloading state
        self.offloaded_layers: Dict[str, torch.Tensor] = {}
        self.layer_devices: Dict[str, str] = {}
        
        logger.info(f"MemoryManager initialized for {device}")
    
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        if self.is_cuda:
            gpu_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
            gpu_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)
            gpu_total = torch.cuda.get_device_properties(self.device).total_memory / (1024**2)
        else:
            gpu_allocated = 0
            gpu_reserved = 0
            gpu_total = 0
        
        # CPU memory (approximate)
        import psutil
        cpu_used = psutil.Process().memory_info().rss / (1024**2)
        
        return MemoryStats(
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            gpu_total_mb=gpu_total,
            cpu_used_mb=cpu_used,
        )
    
    def print_stats(self):
        """Print memory statistics"""
        stats = self.get_stats()
        
        print("\n" + "=" * 50)
        print("🧠 MEMORY STATS")
        print("=" * 50)
        
        if self.is_cuda:
            print(f"GPU Allocated: {stats.gpu_allocated_mb:.1f} MB")
            print(f"GPU Reserved:  {stats.gpu_reserved_mb:.1f} MB")
            print(f"GPU Total:     {stats.gpu_total_mb:.1f} MB")
            print(f"GPU Free:      {stats.gpu_free_mb:.1f} MB")
        
        print(f"CPU Used:      {stats.cpu_used_mb:.1f} MB")
        print("=" * 50 + "\n")
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection"""
        gc.collect()
        
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Memory cache cleared")
    
    def optimize_model_memory(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize model for memory efficiency.
        
        Applies:
        - Inplace operations where safe
        - Memory-efficient parameter storage
        """
        # Enable memory efficient mode for certain operations
        for module in model.modules():
            # Set inplace=True for activations where safe
            if hasattr(module, 'inplace'):
                module.inplace = True
            
            # Disable gradient computation (inference only)
            for param in module.parameters():
                param.requires_grad = False
        
        return model
    
    def offload_layer(self, layer: torch.nn.Module, layer_name: str):
        """Offload a layer to CPU"""
        if layer_name in self.layer_devices and self.layer_devices[layer_name] == "cpu":
            return  # Already offloaded
        
        layer.to("cpu")
        self.layer_devices[layer_name] = "cpu"
        
        if self.is_cuda:
            torch.cuda.empty_cache()
    
    def load_layer(self, layer: torch.nn.Module, layer_name: str):
        """Load a layer back to GPU"""
        if layer_name in self.layer_devices and self.layer_devices[layer_name] == "cuda":
            return  # Already on GPU
        
        layer.to(self.device)
        self.layer_devices[layer_name] = "cuda"
    
    def offload_model_layers(
        self,
        model: torch.nn.Module,
        keep_layers: int = 4,  # Keep first N layers on GPU
    ):
        """
        Offload most layers to CPU, keeping only some on GPU.
        Good for large models on limited VRAM.
        """
        layers = list(model.named_modules())
        
        for i, (name, layer) in enumerate(layers):
            if i >= keep_layers and hasattr(layer, 'weight'):
                self.offload_layer(layer, name)
        
        logger.info(f"Offloaded layers to CPU, keeping {keep_layers} on GPU")
    
    def sequential_forward(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        offload: bool = True,
    ) -> torch.Tensor:
        """
        Memory-efficient forward pass.
        Loads layers one at a time if offloading is enabled.
        """
        if not offload:
            return model(**inputs)
        
        x = inputs.get('input_ids', inputs.get('x'))
        
        # Process layer by layer
        for name, layer in model.named_children():
            # Load layer to GPU
            self.load_layer(layer, name)
            
            # Forward
            x = layer(x)
            
            # Offload layer back to CPU
            self.offload_layer(layer, name)
            
            # Clear cache
            if self.is_cuda:
                torch.cuda.empty_cache()
        
        return x
    
    @staticmethod
    def estimate_model_memory(
        num_params: int,
        dtype: torch.dtype = torch.bfloat16,
        include_optimizer: bool = False,
        include_gradients: bool = False,
    ) -> float:
        """
        Estimate memory required for a model.
        
        Args:
            num_params: Number of parameters
            dtype: Data type
            include_optimizer: Include optimizer states (for training)
            include_gradients: Include gradient storage (for training)
            
        Returns:
            Estimated memory in MB
        """
        bytes_per_param = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }.get(dtype, 2)
        
        # Base model
        memory = num_params * bytes_per_param
        
        # Gradients (same size as model)
        if include_gradients:
            memory *= 2
        
        # Optimizer states (Adam/AdamW: 2x for momentum and variance)
        if include_optimizer:
            memory += num_params * 4 * 2  # Always float32 for optimizer
        
        return memory / (1024**2)


class GradientCheckpointManager:
    """
    Gradient checkpointing for training with limited memory.
    Trades compute for memory.
    """
    
    @staticmethod
    def enable(model: torch.nn.Module, checkpoint_ratio: float = 0.5):
        """
        Enable gradient checkpointing on a fraction of layers.
        
        Args:
            model: The model
            checkpoint_ratio: Fraction of layers to checkpoint (0-1)
        """
        layers = [m for m in model.modules() if hasattr(m, 'gradient_checkpointing')]
        num_to_checkpoint = int(len(layers) * checkpoint_ratio)
        
        for i, layer in enumerate(layers):
            if i < num_to_checkpoint:
                layer.gradient_checkpointing = True
    
    @staticmethod
    def wrap_layer(layer: torch.nn.Module) -> torch.nn.Module:
        """Wrap a layer with gradient checkpointing"""
        from torch.utils.checkpoint import checkpoint
        
        class CheckpointedLayer(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
            
            def forward(self, *args, **kwargs):
                return checkpoint(self.layer, *args, use_reentrant=False, **kwargs)
        
        return CheckpointedLayer(layer)


def get_optimal_batch_size(
    model: torch.nn.Module,
    sample_input: Dict[str, torch.Tensor],
    max_memory_fraction: float = 0.9,
    starting_batch_size: int = 1,
) -> int:
    """
    Find optimal batch size by binary search.
    
    Args:
        model: The model
        sample_input: A sample input (batch size 1)
        max_memory_fraction: Maximum fraction of GPU memory to use
        starting_batch_size: Initial batch size to try
        
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return starting_batch_size
    
    device = next(model.parameters()).device
    total_memory = torch.cuda.get_device_properties(device).total_memory
    max_memory = int(total_memory * max_memory_fraction)
    
    batch_size = starting_batch_size
    
    while True:
        try:
            # Scale inputs to batch size
            batched_input = {
                k: v.expand(batch_size, *v.shape[1:]) if v.dim() > 0 else v
                for k, v in sample_input.items()
            }
            
            torch.cuda.empty_cache()
            
            with torch.inference_mode():
                _ = model(**batched_input)
            
            # Check memory
            if torch.cuda.memory_allocated(device) > max_memory:
                break
            
            batch_size *= 2
            
        except RuntimeError:  # OOM
            break
    
    optimal = max(1, batch_size // 2)
    logger.info(f"Optimal batch size: {optimal}")
    
    return optimal
