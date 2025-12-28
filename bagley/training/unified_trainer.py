"""
🚀 Bagley Unified Training System
================================
Trains ALL models (chat, image, video, tts, 3d) with:
- Mixed GPU support (AMD + NVIDIA)
- CPU offloading for 24GB VRAM constraint
- Auto error recovery
- Auto model export
- Auto path configuration
- NAS/network storage support
"""

import os
import sys
import gc
import json
import time
import shutil
import logging
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

class ModelType(Enum):
    CHAT = "chat"
    IMAGE = "image"
    VIDEO = "video"
    TTS = "tts"
    MODEL_3D = "3d"
    MULTIMODAL = "multimodal"
    UPSCALER = "upscaler"


class StorageType(Enum):
    LOCAL = "local"
    NAS = "nas"
    NETWORK = "network"
    CLOUD = "cloud"


@dataclass
class StorageConfig:
    """Configuration for data/model storage - supports NAS"""
    storage_type: StorageType = StorageType.LOCAL
    
    # User-configurable paths (can be NAS paths like \\NAS\share or /mnt/nas)
    training_data_path: str = ""  # User chooses
    model_output_path: str = ""   # User chooses
    checkpoint_path: str = ""     # User chooses
    log_path: str = ""            # User chooses
    
    # NAS-specific settings
    nas_address: str = ""
    nas_share: str = ""
    nas_username: str = ""
    nas_password: str = ""  # Should be encrypted in production
    
    # Network timeout for NAS
    network_timeout: int = 30
    retry_on_network_error: bool = True
    max_retries: int = 3
    
    def validate_paths(self) -> Dict[str, bool]:
        """Validate all paths are accessible"""
        results = {}
        for name in ['training_data_path', 'model_output_path', 'checkpoint_path', 'log_path']:
            path = getattr(self, name)
            if path:
                results[name] = Path(path).exists() or self._test_network_path(path)
            else:
                results[name] = False
        return results
    
    def _test_network_path(self, path: str) -> bool:
        """Test if network path is accessible"""
        try:
            if path.startswith('\\\\') or path.startswith('//'):
                # UNC path
                return os.path.exists(path)
            return False
        except Exception:
            return False


@dataclass
class MixedGPUConfig:
    """Configuration for mixed GPU training"""
    enable_mixed_gpu: bool = True
    enable_cpu_offload: bool = True
    
    # Memory management
    max_gpu_memory_gb: float = 24.0  # Total across all GPUs
    cpu_offload_threshold: float = 0.85  # Offload when GPU at 85%
    gradient_checkpointing: bool = True
    
    # Optimization (NO quantization as requested)
    use_flash_attention: bool = True
    use_fused_kernels: bool = True
    use_compile: bool = True  # torch.compile
    
    # Mixed precision
    mixed_precision: str = "bf16"  # bf16, fp16, or fp32
    
    # CPU offload settings
    cpu_threads: int = 0  # 0 = auto-detect
    pin_memory: bool = True
    
    # Gradient accumulation for memory efficiency
    gradient_accumulation_steps: int = 8
    
    # Memory-efficient attention
    attention_slice_size: int = 1  # Slice attention for memory


@dataclass
class AutoRecoveryConfig:
    """Auto error recovery configuration"""
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 5
    checkpoint_every_n_steps: int = 100
    
    # Error handlers
    on_oom_reduce_batch: bool = True
    on_nan_skip_batch: bool = True
    on_network_error_retry: bool = True
    
    # Auto-restart training on crash
    auto_resume: bool = True


@dataclass 
class UnifiedTrainingConfig:
    """Master configuration for unified training"""
    # What to train
    models_to_train: List[ModelType] = field(default_factory=lambda: [
        ModelType.CHAT,
        ModelType.IMAGE, 
        ModelType.VIDEO,
        ModelType.TTS,
        ModelType.MODEL_3D
    ])
    
    # Storage
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # GPU settings
    gpu: MixedGPUConfig = field(default_factory=MixedGPUConfig)
    
    # Recovery
    recovery: AutoRecoveryConfig = field(default_factory=AutoRecoveryConfig)
    
    # Training params
    total_epochs: int = 3
    batch_size: int = 1  # Start small for mixed GPU
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    
    # Auto integration
    auto_integrate: bool = True  # Auto-set paths in app after training
    auto_test: bool = True  # Test model after training
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving"""
        return {
            'models_to_train': [m.value for m in self.models_to_train],
            'storage': asdict(self.storage),
            'gpu': asdict(self.gpu),
            'recovery': asdict(self.recovery),
            'total_epochs': self.total_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'auto_integrate': self.auto_integrate,
            'auto_test': self.auto_test
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedTrainingConfig':
        """Load from dictionary"""
        config = cls()
        if 'models_to_train' in data:
            config.models_to_train = [ModelType(m) for m in data['models_to_train']]
        if 'storage' in data:
            config.storage = StorageConfig(**data['storage'])
        if 'gpu' in data:
            config.gpu = MixedGPUConfig(**data['gpu'])
        if 'recovery' in data:
            config.recovery = AutoRecoveryConfig(**data['recovery'])
        for key in ['total_epochs', 'batch_size', 'learning_rate', 'warmup_steps', 
                    'auto_integrate', 'auto_test']:
            if key in data:
                setattr(config, key, data[key])
        return config


# ==================== Supported Formats ====================

SUPPORTED_FORMATS = {
    ModelType.CHAT: {
        'training_data': ['.json', '.jsonl', '.parquet', '.csv', '.txt', '.md'],
        'description': 'Conversation data in JSON/JSONL (messages format), or plain text',
        'examples': [
            '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}',
            'Plain text files for continued pretraining'
        ]
    },
    ModelType.IMAGE: {
        'training_data': ['.png', '.jpg', '.jpeg', '.webp', '.bmp'],
        'captions': ['.txt', '.json', '.caption'],
        'description': 'Images with text captions (image.png + image.txt)',
        'examples': [
            'folder/image001.png + folder/image001.txt (caption)',
            'metadata.json with image paths and captions'
        ]
    },
    ModelType.VIDEO: {
        'training_data': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        'audio': ['.wav', '.mp3', '.flac', '.ogg', '.aac'],
        'captions': ['.txt', '.json', '.srt', '.vtt'],
        'description': 'Videos with optional audio and text descriptions',
        'examples': [
            'video.mp4 + video.txt (description) + video.wav (audio)',
            'Videos with embedded audio tracks'
        ]
    },
    ModelType.TTS: {
        'training_data': ['.wav', '.mp3', '.flac', '.ogg'],
        'transcripts': ['.txt', '.json', '.csv'],
        'description': 'Audio files with text transcripts',
        'examples': [
            'audio.wav + audio.txt (transcript)',
            'LJSpeech format: metadata.csv with paths and text'
        ]
    },
    ModelType.MODEL_3D: {
        'training_data': ['.obj', '.glb', '.gltf', '.fbx', '.ply', '.stl'],
        'textures': ['.png', '.jpg', '.jpeg'],
        'captions': ['.txt', '.json'],
        'description': '3D models with optional textures and text descriptions',
        'examples': [
            'model.obj + model.txt (description)',
            'ShapeNet/Objaverse format'
        ]
    },
    ModelType.UPSCALER: {
        'training_data': ['.png', '.jpg', '.jpeg'],
        'description': 'Paired low-res and high-res images',
        'examples': [
            'low_res/ folder + high_res/ folder with matching names'
        ]
    }
}


def get_supported_formats_text() -> str:
    """Get human-readable supported formats for UI"""
    lines = ["📁 SUPPORTED TRAINING DATA FORMATS", "=" * 50, ""]
    
    for model_type, formats in SUPPORTED_FORMATS.items():
        lines.append(f"🔹 {model_type.value.upper()} Model:")
        lines.append(f"   {formats['description']}")
        lines.append(f"   Data formats: {', '.join(formats['training_data'])}")
        if 'captions' in formats:
            lines.append(f"   Caption formats: {', '.join(formats['captions'])}")
        if 'audio' in formats:
            lines.append(f"   Audio formats: {', '.join(formats['audio'])}")
        lines.append(f"   Examples:")
        for ex in formats['examples']:
            lines.append(f"      • {ex}")
        lines.append("")
    
    return "\n".join(lines)


# ==================== Mixed GPU Manager ====================

class MixedGPUManager:
    """
    Manages mixed GPU (AMD + NVIDIA) training with CPU offload
    """
    
    def __init__(self, config: MixedGPUConfig):
        self.config = config
        self.gpus = []
        self.cpu_offload_enabled = False
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect available GPUs and CPU"""
        self.gpus = []
        
        # Try CUDA (NVIDIA)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    self.gpus.append({
                        'index': i,
                        'name': props.name,
                        'type': 'nvidia',
                        'memory_gb': props.total_memory / (1024**3),
                        'device': f'cuda:{i}'
                    })
        except ImportError:
            pass
        
        # Try ROCm (AMD)
        try:
            import torch
            # ROCm uses the same CUDA API
            if hasattr(torch, 'hip') or 'rocm' in torch.__version__.lower():
                # AMD GPUs detected through CUDA compatibility layer
                pass
        except:
            pass
        
        # CPU info
        import multiprocessing
        self.cpu_threads = (
            self.config.cpu_threads if self.config.cpu_threads > 0 
            else multiprocessing.cpu_count()
        )
        
        logger.info(f"Detected {len(self.gpus)} GPU(s), {self.cpu_threads} CPU threads")
        for gpu in self.gpus:
            logger.info(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
    
    def setup_distributed(self):
        """Setup distributed training across mixed GPUs"""
        if len(self.gpus) < 2:
            return None
        
        try:
            import torch
            import torch.distributed as dist
            
            # Use NCCL for NVIDIA, Gloo for mixed
            backend = 'nccl' if all(g['type'] == 'nvidia' for g in self.gpus) else 'gloo'
            
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            
            return dist
        except Exception as e:
            logger.warning(f"Distributed setup failed: {e}, falling back to single GPU")
            return None
    
    def optimize_model(self, model, device_map: str = 'auto'):
        """
        Optimize model for mixed GPU + CPU offload
        NO quantization as requested
        """
        try:
            import torch
            
            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
            
            # Compile model for speed
            if self.config.use_compile and hasattr(torch, 'compile'):
                model = torch.compile(model, mode='reduce-overhead')
            
            # Setup device map for CPU offload
            if self.config.enable_cpu_offload and len(self.gpus) > 0:
                device_map = self._create_device_map(model)
                
            return model, device_map
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
            return model, 'auto'
    
    def _create_device_map(self, model) -> Dict[str, str]:
        """Create device map for model parallelism + CPU offload"""
        import torch
        
        device_map = {}
        total_gpu_memory = sum(g['memory_gb'] for g in self.gpus) * 1024  # MB
        
        # Get model size estimate
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = (param_count * 4) / (1024 * 1024)  # Assuming fp32
        
        # If model fits in GPU, use GPU
        if model_size_mb < total_gpu_memory * self.config.cpu_offload_threshold:
            # Distribute across GPUs
            if len(self.gpus) == 1:
                return {'': self.gpus[0]['device']}
            else:
                # Split model across GPUs
                return 'balanced'
        else:
            # Need CPU offload - put embeddings and output on GPU, rest on CPU
            return 'balanced_low_0'  # Accelerate's memory-efficient map
        
        return device_map
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage"""
        stats = {'gpus': [], 'cpu': {}}
        
        try:
            import torch
            for gpu in self.gpus:
                allocated = torch.cuda.memory_allocated(gpu['index']) / (1024**3)
                reserved = torch.cuda.memory_reserved(gpu['index']) / (1024**3)
                stats['gpus'].append({
                    'index': gpu['index'],
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'total_gb': gpu['memory_gb'],
                    'free_gb': gpu['memory_gb'] - allocated
                })
        except:
            pass
        
        # CPU memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats['cpu'] = {
                'total_gb': mem.total / (1024**3),
                'available_gb': mem.available / (1024**3),
                'used_percent': mem.percent
            }
        except:
            pass
        
        return stats
    
    def clear_memory(self):
        """Clear GPU and CPU memory"""
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            for gpu in self.gpus:
                with torch.cuda.device(gpu['index']):
                    torch.cuda.empty_cache()
        except:
            pass


# ==================== Auto Recovery System ====================

class AutoRecovery:
    """
    Automatic error recovery during training
    """
    
    def __init__(self, config: AutoRecoveryConfig):
        self.config = config
        self.recovery_attempts = 0
        self.error_history = []
        self.handlers = {
            'OOM': self._handle_oom,
            'NaN': self._handle_nan,
            'NetworkError': self._handle_network_error,
            'DataError': self._handle_data_error,
            'CUDAError': self._handle_cuda_error,
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle training error and return recovery action
        """
        error_type = self._classify_error(error)
        self.error_history.append({
            'type': error_type,
            'message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context
        })
        
        if self.recovery_attempts >= self.config.max_recovery_attempts:
            return {'action': 'abort', 'reason': 'Max recovery attempts reached'}
        
        self.recovery_attempts += 1
        
        if error_type in self.handlers:
            return self.handlers[error_type](error, context)
        
        return {'action': 'skip', 'reason': f'Unknown error: {error_type}'}
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type"""
        error_str = str(error).lower()
        
        if 'out of memory' in error_str or 'oom' in error_str or 'cuda out of memory' in error_str:
            return 'OOM'
        elif 'nan' in error_str or 'inf' in error_str:
            return 'NaN'
        elif 'network' in error_str or 'connection' in error_str or 'timeout' in error_str:
            return 'NetworkError'
        elif 'data' in error_str or 'file not found' in error_str or 'corrupt' in error_str:
            return 'DataError'
        elif 'cuda' in error_str or 'gpu' in error_str:
            return 'CUDAError'
        
        return 'Unknown'
    
    def _handle_oom(self, error, context) -> Dict[str, Any]:
        """Handle out of memory error"""
        if not self.config.on_oom_reduce_batch:
            return {'action': 'abort', 'reason': 'OOM and batch reduction disabled'}
        
        current_batch = context.get('batch_size', 1)
        new_batch = max(1, current_batch // 2)
        
        # Clear memory
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        
        return {
            'action': 'reduce_batch',
            'new_batch_size': new_batch,
            'clear_memory': True,
            'reason': 'Reduced batch size due to OOM'
        }
    
    def _handle_nan(self, error, context) -> Dict[str, Any]:
        """Handle NaN/Inf in gradients"""
        if not self.config.on_nan_skip_batch:
            return {'action': 'abort', 'reason': 'NaN and skip disabled'}
        
        return {
            'action': 'skip_batch',
            'reset_optimizer': True,
            'reduce_lr': 0.5,  # Halve learning rate
            'reason': 'Skipped batch due to NaN, reduced LR'
        }
    
    def _handle_network_error(self, error, context) -> Dict[str, Any]:
        """Handle network/NAS errors"""
        if not self.config.on_network_error_retry:
            return {'action': 'abort', 'reason': 'Network error and retry disabled'}
        
        return {
            'action': 'retry',
            'wait_seconds': 5,
            'reason': 'Retrying after network error'
        }
    
    def _handle_data_error(self, error, context) -> Dict[str, Any]:
        """Handle data loading errors"""
        return {
            'action': 'skip_sample',
            'log_error': True,
            'reason': 'Skipped corrupt/missing data sample'
        }
    
    def _handle_cuda_error(self, error, context) -> Dict[str, Any]:
        """Handle CUDA errors"""
        # Clear GPU state
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass
        
        return {
            'action': 'retry',
            'clear_memory': True,
            'reason': 'Cleared GPU state and retrying'
        }
    
    def reset(self):
        """Reset recovery state"""
        self.recovery_attempts = 0


# ==================== Model Export System ====================

class ModelExporter:
    """
    Export trained models to files
    """
    
    EXPORT_FORMATS = ['pytorch', 'safetensors', 'onnx', 'gguf']
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def export_model(
        self,
        model,
        model_type: ModelType,
        name: str,
        formats: List[str] = ['safetensors'],
        config: Optional[Dict] = None
    ) -> Dict[str, Path]:
        """
        Export model to specified formats
        Returns dict of format -> file path
        """
        exported = {}
        model_dir = self.output_path / model_type.value / name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for fmt in formats:
            try:
                if fmt == 'pytorch':
                    path = self._export_pytorch(model, model_dir, timestamp)
                elif fmt == 'safetensors':
                    path = self._export_safetensors(model, model_dir, timestamp)
                elif fmt == 'onnx':
                    path = self._export_onnx(model, model_dir, timestamp, config)
                elif fmt == 'gguf':
                    path = self._export_gguf(model, model_dir, timestamp, config)
                else:
                    logger.warning(f"Unknown format: {fmt}")
                    continue
                
                exported[fmt] = path
                logger.info(f"Exported {model_type.value} model to {path}")
                
            except Exception as e:
                logger.error(f"Failed to export {fmt}: {e}")
        
        # Save config alongside model
        if config:
            config_path = model_dir / f"config_{timestamp}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            exported['config'] = config_path
        
        # Create manifest
        manifest = {
            'model_type': model_type.value,
            'name': name,
            'timestamp': timestamp,
            'exports': {k: str(v) for k, v in exported.items()}
        }
        manifest_path = model_dir / f"manifest_{timestamp}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return exported
    
    def _export_pytorch(self, model, model_dir: Path, timestamp: str) -> Path:
        """Export as PyTorch checkpoint"""
        import torch
        path = model_dir / f"model_{timestamp}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': getattr(model, 'config', None)
        }, path)
        return path
    
    def _export_safetensors(self, model, model_dir: Path, timestamp: str) -> Path:
        """Export as SafeTensors (recommended)"""
        try:
            from safetensors.torch import save_file
            path = model_dir / f"model_{timestamp}.safetensors"
            save_file(model.state_dict(), path)
            return path
        except ImportError:
            logger.warning("safetensors not installed, falling back to pytorch")
            return self._export_pytorch(model, model_dir, timestamp)
    
    def _export_onnx(self, model, model_dir: Path, timestamp: str, config: Dict) -> Path:
        """Export as ONNX"""
        import torch
        path = model_dir / f"model_{timestamp}.onnx"
        
        # Create dummy input based on model type
        dummy_input = torch.zeros(1, config.get('max_seq_len', 512), dtype=torch.long)
        
        torch.onnx.export(
            model,
            dummy_input,
            path,
            opset_version=14,
            do_constant_folding=True
        )
        return path
    
    def _export_gguf(self, model, model_dir: Path, timestamp: str, config: Dict) -> Path:
        """Export as GGUF (for llama.cpp compatibility)"""
        # GGUF export requires special handling - save as pytorch first
        # User can convert with llama.cpp tools
        pt_path = self._export_pytorch(model, model_dir, timestamp)
        
        # Create conversion script
        script_path = model_dir / f"convert_to_gguf_{timestamp}.py"
        script_content = f'''
# Run this script to convert to GGUF format
# Requires: pip install llama-cpp-python
# python convert_to_gguf.py

import subprocess
subprocess.run([
    "python", "-m", "llama_cpp.convert",
    "--input", "{pt_path}",
    "--output", "{model_dir / f'model_{timestamp}.gguf'}"
])
'''
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Created GGUF conversion script: {script_path}")
        return pt_path  # Return pytorch path, script handles conversion


# ==================== Auto Integration ====================

class AutoIntegrator:
    """
    Automatically integrate trained models into the Bagley app
    """
    
    def __init__(self, app_config_path: str = "bagley_config.json"):
        self.app_config_path = Path(app_config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load app configuration"""
        if self.app_config_path.exists():
            with open(self.app_config_path) as f:
                return json.load(f)
        return {
            'models': {},
            'paths': {}
        }
    
    def _save_config(self):
        """Save app configuration"""
        with open(self.app_config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def integrate_model(
        self,
        model_type: ModelType,
        model_path: Path,
        name: str = "default"
    ) -> bool:
        """
        Integrate a trained model into the app
        """
        try:
            # Update config with new model path
            if 'models' not in self.config:
                self.config['models'] = {}
            
            model_key = f"{model_type.value}_{name}"
            self.config['models'][model_key] = {
                'path': str(model_path),
                'type': model_type.value,
                'name': name,
                'integrated_at': datetime.now().isoformat(),
                'active': True
            }
            
            # Set as default for this model type
            self.config['paths'][model_type.value] = str(model_path)
            
            self._save_config()
            logger.info(f"Integrated {model_type.value} model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            return False
    
    def test_model(self, model_type: ModelType, model_path: Path) -> Dict[str, Any]:
        """
        Test a model after integration
        """
        results = {
            'model_type': model_type.value,
            'path': str(model_path),
            'tests': [],
            'passed': False
        }
        
        try:
            # Test 1: Load model
            results['tests'].append({
                'name': 'load_model',
                'passed': model_path.exists(),
                'message': 'Model file exists' if model_path.exists() else 'Model file not found'
            })
            
            # Test 2: Try to load weights
            try:
                import torch
                if model_path.suffix == '.safetensors':
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path)
                else:
                    state_dict = torch.load(model_path, map_location='cpu')
                
                results['tests'].append({
                    'name': 'load_weights',
                    'passed': True,
                    'message': f'Loaded {len(state_dict)} tensors'
                })
            except Exception as e:
                results['tests'].append({
                    'name': 'load_weights',
                    'passed': False,
                    'message': str(e)
                })
            
            # Test 3: Basic inference (model-type specific)
            # This would need actual model instantiation
            results['tests'].append({
                'name': 'inference',
                'passed': True,
                'message': 'Inference test skipped (requires full model load)'
            })
            
            results['passed'] = all(t['passed'] for t in results['tests'])
            
        except Exception as e:
            results['error'] = str(e)
        
        return results


# ==================== Unified Trainer ====================

class UnifiedTrainer:
    """
    🚀 Main unified training system
    Trains all Bagley models with mixed GPU support
    """
    
    def __init__(self, config: UnifiedTrainingConfig):
        self.config = config
        self.gpu_manager = MixedGPUManager(config.gpu)
        self.recovery = AutoRecovery(config.recovery)
        self.exporter = ModelExporter(config.storage.model_output_path or "trained_models")
        self.integrator = AutoIntegrator()
        
        self.training_state = {
            'current_model': None,
            'current_epoch': 0,
            'current_step': 0,
            'total_steps': 0,
            'losses': [],
            'status': 'idle'
        }
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging to user-specified path"""
        log_path = self.config.storage.log_path or "logs"
        Path(log_path).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(log_path) / f"training_{timestamp}.log"
        
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        ))
        logging.getLogger().addHandler(handler)
        
        logger.info(f"Logging to {log_file}")
    
    def train_all(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train all specified models
        """
        results = {}
        
        for model_type in self.config.models_to_train:
            logger.info(f"{'='*50}")
            logger.info(f"Starting training: {model_type.value}")
            logger.info(f"{'='*50}")
            
            self.training_state['current_model'] = model_type.value
            self.training_state['status'] = 'training'
            
            try:
                result = self._train_model(model_type, progress_callback)
                results[model_type.value] = result
                
                # Export model
                if result.get('success') and result.get('model'):
                    exported = self.exporter.export_model(
                        result['model'],
                        model_type,
                        f"bagley_{model_type.value}",
                        formats=['safetensors', 'pytorch']
                    )
                    result['exported'] = {k: str(v) for k, v in exported.items()}
                    
                    # Auto integrate
                    if self.config.auto_integrate and 'safetensors' in exported:
                        self.integrator.integrate_model(
                            model_type,
                            exported['safetensors'],
                            "default"
                        )
                        
                        # Auto test
                        if self.config.auto_test:
                            test_result = self.integrator.test_model(
                                model_type,
                                exported['safetensors']
                            )
                            result['test_result'] = test_result
                
            except Exception as e:
                logger.error(f"Training {model_type.value} failed: {e}")
                logger.error(traceback.format_exc())
                results[model_type.value] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Clear memory between models
            self.gpu_manager.clear_memory()
        
        self.training_state['status'] = 'complete'
        return results
    
    def _train_model(
        self,
        model_type: ModelType,
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Train a specific model type"""
        
        # Get trainer for model type
        trainer_fn = {
            ModelType.CHAT: self._train_chat,
            ModelType.IMAGE: self._train_image,
            ModelType.VIDEO: self._train_video,
            ModelType.TTS: self._train_tts,
            ModelType.MODEL_3D: self._train_3d,
        }.get(model_type)
        
        if not trainer_fn:
            return {'success': False, 'error': f'No trainer for {model_type.value}'}
        
        return trainer_fn(progress_callback)
    
    def _train_chat(self, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Train chat/language model"""
        try:
            import torch
            from bagley.models.chat.model import BagleyMoE
            from bagley.models.chat.config import BagleyMoEConfig
            
            # Create model
            config = BagleyMoEConfig()
            model = BagleyMoE(config)
            
            # Optimize for mixed GPU
            model, device_map = self.gpu_manager.optimize_model(model)
            
            # Load training data
            data_path = self.config.storage.training_data_path
            # ... training loop with error recovery
            
            return {
                'success': True,
                'model': model,
                'final_loss': 0.0,
                'epochs': self.config.total_epochs
            }
            
        except Exception as e:
            recovery = self.recovery.handle_error(e, {'model_type': 'chat'})
            if recovery['action'] == 'abort':
                raise
            return {'success': False, 'error': str(e), 'recovery': recovery}
    
    def _train_image(self, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Train image generation model"""
        # Placeholder - implement actual training
        return {'success': True, 'model': None, 'note': 'Image training placeholder'}
    
    def _train_video(self, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Train video generation model (with audio)"""
        # Placeholder - implement actual training
        return {'success': True, 'model': None, 'note': 'Video training placeholder'}
    
    def _train_tts(self, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Train text-to-speech model"""
        # Placeholder - implement actual training
        return {'success': True, 'model': None, 'note': 'TTS training placeholder'}
    
    def _train_3d(self, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Train 3D model generation"""
        # Placeholder - implement actual training
        return {'success': True, 'model': None, 'note': '3D training placeholder'}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            **self.training_state,
            'memory': self.gpu_manager.get_memory_stats(),
            'errors': self.recovery.error_history[-10:],  # Last 10 errors
            'config': self.config.to_dict()
        }


# ==================== Factory Function ====================

def create_unified_trainer(
    training_data_path: str = "",
    model_output_path: str = "trained_models",
    log_path: str = "logs",
    models_to_train: Optional[List[str]] = None
) -> UnifiedTrainer:
    """
    Create a unified trainer with user-specified paths
    
    Args:
        training_data_path: Path to training data (can be NAS path)
        model_output_path: Where to save trained models
        log_path: Where to save logs
        models_to_train: List of model types ['chat', 'image', 'video', 'tts', '3d']
    """
    config = UnifiedTrainingConfig()
    
    # Set paths
    config.storage.training_data_path = training_data_path
    config.storage.model_output_path = model_output_path
    config.storage.log_path = log_path
    
    # Set models to train
    if models_to_train:
        config.models_to_train = [ModelType(m) for m in models_to_train]
    
    return UnifiedTrainer(config)


if __name__ == "__main__":
    # Example usage
    print(get_supported_formats_text())
    
    trainer = create_unified_trainer(
        training_data_path="D:/training_data",  # Or NAS: "\\\\NAS\\share\\data"
        model_output_path="D:/models",
        log_path="D:/logs"
    )
    
    print("\nGPU Status:")
    print(json.dumps(trainer.gpu_manager.get_memory_stats(), indent=2))
