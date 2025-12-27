"""
🚀 Bagley Training Architecture
Distributed training for GPU clusters with smart data management
Works with 1 GPU, 2 GPUs, or 100 GPUs - even mixed AMD/NVIDIA
"""

from bagley.training.config import TrainingConfig, DistributedConfig
from bagley.training.trainer import BagleyTrainer
from bagley.training.data import DatasetConfig, DataLoader
from bagley.training.distributed import DistributedManager
from bagley.training.monitor import (
    GPUMonitor,
    SmartDataSorter,
    TrainingOrchestrator,
    TrainingJob,
    DataType,
)
from bagley.training.pipeline import (
    SmartDataPipeline,
    ChatDataProcessor,
    ImageDataProcessor,
    AudioDataProcessor,
)
from bagley.training.flexible_trainer import (
    FlexibleTrainer,
    AutoTrainer,
    SmartLogger,
    HardwareDetector,
    GPUInfo,
    GPUType,
    TrainingCheckpoint,
)

__all__ = [
    # Config
    "TrainingConfig",
    "DistributedConfig",
    # Training
    "BagleyTrainer",
    "DatasetConfig",
    "DataLoader",
    "DistributedManager",
    # Monitoring
    "GPUMonitor",
    "SmartDataSorter",
    "TrainingOrchestrator",
    "TrainingJob",
    "DataType",
    # Pipeline
    "SmartDataPipeline",
    "ChatDataProcessor",
    "ImageDataProcessor",
    "AudioDataProcessor",
    # Flexible Training
    "FlexibleTrainer",
    "AutoTrainer",
    "SmartLogger",
    "HardwareDetector",
    "GPUInfo",
    "GPUType",
    "TrainingCheckpoint",
]
