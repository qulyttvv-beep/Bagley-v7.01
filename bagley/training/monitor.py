"""
🎛️ Bagley Training Monitor & Smart Data System
Auto-sorts data, monitors GPU, prevents fires
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Types of training data"""
    CHAT = "chat"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class TrainingJob:
    """A training job"""
    model_type: str
    data_paths: List[str]
    output_dir: str
    status: str = "pending"  # pending, running, paused, completed, failed
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_loss: float = 0.0
    start_time: Optional[float] = None
    gpu_temps: List[float] = field(default_factory=list)


class GPUMonitor:
    """
    🌡️ GPU Temperature Monitor
    
    Monitors GPU temps and can pause training if overheating.
    Works with both AMD (ROCm) and NVIDIA.
    """
    
    def __init__(
        self,
        max_temp: float = 85.0,  # Celsius
        critical_temp: float = 90.0,
        cooldown_temp: float = 75.0,
        check_interval: float = 5.0,  # seconds
    ):
        self.max_temp = max_temp
        self.critical_temp = critical_temp
        self.cooldown_temp = cooldown_temp
        self.check_interval = check_interval
        
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._on_overheat: Optional[Callable] = None
        self._on_cooldown: Optional[Callable] = None
        
        # Detect GPU type
        self.gpu_type = self._detect_gpu_type()
        logger.info(f"GPU Monitor initialized ({self.gpu_type})")
    
    def _detect_gpu_type(self) -> str:
        """Detect if AMD or NVIDIA"""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0).lower()
                if 'amd' in name or 'radeon' in name:
                    return "amd"
                return "nvidia"
        except:
            pass
        return "unknown"
    
    def get_temps(self) -> List[float]:
        """Get current GPU temperatures"""
        temps = []
        
        if self.gpu_type == "nvidia":
            temps = self._get_nvidia_temps()
        elif self.gpu_type == "amd":
            temps = self._get_amd_temps()
        
        return temps if temps else [0.0]
    
    def _get_nvidia_temps(self) -> List[float]:
        """Get NVIDIA GPU temps using nvidia-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            temps = [float(t.strip()) for t in result.stdout.strip().split('\n') if t.strip()]
            return temps
        except Exception as e:
            logger.debug(f"nvidia-smi failed: {e}")
            return []
    
    def _get_amd_temps(self) -> List[float]:
        """Get AMD GPU temps using rocm-smi"""
        try:
            import subprocess
            result = subprocess.run(
                ['rocm-smi', '--showtemp', '--json'],
                capture_output=True, text=True, timeout=5
            )
            data = json.loads(result.stdout)
            temps = []
            for gpu_id, gpu_data in data.items():
                if 'Temperature' in gpu_data:
                    # Parse temperature value
                    temp_str = gpu_data['Temperature'].get('edge', '0')
                    temp = float(str(temp_str).replace('c', '').replace('C', '').strip())
                    temps.append(temp)
            return temps
        except Exception as e:
            logger.debug(f"rocm-smi failed: {e}")
            # Fallback method
            return self._get_amd_temps_fallback()
    
    def _get_amd_temps_fallback(self) -> List[float]:
        """Fallback AMD temp reading"""
        try:
            import subprocess
            result = subprocess.run(
                ['rocm-smi', '-t'],
                capture_output=True, text=True, timeout=5
            )
            temps = []
            for line in result.stdout.split('\n'):
                if 'Temperature' in line or 'Temp' in line:
                    # Try to extract number
                    import re
                    nums = re.findall(r'(\d+\.?\d*)', line)
                    if nums:
                        temps.append(float(nums[0]))
            return temps
        except:
            return []
    
    def start_monitoring(
        self,
        on_overheat: Optional[Callable] = None,
        on_cooldown: Optional[Callable] = None,
    ):
        """Start background temperature monitoring"""
        self._on_overheat = on_overheat
        self._on_cooldown = on_cooldown
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            temps = self.get_temps()
            max_temp = max(temps) if temps else 0
            
            if max_temp >= self.critical_temp:
                logger.warning(f"🔥 CRITICAL TEMP: {max_temp}°C - Pausing training!")
                self._paused = True
                if self._on_overheat:
                    self._on_overheat(max_temp)
            
            elif max_temp >= self.max_temp and not self._paused:
                logger.warning(f"⚠️ High temp: {max_temp}°C - Pausing training")
                self._paused = True
                if self._on_overheat:
                    self._on_overheat(max_temp)
            
            elif max_temp <= self.cooldown_temp and self._paused:
                logger.info(f"✅ Cooled down: {max_temp}°C - Resuming training")
                self._paused = False
                if self._on_cooldown:
                    self._on_cooldown(max_temp)
            
            time.sleep(self.check_interval)
    
    @property
    def is_paused(self) -> bool:
        return self._paused


class SmartDataSorter:
    """
    🧠 Smart Training Data Sorter
    
    Automatically detects and sorts training data by type:
    - Chat/Text conversations
    - Images with captions
    - Videos
    - Audio/TTS data
    - Code
    """
    
    # File extensions for each type
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb'}
    DATA_EXTENSIONS = {'.json', '.jsonl', '.csv', '.parquet', '.txt'}
    
    def __init__(self):
        self.sorted_data: Dict[DataType, List[Path]] = {
            DataType.CHAT: [],
            DataType.IMAGE: [],
            DataType.VIDEO: [],
            DataType.AUDIO: [],
            DataType.CODE: [],
            DataType.UNKNOWN: [],
        }
        self.stats: Dict[str, int] = {}
    
    def scan_folders(self, folders: List[str]) -> Dict[DataType, List[Path]]:
        """
        Scan folders and sort files by type.
        
        Args:
            folders: List of folder paths to scan
            
        Returns:
            Dictionary mapping DataType to list of file paths
        """
        # Reset
        for key in self.sorted_data:
            self.sorted_data[key] = []
        
        for folder in folders:
            folder_path = Path(folder)
            if not folder_path.exists():
                logger.warning(f"Folder not found: {folder}")
                continue
            
            self._scan_folder(folder_path)
        
        # Calculate stats
        self.stats = {
            dt.value: len(files) for dt, files in self.sorted_data.items()
        }
        
        logger.info(f"Scan complete: {self.stats}")
        return self.sorted_data
    
    def _scan_folder(self, folder: Path):
        """Recursively scan a folder"""
        for item in folder.rglob('*'):
            if item.is_file():
                data_type = self._classify_file(item)
                self.sorted_data[data_type].append(item)
    
    def _classify_file(self, file_path: Path) -> DataType:
        """Classify a file by type"""
        ext = file_path.suffix.lower()
        
        # By extension
        if ext in self.IMAGE_EXTENSIONS:
            return DataType.IMAGE
        elif ext in self.VIDEO_EXTENSIONS:
            return DataType.VIDEO
        elif ext in self.AUDIO_EXTENSIONS:
            return DataType.AUDIO
        elif ext in self.CODE_EXTENSIONS:
            return DataType.CODE
        elif ext in self.DATA_EXTENSIONS:
            # Need to inspect content
            return self._classify_data_file(file_path)
        
        return DataType.UNKNOWN
    
    def _classify_data_file(self, file_path: Path) -> DataType:
        """Classify JSON/JSONL/CSV files by content"""
        try:
            ext = file_path.suffix.lower()
            
            if ext == '.jsonl':
                # Read first line
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    data = json.loads(first_line)
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        data = data[0]
            else:
                return DataType.UNKNOWN
            
            # Classify by fields
            if isinstance(data, dict):
                keys = set(data.keys())
                
                # Chat data
                if 'messages' in keys or 'conversations' in keys:
                    return DataType.CHAT
                if 'instruction' in keys or 'input' in keys or 'output' in keys:
                    return DataType.CHAT
                if 'prompt' in keys and 'response' in keys:
                    return DataType.CHAT
                if 'text' in keys and len(keys) <= 3:
                    return DataType.CHAT
                
                # Image data
                if 'image' in keys or 'image_path' in keys:
                    return DataType.IMAGE
                if 'caption' in keys and ('url' in keys or 'file' in keys):
                    return DataType.IMAGE
                
                # Audio/TTS data
                if 'audio' in keys or 'audio_path' in keys:
                    return DataType.AUDIO
                if 'wav' in keys or 'transcript' in keys:
                    return DataType.AUDIO
                
                # Video data
                if 'video' in keys or 'video_path' in keys:
                    return DataType.VIDEO
                
                # Code data
                if 'code' in keys or 'function' in keys:
                    return DataType.CODE
            
            return DataType.CHAT  # Default to chat for text data
            
        except Exception as e:
            logger.debug(f"Error classifying {file_path}: {e}")
            return DataType.UNKNOWN
    
    def get_training_config(self) -> Dict[str, Any]:
        """Generate training configuration based on sorted data"""
        config = {
            "models_to_train": [],
            "data_config": {},
        }
        
        if self.sorted_data[DataType.CHAT]:
            config["models_to_train"].append("chat")
            config["data_config"]["chat"] = {
                "files": [str(f) for f in self.sorted_data[DataType.CHAT]],
                "count": len(self.sorted_data[DataType.CHAT]),
            }
        
        if self.sorted_data[DataType.IMAGE]:
            config["models_to_train"].append("image")
            config["data_config"]["image"] = {
                "files": [str(f) for f in self.sorted_data[DataType.IMAGE]],
                "count": len(self.sorted_data[DataType.IMAGE]),
            }
        
        if self.sorted_data[DataType.VIDEO]:
            config["models_to_train"].append("video")
            config["data_config"]["video"] = {
                "files": [str(f) for f in self.sorted_data[DataType.VIDEO]],
                "count": len(self.sorted_data[DataType.VIDEO]),
            }
        
        if self.sorted_data[DataType.AUDIO]:
            config["models_to_train"].append("tts")
            config["data_config"]["tts"] = {
                "files": [str(f) for f in self.sorted_data[DataType.AUDIO]],
                "count": len(self.sorted_data[DataType.AUDIO]),
            }
        
        return config


class TrainingOrchestrator:
    """
    🎯 Training Orchestrator
    
    Manages training of all models with:
    - GPU temperature monitoring
    - Auto-pause on overheat
    - Smart data routing
    - Progress tracking
    """
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.data_sorter = SmartDataSorter()
        self.jobs: List[TrainingJob] = []
        self.current_job: Optional[TrainingJob] = None
        self._paused = False
        self._callbacks: Dict[str, List[Callable]] = {
            "on_progress": [],
            "on_overheat": [],
            "on_cooldown": [],
            "on_complete": [],
            "on_error": [],
        }
    
    def add_callback(self, event: str, callback: Callable):
        """Add callback for events"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args):
        """Emit event to callbacks"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def setup_training(self, data_folders: List[str], output_dir: str):
        """
        Setup training from data folders.
        
        Args:
            data_folders: Folders containing training data
            output_dir: Where to save trained models
        """
        # Scan and sort data
        logger.info(f"Scanning {len(data_folders)} folders...")
        self.data_sorter.scan_folders(data_folders)
        
        # Generate training config
        config = self.data_sorter.get_training_config()
        
        # Create jobs for each model type
        self.jobs = []
        for model_type in config["models_to_train"]:
            data_config = config["data_config"][model_type]
            
            job = TrainingJob(
                model_type=model_type,
                data_paths=data_config["files"],
                output_dir=os.path.join(output_dir, model_type),
            )
            self.jobs.append(job)
            logger.info(f"Created {model_type} training job with {data_config['count']} files")
        
        return config
    
    def start_training(self):
        """Start training all models"""
        # Start GPU monitoring
        self.gpu_monitor.start_monitoring(
            on_overheat=self._handle_overheat,
            on_cooldown=self._handle_cooldown,
        )
        
        # Train each model
        for job in self.jobs:
            if job.status == "completed":
                continue
            
            self.current_job = job
            job.status = "running"
            job.start_time = time.time()
            
            try:
                self._train_model(job)
                job.status = "completed"
                self._emit("on_complete", job)
            except Exception as e:
                job.status = "failed"
                logger.error(f"Training failed: {e}")
                self._emit("on_error", job, str(e))
        
        self.gpu_monitor.stop_monitoring()
    
    def _train_model(self, job: TrainingJob):
        """Train a single model"""
        logger.info(f"Starting {job.model_type} training...")
        
        # This would call the actual trainer
        # For now, simulate training loop
        from bagley.training import BagleyTrainer, TrainingConfig, DistributedConfig
        
        # Setup would go here...
        # trainer = BagleyTrainer(model, config, ...)
        # trainer.train()
        
        # Simulated progress for demo
        job.total_steps = 1000
        for step in range(job.total_steps):
            # Check if paused (overheating)
            while self._paused or self.gpu_monitor.is_paused:
                time.sleep(1)
            
            # Training step would go here
            job.current_step = step
            job.progress = step / job.total_steps
            job.gpu_temps = self.gpu_monitor.get_temps()
            
            self._emit("on_progress", job)
    
    def _handle_overheat(self, temp: float):
        """Handle GPU overheat"""
        self._paused = True
        logger.warning(f"🔥 Training paused - GPU at {temp}°C")
        self._emit("on_overheat", temp)
    
    def _handle_cooldown(self, temp: float):
        """Handle GPU cooldown"""
        self._paused = False
        logger.info(f"✅ Training resumed - GPU at {temp}°C")
        self._emit("on_cooldown", temp)
    
    def pause(self):
        """Manually pause training"""
        self._paused = True
    
    def resume(self):
        """Manually resume training"""
        self._paused = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            "paused": self._paused,
            "gpu_temps": self.gpu_monitor.get_temps(),
            "current_job": {
                "model": self.current_job.model_type if self.current_job else None,
                "progress": self.current_job.progress if self.current_job else 0,
                "step": self.current_job.current_step if self.current_job else 0,
                "loss": self.current_job.current_loss if self.current_job else 0,
            } if self.current_job else None,
            "jobs": [
                {
                    "model": j.model_type,
                    "status": j.status,
                    "progress": j.progress,
                }
                for j in self.jobs
            ],
        }
