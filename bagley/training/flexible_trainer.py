"""
🎯 Flexible Training System
Works with 1 GPU, 2 GPUs, or 100 GPUs
Even mixed AMD + NVIDIA (with limitations)
Auto-detects and adapts to your hardware
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class GPUType(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """Information about a GPU"""
    index: int
    name: str
    gpu_type: GPUType
    memory_total: int  # MB
    memory_free: int  # MB
    compute_capability: Optional[str] = None


@dataclass
class TrainingCheckpoint:
    """Checkpoint for resumable training"""
    step: int
    epoch: int
    model_state_path: str
    optimizer_state_path: str
    scheduler_state_path: str
    config: Dict[str, Any]
    timestamp: str
    loss: float
    metrics: Dict[str, float] = field(default_factory=dict)


class SmartLogger:
    """
    📝 Smart Logging System
    
    - Creates numbered log files (log1.txt, log2.txt, etc.)
    - Separates by training run
    - Includes timestamps, metrics, GPU stats
    - Easy to parse and analyze
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_log_num = self._get_next_log_number()
        self.log_file = self.log_dir / f"log{self.current_log_num}.txt"
        self.json_log = self.log_dir / f"log{self.current_log_num}.jsonl"
        
        # Setup file handler
        self.file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        
        logger.addHandler(self.file_handler)
        
        self._write_header()
    
    def _get_next_log_number(self) -> int:
        """Get next log number"""
        existing = list(self.log_dir.glob("log*.txt"))
        if not existing:
            return 1
        
        numbers = []
        for f in existing:
            try:
                num = int(f.stem.replace("log", ""))
                numbers.append(num)
            except ValueError:
                pass
        
        return max(numbers, default=0) + 1
    
    def _write_header(self):
        """Write log header"""
        header = f"""
{'='*60}
BAGLEY TRAINING LOG #{self.current_log_num}
Started: {datetime.now().isoformat()}
{'='*60}
"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        gpu_temps: List[float] = None,
        gpu_mem: List[float] = None,
        metrics: Dict[str, float] = None,
    ):
        """Log a training step"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "loss": loss,
            "lr": lr,
            "gpu_temps": gpu_temps or [],
            "gpu_mem": gpu_mem or [],
            "metrics": metrics or {},
        }
        
        # JSON log
        with open(self.json_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Text log
        msg = f"Step {step:>6} | Loss: {loss:.4f} | LR: {lr:.2e}"
        if gpu_temps:
            msg += f" | GPU Temps: {[f'{t:.0f}°C' for t in gpu_temps]}"
        
        logger.info(msg)
    
    def log_checkpoint(self, checkpoint: TrainingCheckpoint):
        """Log checkpoint save"""
        logger.info(f"💾 Checkpoint saved at step {checkpoint.step} (loss: {checkpoint.loss:.4f})")
    
    def log_event(self, event: str, details: Dict = None):
        """Log an event"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details or {},
        }
        
        with open(self.json_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        
        logger.info(f"📌 {event}")
    
    def close(self):
        """Close the logger"""
        footer = f"""
{'='*60}
TRAINING COMPLETED
Ended: {datetime.now().isoformat()}
{'='*60}
"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(footer)
        
        logger.removeHandler(self.file_handler)


class HardwareDetector:
    """
    🔍 Detects available hardware
    Works with mixed GPU setups
    """
    
    @staticmethod
    def detect_gpus() -> List[GPUInfo]:
        """Detect all available GPUs"""
        gpus = []
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("No CUDA GPUs detected")
                return gpus
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Detect type
                name_lower = props.name.lower()
                if 'amd' in name_lower or 'radeon' in name_lower or 'instinct' in name_lower:
                    gpu_type = GPUType.AMD
                elif 'nvidia' in name_lower or 'geforce' in name_lower or 'rtx' in name_lower or 'tesla' in name_lower:
                    gpu_type = GPUType.NVIDIA
                else:
                    gpu_type = GPUType.UNKNOWN
                
                gpu = GPUInfo(
                    index=i,
                    name=props.name,
                    gpu_type=gpu_type,
                    memory_total=props.total_memory // (1024 * 1024),
                    memory_free=props.total_memory // (1024 * 1024),  # Approximate
                    compute_capability=f"{props.major}.{props.minor}" if gpu_type == GPUType.NVIDIA else None,
                )
                gpus.append(gpu)
                
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
        
        return gpus
    
    @staticmethod
    def get_optimal_config(gpus: List[GPUInfo]) -> Dict[str, Any]:
        """Get optimal training config based on hardware"""
        if not gpus:
            return {"device": "cpu", "batch_size": 1}
        
        total_memory = sum(g.memory_total for g in gpus)
        num_gpus = len(gpus)
        
        # Check for mixed setup
        gpu_types = set(g.gpu_type for g in gpus)
        is_mixed = len(gpu_types) > 1
        
        config = {
            "device": "cuda",
            "num_gpus": num_gpus,
            "is_mixed": is_mixed,
            "gpu_types": [g.gpu_type.value for g in gpus],
        }
        
        # Batch size based on memory
        if total_memory >= 80000:  # 80GB+
            config["batch_size"] = 32
            config["gradient_accumulation"] = 1
        elif total_memory >= 48000:  # 48GB+
            config["batch_size"] = 16
            config["gradient_accumulation"] = 2
        elif total_memory >= 24000:  # 24GB+
            config["batch_size"] = 8
            config["gradient_accumulation"] = 4
        elif total_memory >= 12000:  # 12GB+
            config["batch_size"] = 4
            config["gradient_accumulation"] = 8
        else:  # < 12GB
            config["batch_size"] = 1
            config["gradient_accumulation"] = 32
        
        # Distributed strategy
        if num_gpus == 1:
            config["strategy"] = "single"
        elif is_mixed:
            # Can't use NCCL with mixed GPUs easily
            config["strategy"] = "gloo"  # Fallback
            logger.warning("Mixed GPU setup detected - using GLOO backend (slower)")
        else:
            config["strategy"] = "ddp"
        
        return config


class FlexibleTrainer:
    """
    🚀 Flexible Trainer
    
    - Works with 1, 2, or N GPUs
    - Auto-detects hardware
    - Resumable from checkpoints
    - Auto-train when data detected
    - Smart GPU utilization
    """
    
    def __init__(
        self,
        model_type: str,
        output_dir: str = "./outputs",
        checkpoint_every: int = 1000,
        log_every: int = 10,
    ):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        
        # Detect hardware
        self.gpus = HardwareDetector.detect_gpus()
        self.config = HardwareDetector.get_optimal_config(self.gpus)
        
        # Logging
        self.logger = SmartLogger(str(self.output_dir / "logs"))
        
        # State
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Interruption handling
        self._interrupted = False
        self._setup_interrupt_handler()
        
        logger.info(f"🎯 FlexibleTrainer initialized")
        logger.info(f"   GPUs: {len(self.gpus)}")
        logger.info(f"   Config: {self.config}")
    
    def _setup_interrupt_handler(self):
        """Setup graceful interrupt handling"""
        import signal
        
        def handler(signum, frame):
            logger.warning("⚠️ Interrupt received - saving checkpoint and exiting...")
            self._interrupted = True
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def load_checkpoint(self, checkpoint_path: str = None) -> bool:
        """Load from checkpoint for resuming"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
            if not checkpoints:
                return False
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            import torch
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.current_step = checkpoint.get('step', 0)
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            
            if self.model and 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            
            if self.optimizer and 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            if self.scheduler and 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            logger.info(f"✅ Resumed from checkpoint: step {self.current_step}, epoch {self.current_epoch}")
            self.logger.log_event("checkpoint_loaded", {"path": str(checkpoint_path), "step": self.current_step})
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def save_checkpoint(self, loss: float = None):
        """Save checkpoint"""
        import torch
        
        checkpoint = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }
        
        if self.model:
            checkpoint['model_state'] = self.model.state_dict()
        if self.optimizer:
            checkpoint['optimizer_state'] = self.optimizer.state_dict()
        if self.scheduler:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
        
        # Save numbered checkpoint
        path = self.checkpoint_dir / f"checkpoint_{self.current_step:08d}.pt"
        torch.save(checkpoint, path)
        
        # Save as latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best if improved
        if loss is not None and loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        self.logger.log_checkpoint(TrainingCheckpoint(
            step=self.current_step,
            epoch=self.current_epoch,
            model_state_path=str(path),
            optimizer_state_path=str(path),
            scheduler_state_path=str(path),
            config=self.config,
            timestamp=checkpoint['timestamp'],
            loss=loss or 0.0,
        ))
    
    def setup_model(self, model):
        """Setup model for training"""
        import torch
        
        self.model = model
        
        if self.config["num_gpus"] == 1:
            self.model = self.model.to("cuda:0")
        elif self.config["num_gpus"] > 1:
            if self.config.get("is_mixed"):
                # For mixed GPUs, just use first GPU
                logger.warning("Mixed GPU setup - using single GPU for stability")
                self.model = self.model.to("cuda:0")
            else:
                # DataParallel for simplicity
                self.model = torch.nn.DataParallel(self.model)
                self.model = self.model.to("cuda")
        
        return self.model
    
    def train_step(self, batch) -> float:
        """Single training step"""
        import torch
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
        elif torch.is_tensor(batch):
            batch = batch.to("cuda")
        
        # Forward
        outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        self.current_step += 1
        
        return loss.item()
    
    def train(
        self,
        dataloader,
        epochs: int = 1,
        callbacks: List[Callable] = None,
    ):
        """Main training loop"""
        from bagley.training.monitor import GPUMonitor
        
        callbacks = callbacks or []
        gpu_monitor = GPUMonitor()
        
        self.logger.log_event("training_started", {
            "model_type": self.model_type,
            "epochs": epochs,
            "config": self.config,
        })
        
        try:
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                epoch_loss = 0.0
                num_batches = 0
                
                for batch in dataloader:
                    if self._interrupted:
                        self.save_checkpoint(epoch_loss / max(num_batches, 1))
                        return
                    
                    # Check GPU temp
                    if gpu_monitor.is_paused:
                        logger.warning("⏸ Training paused (GPU overheating)")
                        while gpu_monitor.is_paused:
                            time.sleep(5)
                        logger.info("▶ Training resumed")
                    
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    
                    # Logging
                    if self.current_step % self.log_every == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        temps = gpu_monitor.get_temps()
                        self.logger.log_step(
                            step=self.current_step,
                            loss=loss,
                            lr=lr,
                            gpu_temps=temps,
                        )
                    
                    # Checkpoint
                    if self.current_step % self.checkpoint_every == 0:
                        self.save_checkpoint(loss)
                    
                    # Callbacks
                    for cb in callbacks:
                        cb(self, loss)
                
                avg_loss = epoch_loss / max(num_batches, 1)
                logger.info(f"📊 Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}")
                self.save_checkpoint(avg_loss)
            
            self.logger.log_event("training_completed", {
                "final_step": self.current_step,
                "best_loss": self.best_loss,
            })
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.save_checkpoint()
            raise
        
        finally:
            self.logger.close()


class AutoTrainer:
    """
    🤖 Auto-Trainer
    
    Watches folders for new data and automatically trains.
    """
    
    def __init__(
        self,
        watch_folders: List[str],
        output_dir: str = "./outputs",
        check_interval: float = 30.0,  # seconds
    ):
        self.watch_folders = [Path(f) for f in watch_folders]
        self.output_dir = Path(output_dir)
        self.check_interval = check_interval
        
        self._running = False
        self._thread = None
        self._known_files: set = set()
        
        # Callbacks
        self.on_new_data: Optional[Callable] = None
        self.on_training_start: Optional[Callable] = None
        self.on_training_complete: Optional[Callable] = None
    
    def _scan_for_new_files(self) -> List[Path]:
        """Scan for new training files"""
        new_files = []
        
        for folder in self.watch_folders:
            if not folder.exists():
                continue
            
            for file in folder.rglob("*"):
                if file.is_file() and file not in self._known_files:
                    self._known_files.add(file)
                    new_files.append(file)
        
        return new_files
    
    def _get_file_hash(self, files: List[Path]) -> str:
        """Get hash of file list for detecting changes"""
        content = "".join(sorted(str(f) for f in files))
        return hashlib.md5(content.encode()).hexdigest()
    
    def start(self):
        """Start auto-training watcher"""
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("🔄 Auto-trainer started - watching for new data")
    
    def stop(self):
        """Stop watcher"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _watch_loop(self):
        """Watch loop"""
        while self._running:
            try:
                new_files = self._scan_for_new_files()
                
                if new_files:
                    logger.info(f"📁 Found {len(new_files)} new files")
                    
                    if self.on_new_data:
                        self.on_new_data(new_files)
                    
                    # Trigger training
                    self._trigger_training(new_files)
                
            except Exception as e:
                logger.error(f"Auto-trainer error: {e}")
            
            time.sleep(self.check_interval)
    
    def _trigger_training(self, new_files: List[Path]):
        """Trigger training on new data"""
        from bagley.training.monitor import SmartDataSorter, DataType
        
        # Sort files by type
        sorter = SmartDataSorter()
        
        for file in new_files:
            data_type = sorter._classify_file(file)
            sorter.sorted_data[data_type].append(file)
        
        # Train each model type that has new data
        for data_type, files in sorter.sorted_data.items():
            if not files:
                continue
            
            model_type_map = {
                DataType.CHAT: "chat",
                DataType.IMAGE: "image",
                DataType.VIDEO: "video",
                DataType.AUDIO: "tts",
            }
            
            model_type = model_type_map.get(data_type)
            if model_type:
                logger.info(f"🏋️ Auto-training {model_type} model with {len(files)} files")
                
                if self.on_training_start:
                    self.on_training_start(model_type, files)
                
                # Would start actual training here
                # trainer = FlexibleTrainer(model_type, str(self.output_dir))
                # trainer.train(...)
                
                if self.on_training_complete:
                    self.on_training_complete(model_type)
