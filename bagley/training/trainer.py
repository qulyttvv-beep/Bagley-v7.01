"""
🏋️ Bagley Trainer
Main training loop with distributed support
"""

import os
import math
import logging
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from bagley.training.config import (
    TrainingConfig, 
    DistributedConfig,
    OptimizerType,
    SchedulerType,
    PrecisionType,
)

logger = logging.getLogger(__name__)


class BagleyTrainer:
    """
    🏋️ Main Trainer for Bagley Models
    
    Supports:
    - Distributed training (DeepSpeed, FSDP)
    - Mixed precision (BF16, FP16)
    - Gradient checkpointing
    - Wandb logging
    - Checkpoint management
    - MoE load balancing
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_config: TrainingConfig,
        dist_config: DistributedConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_loss: Optional[Callable] = None,
    ):
        self.model = model
        self.train_config = train_config
        self.dist_config = dist_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_loss = compute_loss or self._default_loss
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup
        self._setup_distributed()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_scaler()
        self._setup_logging()
        
        logger.info(f"Initialized BagleyTrainer for {train_config.model_type}")
    
    def _setup_distributed(self):
        """Setup distributed training"""
        self.is_distributed = self.dist_config.world_size > 1
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = self.dist_config.world_size
        self.is_main = self.local_rank == 0
        
        if self.is_distributed:
            torch.distributed.init_process_group(
                backend=self.dist_config.backend,
                init_method=self.dist_config.init_method,
            )
            torch.cuda.set_device(self.local_rank)
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model = self.model.to(self.device)
        
        # DeepSpeed or FSDP wrapping
        if self.dist_config.use_deepspeed:
            self._setup_deepspeed()
        elif self.dist_config.use_fsdp:
            self._setup_fsdp()
        elif self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.dist_config.find_unused_parameters,
            )
        
        logger.info(f"Distributed setup: rank {self.local_rank}/{self.world_size}")
    
    def _setup_deepspeed(self):
        """Initialize DeepSpeed"""
        try:
            import deepspeed
            
            ds_config = self.dist_config.get_deepspeed_config()
            
            self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
                model=self.model,
                config=ds_config,
            )
            
            self.use_deepspeed = True
            logger.info("DeepSpeed initialized")
            
        except ImportError:
            logger.warning("DeepSpeed not available, falling back to DDP")
            self.use_deepspeed = False
    
    def _setup_fsdp(self):
        """Initialize FSDP"""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            CPUOffload,
            MixedPrecision,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        fsdp_config = self.dist_config.get_fsdp_config()
        
        # Mixed precision
        if self.train_config.precision == PrecisionType.BF16:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            mp_policy = None
        
        # CPU offload
        cpu_offload = CPUOffload(offload_params=fsdp_config["cpu_offload"])
        
        self.model = FSDP(
            self.model,
            cpu_offload=cpu_offload,
            mixed_precision=mp_policy,
            sync_module_states=True,
        )
        
        self.use_fsdp = True
        logger.info("FSDP initialized")
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        if hasattr(self, 'use_deepspeed') and self.use_deepspeed:
            return  # DeepSpeed handles this
        
        # Get parameters with weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'ln' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        # Select optimizer
        if self.train_config.optimizer == OptimizerType.ADAMW:
            self.optimizer = AdamW(
                param_groups,
                lr=self.train_config.learning_rate,
                betas=(self.train_config.adam_beta1, self.train_config.adam_beta2),
                eps=self.train_config.adam_epsilon,
            )
        elif self.train_config.optimizer == OptimizerType.LION:
            try:
                from lion_pytorch import Lion
                self.optimizer = Lion(
                    param_groups,
                    lr=self.train_config.learning_rate,
                )
            except ImportError:
                logger.warning("Lion not available, using AdamW")
                self.optimizer = AdamW(param_groups, lr=self.train_config.learning_rate)
        else:
            self.optimizer = AdamW(param_groups, lr=self.train_config.learning_rate)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if hasattr(self, 'use_deepspeed') and self.use_deepspeed:
            return  # DeepSpeed handles this
        
        num_training_steps = len(self.train_dataloader) * self.train_config.num_epochs
        num_warmup_steps = self.train_config.warmup_steps
        
        if self.train_config.lr_scheduler == SchedulerType.COSINE_WARMUP:
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
            )
    
    def _setup_scaler(self):
        """Setup gradient scaler for mixed precision"""
        if self.train_config.precision == PrecisionType.FP16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _setup_logging(self):
        """Setup logging (wandb, tensorboard)"""
        if self.is_main and self.train_config.log_to_wandb:
            try:
                import wandb
                
                wandb.init(
                    project=self.train_config.wandb_project,
                    entity=self.train_config.wandb_entity,
                    name=self.train_config.run_name,
                    config={
                        "train_config": vars(self.train_config),
                        "dist_config": vars(self.dist_config),
                    },
                )
                self.use_wandb = True
                
            except ImportError:
                logger.warning("Wandb not available")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def _default_loss(self, model_output, batch) -> torch.Tensor:
        """Default loss computation"""
        if hasattr(model_output, 'loss'):
            return model_output.loss
        elif isinstance(model_output, tuple) and len(model_output) > 1:
            return model_output[1]
        else:
            raise ValueError("Could not extract loss from model output")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        total_steps = len(self.train_dataloader) * self.train_config.num_epochs
        
        for epoch in range(self.train_config.num_epochs):
            self.epoch = epoch
            self.model.train()
            
            epoch_loss = 0.0
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                loss = self._training_step(batch)
                epoch_loss += loss.item()
                
                # Logging
                if self.global_step % self.train_config.logging_steps == 0:
                    self._log_metrics({
                        "train/loss": loss.item(),
                        "train/lr": self._get_lr(),
                        "train/epoch": epoch,
                        "train/step": self.global_step,
                    })
                
                # Evaluation
                if (self.eval_dataloader is not None and 
                    self.global_step % self.train_config.eval_steps == 0):
                    eval_loss = self.evaluate()
                    self._log_metrics({"eval/loss": eval_loss})
                
                # Checkpointing
                if self.global_step % self.train_config.save_steps == 0:
                    self.save_checkpoint()
                
                self.global_step += 1
            
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch}: avg_loss = {avg_epoch_loss:.4f}")
        
        # Final save
        self.save_checkpoint(final=True)
        logger.info("Training complete!")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step"""
        # Get autocast context
        autocast_dtype = self._get_autocast_dtype()
        
        # Gradient accumulation
        accumulation_steps = self.train_config.gradient_accumulation_steps
        
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            outputs = self.model(**batch)
            loss = self.compute_loss(outputs, batch)
            loss = loss / accumulation_steps
        
        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        elif hasattr(self, 'use_deepspeed') and self.use_deepspeed:
            self.model.backward(loss)
        else:
            loss.backward()
        
        # Step optimizer
        if (self.global_step + 1) % accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif hasattr(self, 'use_deepspeed') and self.use_deepspeed:
                self.model.step()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.max_grad_norm,
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        
        return loss * accumulation_steps
    
    def evaluate(self) -> float:
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._move_to_device(batch)
                
                with torch.cuda.amp.autocast(dtype=self._get_autocast_dtype()):
                    outputs = self.model(**batch)
                    loss = self.compute_loss(outputs, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.eval_dataloader)
        
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint"""
        if not self.is_main:
            return
        
        output_dir = Path(self.train_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            checkpoint_dir = output_dir / "final"
        else:
            checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"
        
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        if hasattr(self, 'use_deepspeed') and self.use_deepspeed:
            self.model.save_checkpoint(str(checkpoint_dir))
        else:
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            torch.save(model_to_save.state_dict(), checkpoint_dir / "model.pt")
        
        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        if hasattr(self, 'use_deepspeed') and self.use_deepspeed:
            self.model.load_checkpoint(str(checkpoint_dir))
        else:
            state_dict = torch.load(checkpoint_dir / "model.pt", map_location=self.device)
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load.load_state_dict(state_dict)
        
        # Load optimizer
        if (checkpoint_dir / "optimizer.pt").exists():
            self.optimizer.load_state_dict(
                torch.load(checkpoint_dir / "optimizer.pt", map_location=self.device)
            )
        
        # Load training state
        if (checkpoint_dir / "training_state.json").exists():
            with open(checkpoint_dir / "training_state.json") as f:
                state = json.load(f)
                self.global_step = state["global_step"]
                self.epoch = state["epoch"]
                self.best_eval_loss = state["best_eval_loss"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _get_autocast_dtype(self):
        """Get autocast dtype"""
        if self.train_config.precision == PrecisionType.BF16:
            return torch.bfloat16
        elif self.train_config.precision == PrecisionType.FP16:
            return torch.float16
        else:
            return torch.float32
    
    def _get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]["lr"]
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics"""
        if self.use_wandb and self.is_main:
            import wandb
            wandb.log(metrics, step=self.global_step)
        
        if self.is_main:
            logger.info(f"Step {self.global_step}: {metrics}")


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine schedule with warmup"""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)
