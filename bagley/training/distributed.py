"""
🌐 Distributed Training Manager
Handles multi-node, multi-GPU training setup
"""

import os
import socket
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a compute node"""
    hostname: str
    num_gpus: int
    gpu_memory: List[int]  # Memory per GPU in GB
    rank: int


class DistributedManager:
    """
    🌐 Distributed Training Manager
    
    Manages distributed training across:
    - Multiple GPUs on single node
    - Multiple nodes in a cluster
    - Various parallelism strategies
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        master_addr: Optional[str] = None,
        master_port: int = 29500,
    ):
        self.backend = backend
        self.master_addr = master_addr or os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = master_port
        
        # State
        self.initialized = False
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.node_rank = 0
        
        logger.info(f"DistributedManager initialized (backend={backend})")
    
    def setup(self, world_size: Optional[int] = None):
        """
        Initialize distributed process group.
        
        Args:
            world_size: Total number of processes. If None, auto-detect.
        """
        if self.initialized:
            logger.warning("Distributed already initialized")
            return
        
        # Get rank from environment (set by launcher)
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
        
        if self.world_size == 1:
            logger.info("Single GPU training, skipping distributed setup")
            self.initialized = True
            return
        
        # Set environment variables
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            world_size=self.world_size,
            rank=self.rank,
        )
        
        # Set device
        torch.cuda.set_device(self.local_rank)
        
        self.initialized = True
        logger.info(f"Distributed initialized: rank {self.rank}/{self.world_size}, local_rank {self.local_rank}")
    
    def cleanup(self):
        """Cleanup distributed resources"""
        if self.initialized and self.world_size > 1:
            dist.destroy_process_group()
            self.initialized = False
            logger.info("Distributed cleanup complete")
    
    @property
    def is_main(self) -> bool:
        """Check if this is the main process"""
        return self.rank == 0
    
    @property
    def device(self) -> torch.device:
        """Get current device"""
        return torch.device(f"cuda:{self.local_rank}")
    
    def barrier(self):
        """Synchronization barrier"""
        if self.world_size > 1:
            dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM):
        """All-reduce tensor across processes"""
        if self.world_size > 1:
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes"""
        if self.world_size == 1:
            return [tensor]
        
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return gathered
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        """Broadcast tensor from source to all processes"""
        if self.world_size > 1:
            dist.broadcast(tensor, src=src)
        return tensor
    
    def get_node_info(self) -> NodeInfo:
        """Get information about current node"""
        return NodeInfo(
            hostname=socket.gethostname(),
            num_gpus=torch.cuda.device_count(),
            gpu_memory=[
                torch.cuda.get_device_properties(i).total_memory // (1024**3)
                for i in range(torch.cuda.device_count())
            ],
            rank=self.node_rank,
        )
    
    def print_cluster_info(self):
        """Print cluster configuration"""
        if not self.is_main:
            return
        
        node_info = self.get_node_info()
        
        print("\n" + "=" * 50)
        print("🖥️ CLUSTER CONFIGURATION")
        print("=" * 50)
        print(f"World Size: {self.world_size}")
        print(f"Backend: {self.backend}")
        print(f"Master: {self.master_addr}:{self.master_port}")
        print(f"\nNode: {node_info.hostname}")
        print(f"GPUs: {node_info.num_gpus}")
        for i, mem in enumerate(node_info.gpu_memory):
            print(f"  GPU {i}: {mem}GB")
        print("=" * 50 + "\n")


class ParallelismManager:
    """
    Manages different types of parallelism.
    
    Supports:
    - Data Parallelism (DDP)
    - Tensor Parallelism
    - Pipeline Parallelism
    - Expert Parallelism (MoE)
    """
    
    def __init__(
        self,
        world_size: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,
    ):
        self.world_size = world_size
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.expert_parallel_size = expert_parallel_size
        
        # Compute data parallel size
        self.data_parallel_size = world_size // (
            tensor_parallel_size * pipeline_parallel_size
        )
        
        # Validate configuration
        self._validate()
        
        # Create process groups
        self._create_groups()
    
    def _validate(self):
        """Validate parallelism configuration"""
        total = self.tensor_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
        
        if total != self.world_size:
            raise ValueError(
                f"Parallelism config ({self.tensor_parallel_size}x{self.pipeline_parallel_size}x{self.data_parallel_size}) "
                f"doesn't match world size ({self.world_size})"
            )
    
    def _create_groups(self):
        """Create distributed process groups"""
        # Data parallel group
        self.data_parallel_group = None
        
        # Tensor parallel group
        self.tensor_parallel_group = None
        
        # Pipeline parallel group
        self.pipeline_parallel_group = None
        
        # Expert parallel group
        self.expert_parallel_group = None
        
        if not dist.is_initialized():
            return
        
        rank = dist.get_rank()
        
        # Create tensor parallel groups
        if self.tensor_parallel_size > 1:
            for i in range(self.world_size // self.tensor_parallel_size):
                ranks = list(range(
                    i * self.tensor_parallel_size,
                    (i + 1) * self.tensor_parallel_size
                ))
                group = dist.new_group(ranks)
                
                if rank in ranks:
                    self.tensor_parallel_group = group
        
        # Create data parallel groups
        if self.data_parallel_size > 1:
            for i in range(self.tensor_parallel_size):
                ranks = list(range(i, self.world_size, self.tensor_parallel_size))
                group = dist.new_group(ranks)
                
                if rank in ranks:
                    self.data_parallel_group = group
    
    def get_tensor_parallel_rank(self, global_rank: int) -> int:
        """Get tensor parallel rank from global rank"""
        return global_rank % self.tensor_parallel_size
    
    def get_data_parallel_rank(self, global_rank: int) -> int:
        """Get data parallel rank from global rank"""
        return global_rank // self.tensor_parallel_size
    
    def get_pipeline_stage(self, global_rank: int) -> int:
        """Get pipeline stage from global rank"""
        return (global_rank // self.tensor_parallel_size) % self.pipeline_parallel_size


def setup_distributed_environment():
    """
    Setup distributed environment.
    
    Call this at the start of training script.
    """
    # Check for SLURM
    if "SLURM_JOB_ID" in os.environ:
        _setup_slurm()
    # Check for torch.distributed.launch / torchrun
    elif "LOCAL_RANK" in os.environ:
        pass  # Already set
    else:
        # Single GPU
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"


def _setup_slurm():
    """Setup environment variables from SLURM"""
    # Get SLURM job info
    job_id = os.environ.get("SLURM_JOB_ID")
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_PER_NODE", torch.cuda.device_count()))
    node_list = os.environ.get("SLURM_JOB_NODELIST", "localhost")
    node_id = int(os.environ.get("SLURM_NODEID", 0))
    local_id = int(os.environ.get("SLURM_LOCALID", 0))
    
    # Compute ranks
    world_size = num_nodes * gpus_per_node
    rank = node_id * gpus_per_node + local_id
    
    # Set environment
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_id)
    
    # Get master address from node list
    if "[" in node_list:
        # Parse compressed format like node[001-004]
        import re
        match = re.match(r"(.+)\[(\d+)-(\d+)\]", node_list)
        if match:
            prefix = match.group(1)
            start = int(match.group(2))
            master = f"{prefix}{start:03d}"
        else:
            master = node_list.split(",")[0]
    else:
        master = node_list.split(",")[0]
    
    os.environ["MASTER_ADDR"] = master
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    
    logger.info(f"SLURM setup: rank {rank}/{world_size}, master {master}")


def get_optimal_parallelism(
    num_gpus: int,
    model_size_billions: float,
    sequence_length: int,
) -> Dict[str, int]:
    """
    Get optimal parallelism configuration.
    
    Args:
        num_gpus: Number of available GPUs
        model_size_billions: Model size in billions of parameters
        sequence_length: Training sequence length
        
    Returns:
        Dictionary with optimal parallelism settings
    """
    # Heuristics based on model size and resources
    
    if model_size_billions <= 7:
        # Small models: data parallelism only
        return {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus,
        }
    
    elif model_size_billions <= 30:
        # Medium models: some tensor parallelism
        tp = min(4, num_gpus)
        return {
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": 1,
            "data_parallel_size": num_gpus // tp,
        }
    
    else:
        # Large models: full model parallelism
        tp = min(8, num_gpus)
        pp = min(4, num_gpus // tp)
        dp = num_gpus // (tp * pp)
        
        return {
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "data_parallel_size": max(1, dp),
        }
