import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging
import os

@dataclass
class DistributedConfig:
    """Distributed training configuration"""
    num_gpus: int = torch.cuda.device_count()
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"
    
class DistributedModelTrainer:
    """Distributed training manager for LSTM model"""
    
    def __init__(
        self,
        model_trainer: ModelTrainer,
        config: DistributedConfig
    ):
        self.model_trainer = model_trainer
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate distributed configuration"""
        if self.config.num_gpus < 1:
            raise ValueError("No GPU available for distributed training")
        if self.config.num_nodes < 1:
            raise ValueError("Number of nodes must be at least 1")
            
    def setup_distributed(self, rank: int):
        """Setup distributed training environment"""
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method='env://',
            world_size=self.config.num_nodes * self.config.num_gpus,
            rank=rank
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        
    def cleanup(self):
        """Cleanup distributed training resources"""
        dist.destroy_process_group()
        
    async def train_distributed(
        self,
        train_data: torch.Tensor,
        valid_data: torch.Tensor,
        num_epochs: int
    ):
        """Launch distributed training"""
        mp.spawn(
            self._train_worker,
            args=(train_data, valid_data, num_epochs),
            nprocs=self.config.num_gpus,
            join=True
        )
        
    def _train_worker(
        self,
        rank: int,
        train_data: torch.Tensor,
        valid_data: torch.Tensor,
        num_epochs: int
    ):
        """Worker process for distributed training"""
        try:
            # Setup distributed environment
            self.setup_distributed(rank)
            
            # Wrap model in DDP
            model = DDP(
                self.model_trainer.model,
                device_ids=[rank]
            )
            
            # Create distributed samplers
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data,
                num_replicas=self.config.num_gpus,
                rank=rank
            )
            
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_data,
                num_replicas=self.config.num_gpus,
                rank=rank
            )
            
            # Training loop
            for epoch in range(num_epochs):
                train_sampler.set_epoch(epoch)
                train_loss = self._train_epoch(
                    model,
                    train_data,
                    train_sampler,
                    rank
                )
                
                # Validation on rank 0
                if rank == 0:
                    valid_loss = self._validate_epoch(
                        model,
                        valid_data,
                        valid_sampler
                    )
                    logging.info(
                        f"Epoch {epoch}: "
                        f"train_loss={train_loss:.4f}, "
                        f"valid_loss={valid_loss:.4f}"
                    )
                    
                # Synchronize processes
                dist.barrier()
                
        except Exception as e:
            logging.error(f"Error in training worker {rank}: {e}")
            raise
            
        finally:
            self.cleanup()
            
    def _train_epoch(
        self,
        model: DDP,
        data: torch.Tensor,
        sampler: torch.utils.data.distributed.DistributedSampler,
        rank: int
    ) -> float:
        """Train one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in torch.utils.data.DataLoader(
            data,
            batch_size=self.model_trainer.config.batch_size,
            sampler=sampler
        ):
            loss = self.model_trainer._train_batch(
                model,
                batch,
                rank
            )
            total_loss += loss
            num_batches += 1
            
        return total_loss / num_batches 