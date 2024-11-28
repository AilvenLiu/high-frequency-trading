import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import asyncio
import torch
from signal_generation.models.distributed_trainer import (
    DistributedModelTrainer,
    DistributedConfig
)
from signal_generation.models.market_state_validator import (
    MarketStateValidator,
    MarketStateConfig
)

@dataclass
class RetrainingConfig:
    """Retraining trigger configuration"""
    performance_threshold: float = 0.7
    market_change_threshold: float = 0.3
    min_retraining_interval: int = 24  # hours
    max_retraining_interval: int = 168  # hours (1 week)
    batch_size: int = 1024
    validation_window: int = 1000
    
class RetrainingManager:
    """Manages model retraining triggers and execution"""
    
    def __init__(
        self,
        config: RetrainingConfig,
        model_trainer: DistributedModelTrainer,
        market_validator: MarketStateValidator,
        performance_monitor
    ):
        self.config = config
        self.model_trainer = model_trainer
        self.market_validator = market_validator
        self.performance_monitor = performance_monitor
        self.last_retrain_time = datetime.now()
        self._initialize_state()
        
    def _initialize_state(self):
        """Initialize retraining state"""
        self.validation_data = []
        self.market_states = []
        self.retraining_history = []
        
    async def update_state(
        self,
        features: torch.Tensor,
        market_data: Dict
    ):
        """Update validation state with new data"""
        try:
            # Update validation data
            self.validation_data.append(features)
            if len(self.validation_data) > self.config.validation_window:
                self.validation_data.pop(0)
                
            # Update market state
            market_analysis = self.market_validator.analyze_market_states(
                market_data
            )
            self.market_states.append(market_analysis['states'][-1])
            
            # Check retraining triggers
            await self._check_triggers()
            
        except Exception as e:
            logging.error(f"Error updating retraining state: {e}")
            raise
            
    async def _check_triggers(self):
        """Check if retraining should be triggered"""
        try:
            current_time = datetime.now()
            time_since_last = current_time - self.last_retrain_time
            
            # Check minimum interval
            if time_since_last.total_seconds() < self.config.min_retraining_interval * 3600:
                return
                
            should_retrain = False
            trigger_reason = []
            
            # Check performance trigger
            if await self._check_performance_trigger():
                should_retrain = True
                trigger_reason.append("performance_degradation")
                
            # Check market state trigger
            if await self._check_market_trigger():
                should_retrain = True
                trigger_reason.append("market_state_change")
                
            # Check maximum interval
            if time_since_last.total_seconds() >= self.config.max_retraining_interval * 3600:
                should_retrain = True
                trigger_reason.append("max_interval_reached")
                
            if should_retrain:
                await self._execute_retraining(trigger_reason)
                
        except Exception as e:
            logging.error(f"Error checking retraining triggers: {e}")
            raise
            
    async def _check_performance_trigger(self) -> bool:
        """Check if performance triggers retraining"""
        recent_performance = np.mean(
            self.performance_monitor.metrics['accuracy'][-self.config.validation_window:]
        )
        return recent_performance < self.config.performance_threshold
        
    async def _check_market_trigger(self) -> bool:
        """Check if market state changes trigger retraining"""
        if len(self.market_states) < self.config.validation_window:
            return False
            
        # Calculate state distribution change
        recent_states = self.market_states[-self.config.validation_window:]
        old_states = self.market_states[:-self.config.validation_window]
        
        if not old_states:  # Not enough historical data
            return False
            
        recent_dist = np.bincount(recent_states) / len(recent_states)
        old_dist = np.bincount(old_states) / len(old_states)
        
        # Pad distributions to same length if needed
        max_len = max(len(recent_dist), len(old_dist))
        recent_dist = np.pad(recent_dist, (0, max_len - len(recent_dist)))
        old_dist = np.pad(old_dist, (0, max_len - len(old_dist)))
        
        # Calculate distribution difference
        dist_change = np.sum(np.abs(recent_dist - old_dist))
        return dist_change > self.config.market_change_threshold
        
    async def _execute_retraining(self, trigger_reason: List[str]):
        """Execute model retraining"""
        try:
            logging.info(
                f"Starting model retraining. Triggers: {', '.join(trigger_reason)}"
            )
            
            # Prepare training data
            train_data = torch.cat(self.validation_data, dim=0)
            
            # Split into train/validation
            split_idx = int(len(train_data) * 0.8)
            train_subset = train_data[:split_idx]
            valid_subset = train_data[split_idx:]
            
            # Execute distributed training
            await self.model_trainer.train_distributed(
                train_subset,
                valid_subset,
                num_epochs=5  # Adjust based on data size
            )
            
            # Record retraining event
            self.retraining_history.append({
                'timestamp': datetime.now().isoformat(),
                'triggers': trigger_reason,
                'training_size': len(train_data),
                'validation_performance': np.mean(
                    self.performance_monitor.metrics['accuracy']
                )
            })
            
            self.last_retrain_time = datetime.now()
            logging.info("Model retraining completed successfully")
            
        except Exception as e:
            logging.error(f"Error executing retraining: {e}")
            raise 