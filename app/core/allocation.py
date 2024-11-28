from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from decimal import Decimal
import logging

@dataclass
class AccountStatus:
    net_worth: Decimal
    available_margin: Decimal
    positions: Dict[str, 'Position']
    risk_level: float

@dataclass
class Position:
    symbol: str
    size: Decimal
    entry_price: Decimal
    leverage: int
    unrealized_pnl: Decimal
    margin_used: Decimal

class DynamicAllocator:
    def __init__(self, 
                 max_position_size: float = 0.8,
                 min_free_margin: float = 0.2,
                 kelly_fraction: float = 0.5):
        """
        Initialize the dynamic allocator with risk parameters
        
        Args:
            max_position_size: Maximum position size as a fraction of account value
            min_free_margin: Minimum free margin to maintain
            kelly_fraction: Fraction of Kelly criterion to use (conservative approach)
        """
        self.max_position_size = max_position_size
        self.min_free_margin = min_free_margin
        self.kelly_fraction = kelly_fraction
        self.logger = logging.getLogger(__name__)

    def calculate_position_size(self, 
                              account_status: AccountStatus,
                              signal_strength: float,
                              win_rate: float,
                              profit_ratio: float,
                              loss_ratio: float) -> Decimal:
        """
        Calculate optimal position size using Kelly Criterion and account constraints
        
        Args:
            account_status: Current account status
            signal_strength: Signal strength from -1 to 1
            win_rate: Historical win rate for the strategy
            profit_ratio: Average profit ratio
            loss_ratio: Average loss ratio
        
        Returns:
            Optimal position size in base currency
        """
        try:
            # Kelly Criterion calculation
            q = 1 - win_rate
            kelly_size = (win_rate/loss_ratio - q/profit_ratio) * self.kelly_fraction
            
            # Adjust for signal strength
            kelly_size *= abs(signal_strength)
            
            # Apply account constraints
            max_allowed = float(account_status.net_worth) * self.max_position_size
            margin_based_limit = float(account_status.available_margin) * (1 - self.min_free_margin)
            
            # Take the minimum of all constraints
            position_size = min(kelly_size * float(account_status.net_worth),
                              max_allowed,
                              margin_based_limit)
            
            return Decimal(str(max(0, position_size)))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return Decimal('0')

    def allocate_funds(self,
                      account_status: AccountStatus,
                      signals: Dict[str, float],
                      strategy_metrics: Dict[str, Dict]) -> Dict[str, Decimal]:
        """
        Allocate funds across multiple trading opportunities
        
        Args:
            account_status: Current account status
            signals: Dictionary of trading signals per symbol
            strategy_metrics: Performance metrics for each strategy
        
        Returns:
            Dictionary of allocated amounts per symbol
        """
        allocations = {}
        try:
            total_allocation = Decimal('0')
            
            for symbol, signal in signals.items():
                metrics = strategy_metrics.get(symbol, {})
                position_size = self.calculate_position_size(
                    account_status=account_status,
                    signal_strength=signal,
                    win_rate=metrics.get('win_rate', 0.5),
                    profit_ratio=metrics.get('profit_ratio', 1.5),
                    loss_ratio=metrics.get('loss_ratio', 1.0)
                )
                
                # Ensure we don't exceed account limits
                remaining = account_status.net_worth - total_allocation
                position_size = min(position_size, remaining)
                
                if position_size > 0:
                    allocations[symbol] = position_size
                    total_allocation += position_size
                    
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error in fund allocation: {e}")
            return {}

    def rebalance_positions(self,
                          account_status: AccountStatus,
                          target_allocations: Dict[str, Decimal]) -> Dict[str, Decimal]:
        """
        Calculate required position adjustments to match target allocations
        
        Args:
            account_status: Current account status
            target_allocations: Target position sizes per symbol
        
        Returns:
            Dictionary of position adjustments (positive for increase, negative for decrease)
        """
        adjustments = {}
        try:
            current_positions = account_status.positions
            
            for symbol, target_size in target_allocations.items():
                current_size = Decimal('0')
                if symbol in current_positions:
                    current_size = current_positions[symbol].size
                
                adjustment = target_size - current_size
                if abs(adjustment) > Decimal('0'):
                    adjustments[symbol] = adjustment
                    
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error calculating position adjustments: {e}")
            return {} 