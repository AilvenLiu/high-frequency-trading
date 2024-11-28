from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Union, List
import operator
import json
import logging
import time

@dataclass
class SubscriptionRule:
    """Dynamic subscription rule configuration"""
    symbol: str
    conditions: Dict[str, Any]
    callback: Callable
    rule_id: Optional[str] = None
    batch_size: Optional[int] = None
    batch_interval: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert rule to dictionary for persistence"""
        return {
            'symbol': self.symbol,
            'conditions': self.conditions,
            'rule_id': self.rule_id,
            'batch_size': self.batch_size,
            'batch_interval': self.batch_interval
        }
        
    @staticmethod
    def from_dict(data: Dict, callback: Callable) -> 'SubscriptionRule':
        """Create rule from dictionary"""
        return SubscriptionRule(
            symbol=data['symbol'],
            conditions=data['conditions'],
            callback=callback,
            rule_id=data.get('rule_id'),
            batch_size=data.get('batch_size'),
            batch_interval=data.get('batch_interval')
        )

    def __init__(
        self,
        symbol: str,
        conditions: Dict,
        batch_size: Optional[int] = None,
        batch_interval: Optional[float] = None
    ):
        self.symbol = symbol
        self.conditions = conditions
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.last_batch_time = time.time()
        self.current_batch = []
        
    async def process_data(
        self,
        data: Union[Dict, List[Dict]]
    ) -> Optional[Union[Dict, List[Dict]]]:
        """Process data with batching support"""
        try:
            if isinstance(data, list):
                # Handle batch data
                filtered_batch = [
                    item for item in data
                    if self._check_conditions(item)
                ]
                return filtered_batch if filtered_batch else None
            else:
                # Handle single data point
                if self._check_conditions(data):
                    if self.batch_size:
                        self.current_batch.append(data)
                        if len(self.current_batch) >= self.batch_size:
                            return self._flush_batch()
                        return None
                    return data
                return None
                
        except Exception as e:
            logging.error(f"Error processing subscription rule: {e}")
            return None
            
    def _flush_batch(self) -> List[Dict]:
        """Flush current batch and reset"""
        if not self.current_batch:
            return None
            
        batch = self.current_batch
        self.current_batch = []
        self.last_batch_time = time.time()
        return batch

class RuleEngine:
    """Engine for evaluating subscription rules"""
    
    def __init__(self):
        self.operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            'change_pct': self._calculate_change_percent
        }
        
        self.last_values: Dict[str, Dict[str, float]] = {}
        
    def evaluate(self, rule: SubscriptionRule, data: Dict[str, Any]) -> bool:
        """
        Evaluate if data matches subscription rule
        
        Args:
            rule: Subscription rule to evaluate
            data: Market data to check
            
        Returns:
            bool: True if data matches rule conditions
        """
        try:
            for field, condition in rule.conditions.items():
                if field not in data:
                    return False
                    
                op = condition['operator']
                value = condition['value']
                
                if op == 'change_pct':
                    # Handle percentage change calculation
                    if not self._check_price_change(
                        rule.symbol,
                        field,
                        data[field],
                        value
                    ):
                        return False
                else:
                    # Handle direct comparison
                    if not self.operators[op](float(data[field]), value):
                        return False
                        
            return True
            
        except Exception as e:
            logging.error(f"Error evaluating rule: {e}")
            return False
            
    def _calculate_change_percent(
        self,
        current: float,
        previous: float
    ) -> float:
        """Calculate percentage change"""
        if previous == 0:
            return 0.0
        return abs((current - previous) / previous) * 100
        
    def _check_price_change(
        self,
        symbol: str,
        field: str,
        current_value: float,
        threshold: float
    ) -> bool:
        """Check if price change exceeds threshold"""
        if symbol not in self.last_values:
            self.last_values[symbol] = {}
            
        last_value = self.last_values[symbol].get(field)
        self.last_values[symbol][field] = float(current_value)
        
        if last_value is None:
            return False
            
        change_pct = self._calculate_change_percent(
            float(current_value),
            last_value
        )
        return change_pct >= threshold 