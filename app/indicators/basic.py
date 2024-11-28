import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
import logging
from collections import deque

@dataclass
class MACDConfig:
    """MACD indicator configuration"""
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    price_key: str = 'close'
    
class MACDIndicator:
    """Moving Average Convergence Divergence implementation"""
    
    def __init__(self, config: MACDConfig):
        """Initialize MACD indicator"""
        self.config = config
        self._validate_config()
        
        # Initialize state for real-time calculation
        self._price_history = deque(maxlen=self.config.slow_period)
        self._fast_ema = None
        self._slow_ema = None
        self._macd_history = deque(maxlen=self.config.signal_period)
        self._signal_line = None
        
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.fast_period >= self.config.slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if any(p < 1 for p in [
            self.config.fast_period,
            self.config.slow_period,
            self.config.signal_period
        ]):
            raise ValueError("All periods must be positive")
            
    def calculate(
        self,
        data: Union[pd.DataFrame, List[Dict], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD for historical data"""
        try:
            # Extract prices
            prices = self._extract_prices(data)
            
            # Calculate EMAs
            fast_ema = self._calculate_ema(
                prices,
                self.config.fast_period
            )
            slow_ema = self._calculate_ema(
                prices,
                self.config.slow_period
            )
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = self._calculate_ema(
                macd_line,
                self.config.signal_period
            )
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            raise
            
    def update(
        self,
        price: float
    ) -> Optional[Tuple[float, float, float]]:
        """Update MACD with new price data"""
        try:
            self._price_history.append(price)
            
            if len(self._price_history) < self.config.slow_period:
                return None
                
            # Update EMAs
            if self._fast_ema is None:
                self._fast_ema = np.mean(list(self._price_history)[-self.config.fast_period:])
            else:
                alpha = 2 / (self.config.fast_period + 1)
                self._fast_ema = price * alpha + self._fast_ema * (1 - alpha)
                
            if self._slow_ema is None:
                self._slow_ema = np.mean(list(self._price_history))
            else:
                alpha = 2 / (self.config.slow_period + 1)
                self._slow_ema = price * alpha + self._slow_ema * (1 - alpha)
                
            # Calculate MACD line
            macd_line = self._fast_ema - self._slow_ema
            self._macd_history.append(macd_line)
            
            # Update signal line
            if len(self._macd_history) < self.config.signal_period:
                return None
                
            if self._signal_line is None:
                self._signal_line = np.mean(list(self._macd_history))
            else:
                alpha = 2 / (self.config.signal_period + 1)
                self._signal_line = (
                    macd_line * alpha +
                    self._signal_line * (1 - alpha)
                )
                
            # Calculate histogram
            histogram = macd_line - self._signal_line
            
            return macd_line, self._signal_line, histogram
            
        except Exception as e:
            logging.error(f"Error updating MACD: {e}")
            raise
            
    def _extract_prices(
        self,
        data: Union[pd.DataFrame, List[Dict], np.ndarray]
    ) -> np.ndarray:
        """Extract price data from input"""
        if isinstance(data, pd.DataFrame):
            return data[self.config.price_key].values
        elif isinstance(data, list):
            return np.array([d[self.config.price_key] for d in data])
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unsupported data format")
            
    def _calculate_ema(
        self,
        data: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
            
        return ema 