import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class RSIConfig:
    """RSI indicator configuration"""
    period: int = 14
    price_key: str = 'close'
    min_periods: Optional[int] = None
    
class RSIIndicator:
    """Relative Strength Index (RSI) implementation"""
    
    def __init__(self, config: RSIConfig):
        """Initialize RSI indicator"""
        self.config = config
        self.min_periods = config.min_periods or config.period
        self._validate_config()
        
        # Initialize state for real-time calculation
        self._prev_price: Optional[float] = None
        self._gains = []
        self._losses = []
        
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.period < 1:
            raise ValueError("Period must be positive")
        if self.config.min_periods and self.config.min_periods < 1:
            raise ValueError("min_periods must be positive")
            
    def calculate(
        self,
        data: Union[pd.DataFrame, List[Dict], np.ndarray]
    ) -> np.ndarray:
        """Calculate RSI for historical data"""
        try:
            # Convert input to numpy array
            prices = self._extract_prices(data)
            
            # Calculate price changes
            changes = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            # Calculate average gains and losses
            avg_gains = self._calculate_averages(gains)
            avg_losses = self._calculate_averages(losses)
            
            # Calculate RS and RSI
            rs = np.divide(
                avg_gains,
                avg_losses,
                out=np.zeros_like(avg_gains),
                where=avg_losses != 0
            )
            rsi = 100 - (100 / (1 + rs))
            
            # Handle initial values
            rsi[:self.min_periods-1] = np.nan
            
            return rsi
            
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            raise
            
    def update(self, price: float) -> Optional[float]:
        """Update RSI with new price data"""
        try:
            if self._prev_price is None:
                self._prev_price = price
                return None
                
            # Calculate price change
            change = price - self._prev_price
            self._prev_price = price
            
            # Update gains and losses
            gain = max(change, 0)
            loss = max(-change, 0)
            
            self._gains.append(gain)
            self._losses.append(loss)
            
            # Maintain window size
            if len(self._gains) > self.config.period:
                self._gains.pop(0)
                self._losses.pop(0)
                
            # Calculate RSI if we have enough data
            if len(self._gains) >= self.min_periods:
                avg_gain = np.mean(self._gains)
                avg_loss = np.mean(self._losses)
                
                if avg_loss == 0:
                    return 100.0
                    
                rs = avg_gain / avg_loss
                return 100 - (100 / (1 + rs))
                
            return None
            
        except Exception as e:
            logging.error(f"Error updating RSI: {e}")
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
            
    def _calculate_averages(self, data: np.ndarray) -> np.ndarray:
        """Calculate moving averages"""
        window = np.ones(self.config.period) / self.config.period
        return np.convolve(data, window, mode='valid') 