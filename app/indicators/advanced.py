import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
import logging
from collections import deque

@dataclass
class BollingerBandsConfig:
    """Bollinger Bands indicator configuration"""
    window: int = 20
    num_std: float = 2.0
    price_key: str = 'close'
    
class BollingerBandsIndicator:
    """Bollinger Bands implementation"""
    
    def __init__(self, config: BollingerBandsConfig):
        """Initialize Bollinger Bands indicator"""
        self.config = config
        self._validate_config()
        
        # Initialize state for real-time calculation
        self._price_history = deque(maxlen=self.config.window)
        
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.window < 2:
            raise ValueError("Window size must be at least 2")
        if self.config.num_std <= 0:
            raise ValueError("Number of standard deviations must be positive")
            
    def calculate(
        self,
        data: Union[pd.DataFrame, List[Dict], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands for historical data"""
        try:
            # Extract prices
            prices = self._extract_prices(data)
            
            # Calculate middle band (SMA)
            middle_band = self._calculate_sma(prices)
            
            # Calculate standard deviation
            rolling_std = self._calculate_rolling_std(prices)
            
            # Calculate upper and lower bands
            upper_band = middle_band + (rolling_std * self.config.num_std)
            lower_band = middle_band - (rolling_std * self.config.num_std)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            raise
            
    def update(
        self,
        price: float
    ) -> Optional[Tuple[float, float, float]]:
        """Update Bollinger Bands with new price data"""
        try:
            self._price_history.append(price)
            
            if len(self._price_history) < self.config.window:
                return None
                
            # Calculate middle band
            prices = np.array(self._price_history)
            middle_band = np.mean(prices)
            
            # Calculate standard deviation
            std = np.std(prices, ddof=1)
            
            # Calculate bands
            upper_band = middle_band + (std * self.config.num_std)
            lower_band = middle_band - (std * self.config.num_std)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            logging.error(f"Error updating Bollinger Bands: {e}")
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
            
    def _calculate_sma(self, data: np.ndarray) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return pd.Series(data).rolling(
            window=self.config.window,
            min_periods=1
        ).mean().values
        
    def _calculate_rolling_std(self, data: np.ndarray) -> np.ndarray:
        """Calculate rolling standard deviation"""
        return pd.Series(data).rolling(
            window=self.config.window,
            min_periods=1
        ).std(ddof=1).values 