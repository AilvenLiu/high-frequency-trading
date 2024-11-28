import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.cluster import KMeans
from scipy import stats

@dataclass
class MarketStateConfig:
    """Market state validation configuration"""
    min_volatility_percentile: float = 5.0
    max_volatility_percentile: float = 95.0
    trend_window: int = 20
    num_states: int = 5  # Number of market states to identify
    min_state_coverage: float = 0.15  # Minimum coverage for each state
    
class MarketStateValidator:
    """Validate market state coverage in training data"""
    
    def __init__(self, config: MarketStateConfig):
        self.config = config
        self._validate_config()
        self.state_classifier = KMeans(n_clusters=config.num_states)
        
    def _validate_config(self):
        """Validate configuration parameters"""
        if not 0 <= self.config.min_volatility_percentile < self.config.max_volatility_percentile <= 100:
            raise ValueError("Invalid volatility percentile range")
        if self.config.trend_window < 2:
            raise ValueError("Trend window must be at least 2")
        if not 0 < self.config.min_state_coverage <= 1:
            raise ValueError("Invalid minimum state coverage")
            
    def analyze_market_states(
        self,
        data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Analyze market states in the data"""
        try:
            # Calculate features for state classification
            features = self._extract_state_features(data)
            
            # Classify market states
            states = self.state_classifier.fit_predict(features)
            
            # Calculate state coverage
            state_coverage = np.bincount(states) / len(states)
            
            # Identify extreme market conditions
            volatility = self._calculate_volatility(data['close'])
            extreme_periods = self._identify_extreme_periods(
                data['close'],
                volatility
            )
            
            return {
                'states': states,
                'state_coverage': state_coverage,
                'extreme_periods': extreme_periods,
                'features': features
            }
            
        except Exception as e:
            logging.error(f"Error analyzing market states: {e}")
            raise
            
    def validate_coverage(
        self,
        state_coverage: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """Validate market state coverage"""
        issues = []
        valid = True
        
        # Check minimum coverage for each state
        for state, coverage in enumerate(state_coverage):
            if coverage < self.config.min_state_coverage:
                issues.append(
                    f"State {state} has insufficient coverage: {coverage:.2%}"
                )
                valid = False
                
        return valid, issues
        
    def _extract_state_features(
        self,
        data: pd.DataFrame
    ) -> np.ndarray:
        """Extract features for market state classification"""
        # Calculate returns
        returns = np.log(data['close']).diff()
        
        # Calculate volatility
        volatility = returns.rolling(window=self.config.trend_window).std()
        
        # Calculate trend
        trend = data['close'].rolling(
            window=self.config.trend_window
        ).mean().pct_change()
        
        # Calculate volume profile
        volume_ma = data['volume'].rolling(
            window=self.config.trend_window
        ).mean()
        volume_ratio = data['volume'] / volume_ma
        
        # Combine features
        features = np.column_stack([
            returns.fillna(0),
            volatility.fillna(0),
            trend.fillna(0),
            volume_ratio.fillna(1)
        ])
        
        return features
        
    def _calculate_volatility(
        self,
        prices: pd.Series
    ) -> pd.Series:
        """Calculate rolling volatility"""
        returns = np.log(prices).diff()
        volatility = returns.rolling(
            window=self.config.trend_window
        ).std()
        return volatility
        
    def _identify_extreme_periods(
        self,
        prices: pd.Series,
        volatility: pd.Series
    ) -> np.ndarray:
        """Identify periods of extreme market conditions"""
        # Calculate volatility percentiles
        vol_low = np.percentile(
            volatility.dropna(),
            self.config.min_volatility_percentile
        )
        vol_high = np.percentile(
            volatility.dropna(),
            self.config.max_volatility_percentile
        )
        
        # Identify extreme periods
        extreme_periods = np.zeros(len(prices), dtype=bool)
        extreme_periods[volatility > vol_high] = True  # High volatility
        extreme_periods[volatility < vol_low] = True   # Low volatility
        
        return extreme_periods 