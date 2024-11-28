import pytest
import numpy as np
import pandas as pd
import torch
from signal_generation.models.lstm_predictor import (
    LSTMPredictor,
    PredictorConfig,
    ModelTrainer
)
from signal_generation.models.market_state_validator import (
    MarketStateValidator,
    MarketStateConfig
)
from typing import Dict, List, Tuple
import logging

class ModelStabilityTest:
    """Test model stability under extreme market conditions"""
    
    def __init__(
        self,
        model_trainer: ModelTrainer,
        market_validator: MarketStateValidator,
        test_data: pd.DataFrame
    ):
        self.model_trainer = model_trainer
        self.market_validator = market_validator
        self.test_data = test_data
        
    async def run_stability_test(self) -> Dict[str, float]:
        """Run stability test under different market conditions"""
        try:
            # Analyze market states
            market_analysis = self.market_validator.analyze_market_states(
                self.test_data
            )
            
            # Test model performance in different states
            state_performance = await self._test_state_performance(
                market_analysis['states']
            )
            
            # Test model stability in extreme periods
            stability_metrics = await self._test_extreme_stability(
                market_analysis['extreme_periods']
            )
            
            # Combine results
            results = {
                'state_performance': state_performance,
                'stability_metrics': stability_metrics
            }
            
            return results
            
        except Exception as e:
            logging.error(f"Error in stability test: {e}")
            raise
            
    async def _test_state_performance(
        self,
        states: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """Test model performance in different market states"""
        performance = {}
        
        for state in np.unique(states):
            # Get data for this state
            state_mask = states == state
            state_data = self._prepare_state_data(state_mask)
            
            # Evaluate model
            metrics = await self._evaluate_model(state_data)
            performance[int(state)] = metrics
            
        return performance
        
    async def _test_extreme_stability(
        self,
        extreme_periods: np.ndarray
    ) -> Dict[str, float]:
        """Test model stability during extreme market conditions"""
        # Prepare extreme period data
        extreme_data = self._prepare_state_data(extreme_periods)
        normal_data = self._prepare_state_data(~extreme_periods)
        
        # Evaluate model on both datasets
        extreme_metrics = await self._evaluate_model(extreme_data)
        normal_metrics = await self._evaluate_model(normal_data)
        
        # Calculate stability metrics
        stability_metrics = {
            'prediction_stability': self._calculate_stability_score(
                extreme_metrics,
                normal_metrics
            ),
            'extreme_confidence': extreme_metrics['confidence'],
            'normal_confidence': normal_metrics['confidence']
        }
        
        return stability_metrics
        
    def _prepare_state_data(
        self,
        mask: np.ndarray
    ) -> torch.Tensor:
        """Prepare data for specific market state"""
        # Extract features for masked periods
        features = self.test_data[mask]
        
        # Convert to tensor
        return torch.tensor(
            features.values,
            dtype=torch.float32
        ).to(self.model_trainer.device)
        
    async def _evaluate_model(
        self,
        data: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model performance on given data"""
        try:
            with torch.no_grad():
                predictions, attention = self.model_trainer.predict(data)
                
                # Calculate metrics
                confidence = torch.max(predictions, dim=1)[0].mean().item()
                attention_entropy = self._calculate_attention_entropy(attention)
                
                return {
                    'confidence': confidence,
                    'attention_entropy': attention_entropy
                }
                
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            raise
            
    def _calculate_stability_score(
        self,
        extreme_metrics: Dict[str, float],
        normal_metrics: Dict[str, float]
    ) -> float:
        """Calculate model stability score"""
        confidence_ratio = extreme_metrics['confidence'] / normal_metrics['confidence']
        entropy_ratio = extreme_metrics['attention_entropy'] / normal_metrics['attention_entropy']
        
        # Combine metrics (lower score means more stable)
        stability_score = abs(1 - confidence_ratio) + abs(1 - entropy_ratio)
        return stability_score
        
    def _calculate_attention_entropy(
        self,
        attention_weights: torch.Tensor
    ) -> float:
        """Calculate entropy of attention weights"""
        attention_np = attention_weights.cpu().numpy()
        return float(stats.entropy(attention_np + 1e-10, axis=1).mean())

@pytest.fixture
def stability_test(model_trainer, market_validator, test_data):
    return ModelStabilityTest(model_trainer, market_validator, test_data)

async def test_model_stability(stability_test):
    """Test model stability"""
    results = await stability_test.run_stability_test()
    
    # Assert stability requirements
    stability_metrics = results['stability_metrics']
    assert stability_metrics['prediction_stability'] < 0.2  # Max 20% deviation
    assert stability_metrics['extreme_confidence'] > 0.6  # Min 60% confidence
    
    # Check state performance
    state_performance = results['state_performance']
    for state_metrics in state_performance.values():
        assert state_metrics['confidence'] > 0.5  # Min 50% confidence
        
    logging.info(f"Stability test results: {results}") 