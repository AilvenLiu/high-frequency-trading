import pytest
from funds_management.risk_control.risk_controller import RiskController

@pytest.fixture
def risk_controller():
    return RiskController()

def test_assess_market_state(risk_controller):
    market_data = {
        'BTC-USDT': {
            'volatility': 0.8,
            'trend_strength': 0.2
        },
        'ETH-USDT': {
            'volatility': 0.4,
            'trend_strength': 0.6
        }
    }
    
    result = risk_controller.assess_market_state(market_data)
    
    assert result['BTC-USDT'] == 'HIGH'
    assert result['ETH-USDT'] == 'LOW' 

def test_market_state_caching(risk_controller):
    """Test market state assessment with caching"""
    market_data = {
        'BTC-USDT': {
            'volatility': 0.8,
            'trend_strength': 0.2
        }
    }
    
    # First call
    result1 = risk_controller.assess_market_state(market_data)
    # Second call (should use cache)
    result2 = risk_controller.assess_market_state(market_data)
    
    assert result1 == result2
    assert result1['BTC-USDT'] == 'HIGH' 