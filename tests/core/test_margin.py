import pytest
from decimal import Decimal
from funds_management.risk_control.margin_calculator import MarginCalculator

@pytest.fixture
def margin_calculator():
    return MarginCalculator(
        initial_margin_ratio=0.1,
        maintenance_margin_ratio=0.05,
        liquidation_buffer=0.01
    )

def test_calculate_dynamic_margin(margin_calculator):
    position_value = Decimal('10000')
    leverage = 5
    market_volatility = 0.6
    
    result = margin_calculator.calculate_dynamic_margin(position_value, leverage, market_volatility)
    
    assert 'dynamic_initial_margin' in result
    assert 'maintenance_margin' in result
    assert 'liquidation_price_buffer' in result
    assert result['dynamic_initial_margin'] > Decimal('0')
    assert result['maintenance_margin'] > Decimal('0')
    assert result['liquidation_price_buffer'] > Decimal('0') 

def test_dynamic_margin_caching(margin_calculator):
    """Test dynamic margin calculation with caching"""
    position_value = Decimal('10000')
    leverage = 5
    market_volatility = 0.6
    
    # First call
    result1 = margin_calculator.calculate_dynamic_margin(position_value, leverage, market_volatility)
    # Second call (should use cache)
    result2 = margin_calculator.calculate_dynamic_margin(position_value, leverage, market_volatility)
    
    assert result1 == result2
    assert 'dynamic_initial_margin' in result1
    assert 'maintenance_margin' in result1
    assert 'liquidation_price_buffer' in result1 