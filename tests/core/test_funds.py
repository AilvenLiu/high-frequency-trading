import pytest
import asyncio
from decimal import Decimal
from typing import Dict, Any
from main import funds_management_workflow

class TestFundsWorkflow:
    @pytest.fixture
    def market_data(self) -> Dict[str, Any]:
        return {
            'BTC-USDT': {
                'current_price': '50000',
                'volatility': 0.4,
                'liquidity': 0.8,
                'trend_strength': 0.6,
                'position_size': 1.0
            }
        }
    
    @pytest.fixture
    def account_data(self) -> Dict[str, Any]:
        return {
            'net_worth': '100000',
            'initial_equity': '120000',
            'total_positions_value': '50000',
            'used_margin': '4000',
            'available_margin': '5000',
            'equity': '95000',
            'positions': {
                'BTC-USDT': {
                    'size': '1',
                    'entry_price': '48000',
                    'leverage': '5',
                    'margin_mode': 'cross'
                }
            }
        }
    
    @pytest.fixture
    def signals(self) -> Dict[str, float]:
        return {
            'BTC-USDT': 0.8
        }
    
    @pytest.mark.asyncio
    async def test_funds_management_workflow(
        self,
        market_data: Dict[str, Any],
        account_data: Dict[str, Any],
        signals: Dict[str, float]
    ):
        result = await funds_management_workflow(
            market_data=market_data,
            account_data=account_data,
            signals=signals
        )
        
        assert 'allocations' in result
        assert 'leverage_recommendations' in result
        assert 'risk_status' in result
        assert 'margin_status' in result
        assert 'margin_alerts' in result
        
        # Verify allocations
        assert isinstance(result['allocations'], dict)
        assert all(isinstance(v, Decimal) for v in result['allocations'].values())
        
        # Verify leverage recommendations
        assert isinstance(result['leverage_recommendations'], dict)
        assert all(isinstance(v, int) for v in result['leverage_recommendations'].values())