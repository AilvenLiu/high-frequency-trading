import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, patch
from trading.execution.api_connector import (
    APIConnector,
    OrderRequest,
    OrderResponse,
    OrderType,
    OrderSide
)

@pytest.fixture
def api_connector():
    return APIConnector(
        api_key="test_key",
        api_secret="test_secret",
        sandbox=True
    )

@pytest.mark.asyncio
async def test_place_order(api_connector):
    # Mock exchange response
    mock_response = {
        'id': '12345',
        'status': 'open',
        'filled': 0.0,
        'average': None,
        'remaining': 1.0,
        'timestamp': 1234567890000
    }
    
    with patch.object(api_connector.exchange, 'create_order', return_value=mock_response):
        request = OrderRequest(
            symbol='BTC-USDT',
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Decimal('1.0'),
            price=Decimal('50000')
        )
        
        response = await api_connector.place_order(request)
        
        assert isinstance(response, OrderResponse)
        assert response.order_id == '12345'
        assert response.status == 'open'

@pytest.mark.asyncio
async def test_cancel_order(api_connector):
    with patch.object(api_connector.exchange, 'cancel_order', return_value={'status': 'cancelled'}):
        result = await api_connector.cancel_order('12345', 'BTC-USDT')
        assert result is True

@pytest.mark.asyncio
async def test_get_order_status(api_connector):
    mock_response = {
        'id': '12345',
        'status': 'filled',
        'filled': 1.0,
        'average': 50000.0,
        'remaining': 0.0,
        'timestamp': 1234567890000
    }
    
    with patch.object(api_connector.exchange, 'fetch_order', return_value=mock_response):
        status = await api_connector.get_order_status('12345', 'BTC-USDT')
        assert status.status == 'filled'
        assert status.filled_amount == Decimal('1.0') 