import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, patch
from trading.execution.order_executor import OrderExecutor
from trading.execution.api_connector import (
    APIConnector,
    OrderRequest,
    OrderResponse,
    OrderType,
    OrderSide
)

@pytest.fixture
def order_executor():
    api_connector = APIConnector("test_key", "test_secret", sandbox=True)
    return OrderExecutor(api_connector)

@pytest.mark.asyncio
async def test_execute_order(order_executor):
    mock_response = OrderResponse(
        order_id='12345',
        status='open',
        filled_amount=Decimal('0'),
        average_price=None,
        remaining=Decimal('1.0'),
        timestamp=1234567890000
    )
    
    with patch.object(order_executor.api_connector, 'place_order', return_value=mock_response):
        request = OrderRequest(
            symbol='BTC-USDT',
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            amount=Decimal('1.0'),
            price=Decimal('50000')
        )
        
        response = await order_executor.execute_order(request)
        assert response.order_id == '12345'
        assert '12345' in order_executor.active_orders

@pytest.mark.asyncio
async def test_execute_batch_orders(order_executor):
    mock_response = OrderResponse(
        order_id='12345',
        status='open',
        filled_amount=Decimal('0'),
        average_price=None,
        remaining=Decimal('1.0'),
        timestamp=1234567890000
    )
    
    with patch.object(order_executor.api_connector, 'place_order', return_value=mock_response):
        requests = [
            OrderRequest(
                symbol='BTC-USDT',
                order_type=OrderType.LIMIT,
                side=OrderSide.BUY,
                amount=Decimal('1.0'),
                price=Decimal('50000')
            ),
            OrderRequest(
                symbol='ETH-USDT',
                order_type=OrderType.LIMIT,
                side=OrderSide.BUY,
                amount=Decimal('10.0'),
                price=Decimal('3000')
            )
        ]
        
        responses = await order_executor.execute_batch_orders(requests)
        assert len(responses) == 2 