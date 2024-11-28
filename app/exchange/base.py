from typing import Dict, Optional, List
import ccxt
import logging
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import json
import time
import asyncio
from datetime import datetime

class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class OrderRequest:
    symbol: str
    order_type: OrderType
    side: OrderSide
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    
@dataclass
class OrderResponse:
    order_id: str
    status: str
    filled_amount: Decimal
    average_price: Optional[Decimal]
    remaining: Decimal
    timestamp: int

class APIConnector:
    def __init__(self, api_key: str, api_secret: str, sandbox: bool = True):
        self.logger = logging.getLogger(__name__)
        self.exchange = ccxt.okex({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        self.exchange.set_sandbox_mode(sandbox)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup structured logging for API calls"""
        handler = logging.FileHandler('api_connector.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place an order with the exchange"""
        try:
            params = {}
            if request.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                params['stopPrice'] = float(request.stop_price)
                
            order = await self.exchange.create_order(
                symbol=request.symbol,
                type=request.order_type.value,
                side=request.side.value,
                amount=float(request.amount),
                price=float(request.price) if request.price else None,
                params=params
            )
            
            self.logger.info(f"Order placed: {json.dumps(order)}")
            
            return OrderResponse(
                order_id=order['id'],
                status=order['status'],
                filled_amount=Decimal(str(order['filled'])),
                average_price=Decimal(str(order['average'])) if order['average'] else None,
                remaining=Decimal(str(order['remaining'])),
                timestamp=order['timestamp']
            )
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}", exc_info=True)
            raise
            
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order cancelled: {json.dumps(result)}")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}", exc_info=True)
            return False
            
    async def get_order_status(self, order_id: str, symbol: str) -> OrderResponse:
        """Get the current status of an order"""
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            
            return OrderResponse(
                order_id=order['id'],
                status=order['status'],
                filled_amount=Decimal(str(order['filled'])),
                average_price=Decimal(str(order['average'])) if order['average'] else None,
                remaining=Decimal(str(order['remaining'])),
                timestamp=order['timestamp']
            )
        except Exception as e:
            self.logger.error(f"Error fetching order status: {e}", exc_info=True)
            raise
            
    async def get_account_balance(self) -> Dict[str, Decimal]:
        """Get account balance"""
        try:
            balance = await self.exchange.fetch_balance()
            return {
                currency: Decimal(str(amount['free']))
                for currency, amount in balance['total'].items()
                if amount['free'] > 0
            }
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}", exc_info=True)
            raise 
            
    async def verify_connection(self) -> bool:
        """Verify API connection and permissions"""
        try:
            # Test API connectivity
            await self.exchange.fetch_balance()
            
            # Verify trading permissions
            permissions = await self.exchange.fetch_permissions()
            required_permissions = ['trade', 'margin']
            
            has_permissions = all(
                perm in permissions 
                for perm in required_permissions
            )
            
            if not has_permissions:
                self.logger.error("Missing required trading permissions")
                return False
                
            self.logger.info("API connection verified successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"API connection verification failed: {e}")
            return False
            
    async def verify_sandbox_mode(self) -> bool:
        """Verify sandbox mode is properly configured"""
        try:
            # Attempt a small test order
            test_order = await self.place_order(
                OrderRequest(
                    symbol="BTC-USDT",
                    order_type=OrderType.LIMIT,
                    side=OrderSide.BUY,
                    amount=Decimal("0.001"),
                    price=Decimal("1000")  # Unrealistic price for sandbox
                )
            )
            
            # Cancel the test order
            await self.cancel_order(test_order.order_id, "BTC-USDT")
            
            self.logger.info("Sandbox mode verified successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Sandbox verification failed: {e}")
            return False
            
    async def verify_sandbox_connection(self) -> bool:
        """Verify sandbox connection and permissions"""
        try:
            # Test API connectivity in sandbox mode
            if not self.sandbox:
                self.logger.error("Not in sandbox mode")
                return False
                
            # Verify account access
            account = await self.exchange.fetch_balance()
            if not account:
                self.logger.error("Failed to fetch sandbox account balance")
                return False
                
            # Test market data access
            tickers = await self.exchange.fetch_tickers(['BTC-USDT'])
            if not tickers:
                self.logger.error("Failed to fetch sandbox market data")
                return False
                
            self.logger.info("Sandbox connection verified successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Sandbox verification failed: {e}")
            return False