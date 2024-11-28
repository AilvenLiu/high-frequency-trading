from typing import Dict, Optional, List
import asyncio
import logging
from decimal import Decimal
from datetime import datetime
from .api_connector import APIConnector, OrderRequest, OrderResponse, OrderType, OrderSide
from .execution_logger import ExecutionLogger

class OrderExecutor:
    def __init__(self, api_connector: APIConnector):
        self.api_connector = api_connector
        self.logger = ExecutionLogger()
        self.active_orders: Dict[str, OrderRequest] = {}
        self.retry_attempts = 3
        self.retry_delay = 1  # seconds
        
    async def execute_order(self, request: OrderRequest) -> Optional[OrderResponse]:
        """Execute an order with retry mechanism"""
        try:
            for attempt in range(self.retry_attempts):
                try:
                    response = await self.api_connector.place_order(request)
                    self.active_orders[response.order_id] = request
                    await self.logger.log_order_execution(request, response)
                    return response
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    await asyncio.sleep(self.retry_delay)
                    
        except Exception as e:
            await self.logger.log_error(f"Failed to execute order: {e}")
            return None
            
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order with retry mechanism"""
        try:
            for attempt in range(self.retry_attempts):
                try:
                    success = await self.api_connector.cancel_order(order_id, symbol)
                    if success:
                        self.active_orders.pop(order_id, None)
                        await self.logger.log_order_cancellation(order_id)
                        return True
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    await asyncio.sleep(self.retry_delay)
                    
            return False
            
        except Exception as e:
            await self.logger.log_error(f"Failed to cancel order: {e}")
            return False
            
    async def monitor_order(self, order_id: str, symbol: str) -> OrderResponse:
        """Monitor order status until filled or cancelled"""
        try:
            while True:
                status = await self.api_connector.get_order_status(order_id, symbol)
                await self.logger.log_order_status(order_id, status)
                
                if status.status in ['filled', 'cancelled']:
                    self.active_orders.pop(order_id, None)
                    return status
                    
                await asyncio.sleep(1)
                
        except Exception as e:
            await self.logger.log_error(f"Error monitoring order: {e}")
            raise
            
    async def execute_batch_orders(self, requests: List[OrderRequest]) -> Dict[str, OrderResponse]:
        """Execute multiple orders in parallel"""
        try:
            tasks = [self.execute_order(request) for request in requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            results = {}
            for request, response in zip(requests, responses):
                if isinstance(response, Exception):
                    await self.logger.log_error(f"Batch order error: {response}")
                    continue
                if response:
                    results[response.order_id] = response
                    
            return results
            
        except Exception as e:
            await self.logger.log_error(f"Batch execution error: {e}")
            return {} 