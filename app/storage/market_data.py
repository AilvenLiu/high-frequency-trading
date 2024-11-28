import asyncio
import websockets
import json
from typing import Dict, Set, Callable
from datetime import datetime
import logging

class MarketDataSync:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.ws_url = config.api_config['ws_url']
        self.subscribed_symbols: Set[str] = set()
        self.callbacks: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
        self._last_update: Dict[str, datetime] = {}
        
    async def start(self):
        """Start market data synchronization"""
        while True:
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    # Subscribe to market data
                    await self._subscribe(websocket)
                    
                    # Process incoming messages
                    async for message in websocket:
                        await self._handle_message(json.loads(message))
                        
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)  # Reconnection delay
                
    async def _subscribe(self, websocket):
        """Subscribe to market data for configured symbols"""
        for symbol in self.config.trading_params['symbols']:
            subscription = {
                "op": "subscribe",
                "args": [
                    {
                        "channel": "tickers",
                        "instId": symbol
                    },
                    {
                        "channel": "candle1m",
                        "instId": symbol
                    }
                ]
            }
            await websocket.send(json.dumps(subscription))
            self.subscribed_symbols.add(symbol)
            
    async def _handle_message(self, message: Dict):
        """Process incoming market data"""
        try:
            if 'data' not in message:
                return
                
            symbol = message['arg']['instId']
            self._last_update[symbol] = datetime.utcnow()
            
            # Notify callbacks
            if symbol in self.callbacks:
                await self.callbacks[symbol](message['data'])
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            
    def add_callback(self, symbol: str, callback: Callable):
        """Add callback for market data updates"""
        self.callbacks[symbol] = callback
        
    def get_last_update_time(self, symbol: str) -> datetime:
        """Get last update time for a symbol"""
        return self._last_update.get(symbol) 