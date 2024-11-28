import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import time
from .redis_cache_manager import RedisCacheManager
from monitoring.performance_monitor import PerformanceMonitor

class AsyncOKXClient:
    def __init__(
        self,
        symbols: List[str],
        channels: List[str],
        cache_manager: Optional[RedisCacheManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize async OKX client.
        
        Args:
            symbols: List of trading pairs
            channels: List of channel names
            cache_manager: Optional Redis cache manager
            performance_monitor: Optional performance monitor
        """
        self.base_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.symbols = symbols
        self.channels = channels
        self.cache_manager = cache_manager
        self.performance_monitor = performance_monitor
        
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_running = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 16
        
        # Callbacks dictionary
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # 添加新的属性
        self.message_timeout = 5.0  # seconds
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30  # seconds
        
        # 增强连接管理
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.connection_state = ConnectionState.DISCONNECTED
        self.message_timeout = 5.0
        self.callback_timeout = 2.0
        self.max_reconnect_attempts = 10
        self.current_reconnect_attempts = 0
        self.connection_monitor_task = None
        
        # 添加锁保护
        self._lock = asyncio.Lock()
        
        # 消息处理统计
        self._message_stats = {
            'processed': 0,
            'errors': 0,
            'last_error_time': None
        }
        
    async def start(self):
        """Start the WebSocket client"""
        self.is_running = True
        while self.is_running:
            try:
                async with websockets.connect(self.base_url) as websocket:
                    self.websocket = websocket
                    logging.info("WebSocket connection established")
                    
                    # Subscribe to channels
                    await self._subscribe()
                    
                    # Reset reconnect delay on successful connection
                    self.reconnect_delay = 1
                    
                    # Process messages
                    await self._process_messages()
                    
            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket connection closed")
                await self._handle_reconnect()
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await self._handle_reconnect()
                
    async def stop(self):
        """Stop the WebSocket client"""
        self.is_running = False
        if self.websocket:
            await self.websocket.close()
            
    async def _subscribe(self):
        """Subscribe to channels"""
        for symbol in self.symbols:
            for channel in self.channels:
                subscription = {
                    "op": "subscribe",
                    "args": [{
                        "channel": channel,
                        "instId": symbol
                    }]
                }
                await self.websocket.send(json.dumps(subscription))
                logging.info(f"Sent subscription request for {symbol} - {channel}")
                
    async def _process_messages(self):
        """Enhanced message processing with queue management"""
        self.connection_monitor_task = asyncio.create_task(
            self._monitor_connection()
        )
        heartbeat_task = asyncio.create_task(
            self._maintain_heartbeat()
        )
        
        try:
            while self.is_running:
                try:
                    # Check queue size
                    if self.message_queue.qsize() >= self.message_queue.maxsize * 0.8:
                        logging.warning("Message queue near capacity")
                        await self._handle_queue_overflow()
                        
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.message_timeout
                    )
                    
                    start_time = time.time()
                    
                    # Process message
                    await self._handle_message(message, start_time)
                    
                except asyncio.TimeoutError:
                    await self._handle_timeout()
                except Exception as e:
                    logging.error(f"Error in message processing: {e}")
                    await self._handle_reconnect()
                    
        finally:
            # Cleanup tasks
            for task in [heartbeat_task, self.connection_monitor_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                        
    async def _maintain_heartbeat(self):
        """Maintain WebSocket heartbeat"""
        while self.is_running and self.websocket:
            try:
                ping_frame = json.dumps({"op": "ping"})
                await self.websocket.send(ping_frame)
                self.last_heartbeat = time.time()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logging.error(f"Heartbeat error: {e}")
                await self._handle_reconnect()
                
    async def _handle_event(self, data: Dict):
        """Handle WebSocket events"""
        if data["event"] == "error":
            logging.error(f"Error from OKX: {data}")
        elif data["event"] == "subscribe":
            logging.info(f"Successfully subscribed to {data.get('arg', {})}")
            
    async def _handle_data(self, data: Dict):
        """Handle market data with batch processing support"""
        try:
            if not isinstance(data.get("data"), list):
                logging.error("Invalid data format")
                return
            
            for item in data["data"]:
                symbol = item.get("instId")
                if not symbol:
                    continue
                
                # Add to batch processor if available
                if hasattr(self, 'batch_processor') and self.batch_processor:
                    await self.batch_processor.add_data(symbol, item)
                else:
                    # Legacy direct processing
                    await self._process_market_data(symbol, item)
                    
        except Exception as e:
            logging.error(f"Error handling market data: {e}")
            if self.performance_monitor:
                self.performance_monitor.record_error("market_data_error")
                
    async def _safe_callback(
        self,
        callback: Callable,
        symbol: str,
        data: Dict
    ):
        """Enhanced callback execution with timeout and error handling"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await asyncio.wait_for(
                    callback(symbol, data),
                    timeout=self.callback_timeout
                )
            else:
                # Run blocking callbacks in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: callback(symbol, data)
                )
                
        except asyncio.TimeoutError:
            logging.error(f"Callback timeout for {symbol}")
            if self.performance_monitor:
                self.performance_monitor.record_error('callback_timeout')
                
        except Exception as e:
            logging.error(f"Error in callback: {e}")
            if self.performance_monitor:
                self.performance_monitor.record_error('callback_error')
                
    async def _handle_reconnect(self):
        """Handle reconnection with exponential backoff"""
        if self.is_running:
            logging.info(f"Attempting to reconnect in {self.reconnect_delay} seconds")
            await asyncio.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
            
    def subscribe_to_symbol(
        self,
        symbol: str,
        callback: Callable[[str, Dict], None]
    ):
        """Subscribe to symbol updates"""
        if symbol not in self.callbacks:
            self.callbacks[symbol] = []
        self.callbacks[symbol].append(callback)
        logging.info(f"Added subscriber for {symbol}") 
        
    async def _handle_message(self, message: str, start_time: float):
        """
        Handle incoming WebSocket message with proper validation and error handling.
        
        Args:
            message: Raw message string
            start_time: Message receipt timestamp
        """
        try:
            data = json.loads(message)
            
            # Validate basic message structure
            if not isinstance(data, dict):
                raise ValueError("Message must be a JSON object")
            
            # Handle events (including heartbeat)
            if "event" in data:
                event_type = data["event"]
                if event_type == "pong":
                    async with self._lock:  # 保护并发访问
                        self.last_heartbeat = time.time()
                    return
                await self._handle_event(data)
                return
            
            # Handle market data
            if "data" in data:
                if not isinstance(data["data"], list):
                    raise ValueError("Market data must be a list")
                await self._handle_data(data)
                
                # Record performance metrics
                if self.performance_monitor:
                    processing_time = time.time() - start_time
                    await self._record_metrics(processing_time, len(message))
                    
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            if self.performance_monitor:
                self.performance_monitor.record_error("json_decode_error")
        except ValueError as e:
            logging.error(f"Validation error: {e}")
            if self.performance_monitor:
                self.performance_monitor.record_error("validation_error")
        except Exception as e:
            logging.error(f"Message handling error: {e}")
            if self.performance_monitor:
                self.performance_monitor.record_error("message_handling_error")
                
    async def _handle_queue_overflow(self):
        """Handle message queue overflow"""
        try:
            # Remove old messages
            while self.message_queue.qsize() > self.message_queue.maxsize * 0.6:
                try:
                    self.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            logging.warning("Cleared old messages from queue")
            
        except Exception as e:
            logging.error(f"Error handling queue overflow: {e}")
            
    async def _monitor_connection(self):
        """Monitor connection health"""
        while self.is_running:
            try:
                if time.time() - self.last_heartbeat > self.heartbeat_interval * 2:
                    logging.error("Connection appears to be dead")
                    await self._handle_reconnect()
                    
                await asyncio.sleep(self.heartbeat_interval / 2)
                
            except Exception as e:
                logging.error(f"Error in connection monitoring: {e}")