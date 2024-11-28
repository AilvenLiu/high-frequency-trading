import redis
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import queue
from functools import wraps
import asyncio
from .subscription_rules import SubscriptionRule

def with_retry(max_attempts: int = 3, delay: int = 1):
    """Retry decorator for Redis operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except redis.RedisError as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_error
        return wrapper
    return decorator

class RedisCacheManager:
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,
        retry_attempts: int = 3,
        retry_delay: int = 1,
        max_queue_size: int = 10000,
        callback_timeout: float = 2.0,
        password: Optional[str] = None,
        max_connections: int = 10,
        **kwargs
    ):
        """
        Initialize Redis cache manager with enhanced error handling and connection pooling.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Time to live for cached data
            retry_attempts: Number of retry attempts for Redis operations
            retry_delay: Delay between retries in seconds
            max_queue_size: Maximum size of message queue
            callback_timeout: Timeout for subscriber callbacks in seconds
            password: Redis password
            max_connections: Maximum number of Redis connections
            **kwargs: Additional keyword arguments for Redis connection pool
        """
        try:
            self.connection_pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                decode_responses=True,
                **kwargs
            )
            
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            
        except redis.RedisError as e:
            logging.error(f"Failed to initialize Redis connection: {e}")
            raise
        
        self.ttl = ttl
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.max_queue_size = max_queue_size
        self.callback_timeout = callback_timeout
        
        # Enhanced subscriber management
        self.subscribers: Dict[str, List[Dict[str, Any]]] = {}
        self.message_queues: Dict[str, queue.Queue] = {}
        self.pubsub = self.redis_client.pubsub()
        self.pubsub_thread = None
        self.running = True
        
        # Health monitoring
        self.last_heartbeat = time.time()
        self.subscriber_health: Dict[str, Dict[str, Any]] = {}
        
        # Start health monitor
        self.health_monitor_thread = threading.Thread(
            target=self._monitor_health,
            daemon=True
        )
        self.health_monitor_thread.start()
        
    @with_retry()
    def store_market_data(self, symbol: str, data: Dict) -> bool:
        """Store and publish market data with validation"""
        try:
            # Validate data
            if not self._validate_market_data(data):
                logging.warning(f"Invalid market data for {symbol}: {data}")
                return False
                
            # Create cache key
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            cache_key = f"market_data:{symbol}:{timestamp}"
            
            # Store data with TTL
            pipeline = self.redis_client.pipeline()
            pipeline.setex(cache_key, self.ttl, json.dumps(data))
            
            # Publish to subscribers if queue not full
            channel = f"market_data:{symbol}"
            if (symbol not in self.message_queues or 
                self.message_queues[symbol].qsize() < self.max_queue_size):
                pipeline.publish(channel, json.dumps(data))
            else:
                logging.warning(f"Message queue full for {symbol}")
                
            pipeline.execute()
            return True
            
        except Exception as e:
            logging.error(f"Failed to store market data: {e}")
            return False
            
    def subscribe_to_symbol(
        self,
        symbol: str,
        callback: Callable[[str, Dict], None],
        filters: Optional[Dict[str, Any]] = None
    ):
        """Subscribe to market data with enhanced management"""
        try:
            channel = f"market_data:{symbol}"
            
            # Initialize message queue
            if symbol not in self.message_queues:
                self.message_queues[symbol] = queue.Queue(maxsize=self.max_queue_size)
                
            # Add subscriber info
            subscriber_info = {
                'callback': callback,
                'filters': filters,
                'last_active': time.time(),
                'error_count': 0
            }
            
            if symbol not in self.subscribers:
                self.subscribers[symbol] = []
            self.subscribers[symbol].append(subscriber_info)
            
            # Subscribe to Redis channel
            if len(self.subscribers[symbol]) == 1:
                self.pubsub.subscribe(channel)
                self._ensure_pubsub_thread()
                
            logging.info(f"Added subscriber for {symbol}")
            
        except Exception as e:
            logging.error(f"Failed to subscribe: {e}")
            
    def unsubscribe_from_symbol(
        self,
        symbol: str,
        callback: Callable[[str, Dict], None]
    ):
        """Unsubscribe with cleanup"""
        try:
            if symbol in self.subscribers:
                # Remove subscriber
                self.subscribers[symbol] = [
                    sub for sub in self.subscribers[symbol]
                    if sub['callback'] != callback
                ]
                
                # Cleanup if no subscribers left
                if not self.subscribers[symbol]:
                    channel = f"market_data:{symbol}"
                    self.pubsub.unsubscribe(channel)
                    del self.subscribers[symbol]
                    
                    if symbol in self.message_queues:
                        del self.message_queues[symbol]
                        
                    # Stop pubsub thread if no subscribers
                    if not self.subscribers:
                        self._stop_pubsub_thread()
                        
            logging.info(f"Removed subscriber for {symbol}")
            
        except Exception as e:
            logging.error(f"Failed to unsubscribe: {e}")
            
    def _ensure_pubsub_thread(self):
        """Ensure pubsub thread is running"""
        if not self.pubsub_thread or not self.pubsub_thread.is_alive():
            self._start_pubsub_thread()
            
    def _start_pubsub_thread(self):
        """Start pubsub thread with enhanced error handling"""
        def pubsub_handler():
            while self.running:
                try:
                    message = self.pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        self._handle_message(message)
                    self.last_heartbeat = time.time()
                except Exception as e:
                    logging.error(f"Error in pubsub handler: {e}")
                    time.sleep(1)
                    
        self.pubsub_thread = threading.Thread(
            target=pubsub_handler,
            daemon=True
        )
        self.pubsub_thread.start()
        
    def _handle_message(self, message: Dict):
        """Handle incoming message with validation and filtering"""
        try:
            channel = message['channel'].decode()
            data = json.loads(message['data'].decode())
            symbol = channel.split(':')[1]
            
            if symbol in self.subscribers:
                for subscriber in self.subscribers[symbol]:
                    if self._should_notify_subscriber(subscriber, data):
                        self._safe_notify_subscriber(symbol, subscriber, data)
                        
        except Exception as e:
            logging.error(f"Error handling message: {e}")
            
    def _should_notify_subscriber(
        self,
        subscriber: Dict[str, Any],
        data: Dict
    ) -> bool:
        """Check if subscriber should be notified based on filters"""
        filters = subscriber.get('filters')
        if not filters:
            return True
            
        try:
            return all(
                data.get(key) == value
                for key, value in filters.items()
            )
        except Exception:
            return False
            
    def _safe_notify_subscriber(
        self,
        symbol: str,
        subscriber: Dict[str, Any],
        data: Dict
    ):
        """Safely notify subscriber with timeout and error handling"""
        try:
            callback = subscriber['callback']
            
            # Create async task for async callbacks
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(
                    self._async_callback_wrapper(symbol, subscriber, data)
                )
            else:
                # Execute sync callback in thread pool
                threading.Thread(
                    target=self._sync_callback_wrapper,
                    args=(symbol, subscriber, data),
                    daemon=True
                ).start()
                
        except Exception as e:
            logging.error(f"Error notifying subscriber: {e}")
            subscriber['error_count'] += 1
            
    async def _async_callback_wrapper(
        self,
        symbol: str,
        subscriber: Dict[str, Any],
        data: Dict
    ):
        """Wrapper for async callbacks with timeout"""
        try:
            await asyncio.wait_for(
                subscriber['callback'](symbol, data),
                timeout=self.callback_timeout
            )
            subscriber['last_active'] = time.time()
            subscriber['error_count'] = 0
        except Exception as e:
            logging.error(f"Async callback error: {e}")
            subscriber['error_count'] += 1
            
    def _sync_callback_wrapper(
        self,
        symbol: str,
        subscriber: Dict[str, Any],
        data: Dict
    ):
        """Wrapper for sync callbacks with timeout"""
        try:
            subscriber['callback'](symbol, data)
            subscriber['last_active'] = time.time()
            subscriber['error_count'] = 0
        except Exception as e:
            logging.error(f"Sync callback error: {e}")
            subscriber['error_count'] += 1
            
    def _monitor_health(self):
        """Monitor Redis connection and subscriber health"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check Redis connection
                if not self._check_redis_connection():
                    logging.error("Redis connection lost")
                    self._attempt_redis_reconnect()
                    
                # Check subscriber health
                for subscriber_id, health_data in self.subscriber_health.items():
                    last_active = health_data.get('last_active', 0)
                    if current_time - last_active > 60:  # 1 minute timeout
                        logging.warning(f"Subscriber {subscriber_id} appears inactive")
                        self._handle_inactive_subscriber(subscriber_id)
                        
                # Check queue sizes
                for queue_name, message_queue in self.message_queues.items():
                    if message_queue.qsize() > self.max_queue_size * 0.8:
                        logging.warning(f"Queue {queue_name} near capacity")
                        self._handle_queue_overflow(queue_name)
                        
                # Update heartbeat
                self.last_heartbeat = current_time
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Health monitor error: {e}")
                time.sleep(1)
                
    def _check_redis_connection(self) -> bool:
        """Check if Redis connection is alive"""
        try:
            self.redis_client.ping()
            return True
        except redis.ConnectionError:
            return False
        
    def _attempt_redis_reconnect(self):
        """Attempt to reconnect to Redis"""
        for attempt in range(self.retry_attempts):
            try:
                self.redis_client = redis.Redis(
                    connection_pool=self.connection_pool,
                    retry_on_timeout=True
                )
                logging.info("Redis reconnection successful")
                return
            except redis.ConnectionError:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
                
        logging.error("Failed to reconnect to Redis after multiple attempts")
        
    def _handle_inactive_subscriber(self, subscriber_id: str):
        """Handle inactive subscriber"""
        try:
            # Remove subscriber from active list
            if subscriber_id in self.subscribers:
                del self.subscribers[subscriber_id]
                del self.subscriber_health[subscriber_id]
                logging.info(f"Removed inactive subscriber: {subscriber_id}")
                
        except Exception as e:
            logging.error(f"Error handling inactive subscriber: {e}")
            
    def _handle_queue_overflow(self, queue_name: str):
        """Handle queue overflow condition"""
        try:
            queue = self.message_queues[queue_name]
            # Remove oldest messages
            while queue.qsize() > self.max_queue_size * 0.6:
                queue.get_nowait()
            logging.info(f"Cleared queue overflow for {queue_name}")
        except Exception as e:
            logging.error(f"Error handling queue overflow: {e}")
            
    def _validate_market_data(self, data: Dict) -> bool:
        """Validate market data structure and values"""
        required_fields = {'price', 'timestamp', 'volume'}
        
        if not all(field in data for field in required_fields):
            return False
            
        try:
            price = float(data['price'])
            volume = float(data['volume'])
            return price > 0 and volume >= 0
        except (ValueError, TypeError):
            return False
            
    def cleanup(self):
        """Cleanup all resources"""
        try:
            self.running = False
            self._stop_pubsub_thread()
            
            # Clear all subscriptions
            for symbol in list(self.subscribers.keys()):
                self.unsubscribe_from_symbol(symbol, None)
                
            # Close Redis connection
            self.redis_client.close()
            
            # Wait for health monitor to stop
            if self.health_monitor_thread.is_alive():
                self.health_monitor_thread.join(timeout=1)
                
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    def subscribe_with_rule(self, rule: SubscriptionRule) -> bool:
        """
        Subscribe to market data with custom rule
        
        Args:
            rule: Subscription rule configuration
            
        Returns:
            bool: True if subscription was successful
        """
        try:
            # Generate unique rule ID if not provided
            if not rule.rule_id:
                rule.rule_id = f"{rule.symbol}_{int(time.time())}"
            
            # Store rule in Redis for persistence
            rule_key = f"subscription_rules:{rule.rule_id}"
            self.redis_client.set(
                rule_key,
                json.dumps(rule.to_dict()),
                ex=86400  # 24 hour expiry
            )
            
            # Add to active subscriptions
            if rule.symbol not in self.subscribers:
                self.subscribers[rule.symbol] = []
            
            self.subscribers[rule.symbol].append({
                'rule': rule,
                'callback': rule.callback,
                'last_active': time.time(),
                'error_count': 0
            })
            
            logging.info(f"Added subscription rule: {rule.rule_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add subscription rule: {e}")
            return False
        
    def restore_subscriptions(self, callback_factory: Callable):
        """
        Restore persisted subscription rules
        
        Args:
            callback_factory: Function to create callback for restored rules
        """
        try:
            # Find all persisted rules
            rule_keys = self.redis_client.keys("subscription_rules:*")
            
            for key in rule_keys:
                rule_data = self.redis_client.get(key)
                if rule_data:
                    # Restore rule
                    rule_dict = json.loads(rule_data)
                    callback = callback_factory(rule_dict['symbol'])
                    
                    rule = SubscriptionRule.from_dict(rule_dict, callback)
                    self.subscribe_with_rule(rule)
                    
            logging.info(f"Restored {len(rule_keys)} subscription rules")
            
        except Exception as e:
            logging.error(f"Failed to restore subscriptions: {e}")
        
    def _evaluate_rules(self, symbol: str, data: Dict):
        """Evaluate subscription rules for market data"""
        if symbol not in self.subscribers:
            return
        
        for subscriber in self.subscribers[symbol]:
            rule = subscriber['rule']
            if self.rule_engine.evaluate(rule, data):
                try:
                    subscriber['callback'](symbol, data)
                    subscriber['last_active'] = time.time()
                except Exception as e:
                    logging.error(f"Error in rule callback: {e}")
                    subscriber['error_count'] += 1

    async def store_market_data_batch(
        self,
        symbol: str,
        data_batch: List[Dict]
    ) -> bool:
        """
        Store batch of market data in Redis.
        
        Args:
            symbol: Trading pair symbol
            data_batch: List of market data items
            
        Returns:
            bool: Success status
        """
        try:
            pipeline = self.redis_client.pipeline()
            
            for data in data_batch:
                key = f"market_data:{symbol}:{data.get('timestamp', time.time())}"
                pipeline.setex(
                    key,
                    self.ttl,
                    json.dumps(data)
                )
                
                # Update latest price
                if "last" in data:
                    pipeline.setex(
                        f"latest_price:{symbol}",
                        self.ttl,
                        data["last"]
                    )
                    
            # Execute pipeline
            results = await pipeline.execute()
            
            # Notify subscribers
            if self.pubsub and data_batch:
                await self.publish_batch(symbol, data_batch)
                
            return all(results)
            
        except redis.RedisError as e:
            logging.error(f"Redis batch storage error for {symbol}: {e}")
            if self.retry_attempts > 0:
                return await self._retry_operation(
                    self.store_market_data_batch,
                    symbol,
                    data_batch
                )
            return False
            
    async def publish_batch(
        self,
        symbol: str,
        data_batch: List[Dict]
    ):
        """Publish batch of market data to subscribers"""
        try:
            message = {
                "symbol": symbol,
                "timestamp": time.time(),
                "batch_size": len(data_batch),
                "data": data_batch
            }
            
            await self.pubsub.publish(
                f"market_data:{symbol}",
                json.dumps(message)
            )
            
        except Exception as e:
            logging.error(f"Error publishing batch for {symbol}: {e}")
            if self.performance_monitor:
                self.performance_monitor.record_error("batch_publish_error")
