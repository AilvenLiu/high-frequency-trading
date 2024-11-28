import pytest
import redis
import json
import time
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from data_collection.redis_cache_manager import RedisCacheManager, with_retry

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('redis.Redis') as mock:
        # Setup basic mock behaviors
        mock.return_value.ping.return_value = True
        mock.return_value.pipeline.return_value.execute.return_value = [True]
        mock.return_value.pubsub.return_value.subscribe.return_value = None
        yield mock

@pytest.fixture
def cache_manager(mock_redis):
    """Create cache manager instance for testing"""
    manager = RedisCacheManager(
        host='localhost',
        port=6379,
        max_queue_size=100,
        callback_timeout=1.0
    )
    yield manager
    manager.cleanup()

class TestRedisCacheManager:
    """Test suite for RedisCacheManager"""
    
    def test_initialization(self, cache_manager, mock_redis):
        """Test cache manager initialization"""
        assert cache_manager.host == 'localhost'
        assert cache_manager.port == 6379
        assert cache_manager.running is True
        mock_redis.assert_called_once()
        
    def test_store_market_data_success(self, cache_manager):
        """Test successful market data storage"""
        test_data = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        result = cache_manager.store_market_data('BTC-USD', test_data)
        assert result is True
        
    def test_store_invalid_market_data(self, cache_manager):
        """Test storing invalid market data"""
        invalid_data = {
            'price': -100.0,  # Invalid price
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        result = cache_manager.store_market_data('BTC-USD', invalid_data)
        assert result is False
        
    @pytest.mark.asyncio
    async def test_async_subscription(self, cache_manager):
        """Test async subscription handling"""
        received_data = []
        
        async def async_callback(symbol: str, data: dict):
            received_data.append((symbol, data))
            
        # Subscribe to test channel
        cache_manager.subscribe_to_symbol('BTC-USD', async_callback)
        
        # Simulate message reception
        test_data = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Manually trigger message handling
        message = {
            'type': 'message',
            'channel': b'market_data:BTC-USD',
            'data': json.dumps(test_data).encode()
        }
        
        await cache_manager._handle_message(message)
        await asyncio.sleep(0.1)  # Allow async callback to complete
        
        assert len(received_data) == 1
        assert received_data[0][0] == 'BTC-USD'
        assert received_data[0][1] == test_data
        
    def test_sync_subscription(self, cache_manager):
        """Test synchronous subscription handling"""
        received_data = []
        
        def sync_callback(symbol: str, data: dict):
            received_data.append((symbol, data))
            
        # Subscribe to test channel
        cache_manager.subscribe_to_symbol('ETH-USD', sync_callback)
        
        # Simulate message reception
        test_data = {
            'price': 200.0,
            'volume': 500.0,
            'timestamp': datetime.now().isoformat()
        }
        
        message = {
            'type': 'message',
            'channel': b'market_data:ETH-USD',
            'data': json.dumps(test_data).encode()
        }
        
        cache_manager._handle_message(message)
        time.sleep(0.1)  # Allow sync callback to complete
        
        assert len(received_data) == 1
        assert received_data[0][0] == 'ETH-USD'
        assert received_data[0][1] == test_data
        
    def test_subscription_with_filters(self, cache_manager):
        """Test subscription with data filters"""
        received_data = []
        
        def callback(symbol: str, data: dict):
            received_data.append((symbol, data))
            
        # Subscribe with price filter
        filters = {'price': 100.0}
        cache_manager.subscribe_to_symbol('BTC-USD', callback, filters=filters)
        
        # Test matching data
        matching_data = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Test non-matching data
        non_matching_data = {
            'price': 200.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Simulate messages
        for data in [matching_data, non_matching_data]:
            message = {
                'type': 'message',
                'channel': b'market_data:BTC-USD',
                'data': json.dumps(data).encode()
            }
            cache_manager._handle_message(message)
            
        time.sleep(0.1)
        assert len(received_data) == 1  # Only matching data should be received
        
    @pytest.mark.asyncio
    async def test_callback_timeout(self, cache_manager):
        """Test callback timeout handling"""
        async def slow_callback(symbol: str, data: dict):
            await asyncio.sleep(2.0)  # Longer than timeout
            
        cache_manager.subscribe_to_symbol('BTC-USD', slow_callback)
        
        test_data = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        message = {
            'type': 'message',
            'channel': b'market_data:BTC-USD',
            'data': json.dumps(test_data).encode()
        }
        
        # Should not raise exception but log error
        await cache_manager._handle_message(message)
        
    def test_connection_failure_recovery(self, cache_manager, mock_redis):
        """Test Redis connection failure recovery"""
        # Simulate connection failure
        mock_redis.return_value.ping.side_effect = [
            redis.ConnectionError,
            True  # Recovers on second attempt
        ]
        
        # Should recover automatically
        assert not cache_manager._check_redis_connection()
        cache_manager._handle_connection_failure()
        assert cache_manager._check_redis_connection()
        
    def test_cleanup(self, cache_manager):
        """Test cleanup process"""
        def callback(symbol: str, data: dict):
            pass
            
        # Add some subscriptions
        cache_manager.subscribe_to_symbol('BTC-USD', callback)
        cache_manager.subscribe_to_symbol('ETH-USD', callback)
        
        # Perform cleanup
        cache_manager.cleanup()
        
        assert not cache_manager.running
        assert not cache_manager.subscribers
        assert cache_manager.pubsub_thread is None
        
    @pytest.mark.parametrize("test_data,expected", [
        (
            {'price': 100.0, 'volume': 1000.0, 'timestamp': '2024-01-01T00:00:00'},
            True
        ),
        (
            {'price': -100.0, 'volume': 1000.0, 'timestamp': '2024-01-01T00:00:00'},
            False
        ),
        (
            {'price': 100.0, 'volume': -1.0, 'timestamp': '2024-01-01T00:00:00'},
            False
        ),
        (
            {'price': 'invalid', 'volume': 1000.0, 'timestamp': '2024-01-01T00:00:00'},
            False
        ),
        (
            {'volume': 1000.0, 'timestamp': '2024-01-01T00:00:00'},  # Missing price
            False
        ),
    ])
    def test_market_data_validation(self, cache_manager, test_data, expected):
        """Test market data validation with various inputs"""
        assert cache_manager._validate_market_data(test_data) is expected
        
    def test_retry_decorator(self):
        """Test retry decorator functionality"""
        mock_func = Mock(side_effect=[
            redis.ConnectionError,
            redis.ConnectionError,
            True
        ])
        
        @with_retry(max_attempts=3, delay=0.1)
        def test_func():
            return mock_func()
            
        result = test_func()
        assert result is True
        assert mock_func.call_count == 3
        
    @pytest.mark.asyncio
    async def test_queue_overflow(self, cache_manager):
        """Test message queue overflow handling"""
        received_count = 0
        
        async def slow_callback(symbol: str, data: dict):
            nonlocal received_count
            await asyncio.sleep(0.1)  # Simulate slow processing
            received_count += 1
            
        # Set small queue size for testing
        cache_manager.max_queue_size = 5
        cache_manager.subscribe_to_symbol('BTC-USD', slow_callback)
        
        # Send more messages than queue size
        test_data = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        messages = []
        for i in range(10):  # Send 10 messages to queue of size 5
            message = {
                'type': 'message',
                'channel': b'market_data:BTC-USD',
                'data': json.dumps({**test_data, 'seq': i}).encode()
            }
            messages.append(message)
            
        # Send messages rapidly
        for msg in messages:
            await cache_manager._handle_message(msg)
            
        await asyncio.sleep(1.5)  # Allow processing to complete
        
        # Should have processed some messages and dropped others
        assert received_count < 10
        assert received_count > 0
        
    def test_subscriber_health_check(self, cache_manager):
        """Test subscriber health monitoring"""
        error_count = 0
        
        def failing_callback(symbol: str, data: dict):
            nonlocal error_count
            error_count += 1
            raise Exception("Simulated callback failure")
            
        cache_manager.subscribe_to_symbol('BTC-USD', failing_callback)
        
        # Simulate multiple message failures
        test_data = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        for _ in range(15):  # More than error threshold
            message = {
                'type': 'message',
                'channel': b'market_data:BTC-USD',
                'data': json.dumps(test_data).encode()
            }
            cache_manager._handle_message(message)
            
        # Force health check
        cache_manager._check_subscriber_health()
        
        # Subscriber should have been removed due to errors
        assert 'BTC-USD' not in cache_manager.subscribers
        
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, cache_manager):
        """Test concurrent subscribe/unsubscribe operations"""
        async def subscribe_unsubscribe():
            callback = lambda s, d: None
            cache_manager.subscribe_to_symbol('BTC-USD', callback)
            await asyncio.sleep(0.1)
            cache_manager.unsubscribe_from_symbol('BTC-USD', callback)
            
        # Run multiple concurrent operations
        tasks = [subscribe_unsubscribe() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Should have cleaned up properly
        assert 'BTC-USD' not in cache_manager.subscribers
        
    def test_memory_management(self, cache_manager):
        """Test memory management under load"""
        large_data = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat(),
            'extra_data': 'x' * 1000000  # 1MB of extra data
        }
        
        # Store multiple large messages
        for i in range(10):
            cache_manager.store_market_data('BTC-USD', large_data)
            
        # Memory usage should be managed
        import psutil
        process = psutil.Process()
        mem_usage = process.memory_percent()
        assert mem_usage < 90.0  # Should not consume excessive memory
        
    @pytest.mark.asyncio
    async def test_network_latency(self, cache_manager):
        """Test behavior under network latency"""
        received_timestamps = []
        
        async def latency_callback(symbol: str, data: dict):
            received_timestamps.append(time.time())
            await asyncio.sleep(0.5)  # Simulate network latency
            
        cache_manager.subscribe_to_symbol('BTC-USD', latency_callback)
        
        # Send messages with varying delays
        test_data = {
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        
        send_timestamps = []
        for i in range(5):
            send_timestamps.append(time.time())
            message = {
                'type': 'message',
                'channel': b'market_data:BTC-USD',
                'data': json.dumps(test_data).encode()
            }
            await cache_manager._handle_message(message)
            await asyncio.sleep(0.1)
            
        await asyncio.sleep(3)  # Allow all processing to complete
        
        # Verify message ordering and processing
        assert len(received_timestamps) == len(send_timestamps)
        for send_time, receive_time in zip(send_timestamps, received_timestamps):
            assert receive_time >= send_time
            
    def test_resource_cleanup(self, cache_manager):
        """Test resource cleanup under various conditions"""
        import gc
        
        # Add some subscriptions
        callbacks = [lambda s, d: None for _ in range(10)]
        for callback in callbacks:
            cache_manager.subscribe_to_symbol('BTC-USD', callback)
            
        # Force garbage collection of callbacks
        callbacks = None
        gc.collect()
        
        # Cleanup should handle missing callbacks
        cache_manager.cleanup()
        
        # Verify all resources are cleaned up
        assert not cache_manager.subscribers
        assert not cache_manager.message_queues
        assert not cache_manager.running
        
    @pytest.mark.parametrize("network_error", [
        redis.ConnectionError,
        redis.TimeoutError,
        redis.RedisError
    ])
    def test_error_recovery(self, cache_manager, mock_redis, network_error):
        """Test recovery from various network errors"""
        # Simulate specific network error
        mock_redis.return_value.ping.side_effect = [
            network_error(),
            network_error(),
            True  # Recovers on third attempt
        ]
        
        # Should recover automatically
        assert not cache_manager._check_redis_connection()
        cache_manager._handle_connection_failure()
        assert cache_manager._check_redis_connection()
        
    def test_health_monitoring(self, cache_manager):
        # Mock Redis client
        mock_redis = Mock()
        cache_manager.redis_client = mock_redis
        
        # Test connection check
        mock_redis.ping.return_value = True
        assert cache_manager._check_redis_connection() is True
        
        mock_redis.ping.side_effect = redis.ConnectionError
        assert cache_manager._check_redis_connection() is False
        
        # Test queue overflow handling
        test_queue = asyncio.Queue(maxsize=100)
        for i in range(90):  # Fill to 90%
            test_queue.put_nowait(f"test_message_{i}")
        
        cache_manager.message_queues["test_queue"] = test_queue
        cache_manager._handle_queue_overflow("test_queue")
        assert test_queue.qsize() < 70  # Should be reduced to below 70%

    def test_subscriber_management(self, cache_manager):
        # Add test subscriber
        subscriber_id = "test_sub_1"
        cache_manager.subscribers[subscriber_id] = {
            "callback": Mock(),
            "filters": None
        }
        cache_manager.subscriber_health[subscriber_id] = {
            "last_active": time.time() - 120  # 2 minutes ago
        }
        
        # Test inactive subscriber handling
        cache_manager._handle_inactive_subscriber(subscriber_id)
        assert subscriber_id not in cache_manager.subscribers
        assert subscriber_id not in cache_manager.subscriber_health

    @pytest.mark.asyncio
    async def test_redis_reconnection(self, cache_manager):
        with patch('redis.Redis') as mock_redis:
            # Simulate failed connection attempts
            mock_redis.side_effect = [
                redis.ConnectionError,
                redis.ConnectionError,
                Mock()  # Successful on third attempt
            ]
            
            await cache_manager._attempt_redis_reconnect()
            assert mock_redis.call_count == 3