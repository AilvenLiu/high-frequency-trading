import pytest
import asyncio
import json
import websockets
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from data_collection.async_okx_client import AsyncOKXClient
from data_collection.redis_cache_manager import RedisCacheManager
from monitoring.performance_monitor import PerformanceMonitor

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    mock = AsyncMock()
    mock.send = AsyncMock()
    mock.recv = AsyncMock()
    mock.close = AsyncMock()
    return mock

@pytest.fixture
def mock_cache_manager():
    """Mock Redis cache manager"""
    mock = Mock(spec=RedisCacheManager)
    mock.store_market_data = Mock(return_value=True)
    return mock

@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitor"""
    mock = Mock(spec=PerformanceMonitor)
    mock.record_message = Mock()
    mock.record_error = Mock()
    return mock

@pytest.fixture
async def okx_client(mock_cache_manager, mock_performance_monitor):
    """Create OKX client instance"""
    client = AsyncOKXClient(
        symbols=['BTC-USDT', 'ETH-USDT'],
        channels=['tickers'],
        cache_manager=mock_cache_manager,
        performance_monitor=mock_performance_monitor
    )
    yield client
    await client.stop()

class TestAsyncOKXClient:
    """Test suite for AsyncOKXClient"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, okx_client):
        """Test client initialization"""
        assert okx_client.symbols == ['BTC-USDT', 'ETH-USDT']
        assert okx_client.channels == ['tickers']
        assert okx_client.is_running is False
        assert okx_client.websocket is None
        
    @pytest.mark.asyncio
    async def test_subscription(self, okx_client, mock_websocket):
        """Test WebSocket subscription"""
        with patch('websockets.connect', return_value=mock_websocket):
            # Start client in background
            task = asyncio.create_task(okx_client.start())
            await asyncio.sleep(0.1)
            
            # Verify subscription messages
            subscription_calls = mock_websocket.send.call_args_list
            assert len(subscription_calls) == 2  # One for each symbol
            
            # Verify subscription format
            for call in subscription_calls:
                sub_msg = json.loads(call.args[0])
                assert sub_msg['op'] == 'subscribe'
                assert 'channel' in sub_msg['args'][0]
                assert 'instId' in sub_msg['args'][0]
                
            await okx_client.stop()
            await task
            
    @pytest.mark.asyncio
    async def test_message_processing(self, okx_client, mock_websocket):
        """Test market data message processing"""
        test_message = {
            'event': 'update',
            'data': [{
                'instId': 'BTC-USDT',
                'price': '50000',
                'volume': '1.5',
                'timestamp': datetime.now().isoformat()
            }]
        }
        
        mock_websocket.recv.return_value = json.dumps(test_message)
        
        with patch('websockets.connect', return_value=mock_websocket):
            task = asyncio.create_task(okx_client.start())
            await asyncio.sleep(0.1)
            
            # Verify data handling
            assert okx_client.cache_manager.store_market_data.called
            assert okx_client.performance_monitor.record_message.called
            
            await okx_client.stop()
            await task
            
    @pytest.mark.asyncio
    async def test_heartbeat(self, okx_client, mock_websocket):
        """Test heartbeat mechanism"""
        responses = [
            json.dumps({'event': 'pong'}),
            json.dumps({'event': 'error'})
        ]
        mock_websocket.recv.side_effect = responses
        
        with patch('websockets.connect', return_value=mock_websocket):
            task = asyncio.create_task(okx_client.start())
            await asyncio.sleep(0.1)
            
            # Verify heartbeat handling
            assert time.time() - okx_client.last_heartbeat < okx_client.heartbeat_interval
            
            await okx_client.stop()
            await task
            
    @pytest.mark.asyncio
    async def test_reconnection(self, okx_client, mock_websocket):
        """Test reconnection mechanism"""
        mock_websocket.recv.side_effect = [
            websockets.exceptions.ConnectionClosed(1006, "Connection lost"),
            json.dumps({'event': 'pong'})
        ]
        
        with patch('websockets.connect', return_value=mock_websocket):
            task = asyncio.create_task(okx_client.start())
            await asyncio.sleep(0.1)
            
            # Verify reconnection attempt
            assert okx_client.reconnect_delay > 1
            assert mock_websocket.close.called
            
            await okx_client.stop()
            await task
            
    @pytest.mark.asyncio
    async def test_callback_handling(self, okx_client, mock_websocket):
        """Test callback execution"""
        callback_called = asyncio.Event()
        
        async def test_callback(symbol: str, data: dict):
            callback_called.set()
            
        okx_client.subscribe_to_symbol('BTC-USDT', test_callback)
        
        test_message = {
            'data': [{
                'instId': 'BTC-USDT',
                'price': '50000',
                'volume': '1.5'
            }]
        }
        mock_websocket.recv.return_value = json.dumps(test_message)
        
        with patch('websockets.connect', return_value=mock_websocket):
            task = asyncio.create_task(okx_client.start())
            await asyncio.wait_for(callback_called.wait(), timeout=1)
            
            await okx_client.stop()
            await task
            
    @pytest.mark.asyncio
    async def test_error_handling(self, okx_client, mock_websocket):
        """Test error handling"""
        mock_websocket.recv.side_effect = Exception("Test error")
        
        with patch('websockets.connect', return_value=mock_websocket):
            task = asyncio.create_task(okx_client.start())
            await asyncio.sleep(0.1)
            
            assert okx_client.performance_monitor.record_error.called
            
            await okx_client.stop()
            await task
            
    @pytest.mark.asyncio
    async def test_queue_overflow(self, okx_client, mock_websocket):
        """Test message queue overflow handling"""
        # Fill queue to capacity
        for _ in range(okx_client.message_queue.maxsize + 1):
            await okx_client.message_queue.put("test")
            
        assert okx_client.message_queue.full()
        await okx_client._handle_queue_overflow()
        assert not okx_client.message_queue.full() 

@pytest.mark.asyncio
async def test_message_handling(okx_client):
    # Test market data message
    test_message = json.dumps({
        "data": [{
            "instId": "BTC-USDT",
            "last": "50000",
            "lastSz": "0.1",
            "askPx": "50001",
            "askSz": "1.0",
            "bidPx": "49999",
            "bidSz": "1.0",
            "open24h": "49000",
            "high24h": "51000",
            "low24h": "48000",
            "volCcy24h": "1000",
            "vol24h": "20"
        }]
    })
    
    start_time = asyncio.get_event_loop().time()
    await okx_client._handle_message(test_message, start_time)
    
    # Test heartbeat message
    heartbeat_message = json.dumps({"event": "pong"})
    await okx_client._handle_message(heartbeat_message, start_time)
    
    # Test invalid message
    with pytest.raises(json.JSONDecodeError):
        await okx_client._handle_message("invalid json", start_time)

@pytest.mark.asyncio
async def test_subscription():
    client = AsyncOKXClient(
        symbols=['BTC-USDT', 'ETH-USDT'],
        channels=['tickers', 'trades']
    )
    
    with patch('websockets.connect') as mock_connect:
        mock_ws = Mock()
        mock_connect.return_value.__aenter__.return_value = mock_ws
        
        # Start client
        client_task = asyncio.create_task(client.start())
        await asyncio.sleep(0.1)  # Allow time for subscription
        
        # Verify subscription messages
        subscription_calls = mock_ws.send.call_args_list
        assert len(subscription_calls) == 4  # 2 symbols * 2 channels
        
        # Verify subscription format
        for call in subscription_calls:
            subscription = json.loads(call[0][0])
            assert subscription["op"] == "subscribe"
            assert "channel" in subscription["args"][0]
            assert "instId" in subscription["args"][0]
            
        # Cleanup
        client.is_running = False
        await client_task 

@pytest.mark.asyncio
async def test_extreme_conditions(okx_client, mock_websocket):
    """Test client behavior under extreme conditions"""
    # Test extremely large message
    large_message = {
        'data': [{
            'instId': 'BTC-USDT',
            'price': '50000',
            'volume': '1.5',
            'extra_data': 'x' * 1000000  # 1MB of extra data
        }]
    }
    
    # Test high-frequency messages
    messages = [json.dumps(large_message) for _ in range(1000)]
    mock_websocket.recv.side_effect = messages
    
    with patch('websockets.connect', return_value=mock_websocket):
        task = asyncio.create_task(okx_client.start())
        await asyncio.sleep(0.1)
        
        # Verify client handles large messages
        assert okx_client.message_queue.qsize() > 0
        assert not okx_client.message_queue.full()
        
        await okx_client.stop()
        await task 