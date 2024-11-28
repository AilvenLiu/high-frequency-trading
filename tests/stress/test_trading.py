import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from stress_testing.load_tests.data_collection_load_test import (
    DataCollectionLoadTest,
    LoadTestConfig
)

@pytest.fixture
def mock_client():
    client = Mock()
    client._handle_message = AsyncMock()
    client.subscribe_to_symbol = AsyncMock()
    client.unsubscribe_from_symbol = AsyncMock()
    return client

@pytest.fixture
def mock_filter_manager():
    manager = Mock()
    manager.apply_filters = AsyncMock()
    return manager

@pytest.fixture
def mock_batch_processor():
    processor = Mock()
    processor.add_data = AsyncMock()
    processor.config = Mock()
    return processor

@pytest.fixture
def mock_performance_monitor():
    monitor = Mock()
    monitor.get_current_metrics = Mock(return_value={
        'latency': 0.1,
        'throughput': 1000
    })
    return monitor

@pytest.fixture
def load_test(
    mock_client,
    mock_filter_manager,
    mock_batch_processor,
    mock_performance_monitor
):
    config = LoadTestConfig(
        duration_seconds=60,
        max_symbols=5,
        max_subscriptions=100,
        batch_sizes=[10, 50],
        message_rates=[10, 50]
    )
    
    test = DataCollectionLoadTest(
        config,
        mock_client,
        mock_filter_manager,
        mock_batch_processor,
        mock_performance_monitor
    )

    # await test.initialize()
    
    return test

@pytest.mark.asyncio
async def test_subscription_scaling(load_test):
    await load_test._run_subscription_scaling_test()
    
    # Verify subscriptions were added and removed
    assert load_test.client.subscribe_to_symbol.called
    assert load_test.client.unsubscribe_from_symbol.called
    
    # Check test results
    results = [r for r in load_test.test_results 
               if r['test_type'] == 'subscription_scaling']
    assert len(results) > 0

@pytest.mark.asyncio
async def test_batch_size_test(load_test):
    await load_test._run_batch_size_test()
    
    # Verify batch processing
    assert load_test.batch_processor.add_data.called
    
    # Check test results
    results = [r for r in load_test.test_results 
               if r['test_type'] == 'batch_size']
    assert len(results) > 0

@pytest.mark.asyncio
async def test_message_rate_test(load_test):
    await load_test._run_message_rate_test()
    
    # Verify message handling
    assert load_test.client._handle_message.called
    
    # Check test results
    results = [r for r in load_test.test_results 
               if r['test_type'] == 'message_rate']
    assert len(results) > 0

@pytest.mark.asyncio
async def test_error_handling_test(load_test):
    await load_test._run_error_handling_test()
    
    # Verify error injection
    assert load_test.client._handle_message.called
    
    # Check test results
    results = [r for r in load_test.test_results 
               if r['test_type'] == 'error_handling']
    assert len(results) > 0 