# main.py
import asyncio
import logging
import sys
from typing import Dict, Any
from datetime import datetime, timedelta
from data_collection.async_okx_client import AsyncOKXClient
from data_collection.redis_cache_manager import RedisCacheManager
from monitoring.performance_monitor import PerformanceMonitor, PerformanceThresholds
from data_collection.subscription_rules import SubscriptionRule
from utils.logger_config import setup_logger
from monitoring.metrics_collectors import (
    LatencyCollector,
    BandwidthCollector,
    TaskMetricsCollector
)
from monitoring.queue_monitor import QueueMonitor
from monitoring.report_generator import MetricsReportGenerator
from stress_testing.metrics.performance_metrics import PerformanceMetricsCollector
from stress_testing.metrics.resource_monitor import ResourceMonitor
from stress_testing.generators.data_generator import DataGenerator, DataGeneratorConfig
from stress_testing.generators.client_simulator import ClientConfig
from stress_testing.generators.client_manager import ClientManager
from stress_testing.reporting.report_generator import StressTestReportGenerator
from data_collection.filters.filter_manager import FilterManager, FilterConfig, FilterType
import time
from stress_testing.load_tests.data_collection_load_test import (
    DataCollectionLoadTest,
    LoadTestConfig
)
from stress_testing.stability_tests.long_running_test import (
    LongRunningStabilityTest,
    StabilityTestConfig
)
import json
from signal_generation.indicators.rsi import RSIIndicator, RSIConfig
from signal_generation.indicators.macd import MACDConfig, MACDIndicator
from signal_generation.indicators.bollinger_bands import BollingerBandsConfig, BollingerBandsIndicator
from signal_generation.features.feature_extractor import FeatureExtractor, FeatureConfig
from signal_generation.models.lstm_predictor import ModelTrainer
from signal_generation.models.distributed_trainer import (
    DistributedModelTrainer,
    DistributedConfig
)
import torch
from signal_generation.monitoring.model_performance_monitor import (
    ModelPerformanceMonitor,
    PerformanceMonitorConfig
)
from funds_management.funds_manager import FundsManager
from trading.signal_generation.signal_generator import SignalGenerator, SignalConfig
from decimal import Decimal
from trading.execution.api_connector import APIConnector, OrderRequest, OrderType, OrderSide
from trading.execution.order_executor import OrderExecutor
from funds_management.risk_control.margin_calculator import MarginCalculator
from funds_management.risk_control.risk_controller import RiskController
from trading.data.market_data_sync import MarketDataSync
from trading.monitoring.health_check import HealthCheck

async def main():
    """Main application entry point"""
    # Setup logging
    setup_logger(console_level=logging.INFO)
    
    # Load centralized configuration
    config_manager = ConfigManager()
    
    # Initialize metrics collectors
    performance_metrics = PerformanceMetricsCollector()
    resource_monitor = ResourceMonitor()
    
    # Start monitoring
    monitor_task = asyncio.create_task(resource_monitor.start())
    
    try:
        # Initialize components with configuration
        cache_manager = RedisCacheManager(
            host=config_manager.data_collection_config['cache']['redis_host'],
            port=config_manager.data_collection_config['cache']['redis_port']
        )
        
        performance_monitor = PerformanceMonitor(
            config_manager.monitoring_config['metrics']
        )
        
        client = AsyncOKXClient(
            symbols=config_manager.data_collection_config['symbols'],
            channels=config_manager.data_collection_config['channels'],
            cache_manager=cache_manager,
            performance_monitor=performance_monitor
        )
        
        # Setup subscription rules
        await setup_subscription_rules(cache_manager)
        
        # Start monitoring tasks
        monitoring_task = asyncio.create_task(
            monitor_metrics(
                LatencyCollector(),
                BandwidthCollector(),
                TaskMetricsCollector(),
                QueueMonitor()
            )
        )
        
        # Start data collection
        collection_task = asyncio.create_task(client.start())
        
        # Initialize report generators
        metrics_report_generator = MetricsReportGenerator()
        stress_test_report_generator = StressTestReportGenerator()
        
        # Schedule periodic report generation
        report_task = asyncio.create_task(
            periodic_report_generation(metrics_report_generator)
        )
        
        # Initialize stress testing components
        data_generator_config = DataGeneratorConfig.from_yaml('config/generator_config.yaml')
        data_generator = DataGenerator(data_generator_config)
        
        client_manager = ClientManager(ws_url="ws://localhost:8765")  # WebSocket服务地址
        
        # Start stress testing tasks
        stress_test_task = asyncio.create_task(
            run_stress_test(
                data_generator,
                client_manager,
                performance_metrics,
                resource_monitor,
                stress_test_report_generator
            )
        )
        
        # Run comprehensive system tests
        await run_comprehensive_tests(
            client,
            FilterManager(),
            BatchProcessor(),
            performance_monitor,
            resource_monitor
        )
        
        # Initialize technical indicators and prediction model
        indicators = await initialize_indicators('config/indicators_config.json')
        model_trainer = await initialize_model('config/model_config.json')
        
        # Initialize feature extractor
        feature_extractor = await initialize_feature_extractor('config/feature_config.json')
        
        # Start signal generation task
        signal_generation_task = asyncio.create_task(
            run_signal_generation(
                client,
                indicators,
                feature_extractor,
                model_trainer,
                cache_manager,
                performance_monitor
            )
        )
        
        # Run until interrupted
        try:
            await asyncio.gather(
                collection_task,
                monitoring_task,
                report_task,
                stress_test_task,
                signal_generation_task
            )
        except KeyboardInterrupt:
            logging.info("Shutting down gracefully...")
        finally:
            await cleanup(
                client,
                cache_manager,
                performance_monitor,
                [LatencyCollector(), BandwidthCollector(), TaskMetricsCollector()],
                metrics_report_generator,
                data_generator,
                client_manager,
                performance_metrics,
                resource_monitor,
                stress_test_report_generator,
                indicators,
                feature_extractor,
                model_trainer
            )
            
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        logging.exception("Detailed error information:")
        sys.exit(1)

async def initialize_cache_manager() -> RedisCacheManager:
    """Initialize Redis cache manager with retry logic"""
    try:
        cache_manager = RedisCacheManager(
            host='localhost',
            port=6379,
            retry_attempts=3,
            retry_delay=1,
            ttl=3600
        )
        logging.info("Successfully initialized Redis Cache")
        return cache_manager
    except Exception as e:
        logging.error(f"Failed to initialize Redis cache: {e}")
        raise

def initialize_performance_monitor(
    latency_collector,
    bandwidth_collector,
    task_collector,
    queue_monitor
) -> PerformanceMonitor:
    """Initialize performance monitoring"""
    thresholds = PerformanceThresholds(
        max_latency_ms=100.0,
        max_cpu_usage=80.0,
        max_memory_usage=80.0,
        max_queue_size=1000,
        max_error_rate=0.01
    )
    
    return PerformanceMonitor(
        metrics_window=1000,
        thresholds=thresholds,
        alert_callback=handle_performance_alert,
        latency_collector=latency_collector,
        bandwidth_collector=bandwidth_collector,
        task_collector=task_collector,
        queue_monitor=queue_monitor
    )

async def initialize_okx_client(
    cache_manager: RedisCacheManager,
    performance_monitor: PerformanceMonitor,
    batch_processor: Optional[BatchProcessor] = None
) -> AsyncOKXClient:
    """Initialize OKX client with monitoring"""
    try:
        symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
        channels = ['tickers', 'trades', 'books']
        
        client = AsyncOKXClient(
            symbols=symbols,
            channels=channels,
            cache_manager=cache_manager,
            performance_monitor=performance_monitor,
            batch_processor=batch_processor
        )
        
        # Verify client initialization
        if not await client.verify_connection():
            raise ConnectionError("Failed to verify OKX client connection")
            
        logging.info("OKX client initialized successfully")
        return client
        
    except Exception as e:
        logging.error(f"Failed to initialize OKX client: {e}")
        raise

async def setup_subscription_rules(cache_manager: RedisCacheManager):
    """Setup market data subscription rules"""
    # Example price change rule
    btc_rule = SubscriptionRule(
        symbol='BTC-USDT',
        conditions={
            'price': {
                'operator': 'change_pct',
                'value': 1.0  # 1% price change
            }
        },
        callback=handle_significant_price_change
    )
    
    # Example volume spike rule
    eth_rule = SubscriptionRule(
        symbol='ETH-USDT',
        conditions={
            'volume': {
                'operator': '>',
                'value': 1000.0
            }
        },
        callback=handle_volume_spike
    )
    
    # Register rules
    cache_manager.subscribe_with_rule(btc_rule)
    cache_manager.subscribe_with_rule(eth_rule)
    
    # Restore any previously saved rules
    cache_manager.restore_subscriptions(create_callback_for_symbol)

async def monitor_metrics(
    latency_collector,
    bandwidth_collector,
    task_collector,
    queue_monitor
):
    """Monitor and report metrics"""
    while True:
        try:
            # Collect metrics
            latency_metrics = latency_collector.get_metrics()
            bandwidth_metrics = bandwidth_collector.get_metrics()
            queue_metrics = await queue_monitor.monitor_queue("market_data")
            
            # Enhanced monitoring with new implementations
            cpu_usage = performance_monitor.get_metric_average("cpu_usage", window=5)
            memory_usage = performance_monitor.get_metric_average("memory_usage", window=5)
            error_rate = performance_monitor.get_error_rate()
            
            logging.info(
                f"System Metrics - CPU: {cpu_usage:.1f}%, "
                f"Memory: {memory_usage:.1f}%, "
                f"Error Rate: {error_rate:.2f}"
            )
            
            if latency_metrics:
                logging.info(
                    f"Latency - Avg: {latency_metrics.avg_ms:.2f}ms, "
                    f"Max: {latency_metrics.max_ms:.2f}ms"
                )
            
            if bandwidth_metrics:
                logging.info(
                    f"Bandwidth - In: {bandwidth_metrics.inbound_kbps:.2f}KB/s, "
                    f"Out: {bandwidth_metrics.outbound_kbps:.2f}KB/s"
                )
            
            if queue_metrics:
                logging.info(
                    f"Queue Length: {queue_metrics.length}, "
                    f"Processing Rate: {queue_metrics.processing_rate:.2f}/s"
                )
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logging.error(f"Error in metrics monitoring: {e}")
            await asyncio.sleep(1)

async def cleanup(
    client: AsyncOKXClient,
    cache_manager: RedisCacheManager,
    performance_monitor: PerformanceMonitor,
    collectors: list,
    metrics_report_generator: MetricsReportGenerator,
    data_generator: DataGenerator,
    client_manager: ClientManager,
    performance_metrics: PerformanceMetricsCollector,
    resource_monitor: ResourceMonitor,
    stress_test_report_generator: StressTestReportGenerator,
    batch_processor: Optional[BatchProcessor] = None,
    indicators: Optional[Dict] = None,
    feature_extractor: Optional[FeatureExtractor] = None,
    model_trainer: Optional[ModelTrainer] = None
):
    """Cleanup all resources"""
    try:
        await client.stop()
        if batch_processor:
            await batch_processor.stop()
        cache_manager.cleanup()
        performance_monitor.cleanup()
        for collector in collectors:
            collector.cleanup()
            
        # Generate performance report
        await generate_performance_report(
            datetime.now() - timedelta(hours=1),
            datetime.now(),
            metrics_report_generator
        )
        
        # Stop stress testing components
        await data_generator.stop()
        await client_manager.stop_all_clients()
        await resource_monitor.stop()
        
        # Generate final stress test report
        await generate_final_stress_test_report(
            client_manager,
            performance_metrics,
            resource_monitor,
            stress_test_report_generator
        )
        
        # Cleanup signal generation components
        if indicators:
            for indicator in indicators.values():
                indicator.cleanup()
        if feature_extractor:
            feature_extractor.cleanup()
        if model_trainer:
            model_trainer.cleanup()
        
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
        raise
    finally:
        logging.info("Cleanup completed")

def handle_performance_alert(alerts: list):
    """Handle performance alerts"""
    for alert in alerts:
        logging.warning(f"Performance Alert: {alert}")

async def handle_significant_price_change(symbol: str, data: Dict):
    """Handle significant price changes"""
    logging.info(
        f"Significant price change for {symbol}: "
        f"Price={data['price']}, Change={data.get('change_percent', 0):.2f}%"
    )

async def handle_volume_spike(symbol: str, data: Dict):
    """Handle trading volume spikes"""
    logging.info(
        f"Volume spike detected for {symbol}: "
        f"Volume={data['volume']}"
    )

def create_callback_for_symbol(symbol: str):
    """Create callback function for restored subscriptions"""
    async def callback(symbol: str, data: Dict):
        logging.info(f"Restored subscription update for {symbol}: {data}")
    return callback

async def generate_performance_report(
    start_time: datetime,
    end_time: datetime,
    report_generator: MetricsReportGenerator
):
    """Generate performance report"""
    try:
        report_path = await report_generator.generate_report(
            start_time,
            end_time
        )
        
        if report_path:
            logging.info(f"Performance report generated: {report_path}")
        else:
            logging.error("Failed to generate performance report")
            
    except Exception as e:
        logging.error(f"Error generating report: {e}")

async def periodic_report_generation(report_generator: MetricsReportGenerator):
    """Generate reports periodically"""
    missed_reports = []
    
    while True:
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            
            # First try to generate any missed reports
            while missed_reports:
                missed_report = missed_reports[0]
                try:
                    await generate_performance_report(
                        missed_report['start'],
                        missed_report['end'],
                        report_generator
                    )
                    missed_reports.pop(0)
                except Exception:
                    break  # Stop trying if still failing
                    
            # Generate current report
            await generate_performance_report(
                start_time,
                end_time,
                report_generator
            )
            
            await asyncio.sleep(300)  # Generate report every 5 minutes
            
        except Exception as e:
            logging.error(f"Error in periodic report generation: {e}")
            # Store failed report details for retry
            missed_reports.append({
                'start': start_time,
                'end': end_time,
                'timestamp': time.time()
            })
            # Cleanup old missed reports (older than 24h)
            current_time = time.time()
            missed_reports = [
                r for r in missed_reports
                if current_time - r['timestamp'] < 86400
            ]
            await asyncio.sleep(60)

async def run_stress_test(
    data_generator: DataGenerator,
    client_manager: ClientManager,
    performance_metrics: PerformanceMetricsCollector,
    resource_monitor: ResourceMonitor,
    report_generator: StressTestReportGenerator
):
    """Run stress test with monitoring"""
    try:
        start_time = datetime.now()
        last_report_time = time.time()  # Initialize last_report_time
        
        # Start data generation
        generator_task = asyncio.create_task(data_generator.start())
        
        try:
            # Start client simulation
            await client_manager.start_clients('config/clients_config.yaml')
            
            while True:
                # Record metrics
                metrics_data = {
                    'total_clients': len(client_manager.clients),
                    'client_stats': client_manager.get_all_statistics(),
                    'latency_stats': performance_metrics.get_latency_stats().__dict__,
                    'throughput_stats': performance_metrics.get_throughput_stats().__dict__,
                    'error_stats': performance_metrics.get_error_stats().__dict__
                }
                
                resource_data = [resource_monitor.get_stats().__dict__]
                
                # Generate periodic report
                current_time = time.time()
                if current_time - last_report_time > 300:  # Every 5 minutes
                    await report_generator.generate_stress_test_report(
                        metrics_data,
                        resource_data,
                        start_time,
                        datetime.now()
                    )
                    last_report_time = current_time
                
                await asyncio.sleep(1)  # Metrics collection interval
                
        finally:
            # Ensure generator task is properly cleaned up
            if not generator_task.done():
                generator_task.cancel()
                try:
                    await generator_task
                except asyncio.CancelledError:
                    pass
                
    except Exception as e:
        logging.error(f"Error in stress test: {e}")
        raise

async def initialize_filter_manager(
    performance_monitor: PerformanceMonitor
) -> FilterManager:
    """Initialize and configure filter manager"""
    filter_manager = FilterManager(performance_monitor)
    
    # Configure default filters
    price_range_config = FilterConfig(
        filter_type=FilterType.PRICE_RANGE,
        parameters={'min_price': 45000, 'max_price': 55000}
    )
    
    price_change_config = FilterConfig(
        filter_type=FilterType.PRICE_CHANGE,
        parameters={'threshold': 0.02}
    )
    
    multi_asset_config = FilterConfig(
        filter_type=FilterType.MULTI_ASSET,
        parameters={'correlation_threshold': 0.7},
        window_size=10,
        assets=['BTC-USDT', 'ETH-USDT']
    )
    
    # Add filters for each symbol
    for symbol in ['BTC-USDT', 'ETH-USDT']:
        await filter_manager.add_filter(symbol, price_range_config)
        await filter_manager.add_filter(symbol, price_change_config)
        
    await filter_manager.add_filter('ETH-USDT', multi_asset_config)
    
    return filter_manager

async def run_comprehensive_tests(
    client,
    filter_manager,
    batch_processor,
    performance_monitor,
    resource_monitor
):
    """Run comprehensive system tests"""
    try:
        # Initialize load test
        load_config = LoadTestConfig(
            duration_seconds=3600,  # 1 hour
            max_symbols=20,
            max_subscriptions=2000,
            batch_sizes=[100, 500, 1000, 5000],
            message_rates=[100, 500, 1000, 5000]
        )
        
        load_test = DataCollectionLoadTest(
            load_config,
            client,
            filter_manager,
            batch_processor,
            performance_monitor
        )
        
        # Initialize stability test
        stability_config = StabilityTestConfig(
            duration_hours=24,
            check_interval=300
        )
        
        stability_test = LongRunningStabilityTest(
            stability_config,
            client,
            filter_manager,
            batch_processor,
            performance_monitor,
            resource_monitor
        )
        
        # Run tests
        logging.info("Starting comprehensive system tests")
        
        # Run load test
        await load_test.run_load_test()
        
        # Run stability test if load test passes
        await stability_test.run_stability_test()
        
        logging.info("Comprehensive system tests completed")
        
    except Exception as e:
        logging.error(f"Error in comprehensive testing: {e}")
        raise

async def initialize_indicators(config_path: str):
    """Initialize technical indicators"""
    try:
        # Load indicator configurations
        with open(config_path, 'r') as f:
            configs = json.load(f)
            
        # Initialize indicators
        rsi_config = RSIConfig(**configs.get('rsi', {}))
        macd_config = MACDConfig(**configs.get('macd', {}))
        bb_config = BollingerBandsConfig(**configs.get('bollinger_bands', {}))
        
        indicators = {
            'rsi': RSIIndicator(rsi_config),
            'macd': MACDIndicator(macd_config),
            'bollinger_bands': BollingerBandsIndicator(bb_config)
        }
        
        logging.info("Technical indicators initialized successfully")
        return indicators
        
    except Exception as e:
        logging.error(f"Error initializing indicators: {e}")
        raise

async def initialize_model(config_path: str):
    """Initialize prediction model"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        model_config = ModelConfig(**config_dict)
        model_trainer = ModelTrainer(model_config)
        
        # Initialize distributed training if GPU available
        if torch.cuda.is_available():
            dist_config = DistributedConfig(
                num_gpus=torch.cuda.device_count()
            )
            model_trainer = DistributedModelTrainer(
                model_trainer,
                dist_config
            )
            
        logging.info("Model trainer initialized with distributed support")
        return model_trainer
        
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        raise

async def initialize_feature_extractor(config_path: str):
    """Initialize feature extractor"""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            
        config = FeatureConfig(**config_dict)
        extractor = FeatureExtractor(config)
        
        logging.info("Feature extractor initialized successfully")
        return extractor
        
    except Exception as e:
        logging.error(f"Error initializing feature extractor: {e}")
        raise

async def run_signal_generation(
    client: AsyncOKXClient,
    indicators: Dict,
    feature_extractor: FeatureExtractor,
    model_trainer: ModelTrainer,
    cache_manager: RedisCacheManager,
    performance_monitor: ModelPerformanceMonitor
):
    """Run signal generation loop"""
    try:
        while True:
            start_time = time.time()
            
            # Get latest market data
            market_data = await client.get_latest_data()
            
            # Calculate technical indicators
            indicator_values = {}
            for name, indicator in indicators.items():
                value = indicator.update(market_data['close'])
                if value is not None:
                    indicator_values[name] = value
                    
            # Extract features
            if indicator_values:
                features = feature_extractor.update(indicator_values)
                
                if features is not None:
                    # Generate predictions
                    prediction_results = await process_features(
                        features,
                        model_trainer
                    )
                    
                    # Cache prediction results
                    await cache_manager.set_async(
                        f"prediction:{market_data['symbol']}",
                        prediction_results,
                        expire=300  # 5 minutes
                    )
                    
                    # Log significant predictions
                    if prediction_results['confidence'] > 0.8:
                        logging.info(
                            f"High confidence prediction for "
                            f"{market_data['symbol']}: {prediction_results}"
                        )
                        
            # 添加性能监控
            latency = time.time() - start_time
            await performance_monitor.update_metrics(
                prediction_results,
                actual=None,  # 实时预测时暂无实际值
                latency=latency
            )
            
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logging.info("Signal generation task cancelled")
    except Exception as e:
        logging.error(f"Error in signal generation: {e}")
        raise

async def process_market_data(
    data: Dict,
    indicators: Dict,
    symbol: str
):
    """Process market data with technical indicators"""
    try:
        results = {
            'timestamp': data.get('timestamp'),
            'symbol': symbol,
            'price': data.get('close'),
            'indicators': {}
        }
        
        # Update RSI
        if 'rsi' in indicators:
            rsi = indicators['rsi'].update(data['close'])
            if rsi is not None:
                results['indicators']['rsi'] = rsi
                
        # Update MACD
        if 'macd' in indicators:
            macd = indicators['macd'].update(data['close'])
            if macd is not None:
                macd_line, signal_line, histogram = macd
                results['indicators']['macd'] = {
                    'macd_line': macd_line,
                    'signal_line': signal_line,
                    'histogram': histogram
                }
                
        # Update Bollinger Bands
        if 'bollinger_bands' in indicators:
            bb = indicators['bollinger_bands'].update(data['close'])
            if bb is not None:
                upper, middle, lower = bb
                results['indicators']['bollinger_bands'] = {
                    'upper': upper,
                    'middle': middle,
                    'lower': lower
                }
                
        return results
        
    except Exception as e:
        logging.error(f"Error processing market data: {e}")
        raise

async def funds_management_workflow(market_data: Dict[str, Any], account_data: Dict[str, Any], signals: Dict[str, float]) -> Dict[str, Any]:
    try:
        # 使用 FundsManager 替代单独的组件初始化
        funds_manager = FundsManager()
        
        # 使用优化后的 manage_funds 方法
        result = await funds_manager.manage_funds(
            market_data=market_data,
            account_data=account_data,
            signals=signals
        )
        
        return {
            'allocations': result.allocations,
            'leverage_recommendations': result.leverage_recommendations,
            'risk_status': result.risk_status,
            'margin_status': result.margin_status,
            'margin_alerts': result.margin_alerts
        }
        
    except Exception as e:
        logging.error(f"Error in funds management workflow: {e}")
        raise

async def trading_workflow(market_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Initialize signal generator
        signal_generator = SignalGenerator(SignalConfig())
        
        # Generate signals with improved performance
        signals = {}
        batch_size = 32
        symbols = list(market_data.keys())
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            batch_data = {
                symbol: market_data[symbol]
                for symbol in batch_symbols
            }
            batch_signals = signal_generator.generate_signals(batch_data)
            signals.update(batch_signals)
        
        # Process signals through funds management
        funds_management_results = await funds_management_workflow(
            market_data=market_data,
            account_data=get_account_data(),
            signals=signals
        )
        
        return {
            'signals': signals,
            'funds_management': funds_management_results
        }
        
    except Exception as e:
        logging.error(f"Error in trading workflow: {e}")
        raise

async def execution_workflow(
    signals: Dict[str, float],
    market_data: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute trades based on signals"""
    try:
        # Initialize components
        api_connector = APIConnector(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            sandbox=config.get('sandbox', True)
        )
        
        order_executor = OrderExecutor(api_connector)
        
        # Process signals and create order requests
        order_requests = []
        for symbol, signal in signals.items():
            if abs(signal) > config['signal_threshold']:
                order_requests.append(
                    OrderRequest(
                        symbol=symbol,
                        order_type=OrderType.LIMIT,
                        side=OrderSide.BUY if signal > 0 else OrderSide.SELL,
                        amount=Decimal(str(market_data[symbol]['position_size'])),
                        price=Decimal(str(market_data[symbol]['current_price']))
                    )
        
        # Execute orders
        execution_results = await order_executor.execute_batch_orders(order_requests)
        
        return {
            'execution_results': execution_results,
            'order_count': len(execution_results)
        }
        
    except Exception as e:
        logging.error(f"Error in execution workflow: {e}", exc_info=True)
        raise

# Update main workflow
async def main_workflow():
    try:
        # Initialize configuration
        config = load_config()
        
        # Generate signals
        market_data = await fetch_market_data()
        signals = await trading_workflow(market_data)
        
        # Execute trades
        execution_results = await execution_workflow(
            signals=signals['signals'],
            market_data=market_data,
            config=config
        )
        
        logging.info(f"Execution completed: {execution_results}")
        
    except Exception as e:
        logging.error(f"Error in main workflow: {e}", exc_info=True)
        raise

async def initialize_trading_system() -> bool:
    """Initialize and verify trading system components"""
    try:
        # Get existing config
        config_manager = ConfigManager()
        
        # Initialize API connector with sandbox mode
        api_connector = APIConnector(
            api_key=config_manager.okx_config['api_key'],
            api_secret=config_manager.okx_config['secret_key'],
            sandbox=config_manager.okx_config['trading_mode'] == 'demo'
        )
        
        # Verify sandbox connection
        if not await api_connector.verify_sandbox_connection():
            logging.error("Failed to verify sandbox connection")
            return False
            
        # Initialize health check
        health_check = HealthCheck(api_connector)
        
        # Perform initial health check
        health_status = await health_check.check_trading_system_health()
        if not all(health_status.values()):
            logging.error(f"System health check failed: {health_status}")
            return False
            
        logging.info("Trading system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to initialize trading system: {e}")
        return False

async def periodic_health_check(health_check: HealthCheck):
    """Perform periodic health checks"""
    while True:
        await health_check.check_system_health()
        await asyncio.sleep(60)  # Check every minute

if __name__ == "__main__":
    try:
        # Initialize trading system
        components = asyncio.run(initialize_trading_system())
        
        # Start main trading loop
        asyncio.run(main())
        
    except Exception as e:
        logging.error(f"Trading system failed: {e}")
        raise