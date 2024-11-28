import asyncio
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

@dataclass
class StabilityTestConfig:
    """Stability test configuration"""
    duration_hours: int = 24
    check_interval: int = 300  # 5 minutes
    memory_threshold: float = 0.85  # 85% memory usage alert
    cpu_threshold: float = 0.80  # 80% CPU usage alert
    error_threshold: int = 100  # errors per hour threshold

class LongRunningStabilityTest:
    """Long-running stability test implementation"""
    
    def __init__(
        self,
        config: StabilityTestConfig,
        client,
        filter_manager,
        batch_processor,
        performance_monitor,
        resource_monitor
    ):
        self.config = config
        self.client = client
        self.filter_manager = filter_manager
        self.batch_processor = batch_processor
        self.performance_monitor = performance_monitor
        self.resource_monitor = resource_monitor
        
        self.is_running = False
        self.stability_metrics = []
        self._start_time = None
        self._error_count = 0
        
    async def run_stability_test(self):
        """Run long-running stability test"""
        self.is_running = True
        self._start_time = time.time()
        
        try:
            test_end_time = self._start_time + (self.config.duration_hours * 3600)
            
            # Start continuous monitoring
            monitor_task = asyncio.create_task(
                self._monitor_system_stability()
            )
            
            # Start test workload
            workload_task = asyncio.create_task(
                self._generate_test_workload()
            )
            
            while time.time() < test_end_time and self.is_running:
                # Check system health
                await self._check_system_health()
                
                # Generate stability report
                if len(self.stability_metrics) % 12 == 0:  # Every hour
                    self._generate_stability_report()
                    
                await asyncio.sleep(self.config.check_interval)
                
            # Clean up tasks
            monitor_task.cancel()
            workload_task.cancel()
            try:
                await asyncio.gather(monitor_task, workload_task)
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            logging.error(f"Stability test error: {e}")
            raise
        finally:
            self.is_running = False
            self._generate_final_report()
            
    async def _monitor_system_stability(self):
        """Monitor system stability metrics"""
        while self.is_running:
            try:
                # Collect system metrics
                cpu_usage = self.resource_monitor.get_cpu_usage()
                memory_usage = self.resource_monitor.get_memory_usage()
                
                # Collect performance metrics
                perf_metrics = self.performance_monitor.get_current_metrics()
                
                # Record stability metrics
                self.stability_metrics.append({
                    'timestamp': time.time(),
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'performance_metrics': perf_metrics,
                    'error_count': self._error_count
                })
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logging.error(f"Error in stability monitoring: {e}")
                self._error_count += 1
                
    async def _generate_test_workload(self):
        """Generate continuous test workload"""
        while self.is_running:
            try:
                # Generate mixed workload
                await self._generate_market_data()
                await self._test_subscriptions()
                await self._test_filters()
                await self._test_batch_processing()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"Error in workload generation: {e}")
                self._error_count += 1
                
    async def _check_system_health(self):
        """Check system health against thresholds"""
        if not self.stability_metrics:
            return
            
        latest_metrics = self.stability_metrics[-1]
        
        # Check resource usage
        if latest_metrics['cpu_usage'] > self.config.cpu_threshold:
            logging.warning(f"High CPU usage: {latest_metrics['cpu_usage']:.2f}")
            
        if latest_metrics['memory_usage'] > self.config.memory_threshold:
            logging.warning(f"High memory usage: {latest_metrics['memory_usage']:.2f}")
            
        # Check error rate
        errors_per_hour = self._calculate_error_rate()
        if errors_per_hour > self.config.error_threshold:
            logging.warning(f"High error rate: {errors_per_hour} errors/hour")
            
    async def _generate_market_data(self):
        """Generate test market data"""
        message = {
            'symbol': 'TEST-STABILITY',
            'data': {
                'price': 50000 + np.random.normal(0, 100),
                'volume': np.random.exponential(100)
            },
            'timestamp': time.time()
        }
        await self.client._handle_message(str(message), time.time())
        
    async def _test_subscriptions(self):
        """Test subscription mechanism"""
        symbol = f"TEST-{int(time.time()) % 100}"
        await self.client.subscribe_to_symbol(symbol, self._dummy_callback)
        await asyncio.sleep(0.1)
        await self.client.unsubscribe_from_symbol(symbol)
        
    async def _test_filters(self):
        """Test filter processing"""
        test_data = {
            'last': str(50000 + np.random.normal(0, 100)),
            'timestamp': time.time()
        }
        await self.filter_manager.apply_filters(test_data, 'TEST-STABILITY')
        
    async def _test_batch_processing(self):
        """Test batch processing"""
        batch = [{
            'price': 50000 + np.random.normal(0, 100),
            'volume': np.random.exponential(100),
            'timestamp': time.time()
        } for _ in range(100)]
        await self.batch_processor.add_data('TEST-STABILITY', batch)
        
    async def _dummy_callback(self, symbol: str, data: Dict):
        """Dummy callback for testing"""
        await asyncio.sleep(0.001)
        
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate per hour"""
        if not self.stability_metrics:
            return 0.0
            
        hour_ago = time.time() - 3600
        recent_metrics = [m for m in self.stability_metrics 
                         if m['timestamp'] > hour_ago]
        
        if not recent_metrics:
            return 0.0
            
        return sum(m['error_count'] for m in recent_metrics)
        
    def _generate_stability_report(self):
        """Generate periodic stability report"""
        if not self.stability_metrics:
            return
            
        report = {
            'timestamp': time.time(),
            'uptime_hours': (time.time() - self._start_time) / 3600,
            'error_rate': self._calculate_error_rate(),
            'resource_usage': {
                'cpu_avg': np.mean([m['cpu_usage'] for m in self.stability_metrics[-60:]]),
                'memory_avg': np.mean([m['memory_usage'] for m in self.stability_metrics[-60:]])
            },
            'performance_metrics': self.performance_monitor.get_current_metrics()
        }
        
        logging.info(f"Stability Report: {report}")
        return report
        
    def _generate_final_report(self):
        """Generate final stability test report"""
        report = {
            'test_duration': time.time() - self._start_time,
            'total_errors': self._error_count,
            'avg_error_rate': self._error_count / (self.config.duration_hours or 1),
            'resource_usage': {
                'cpu_max': max(m['cpu_usage'] for m in self.stability_metrics),
                'memory_max': max(m['memory_usage'] for m in self.stability_metrics),
                'cpu_avg': np.mean([m['cpu_usage'] for m in self.stability_metrics]),
                'memory_avg': np.mean([m['memory_usage'] for m in self.stability_metrics])
            },
            'performance_summary': self._generate_performance_summary()
        }
        
        logging.info("Final Stability Test Report Generated")
        return report
        
    def _generate_performance_summary(self) -> Dict:
        """Generate performance metrics summary"""
        perf_metrics = [m['performance_metrics'] for m in self.stability_metrics]
        
        return {
            'latency': {
                'avg': np.mean([m['latency'] for m in perf_metrics if 'latency' in m]),
                'max': max(m['latency'] for m in perf_metrics if 'latency' in m)
            },
            'throughput': {
                'avg': np.mean([m['throughput'] for m in perf_metrics if 'throughput' in m]),
                'min': min(m['throughput'] for m in perf_metrics if 'throughput' in m)
            }
        } 