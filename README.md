# High-Frequency Crypto Trading System

A production-ready cryptocurrency trading system designed for high-frequency trading on OKX platform, featuring real-time data collection, AI-driven signal generation, and automated trading execution.

## System Overview

This system is designed to perform high-frequency trading on cryptocurrency markets, with a focus on the OKX exchange. It combines real-time market data processing, machine learning-based signal generation, and automated trade execution.

### Key Features

- Real-time market data collection via WebSocket
- AI-driven trading signal generation using LSTM with attention mechanism
- Parallel processing of technical indicators
- Redis-based market data caching
- Comprehensive performance monitoring
- Robust error handling and recovery mechanisms
- Extensive testing framework

## System Architecture

### 1. Data Collection Module

The data collection module maintains WebSocket connections to OKX's real-time data feeds, handling:

- Multiple trading pairs (BTC-USDT, ETH-USDT, SOL-USDT)
- Multiple data channels (trades, order books, tickers)
- Automatic reconnection with exponential backoff
- Message queuing and processing
- Redis-based data caching

Key components:
- `AsyncOKXClient`: Manages WebSocket connections and data streaming
- `RedisCacheManager`: Handles market data caching
- `PerformanceMonitor`: Tracks system metrics

### 2. Signal Generation Module

Implements machine learning models and technical analysis to generate trading signals:

- LSTM model with attention mechanism for price prediction
- Technical indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
- Parallel processing for indicator calculation
- Feature extraction and normalization
- Signal validation and filtering

### 3. Trading Execution Module

Handles order execution and position management:

- Signal-based order execution
- Dynamic position sizing
- Risk management integration
- Order tracking and management
- P&L monitoring

### 4. Performance Monitoring

Comprehensive system monitoring including:

- System metrics (CPU, memory, latency)
- Trading metrics (signal accuracy, execution speed)
- Queue monitoring
- Error rate tracking
- Resource utilization

## Configuration System

The system uses YAML-based configuration files for different components:

### Environment Configuration
```yaml
system:
  mode: "production"  # production/development/test
  log_level: "INFO"
  max_memory_usage: 85.0
```

### Trading Configuration
```yaml
trading:
  symbols: ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
  timeframes: [1s, 3s, 1m]
  channels: [trades, books, tickers]
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Redis server
- CUDA-capable GPU (optional, for distributed training)

### Redis Setup on MacOS

1. Installation and Basic Setup:
```bash
brew install redis
brew services start redis
redis-cli ping  # Should return "PONG"
```

2. Configuration:
- Edit `/usr/local/etc/redis.conf`
- Key settings:
  - `maxmemory 2gb`
  - `maxmemory-policy allkeys-lru`
  - `appendonly yes`

### Project Setup
```bash
git clone <repository>
cd high-frequency-trading
pip install -r requirements.txt
```

## Testing Framework

The system includes comprehensive testing capabilities:

1. **Unit Tests**: Component-level testing using pytest
2. **Integration Tests**: System workflow validation
3. **Stress Tests**: 
   - Load testing with configurable parameters
   - Long-running stability tests
   - Performance metrics collection

## Current Limitations

1. **Performance Bottlenecks**
   - WebSocket connection stability under high load
   - Redis cache management during peak periods
   - Model inference latency

2. **Risk Management**
   - Basic implementation of stop-loss
   - Limited position sizing strategies
   - Simplified risk exposure calculations

## Roadmap

1. **Short-term (1-2 months)**
   - Complete trading execution module
   - Enhance risk management system
   - Implement advanced position sizing

2. **Medium-term (3-6 months)**
   - Add multi-exchange support
   - Implement portfolio optimization
   - Enhance model training pipeline

3. **Long-term (6+ months)**
   - Implement advanced risk models
   - Add market making capabilities
   - Develop GUI interface

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## License

MIT License - See LICENSE file for details

## Support

For support and questions, please open an issue in the GitHub repository or contact the maintainers directly.