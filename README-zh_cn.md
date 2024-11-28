# 高频加密货币交易系统

一个面向生产环境的加密货币交易系统，专为 OKX 平台的高频交易设计，具备实时数据采集、AI 驱动的信号生成和自动化交易执行功能。

## 系统概述

本系统专为加密货币市场的高频交易而设计，重点支持 OKX 交易所。它结合了实时市场数据处理、机器学习信号生成和自动化交易执行功能。

### 核心特性

- 基于 WebSocket 的实时市场数据采集
- 使用带注意力机制的 LSTM 进行 AI 驱动的交易信号生成
- 技术指标的并行处理
- 基于 Redis 的市场数据缓存
- 全面的性能监控
- 健壮的错误处理和恢复机制
- 完整的测试框架

## 系统架构

### 1. 数据采集模块

数据采集模块维护与 OKX 实时数据源的 WebSocket 连接，处理：

- 多交易对支持 (BTC-USDT, ETH-USDT, SOL-USDT)
- 多数据通道 (交易、订单簿、行情)
- 自动重连机制（指数退避）
- 消息队列和处理
- 基于 Redis 的数据缓存

核心组件：
- `AsyncOKXClient`: 管理 WebSocket 连接和数据流
- `RedisCacheManager`: 处理市场数据缓存
- `PerformanceMonitor`: 跟踪系统指标

### 2. 信号生成模块

实现机器学习模型和技术分析以生成交易信号：

- 带注意力机制的 LSTM 模型用于价格预测
- 技术指标：
  - RSI (相对强弱指标)
  - MACD (移动平均线趋同/背离指标)
  - 布林带
- 指标计算的并行处理
- 特征提取和归一化
- 信号验证和过滤

### 3. 交易执行模块

处理订单执行和仓位管理：

- 基于信号的订单执行
- 动态仓位管理
- 风险管理集成
- 订单跟踪和管理
- 盈亏监控

### 4. 性能监控

全面的系统监控，包括：

- 系统指标 (CPU、内存、延迟)
- 交易指标 (信号准确度、执行速度)
- 队列监控
- 错误率跟踪
- 资源利用率

## 配置系统

系统使用 YAML 格式的配置文件：

### 环境配置
````yaml
system:
  mode: "production"  # production/development/test
  log_level: "INFO"
  max_memory_usage: 85.0
````

### 交易配置
````yaml
trading:
  symbols: ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
  timeframes: [1s, 3s, 1m]
  channels: [trades, books, tickers]
````

## 安装与设置

### 系统要求
- Python 3.8+
- Redis 服务器
- CUDA 兼容的 GPU (可选，用于分布式训练)

### MacOS 下的 Redis 设置

1. 安装和基本设置：
````bash
brew install redis
brew services start redis
redis-cli ping  # 应返回 "PONG"
````

2. 配置：
- 编辑 `/usr/local/etc/redis.conf`
- 关键设置：
  - `maxmemory 2gb`
  - `maxmemory-policy allkeys-lru`
  - `appendonly yes`

### 项目设置
````bash
git clone <repository>
cd high-frequency-trading
pip install -r requirements.txt
````

## 测试框架

系统包含全面的测试能力：

1. **单元测试**：使用 pytest 进行组件级测试
2. **集成测试**：系统工作流验证
3. **压力测试**：
   - 可配置参数的负载测试
   - 长期稳定性测试
   - 性能指标收集

## 当前限制

1. **性能瓶颈**
   - 高负载下的 WebSocket 连接稳定性
   - 高峰期的 Redis 缓存管理
   - 模型推理延迟

2. **风险管理**
   - 基础的止损实现
   - 有限的仓位管理策略
   - 简化的风险敞口计算

## 路线图

1. **短期目标 (1-2个月)**
   - 完善交易执行模块
   - 增强风险管理系统
   - 实现高级仓位管理

2. **中期目标 (3-6个月)**
   - 添加多交易所支持
   - 实现投资组合优化
   - 增强模型训练流程

3. **长期目标 (6个月以上)**
   - 实现高级风险模型
   - 添加做市商功能
   - 开发图形界面

## 参与贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m '添加 AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 许可证

MIT 许可证 - 详见 LICENSE 文件

## 支持

如需支持和咨询，请在 GitHub 仓库提交 Issue 或直接联系维护者。