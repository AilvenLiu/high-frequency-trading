import pytest
from decimal import Decimal
from typing import Dict
from funds_management.allocation.dynamic_allocator import DynamicAllocator, AccountStatus, Position
from funds_management.allocation.leverage_manager import LeverageManager, MarketCondition

@pytest.fixture
def account_status():
    return AccountStatus(
        net_worth=Decimal('10000'),
        available_margin=Decimal('8000'),
        positions={
            'BTC-USDT': Position(
                symbol='BTC-USDT',
                size=Decimal('0.1'),
                entry_price=Decimal('50000'),
                leverage=5,
                unrealized_pnl=Decimal('100'),
                margin_used=Decimal('1000')
            )
        },
        risk_level=0.3
    )

@pytest.fixture
def market_condition():
    return MarketCondition(
        volatility=0.4,
        liquidity=0.8,
        trend_strength=0.6
    )

class TestDynamicAllocator:
    def test_calculate_position_size(self, account_status):
        allocator = DynamicAllocator()
        position_size = allocator.calculate_position_size(
            account_status=account_status,
            signal_strength=0.8,
            win_rate=0.6,
            profit_ratio=1.5,
            loss_ratio=1.0
        )
        
        assert isinstance(position_size, Decimal)
        assert position_size > 0
        assert position_size <= account_status.net_worth * Decimal(str(allocator.max_position_size))

    def test_allocate_funds(self, account_status):
        allocator = DynamicAllocator()
        signals = {
            'BTC-USDT': 0.8,
            'ETH-USDT': 0.6
        }
        strategy_metrics = {
            'BTC-USDT': {'win_rate': 0.6, 'profit_ratio': 1.5, 'loss_ratio': 1.0},
            'ETH-USDT': {'win_rate': 0.55, 'profit_ratio': 1.4, 'loss_ratio': 1.0}
        }
        
        allocations = allocator.allocate_funds(account_status, signals, strategy_metrics)
        
        assert isinstance(allocations, dict)
        assert all(isinstance(v, Decimal) for v in allocations.values())
        assert sum(allocations.values()) <= account_status.net_worth

    def test_rebalance_positions(self, account_status):
        allocator = DynamicAllocator()
        target_allocations = {
            'BTC-USDT': Decimal('0.15'),
            'ETH-USDT': Decimal('0.5')
        }
        
        adjustments = allocator.rebalance_positions(account_status, target_allocations)
        
        assert isinstance(adjustments, dict)
        assert all(isinstance(v, Decimal) for v in adjustments.values())
        assert 'BTC-USDT' in adjustments

    def test_edge_cases(self, account_status):
        allocator = DynamicAllocator()
        
        # Test zero signal strength
        position_size = allocator.calculate_position_size(
            account_status=account_status,
            signal_strength=0,
            win_rate=0.6,
            profit_ratio=1.5,
            loss_ratio=1.0
        )
        assert position_size == Decimal('0')
        
        # Test maximum signal strength
        position_size = allocator.calculate_position_size(
            account_status=account_status,
            signal_strength=1.0,
            win_rate=0.6,
            profit_ratio=1.5,
            loss_ratio=1.0
        )
        assert position_size <= account_status.net_worth * Decimal(str(allocator.max_position_size))
        
        # Test insufficient margin
        low_margin_status = AccountStatus(
            net_worth=account_status.net_worth,
            available_margin=Decimal('100'),
            positions=account_status.positions,
            risk_level=0.8
        )
        position_size = allocator.calculate_position_size(
            account_status=low_margin_status,
            signal_strength=0.8,
            win_rate=0.6,
            profit_ratio=1.5,
            loss_ratio=1.0
        )
        assert position_size <= low_margin_status.available_margin * (1 - allocator.min_free_margin)

class TestLeverageManager:
    def test_calculate_safe_leverage(self, market_condition):
        manager = LeverageManager()
        leverage = manager.calculate_safe_leverage(
            market_condition=market_condition,
            signal_strength=0.8,
            account_equity=Decimal('10000')
        )
        
        assert isinstance(leverage, int)
        assert manager.min_leverage <= leverage <= manager.max_leverage

    def test_adjust_leverage(self, market_condition):
        manager = LeverageManager()
        adjusted = manager.adjust_leverage(
            current_leverage=5,
            target_leverage=8,
            position_size=Decimal('1000'),
            market_condition=market_condition
        )
        
        assert isinstance(adjusted, int) or adjusted is None
        if adjusted is not None:
            assert manager.min_leverage <= adjusted <= manager.max_leverage

    def test_get_leverage_recommendations(self, market_condition):
        manager = LeverageManager()
        positions = {
            'BTC-USDT': {'leverage': 5, 'size': Decimal('0.1')},
            'ETH-USDT': {'leverage': 3, 'size': Decimal('1.0')}
        }
        market_conditions = {
            'BTC-USDT': market_condition,
            'ETH-USDT': market_condition
        }
        signals = {
            'BTC-USDT': 0.8,
            'ETH-USDT': 0.6
        }
        
        recommendations = manager.get_leverage_recommendations(
            positions=positions,
            market_conditions=market_conditions,
            signals=signals,
            account_equity=Decimal('10000')
        )
        
        assert isinstance(recommendations, dict)
        assert all(isinstance(v, int) for v in recommendations.values())
        assert all(manager.min_leverage <= v <= manager.max_leverage 
                  for v in recommendations.values())

    def test_edge_cases(self, market_condition):
        manager = LeverageManager()
        
        # Test high volatility
        high_vol_condition = MarketCondition(
            volatility=0.9,
            liquidity=market_condition.liquidity,
            trend_strength=market_condition.trend_strength
        )
        leverage = manager.calculate_safe_leverage(
            market_condition=high_vol_condition,
            signal_strength=0.8,
            account_equity=Decimal('10000')
        )
        assert leverage <= manager.base_leverage
        
        # Test low liquidity
        low_liq_condition = MarketCondition(
            volatility=market_condition.volatility,
            liquidity=0.1,
            trend_strength=market_condition.trend_strength
        )
        leverage = manager.calculate_safe_leverage(
            market_condition=low_liq_condition,
            signal_strength=0.8,
            account_equity=Decimal('10000')
        )
        assert leverage <= manager.base_leverage 