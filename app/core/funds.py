from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
from decimal import Decimal
import asyncio
import logging
from dataclasses import dataclass
from funds_management.allocation.dynamic_allocator import DynamicAllocator
from funds_management.allocation.leverage_manager import LeverageManager
from funds_management.risk_control.risk_controller import RiskController
from funds_management.risk_control.margin_calculator import MarginCalculator
from config.config_manager import ConfigManager
from functools import lru_cache
import hashlib
import json

@dataclass
class FundsManagementResult:
    allocations: Dict[str, Decimal]
    leverage_recommendations: Dict[str, int]
    risk_status: Dict[str, Any]
    margin_status: Dict[str, Any]
    margin_alerts: List[str]

class FundsManager:
    def __init__(self):
        self.config_manager = ConfigManager()
        self._setup_components()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
    def _setup_components(self):
        """Initialize all required components"""
        config = self.config_manager.funds_management_config
        
        self.allocator = DynamicAllocator(
            max_position_size=config['allocation']['max_position_size'],
            min_free_margin=config['allocation']['min_free_margin'],
            kelly_fraction=config['allocation']['kelly_fraction']
        )
        
        self.leverage_manager = LeverageManager(
            max_leverage=config['leverage']['max_leverage'],
            min_leverage=config['leverage']['min_leverage'],
            volatility_threshold=config['leverage']['volatility_threshold'],
            base_leverage=config['leverage']['base_leverage']
        )
        
        self.risk_controller = RiskController()
        self.margin_calculator = MarginCalculator(
            initial_margin_ratio=config['margin']['initial_ratio'],
            maintenance_margin_ratio=config['margin']['maintenance_ratio'],
            liquidation_buffer=config['margin']['liquidation_buffer']
        )

    @lru_cache(maxsize=1000)
    def _calculate_risk_hash(self, **kwargs) -> str:
        """Calculate hash for risk calculation inputs"""
        sorted_items = sorted(kwargs.items())
        return hashlib.md5(
            json.dumps(sorted_items).encode()
        ).hexdigest()
        
    async def manage_funds(self, 
                         market_data: Dict[str, Any], 
                         account_data: Dict[str, Any], 
                         signals: Dict[str, float]) -> FundsManagementResult:
        """Optimized concurrent funds management"""
        try:
            # Calculate cache key
            risk_hash = self._calculate_risk_hash(
                net_worth=account_data['net_worth'],
                initial_equity=account_data['initial_equity'],
                total_positions_value=account_data['total_positions_value'],
                used_margin=account_data['used_margin'],
                available_margin=account_data['available_margin']
            )
            
            # Use cached results if available
            if hasattr(self, '_risk_cache') and risk_hash in self._risk_cache:
                risk_status = self._risk_cache[risk_hash]
            else:
                risk_status = await self._calculate_risk_status(account_data)
                self._risk_cache = {risk_hash: risk_status}
            
            # Create tasks for parallel execution
            margin_task = self.executor.submit(
                self.margin_calculator.calculate_account_margin_status,
                equity=Decimal(str(account_data['equity'])),
                positions=account_data['positions'],
                prices={symbol: Decimal(str(data['current_price'])) 
                       for symbol, data in market_data.items()}
            )
            
            # Wait for margin calculation
            margin_status = margin_task.result()
            
            # Process margin alerts
            margin_alerts = self.margin_calculator.check_margin_alerts(
                margin_status.margin_ratio,
                margin_status.used_margin,
                margin_status.available_margin
            )
            
            # Parallel allocation and leverage calculations
            alloc_task = self.executor.submit(
                self.allocator.allocate_funds,
                account_data,
                signals,
                market_data
            )
            
            leverage_task = self.executor.submit(
                self.leverage_manager.get_leverage_recommendations,
                account_data['positions'],
                market_data,
                signals,
                Decimal(str(account_data['net_worth']))
            )
            
            allocations = alloc_task.result()
            leverage_recommendations = leverage_task.result()
            
            return FundsManagementResult(
                allocations=allocations,
                leverage_recommendations=leverage_recommendations,
                risk_status=risk_status,
                margin_status=margin_status,
                margin_alerts=margin_alerts
            )
            
        except Exception as e:
            self.logger.error(f"Error in funds management: {e}", exc_info=True)
            raise 