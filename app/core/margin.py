from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal
import logging
from enum import Enum
from functools import lru_cache

class MarginMode(Enum):
    ISOLATED = "isolated"
    CROSS = "cross"

@dataclass
class MarginRequirement:
    initial_margin: Decimal
    maintenance_margin: Decimal
    liquidation_price: Decimal

@dataclass
class MarginStatus:
    total_margin: Decimal
    used_margin: Decimal
    available_margin: Decimal
    margin_ratio: Decimal
    risk_level: str

class MarginCalculator:
    def __init__(self,
                 initial_margin_ratio: float = 0.1,
                 maintenance_margin_ratio: float = 0.05,
                 liquidation_buffer: float = 0.01):
        """
        Initialize margin calculator with risk parameters
        
        Args:
            initial_margin_ratio: Initial margin requirement ratio
            maintenance_margin_ratio: Maintenance margin requirement ratio
            liquidation_buffer: Buffer before liquidation price
        """
        self.initial_margin_ratio = Decimal(str(initial_margin_ratio))
        self.maintenance_margin_ratio = Decimal(str(maintenance_margin_ratio))
        self.liquidation_buffer = Decimal(str(liquidation_buffer))
        self.logger = logging.getLogger(__name__)

    def calculate_position_margin(self,
                                position_size: Decimal,
                                entry_price: Decimal,
                                leverage: int,
                                mode: MarginMode = MarginMode.CROSS) -> MarginRequirement:
        """Calculate margin requirements for a position"""
        try:
            # Validate inputs
            if position_size <= Decimal('0'):
                raise ValueError("Position size cannot be negative or zero")
            if leverage <= 0:
                raise ValueError("Invalid leverage value")
            if entry_price <= Decimal('0'):
                raise ValueError("Entry price cannot be negative or zero")
            
            position_value = position_size * entry_price
            
            if mode == MarginMode.ISOLATED:
                initial_margin = position_value / Decimal(str(leverage))
                maintenance_margin = initial_margin * self.maintenance_margin_ratio
            else:  # CROSS mode
                initial_margin = position_value * self.initial_margin_ratio
                maintenance_margin = position_value * self.maintenance_margin_ratio
            
            # Calculate liquidation price with safety checks
            buffer = position_value * self.liquidation_buffer
            liquidation_denominator = position_value
            if liquidation_denominator == Decimal('0'):
                raise ValueError("Cannot calculate liquidation price with zero position value")
            
            liquidation_price = entry_price * (Decimal('1') - (initial_margin - buffer) / liquidation_denominator)
            
            return MarginRequirement(
                initial_margin=initial_margin,
                maintenance_margin=maintenance_margin,
                liquidation_price=max(Decimal('0'), liquidation_price)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position margin: {e}")
            raise

    def calculate_account_margin_status(self,
                                      equity: Decimal,
                                      positions: Dict[str, Dict],
                                      prices: Dict[str, Decimal]) -> MarginStatus:
        """Calculate overall account margin status"""
        try:
            total_initial_margin = Decimal('0')
            total_maintenance_margin = Decimal('0')
            
            for symbol, position in positions.items():
                if symbol not in prices:
                    continue
                    
                current_price = prices[symbol]
                position_size = Decimal(str(position['size']))
                leverage = Decimal(str(position['leverage']))
                mode = MarginMode(position.get('margin_mode', 'cross'))
                
                margin_req = self.calculate_position_margin(
                    position_size=position_size,
                    entry_price=current_price,
                    leverage=int(leverage),
                    mode=mode
                )
                
                total_initial_margin += margin_req.initial_margin
                total_maintenance_margin += margin_req.maintenance_margin
            
            used_margin = total_initial_margin
            available_margin = max(Decimal('0'), equity - used_margin)
            
            if used_margin > Decimal('0'):
                margin_ratio = equity / used_margin
            else:
                margin_ratio = Decimal('999.99')
            
            # Determine risk level
            if margin_ratio >= Decimal('3'):
                risk_level = "LOW"
            elif margin_ratio >= Decimal('1.5'):
                risk_level = "MEDIUM"
            elif margin_ratio >= Decimal('1.1'):
                risk_level = "HIGH"
            else:
                risk_level = "EXTREME"
            
            return MarginStatus(
                total_margin=equity,
                used_margin=used_margin,
                available_margin=available_margin,
                margin_ratio=margin_ratio,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating account margin status: {e}")
            raise

    def validate_margin_requirements(self,
                                   margin_status: MarginStatus,
                                   min_margin_ratio: float = 1.2) -> bool:
        """Validate if account meets margin requirements"""
        try:
            return float(margin_status.margin_ratio) >= min_margin_ratio
        except Exception as e:
            self.logger.error(f"Error validating margin requirements: {e}")
            return False

    @lru_cache(maxsize=1000)
    def calculate_dynamic_margin(self, position_value: Decimal, leverage: int, market_volatility: float) -> Dict[str, Decimal]:
        """Calculate dynamic margin requirements with caching"""
        try:
            # Adjust initial margin based on market volatility
            volatility_factor = Decimal(str(max(0.5, min(1.5, 1 + market_volatility))))
            dynamic_initial_margin = position_value * self.initial_margin_ratio * volatility_factor
            
            # Calculate maintenance margin
            maintenance_margin = position_value * self.maintenance_margin_ratio
            
            # Calculate liquidation price buffer
            liquidation_price_buffer = position_value * self.liquidation_buffer
            
            return {
                'dynamic_initial_margin': dynamic_initial_margin,
                'maintenance_margin': maintenance_margin,
                'liquidation_price_buffer': liquidation_price_buffer
            }
        except Exception as e:
            self.logger.error(f"Error calculating dynamic margin: {e}")
            raise 