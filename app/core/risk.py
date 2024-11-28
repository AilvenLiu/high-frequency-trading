from dataclasses import dataclass
from typing import Dict, Optional, List
from decimal import Decimal
import logging
from enum import Enum
from functools import lru_cache

@dataclass
class RiskThresholds:
    max_account_drawdown: float = 0.2  # 最大账户回撤
    max_position_size: float = 0.3     # 单个仓位最大占比
    max_total_leverage: float = 10.0   # 最大总杠杆
    margin_warning_ratio: float = 0.5  # 保证金警告比例
    margin_danger_ratio: float = 0.8   # 保证金危险比例

class RiskLevel(Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"
    CRITICAL = "CRITICAL"

@dataclass
class RiskStatus:
    level: RiskLevel
    warnings: List[str]
    recommended_actions: List[str]

class RiskController:
    def __init__(self, thresholds: Optional[RiskThresholds] = None):
        self.thresholds = thresholds or RiskThresholds()
        self.logger = logging.getLogger(__name__)
        
    def calculate_account_risk(self,
                             current_equity: Decimal,
                             initial_equity: Decimal,
                             total_positions_value: Decimal,
                             used_margin: Decimal,
                             available_margin: Decimal) -> RiskStatus:
        """Calculate overall account risk level"""
        warnings = []
        actions = []
        risk_level = RiskLevel.SAFE
        
        # Check drawdown
        drawdown = 1 - (float(current_equity) / float(initial_equity))
        if drawdown > self.thresholds.max_account_drawdown:
            warnings.append(f"Account drawdown ({drawdown:.2%}) exceeds threshold")
            actions.append("Consider reducing position sizes")
            risk_level = RiskLevel.WARNING
            
        # Check leverage
        if total_positions_value > Decimal('0'):
            current_leverage = total_positions_value / current_equity
            if float(current_leverage) > self.thresholds.max_total_leverage:
                warnings.append(f"Total leverage ({float(current_leverage):.2f}x) too high")
                actions.append("Reduce leverage on existing positions")
                risk_level = max(risk_level, RiskLevel.DANGER)
                
        # Check margin usage
        if used_margin > Decimal('0'):
            margin_usage_ratio = used_margin / (used_margin + available_margin)
            if float(margin_usage_ratio) > self.thresholds.margin_danger_ratio:
                warnings.append(f"Margin usage ({float(margin_usage_ratio):.2%}) critical")
                actions.append("Immediate position reduction required")
                risk_level = RiskLevel.CRITICAL
            elif float(margin_usage_ratio) > self.thresholds.margin_warning_ratio:
                warnings.append(f"High margin usage ({float(margin_usage_ratio):.2%})")
                actions.append("Monitor margin usage closely")
                risk_level = max(risk_level, RiskLevel.WARNING)
                
        return RiskStatus(level=risk_level, warnings=warnings, recommended_actions=actions)
        
    def validate_new_position(self,
                            account_equity: Decimal,
                            position_size: Decimal,
                            leverage: int,
                            existing_positions: Dict[str, Decimal]) -> bool:
        """Validate if new position meets risk requirements"""
        try:
            # Check position size limit
            total_existing = sum(existing_positions.values())
            new_total = total_existing + position_size
            if float(new_total / account_equity) > self.thresholds.max_position_size:
                self.logger.warning("New position would exceed maximum position size")
                return False
                
            # Check leverage limit
            if leverage > self.thresholds.max_total_leverage:
                self.logger.warning("Requested leverage exceeds maximum allowed")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating new position: {e}")
            return False
            
    def get_risk_adjusted_size(self,
                              original_size: Decimal,
                              risk_level: RiskLevel,
                              volatility: float) -> Decimal:
        """Adjust position size based on risk level and market conditions"""
        try:
            adjustment_factors = {
                RiskLevel.SAFE: 1.0,
                RiskLevel.WARNING: 0.7,
                RiskLevel.DANGER: 0.4,
                RiskLevel.CRITICAL: 0.0
            }
            
            # Adjust for risk level
            base_adjustment = adjustment_factors[risk_level]
            
            # Further adjust for volatility
            volatility_factor = max(0.2, 1 - volatility)
            
            final_adjustment = base_adjustment * volatility_factor
            return original_size * Decimal(str(final_adjustment))
            
        except Exception as e:
            self.logger.error(f"Error adjusting position size: {e}")
            return Decimal('0')
        
    @lru_cache(maxsize=1000)
    def assess_market_state(self, market_data: Dict[str, Dict]) -> Dict[str, str]:
        """Assess market state with caching"""
        try:
            market_state = {}
            for symbol, data in market_data.items():
                volatility = data.get('volatility', 0)
                trend_strength = data.get('trend_strength', 0)
                
                # Determine risk level based on volatility and trend strength
                if volatility > 0.7 and trend_strength < 0.3:
                    risk_level = 'HIGH'
                elif volatility > 0.5:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
                
                market_state[symbol] = risk_level
                self.logger.info(f"Market state for {symbol}: {risk_level}")
            
            return market_state
        except Exception as e:
            self.logger.error(f"Error assessing market state: {e}")
            raise 