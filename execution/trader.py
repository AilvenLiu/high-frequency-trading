import ccxt
import logging
from config.config_manager import ConfigManager
from typing import Optional


class Trader:
    """Handles trade execution on OKX."""

    def __init__(self, config: ConfigManager):
        self.config = config
        okx_config = self.config.okx_config
        exchange_params = {
            'apiKey': okx_config.get('api_key'),
            'secret': okx_config.get('secret_key'),
            'password': okx_config.get('passphrase'),
            'timeout': 30000,
            'enableRateLimit': True,
        }

        # 启用沙盒模式
        if okx_config.get('trading_mode') == 'demo':
            exchange_params['sandbox'] = True

        self.exchange = ccxt.okx(exchange_params)

        # 转换符号格式为 CCXT 兼容的格式
        self.symbol = self.config.data_collection_config['symbols'][0].replace('-', '/')

    def place_order(self, side: str, amount: float):
        """
        Places an order on OKX.
        :param side: 'buy' or 'sell'
        :param amount: Amount to trade
        """
        try:
            order = self.exchange.create_market_order(self.symbol, side, amount)
            logging.info(f"Placed {side} order: {order}")
            return order
        except Exception as e:
            logging.error(f"Error placing {side} order: {e}")
            return None

    def get_balance(self) -> Optional[dict]:
        """
        Retrieves account balance.
        :return: Balance dictionary if successful, None otherwise
        """
        try:
            balance = self.exchange.fetch_balance()
            logging.info("Fetched account balance.")
            return balance
        except Exception as e:
            logging.error(f"Error fetching balance: {e}")
            return None

    def execute_signal(self, signal: int, amount: float):
        """
        Executes trade based on the signal.
        :param signal: 1 for buy, -1 for sell, 0 for hold
        :param amount: Amount to trade
        """
        if signal == 1:
            self.place_order('buy', amount)
        elif signal == -1:
            self.place_order('sell', amount)
        else:
            logging.info("No action for hold signal.")