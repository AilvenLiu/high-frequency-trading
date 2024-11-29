import ccxt
import logging
import time
from typing import List, Dict
import pandas as pd
from config.config_manager import ConfigManager


class DataCollector:
    """Collects market data from OKX API."""

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
        self.symbols = [symbol.replace('-', '/') for symbol in self.config.data_collection_config['symbols']]
        self.timeframes = ['1m', '5m', '15m']

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int = None, limit: int = 100) -> pd.DataFrame:
        """
        Fetches OHLCV data for a given symbol and timeframe.
        :param symbol: Trading pair symbol e.g., 'BTC/USDT'
        :param timeframe: Timeframe e.g., '1m'
        :param since: Timestamp in milliseconds to start fetching from
        :param limit: Number of data points to fetch
        :return: DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if isinstance(ohlcv, list):
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            else:
                logging.error(f"Unexpected response format for {symbol} {timeframe}: {ohlcv}")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def collect_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect OHLCV data for all symbols and timeframes.
        :return: Nested dictionary containing DataFrames for each symbol and timeframe
        """
        data = {}
        for symbol in self.symbols:
            data[symbol] = {}
            for timeframe in self.timeframes:
                logging.info(f"Fetching {timeframe} data for {symbol}")
                df = self.fetch_ohlcv(symbol, timeframe)
                if not df.empty:
                    data[symbol][timeframe] = df
                time.sleep(self.exchange.rateLimit / 1000)  # Respect rate limit
        return data