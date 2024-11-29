import pandas as pd
import logging
from typing import Dict
from config.config_manager import ConfigManager


class SimpleStrategy:
    """Generates trading signals based on RSI and MACD indicators."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.indicators_config = self.config.signal_generation_config['indicators']

    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.indicators_config['rsi']['period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.indicators_config['rsi']['period']).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the MACD indicator."""
        fast = df['close'].ewm(span=self.indicators_config['macd']['fast_period'], adjust=False).mean()
        slow = df['close'].ewm(span=self.indicators_config['macd']['slow_period'], adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=self.indicators_config['macd']['signal_period'], adjust=False).mean()
        df['macd'] = macd
        df['signal'] = signal
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates buy/sell signals based on RSI and MACD.
        :param df: DataFrame with OHLCV data
        :return: Series with signals: 1 for buy, -1 for sell, 0 for hold
        """
        df['rsi'] = self.calculate_rsi(df)
        df = self.calculate_macd(df)

        # Initialize signal column
        df['signal'] = 0

        # Buy signal: RSI < oversold and MACD crosses above signal line
        buy_condition = (
            (df['rsi'] < self.indicators_config['rsi']['oversold']) &
            (df['macd'] > df['signal']) &
            (df['macd'].shift(1) <= df['signal'].shift(1))
        )

        # Sell signal: RSI > overbought and MACD crosses below signal line
        sell_condition = (
            (df['rsi'] > self.indicators_config['rsi']['overbought']) &
            (df['macd'] < df['signal']) &
            (df['macd'].shift(1) >= df['signal'].shift(1))
        )

        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1

        logging.info("Signals generated based on RSI and MACD.")
        return df['signal']