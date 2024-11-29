import sqlite3
import logging
from typing import Dict
import pandas as pd
from config.config_manager import ConfigManager
from pathlib import Path


class DataPersistence:
    """Handles data storage in SQLite database."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.db_path = Path(__file__).parent.parent / 'data' / 'market_data.db'
        self._initialize_db()

    def _sanitize_symbol(self, symbol: str) -> str:
        """Sanitize symbol for use in table names."""
        return symbol.replace('-', '_').replace('/', '_')

    def _initialize_db(self):
        """Initializes the SQLite database and creates necessary tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for symbol in self.config.data_collection_config['symbols']:
                sanitized_symbol = self._sanitize_symbol(symbol)
                for timeframe in ['1m', '5m', '15m']:
                    table_name = f"{sanitized_symbol}_{timeframe}"
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS "{table_name}" (
                            timestamp TEXT PRIMARY KEY,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            volume REAL
                        )
                    """)
            conn.commit()
            conn.close()
            logging.info("Database initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing database: {e}")

    def save_ohlcv(self, data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Saves OHLCV data to the SQLite database.
        :param data: Nested dictionary with symbols, timeframes, and DataFrames
        """
        try:
            conn = sqlite3.connect(self.db_path)
            for symbol, timeframes in data.items():
                for timeframe, df in timeframes.items():
                    table_name = f"{self._sanitize_symbol(symbol)}_{timeframe}"
                    df.to_sql(table_name, conn, if_exists='append', index=False,
                              dtype={
                                  'timestamp': 'TEXT',
                                  'open': 'REAL',
                                  'high': 'REAL',
                                  'low': 'REAL',
                                  'close': 'REAL',
                                  'volume': 'REAL'
                              })
                    logging.info(f"Saved data to {table_name}")
            conn.close()
        except Exception as e:
            logging.error(f"Error saving data to database: {e}")

    def load_latest_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Loads the latest OHLCV data from the database.
        :param symbol: Trading pair symbol e.g., 'BTC/USDT'
        :param timeframe: Timeframe e.g., '1m'
        :return: DataFrame with OHLCV data
        """
        try:
            original_symbol = symbol.replace('/', '-')
            sanitized_symbol = self._sanitize_symbol(original_symbol)
            table_name = f"{sanitized_symbol}_{timeframe}"
            conn = sqlite3.connect(self.db_path)
            query = f'SELECT * FROM "{table_name}" ORDER BY timestamp DESC LIMIT 100'
            df = pd.read_sql_query(query, conn)
            conn.close()
            if not df.empty:
                df = df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological order
                logging.info(f"Loaded latest data from {table_name}")
                return df
            else:
                logging.warning(f"No data found in {table_name}")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading data from database: {e}")
            return pd.DataFrame()