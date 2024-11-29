import logging
import time
from config.config_manager import ConfigManager
from data.data_collector import DataCollector
from data.data_persistence import DataPersistence
from strategies.simple_strategy import SimpleStrategy
from execution.trader import Trader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("project.log"),
        logging.StreamHandler()
    ]
)

def main():
    # Load configuration
    config = ConfigManager(config_path='config/config.yaml')
    logging.info("Configuration loaded.")

    # Initialize modules
    collector = DataCollector(config)
    storage = DataPersistence(config)
    strategy = SimpleStrategy(config)
    trader = Trader(config)
    logging.info("Modules initialized.")

    # Main loop
    while True:
        logging.info("Starting data collection cycle.")
        # Collect data
        data = collector.collect_data()
        # Persist data
        storage.save_ohlcv(data)
        # Generate signals for the first symbol and timeframe
        symbol = config.data_collection_config['symbols'][0]
        timeframe = '1m'  # Example timeframe
        # Load latest data from DB
        df = storage.load_latest_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            signals = strategy.generate_signals(df)
            latest_signal = signals.iloc[-1]
            logging.info(f"Latest signal: {latest_signal}")
            # Execute trade based on signal
            if latest_signal != 0:
                balance = get_balance(config)
                amount = config.funds_management_config['allocation']['max_position_size'] * balance
                trader.execute_signal(latest_signal, amount)
        else:
            logging.warning("No data available for strategy.")

        # Wait for the next cycle
        time.sleep(60)  # Wait for 1 minute


def get_balance(config: ConfigManager) -> float:
    """Fetches the available balance."""
    trader = Trader(config)
    balance = trader.get_balance()
    if balance and 'USDT' in balance:
        return balance['USDT']['free']
    return 0.0


if __name__ == "__main__":
    main()