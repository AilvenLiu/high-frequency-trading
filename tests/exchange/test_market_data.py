import unittest
from datetime import datetime
from data_collection.okx_market_data import MarketDataProcessor, Ticker, Trade, OrderBook

class TestMarketDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = MarketDataProcessor(max_cache_size=100)
        
    def test_process_ticker(self):
        ticker_data = {
            'instId': 'BTC-USDT',
            'last': '50000',
            'lastSz': '0.1',
            'askPx': '50001',
            'askSz': '1.5',
            'bidPx': '49999',
            'bidSz': '2.0',
            'open24h': '49000',
            'high24h': '51000',
            'low24h': '48000',
            'volCcy24h': '1000',
            'vol24h': '20',
            'ts': '1650000000000'
        }
        
        ticker = self.processor.process_ticker(ticker_data)
        self.assertEqual(ticker.instId, 'BTC-USDT')
        self.assertEqual(ticker.last, 50000)
        self.assertEqual(len(self.processor.tickers['BTC-USDT']), 1)
        
    def test_process_trade(self):
        trade_data = {
            'instId': 'BTC-USDT',
            'px': '50000',
            'sz': '0.1',
            'side': 'buy',
            'ts': '1650000000000'
        }
        
        trade = self.processor.process_trade(trade_data)
        self.assertEqual(trade.instId, 'BTC-USDT')
        self.assertEqual(trade.price, 50000)
        self.assertEqual(len(self.processor.trades['BTC-USDT']), 1)
        
    def test_process_order_book(self):
        order_book_data = {
            'instId': 'BTC-USDT',
            'asks': [['50000', '1.0', '1']],
            'bids': [['49999', '2.0', '2']],
            'ts': '1650000000000'
        }
        
        order_book = self.processor.process_order_book(order_book_data)
        self.assertEqual(order_book.instId, 'BTC-USDT')
        self.assertEqual(len(order_book.asks), 1)
        self.assertEqual(len(order_book.bids), 1)
        
    def test_vwap_calculation(self):
        trade_data = [
            {'instId': 'BTC-USDT', 'px': '50000', 'sz': '1.0', 'side': 'buy', 'ts': '1650000000000'},
            {'instId': 'BTC-USDT', 'px': '51000', 'sz': '2.0', 'side': 'buy', 'ts': '1650000001000'}
        ]
        
        for data in trade_data:
            self.processor.process_trade(data)
            
        vwap = self.processor.get_vwap('BTC-USDT', window=2)
        self.assertEqual(vwap, 50666.666666666664)

if __name__ == '__main__':
    unittest.main()
