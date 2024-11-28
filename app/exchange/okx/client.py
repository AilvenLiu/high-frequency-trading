from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import logging
from collections import deque
from .redis_cache_manager import RedisCacheManager

@dataclass
class Ticker:
    instId: str
    last: float
    lastSz: float
    askPx: float
    askSz: float
    bidPx: float
    bidSz: float
    open24h: float
    high24h: float
    low24h: float
    volCcy24h: float
    vol24h: float
    timestamp: datetime

@dataclass
class Trade:
    instId: str
    price: float
    size: float
    side: str
    timestamp: datetime

@dataclass
class OrderBookLevel:
    price: float
    size: float
    orders: int
    
@dataclass
class OrderBook:
    instId: str
    asks: List[OrderBookLevel]
    bids: List[OrderBookLevel]
    timestamp: datetime

class MarketDataProcessor:
    def __init__(
        self,
        max_cache_size: int = 1000,
        cache_manager: Optional[RedisCacheManager] = None
    ):
        """
        Initialize the market data processor.
        
        Args:
            max_cache_size: Maximum number of data points to keep in memory
            cache_manager: Optional Redis cache manager for persistent storage
        """
        self.max_cache_size = max_cache_size
        self.cache_manager = cache_manager
        self.data_cache: Dict[str, deque] = {}
        self.vwap_cache: Dict[str, float] = {}
        
    def process_ticker(self, data: dict) -> Ticker:
        ticker = Ticker(
            instId=data['instId'],
            last=float(data['last']),
            lastSz=float(data['lastSz']),
            askPx=float(data['askPx']),
            askSz=float(data['askSz']),
            bidPx=float(data['bidPx']),
            bidSz=float(data['bidSz']),
            open24h=float(data['open24h']),
            high24h=float(data['high24h']),
            low24h=float(data['low24h']),
            volCcy24h=float(data['volCcy24h']),
            vol24h=float(data['vol24h']),
            timestamp=datetime.fromtimestamp(int(data['ts']) / 1000)
        )
        
        if ticker.instId not in self.data_cache:
            self.data_cache[ticker.instId] = deque(maxlen=self.max_cache_size)
            
        self.data_cache[ticker.instId].append(ticker)
        
        # Update VWAP
        if 'last' in data and 'volume' in data:
            try:
                price = float(data['last'])
                volume = float(data['volume'])
                self.vwap_cache[ticker.instId] = self._calculate_vwap(ticker.instId, price, volume)
            except (ValueError, KeyError) as e:
                logging.error(f"Error calculating VWAP for {ticker.instId}: {e}")
                
        # Store in Redis if cache manager is available
        if self.cache_manager:
            try:
                self.cache_manager.store_market_data(ticker.instId, data)
            except Exception as e:
                logging.error(f"Failed to store data in Redis: {e}")
                
        return ticker
    
    def process_trade(self, data: dict) -> Trade:
        trade = Trade(
            instId=data['instId'],
            price=float(data['px']),
            size=float(data['sz']),
            side=data['side'],
            timestamp=datetime.fromtimestamp(int(data['ts']) / 1000)
        )
        
        if trade.instId not in self.data_cache:
            self.data_cache[trade.instId] = deque(maxlen=self.max_cache_size)
            
        self.data_cache[trade.instId].append(trade)
        
        # Update VWAP
        if 'last' in data and 'volume' in data:
            try:
                price = float(data['last'])
                volume = float(data['volume'])
                self.vwap_cache[trade.instId] = self._calculate_vwap(trade.instId, price, volume)
            except (ValueError, KeyError) as e:
                logging.error(f"Error calculating VWAP for {trade.instId}: {e}")
                
        # Store in Redis if cache manager is available
        if self.cache_manager:
            try:
                self.cache_manager.store_market_data(trade.instId, data)
            except Exception as e:
                logging.error(f"Failed to store data in Redis: {e}")
                
        return trade
    
    def process_order_book(self, data: dict) -> OrderBook:
        def parse_book_level(level: List) -> OrderBookLevel:
            return OrderBookLevel(
                price=float(level[0]),
                size=float(level[1]),
                orders=int(level[2])
            )
            
        order_book = OrderBook(
            instId=data['instId'],
            asks=[parse_book_level(level) for level in data['asks']],
            bids=[parse_book_level(level) for level in data['bids']],
            timestamp=datetime.fromtimestamp(int(data['ts']) / 1000)
        )
        
        if order_book.instId not in self.data_cache:
            self.data_cache[order_book.instId] = deque(maxlen=self.max_cache_size)
            
        self.data_cache[order_book.instId].append(order_book)
        
        # Update VWAP
        if 'last' in data and 'volume' in data:
            try:
                price = float(data['last'])
                volume = float(data['volume'])
                self.vwap_cache[order_book.instId] = self._calculate_vwap(order_book.instId, price, volume)
            except (ValueError, KeyError) as e:
                logging.error(f"Error calculating VWAP for {order_book.instId}: {e}")
                
        # Store in Redis if cache manager is available
        if self.cache_manager:
            try:
                self.cache_manager.store_market_data(order_book.instId, data)
            except Exception as e:
                logging.error(f"Failed to store data in Redis: {e}")
                
        return order_book
    
    def get_vwap(self, symbol: str) -> Optional[float]:
        """Get the current VWAP for a symbol."""
        return self.vwap_cache.get(symbol)
        
    def _calculate_vwap(self, symbol: str, price: float, volume: float) -> float:
        """Calculate Volume Weighted Average Price."""
        current_vwap = self.vwap_cache.get(symbol, 0.0)
        if current_vwap == 0.0:
            return price
            
        # Simple moving VWAP calculation
        alpha = 0.1  # Smoothing factor
        return current_vwap * (1 - alpha) + price * alpha
        
    def get_historical_data(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            start_time: Optional start timestamp (milliseconds)
            end_time: Optional end timestamp (milliseconds)
            
        Returns:
            List of historical data points
        """
        if self.cache_manager:
            try:
                return self.cache_manager.get_market_data(symbol, start_time, end_time)
            except Exception as e:
                logging.error(f"Failed to get historical data from Redis: {e}")
                
        # Fallback to in-memory cache
        return list(self.data_cache.get(symbol, []))

    def process_market_data(self, symbol: str, data: Dict):
        """Process incoming market data."""
        if symbol not in self.data_cache:
            self.data_cache[symbol] = deque(maxlen=self.max_cache_size)
            
        self.data_cache[symbol].append(data)
        
        # Update VWAP
        if 'last' in data and 'volume' in data:
            try:
                price = float(data['last'])
                volume = float(data['volume'])
                self.vwap_cache[symbol] = self._calculate_vwap(symbol, price, volume)
                logging.debug(f"Updated VWAP for {symbol}: {self.vwap_cache[symbol]:.2f}")
            except (ValueError, KeyError) as e:
                logging.error(f"Error calculating VWAP for {symbol}: {e}")
                
        # Store in Redis if cache manager is available
        if self.cache_manager:
            try:
                success = self.cache_manager.store_market_data(symbol, data)
                if not success:
                    logging.warning(f"Failed to store data in Redis for {symbol}")
            except Exception as e:
                logging.error(f"Failed to store data in Redis: {e}")
