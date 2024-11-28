import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
import aioredis
import json
import sqlite3
import aiosqlite
from pathlib import Path

@dataclass
class StorageConfig:
    """Prediction storage configuration"""
    redis_url: str = "redis://localhost:6379/0"
    sqlite_path: str = "data/predictions.db"
    cache_expiry: int = 3600  # 1 hour
    batch_size: int = 100
    
class PredictionStorage:
    """Manages prediction result storage and retrieval"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self._initialize_storage()
        
    async def _initialize_storage(self):
        """Initialize storage backends"""
        try:
            # Initialize Redis connection
            self.redis = await aioredis.create_redis_pool(self.config.redis_url)
            
            # Initialize SQLite database
            await self._setup_database()
            
            logging.info("Prediction storage initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing storage: {e}")
            raise
            
    async def _setup_database(self):
        """Setup SQLite database schema"""
        try:
            async with aiosqlite.connect(self.config.sqlite_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        prediction INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        market_state INTEGER,
                        features TEXT,
                        actual_value REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
                    ON predictions(timestamp)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_symbol 
                    ON predictions(symbol)
                """)
                await db.commit()
                
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            raise
            
    async def store_prediction(
        self,
        prediction: Dict,
        features: Optional[Dict] = None,
        market_state: Optional[int] = None
    ):
        """Store prediction results"""
        try:
            # Store in Redis for quick access
            cache_key = f"pred:{prediction['symbol']}:{prediction['timestamp']}"
            await self.redis.set(
                cache_key,
                json.dumps(prediction),
                expire=self.config.cache_expiry
            )
            
            # Store in SQLite for persistence
            async with aiosqlite.connect(self.config.sqlite_path) as db:
                await db.execute("""
                    INSERT INTO predictions (
                        timestamp,
                        symbol,
                        prediction,
                        confidence,
                        market_state,
                        features
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    prediction['timestamp'],
                    prediction['symbol'],
                    prediction['prediction'],
                    prediction['confidence'],
                    market_state,
                    json.dumps(features) if features else None
                ))
                await db.commit()
                
        except Exception as e:
            logging.error(f"Error storing prediction: {e}")
            raise
            
    async def update_actual_value(
        self,
        symbol: str,
        timestamp: str,
        actual_value: float
    ):
        """Update prediction with actual value"""
        try:
            async with aiosqlite.connect(self.config.sqlite_path) as db:
                await db.execute("""
                    UPDATE predictions 
                    SET actual_value = ? 
                    WHERE symbol = ? AND timestamp = ?
                """, (actual_value, symbol, timestamp))
                await db.commit()
                
        except Exception as e:
            logging.error(f"Error updating actual value: {e}")
            raise
            
    async def get_recent_predictions(
        self,
        symbol: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get recent predictions from cache"""
        try:
            # Try Redis first
            cache_pattern = f"pred:{symbol}:*"
            keys = await self.redis.keys(cache_pattern)
            
            if keys:
                predictions = []
                for key in sorted(keys)[-limit:]:
                    value = await self.redis.get(key)
                    if value:
                        predictions.append(json.loads(value))
                return predictions
                
            # Fallback to SQLite
            async with aiosqlite.connect(self.config.sqlite_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("""
                    SELECT * FROM predictions 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (symbol, limit)) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
                    
        except Exception as e:
            logging.error(f"Error retrieving predictions: {e}")
            raise
            
    async def get_performance_metrics(
        self,
        symbol: str,
        start_time: str,
        end_time: str
    ) -> Dict:
        """Get prediction performance metrics"""
        try:
            async with aiosqlite.connect(self.config.sqlite_path) as db:
                async with db.execute("""
                    SELECT 
                        COUNT(*) as total,
                        AVG(CASE WHEN actual_value IS NOT NULL 
                            AND SIGN(actual_value) = prediction 
                            THEN 1 ELSE 0 END) as accuracy,
                        AVG(confidence) as avg_confidence
                    FROM predictions 
                    WHERE symbol = ? 
                    AND timestamp BETWEEN ? AND ?
                """, (symbol, start_time, end_time)) as cursor:
                    row = await cursor.fetchone()
                    return {
                        'total_predictions': row[0],
                        'accuracy': row[1],
                        'avg_confidence': row[2]
                    }
                    
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            raise
            
    async def cleanup(self):
        """Cleanup storage connections"""
        try:
            self.redis.close()
            await self.redis.wait_closed()
            
        except Exception as e:
            logging.error(f"Error cleaning up storage: {e}")
            raise 