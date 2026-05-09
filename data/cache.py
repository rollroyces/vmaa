#!/usr/bin/env python3
"""
VMAA Data Cache Layer
=====================
SQLite-based local cache for financial data.

Purpose:
  - Avoid re-fetching data multiple times per day
  - Each ticker + data_type is cached once per day
  - Auto-expires data older than 720 days (~2 years)
  - Zero dependency (SQLite built into Python)

Usage:
  from data.cache import cache_get, cache_set, cache_cleanup

  # Store
  cache_set('AAPL', 'fundamentals', {'marketCap': 3e12, ...})
  
  # Retrieve (same day only)
  data = cache_get('AAPL', 'fundamentals')
  if data:
      print('Using cached data')
  else:
      # Fetch from API
      data = fetch_from_api('AAPL')
      cache_set('AAPL', 'fundamentals', data)

  # Cleanup old data (run periodically)
  cache_cleanup()
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("vmaa.cache")

# ── Configuration ──────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DB = CACHE_DIR / "data_cache.db"
MAX_AGE_DAYS = 720  # Keep data for 720 days (~2 years)
CURRENT_DAY = datetime.now().strftime("%Y-%m-%d")  # Today's date for cache key


# ═══════════════════════════════════════════════════════════════
# Database Setup
# ═══════════════════════════════════════════════════════════════

def _get_db() -> sqlite3.Connection:
    """Get SQLite connection with WAL mode for concurrent reads."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_cache (
            ticker       TEXT NOT NULL,
            data_type    TEXT NOT NULL,
            fetch_date   TEXT NOT NULL,  -- YYYY-MM-DD
            data         TEXT NOT NULL,  -- JSON blob
            created_at   TEXT NOT NULL,  -- ISO datetime
            expires_at   TEXT NOT NULL,  -- ISO datetime
            PRIMARY KEY (ticker, data_type, fetch_date)
        )
    """)
    # Index for fast expiry cleanup
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_expires 
        ON data_cache(expires_at)
    """)
    # Index for fetching all data of a type (e.g., sector medians)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_type_date 
        ON data_cache(data_type, fetch_date)
    """)
    conn.commit()


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════

def cache_get(ticker: str, data_type: str) -> Optional[Any]:
    """
    Get cached data for a ticker + type from TODAY.
    Returns parsed JSON data, or None if not cached today.
    
    data_type options:
      - 'fundamentals'  : Finnhub metrics / yfinance info
      - 'bars'          : OHLCV price history
      - 'price'         : Current price + volume
      - 'sector'        : Industry classification
      - 'analyst'       : Analyst recommendations
      - 'earnings'      : Quarterly earnings data
      - 'sec_edgar'     : SEC EDGAR fundamentals
    """
    try:
        conn = _get_db()
        cursor = conn.execute(
            "SELECT data FROM data_cache WHERE ticker=? AND data_type=? AND fetch_date=?",
            (ticker.upper(), data_type, CURRENT_DAY)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row['data'])
    except Exception as e:
        logger.debug(f"Cache read error for {ticker}/{data_type}: {e}")
    return None


def cache_set(ticker: str, data_type: str, data: Any) -> bool:
    """
    Store data in cache. Overwrites any existing entry for same ticker+type+today.
    Returns True if stored successfully.
    """
    try:
        conn = _get_db()
        expires = (datetime.now() + timedelta(days=MAX_AGE_DAYS)).isoformat()
        now = datetime.now().isoformat()
        
        conn.execute("""
            INSERT OR REPLACE INTO data_cache (ticker, data_type, fetch_date, data, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            ticker.upper(),
            data_type,
            CURRENT_DAY,
            json.dumps(data, default=str),
            now,
            expires
        ))
        conn.commit()
        return True
    except Exception as e:
        logger.debug(f"Cache write error for {ticker}/{data_type}: {e}")
        return False


def cache_get_batch(tickers: List[str], data_type: str) -> Dict[str, Any]:
    """
    Batch get cached data for multiple tickers of the same type.
    Returns {ticker: data_dict} for entries found in today's cache.
    """
    if not tickers:
        return {}
    try:
        conn = _get_db()
        placeholders = ','.join(['?'] * len(tickers))
        cursor = conn.execute(
            f"SELECT ticker, data FROM data_cache "
            f"WHERE ticker IN ({placeholders}) AND data_type=? AND fetch_date=?",
            [t.upper() for t in tickers] + [data_type, CURRENT_DAY]
        )
        results = {}
        for row in cursor.fetchall():
            try:
                results[row['ticker']] = json.loads(row['data'])
            except (json.JSONDecodeError, TypeError):
                pass
        return results
    except Exception as e:
        logger.debug(f"Cache batch read error for {data_type}: {e}")
    return {}


def cache_exists(ticker: str, data_type: str) -> bool:
    """Check if today's cache exists for a ticker+type."""
    try:
        conn = _get_db()
        cursor = conn.execute(
            "SELECT 1 FROM data_cache WHERE ticker=? AND data_type=? AND fetch_date=?",
            (ticker.upper(), data_type, CURRENT_DAY)
        )
        return cursor.fetchone() is not None
    except Exception:
        return False


def cache_cleanup(max_age_days: int = MAX_AGE_DAYS) -> int:
    """
    Remove data older than max_age_days.
    Returns number of rows deleted.
    """
    try:
        conn = _get_db()
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        cursor = conn.execute(
            "DELETE FROM data_cache WHERE expires_at < ?",
            (cutoff,)
        )
        conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cache cleanup: removed {deleted} expired entries")
        return deleted
    except Exception as e:
        logger.warning(f"Cache cleanup failed: {e}")
        return 0


def cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    try:
        conn = _get_db()
        cursor = conn.execute("SELECT COUNT(*) as total FROM data_cache")
        total = cursor.fetchone()['total']
        
        cursor = conn.execute(
            "SELECT data_type, COUNT(*) as cnt FROM data_cache GROUP BY data_type"
        )
        by_type = {row['data_type']: row['cnt'] for row in cursor.fetchall()}
        
        cursor = conn.execute(
            "SELECT fetch_date, COUNT(*) as cnt FROM data_cache GROUP BY fetch_date ORDER BY fetch_date"
        )
        by_date = {row['fetch_date']: row['cnt'] for row in cursor.fetchall()}
        
        # Size estimate
        cursor = conn.execute("SELECT SUM(LENGTH(data)) as total_bytes FROM data_cache")
        size = cursor.fetchone()['total_bytes'] or 0
        
        return {
            'total_entries': total,
            'entries_by_type': by_type,
            'entries_by_date': by_date,
            'total_size_mb': round(size / 1024 / 1024, 2),
            'db_path': str(CACHE_DB),
            'max_age_days': MAX_AGE_DAYS,
        }
    except Exception as e:
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════
# Cache Wrapper (decorator pattern)
# ═══════════════════════════════════════════════════════════════

def cached(data_type: str):
    """
    Decorator: wraps a fetch function with cache.
    
    Usage:
        @cached('fundamentals')
        def fetch_fundamentals(ticker):
            # ... expensive API call ...
            return data
            
        data = fetch_fundamentals('AAPL')  # Uses cache if available
    """
    def decorator(func):
        def wrapper(ticker: str, *args, **kwargs):
            # Try cache first
            cached_data = cache_get(ticker, data_type)
            if cached_data is not None:
                return cached_data
            
            # Fetch fresh data
            data = func(ticker, *args, **kwargs)
            
            # Store in cache (only if non-empty)
            if data:
                cache_set(ticker, data_type, data)
            
            return data
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Cache Test ===")
    
    # Store
    cache_set('AAPL', 'test', {'price': 290, 'source': 'test'})
    cache_set('MSFT', 'test', {'price': 415, 'source': 'test'})
    
    # Retrieve
    data = cache_get('AAPL', 'test')
    print(f"Get AAPL: {data}")
    
    # Batch
    batch = cache_get_batch(['AAPL', 'MSFT', 'INMD'], 'test')
    print(f"Batch: {list(batch.keys())}")
    
    # Stats
    stats = cache_stats()
    print(f"Stats: {stats['total_entries']} entries, {stats['total_size_mb']}MB")
    
    # Cleanup test data
    cache_set('AAPL', 'test', {'cleanup': True})
    print("Done")
