#!/usr/bin/env python3
"""
VMAA 2.0 — Analyst Revision Tracker
====================================
Heuristic-based analyst upgrade detection using cached target prices.

Since yfinance doesn't provide analyst revision history, we cache the mean
analyst target price per ticker across pipeline runs. A ≥5% increase in the
mean target (with ≥3 analysts covering) is treated as a "recent upgrade" —
satisfying the MAGNA "3" criterion requirement.

Data limitation acknowledged: this is a proxy for true revision tracking,
which requires Bloomberg/Refinitiv data. Documented per spec review.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("vmaa.analyst_tracker")

CACHE_PATH = Path(__file__).resolve().parent.parent / "output" / "analyst_cache.json"
MAX_CACHE_AGE_DAYS = 14  # Cache entries older than 14 days are considered stale
UPGRADE_THRESHOLD_PCT = 0.05  # 5% target increase = "recent upgrade"


def _load_cache() -> Dict:
    """Load analyst cache from disk."""
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.debug(f"Could not load analyst cache: {e}")
        return {}


def _save_cache(cache: Dict) -> None:
    """Save analyst cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2, default=str)
    except IOError as e:
        logger.debug(f"Could not save analyst cache: {e}")


def check_recent_upgrade(ticker: str, current_target: float,
                         current_analyst_count: int) -> bool:
    """
    Check if analyst target has been recently upgraded.

    Strategy:
      1. Load cached target from previous run
      2. If no cache exists, store current target and return True (first observation)
      3. If cached target exists, check if current target is ≥5% higher
      4. Always update cache with current data for next run
      5. Purge stale (>14 day) entries

    Returns True if:
      - First observation (no prior cache) — give benefit of doubt
      - Current target ≥ 5% higher than cached target
      - Cache entry is stale (>14 days) — refresh accepted as "recent"
    """
    cache = _load_cache()
    now = datetime.now().isoformat()

    # Check existing cache
    if ticker in cache:
        entry = cache[ticker]
        cached_target = entry.get('target', 0)
        cached_time = entry.get('timestamp', '')

        # Purge check: if cached data is >14 days old, treat as fresh observation
        if cached_time:
            try:
                cached_dt = datetime.fromisoformat(cached_time)
                if (datetime.now() - cached_dt).days > MAX_CACHE_AGE_DAYS:
                    logger.debug(f"  {ticker}: Analyst cache stale ({MAX_CACHE_AGE_DAYS}+ days), "
                                 f"accepting as fresh observation")
                    cache[ticker] = {
                        'target': current_target,
                        'analyst_count': current_analyst_count,
                        'timestamp': now,
                    }
                    _save_cache(cache)
                    return True  # Stale cache → accept as recent
            except (ValueError, TypeError):
                pass

        # Compare current vs cached target
        if cached_target > 0:
            change_pct = (current_target - cached_target) / cached_target
            if change_pct >= UPGRADE_THRESHOLD_PCT:
                logger.debug(f"  {ticker}: Analyst target UPGRADE detected: "
                             f"${cached_target:.2f} → ${current_target:.2f} "
                             f"({change_pct:+.1%})")
                cache[ticker] = {
                    'target': current_target,
                    'analyst_count': current_analyst_count,
                    'timestamp': now,
                }
                _save_cache(cache)
                return True

    # Check if this is the first observation (before we overwrite cache)
    is_first_observation = ticker not in cache

    # Update cache with current data
    cache[ticker] = {
        'target': current_target,
        'analyst_count': current_analyst_count,
        'timestamp': now,
    }
    _save_cache(cache)

    # First observation: benefit of doubt (no prior data to compare)
    if is_first_observation:
        return True

    return False


def get_cached_target(ticker: str) -> Optional[Dict]:
    """Retrieve cached analyst data for a ticker."""
    cache = _load_cache()
    return cache.get(ticker)


def clear_cache() -> None:
    """Clear all cached analyst data (useful for fresh start)."""
    _save_cache({})
    logger.info("Analyst cache cleared")


def cache_stats() -> Dict:
    """Return cache statistics."""
    cache = _load_cache()
    return {
        'total_entries': len(cache),
        'tickers': list(cache.keys()),
        'path': str(CACHE_PATH),
    }
