#!/usr/bin/env python3
"""
HK Data Loader
==============
Historical data fetcher for HK stocks using yfinance with `.HK` suffix.

Wraps the existing US HistoricalDataLoader for price/volume data and
adds HK-specific helpers:
  - HSI benchmark (^HSI + 2800.HK)
  - HK stock universe loading
  - Sector classification from yfinance info
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from backtest.hk.hk_config import HKBacktestConfig, HKC
from backtest.data import HistoricalDataLoader, HistoricalSnapshot

logger = logging.getLogger("vmaa.backtest.hk.data")


class HKDataLoader:
    """
    HK stock data loader reusing the existing HistoricalDataLoader internally.

    Usage:
        loader = HKDataLoader(config)
        loader.fetch_all(tickers, "2023-01-01", "2025-12-31")
        snapshot = loader.get_snapshot("0700.HK", "2024-06-15")
        hsi_hist = loader.get_benchmark_history()
    """

    def __init__(self, config: HKBacktestConfig = HKC):
        self.config = config
        # Reuse existing US data loader for price caching
        from backtest.config import BacktestConfig
        us_config = BacktestConfig(
            cache_dir=config.cache_dir,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        self._loader = HistoricalDataLoader(us_config)

        # HK-specific caches
        self._hk_info_cache: Dict[str, dict] = {}
        self._benchmark_hist: Optional[pd.DataFrame] = None
        self._hsi_hist: Optional[pd.DataFrame] = None

    def fetch_all(self, tickers: List[str],
                  start_date: str, end_date: str) -> None:
        """
        Fetch all data for HK tickers.

        Args:
            tickers: List of HK tickers (e.g., ['0700.HK', '0016.HK'])
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
        """
        logger.info(f"Loading HK data for {len(tickers)} tickers...")

        # Use existing loader for price + financial data
        all_tickers = list(dict.fromkeys(
            ["2800.HK", "^HSI"] + [t for t in tickers if t not in ("2800.HK", "^HSI")]
        ))
        self._loader.fetch_all(all_tickers, start_date, end_date)

        # Pre-fetch HK info for sector classification
        for ticker in tickers:
            try:
                tk = yf.Ticker(ticker)
                self._hk_info_cache[ticker] = tk.info
            except Exception:
                pass
            time.sleep(0.05)

        # Cache benchmark data
        self._benchmark_hist = self._loader.get_price_history("2800.HK")
        self._hsi_hist = self._loader.get_price_history("^HSI")

        logger.info(
            f"HK data loaded: {len(self._loader._price_cache)} tickers with price data, "
            f"{len(self._hk_info_cache)} with HK info"
        )

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get full daily price history for a ticker."""
        return self._loader.get_price_history(ticker)

    def get_snapshot(self, ticker: str, date_str: str) -> Optional[HistoricalSnapshot]:
        """Get a point-in-time snapshot for a ticker."""
        return self._loader.get_snapshot(ticker, date_str)

    def get_benchmark_history(self) -> Optional[pd.DataFrame]:
        """Get 2800.HK benchmark price history."""
        return self._benchmark_hist

    def get_hsi_history(self) -> Optional[pd.DataFrame]:
        """Get ^HSI index history (for regime detection)."""
        return self._hsi_hist

    def get_hk_info(self, ticker: str) -> dict:
        """Get cached yfinance info dict for HK ticker."""
        return self._hk_info_cache.get(ticker, {})

    def get_sector(self, ticker: str) -> str:
        """Get sector classification for HK ticker."""
        return self._hk_info_cache.get(ticker, {}).get('sector', 'Unknown')

    def get_industry(self, ticker: str) -> str:
        """Get industry classification for HK ticker."""
        return self._hk_info_cache.get(ticker, {}).get('industry', 'Unknown')

    def get_trading_days(self, start: str, end: str) -> List[str]:
        """Get all HK trading days in range from 2800.HK."""
        hist = self._benchmark_hist
        if hist is not None and not hist.empty:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            # Handle tz-aware index
            idx = hist.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                idx = idx.tz_localize(None)
            mask = (idx >= start_ts) & (idx <= end_ts)
            return [d.strftime('%Y-%m-%d') for d in hist.index[mask]]
        # Fallback: business days
        return [d.strftime('%Y-%m-%d') for d in pd.bdate_range(start, end)]

    def get_rebalance_dates(self, start: str, end: str) -> List[str]:
        """Get monthly rebalance dates."""
        freq_map = {'daily': 'B', 'weekly': 'W-FRI', 'monthly': 'BME'}
        freq = freq_map.get(self.config.rebalance_frequency, 'BME')
        all_dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq)
        return [d.strftime('%Y-%m-%d') for d in all_dates]
