#!/usr/bin/env python3
"""
VMAA-HK Backtest Engine
========================
Hong Kong stock backtesting with HK-specific screening thresholds,
HKD currency, and HSI benchmark comparison.

Exports:
  HKBacktestConfig  — HK-specific backtest parameters
  HKDataLoader      — HK stock data fetcher
  HKScreener        — HK Part 1 + Part 2 screening wrapper
  HKBacktestEngine  — Walk-forward HK backtest simulator
  HKMetrics         — HK performance analytics
"""
from __future__ import annotations

from backtest.hk.hk_config import HKBacktestConfig, HKC
from backtest.hk.hk_data import HKDataLoader
from backtest.hk.hk_screener import HKScreener
from backtest.hk.hk_backtest import HKBacktestEngine
from backtest.hk.hk_metrics import HKMetricsCalculator

__all__ = [
    "HKBacktestConfig",
    "HKC",
    "HKDataLoader",
    "HKScreener",
    "HKBacktestEngine",
    "HKMetricsCalculator",
]
