#!/usr/bin/env python3
"""
VMAA Chip Analysis Engine — Orchestrator
=========================================
ChipEngine: central orchestrator that coordinates all chip analysis modules.

Usage:
  from engine.chip.engine import ChipEngine
  engine = ChipEngine()
  report = engine.analyze("AAPL", period="1y")
  sr = engine.get_support_resistance("AAPL")
  cost = engine.get_cost_basis("AAPL")
  flow = engine.get_money_flow("AAPL")
  dist = engine.get_distribution("AAPL")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.chip.config import ChipConfig, ChipConfigManager, get_chip_config
from engine.chip.distribution import (
    VolumeDistributionResult,
    analyze_distribution,
    build_volume_profile,
    compute_rvol,
)
from engine.chip.concentration import (
    ConcentrationResult,
    SupportResistanceResult,
    compute_concentration,
    detect_support_resistance,
)
from engine.chip.cost import (
    CostBasisResult,
    analyze_cost_basis,
)
from engine.chip.profitability import (
    ProfitabilityResult,
    compute_full_profitability,
)

logger = logging.getLogger("vmaa.engine.chip.engine")


# ═══════════════════════════════════════════════════════════════════
# Main Report Dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ChipReport:
    """Complete chip analysis report for a single ticker."""
    ticker: str
    timestamp: str
    period: str
    data_points: int
    current_price: float
    
    # Distribution
    distribution: Optional[Dict[str, Any]] = None
    
    # Concentration
    concentration: Optional[Dict[str, Any]] = None
    
    # Cost basis
    cost_basis: Optional[Dict[str, Any]] = None
    
    # Profitability
    profitability: Optional[Dict[str, Any]] = None
    
    # Support/Resistance
    support_resistance: Optional[Dict[str, Any]] = None
    
    # Error info (if any part failed)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Data Loader
# ═══════════════════════════════════════════════════════════════════

def _fetch_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker using yfinance.

    Args:
        ticker: Ticker symbol (e.g., 'AAPL', '0700.HK')
        period: yfinance period string

    Returns:
        DataFrame with OHLCV columns
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required for chip analysis. Install: pip install yfinance")

    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} (period={period})")

    # Ensure standard columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in data for {ticker}")

    logger.info(f"Fetched {len(df)} rows for {ticker} ({period})")
    return df


# ═══════════════════════════════════════════════════════════════════
# JSON Serializer
# ═══════════════════════════════════════════════════════════════════

def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and numpy types to JSON-serializable dicts."""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _to_dict(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) and not np.isinf(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj


# ═══════════════════════════════════════════════════════════════════
# Chip Engine
# ═══════════════════════════════════════════════════════════════════

class ChipEngine:
    """
    Chip Analysis Engine — analyzes capital/volume distribution,
    concentration, cost basis, profitability, and support/resistance.

    Example:
        >>> engine = ChipEngine()
        >>> report = engine.analyze("AAPL")
        >>> print(report.to_json(indent=2))
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_manager = ChipConfigManager(config_path)
        self._cfg = self._config_manager.to_dataclass()
        logger.info("ChipEngine initialized")

    @property
    def config(self) -> ChipConfig:
        return self._cfg

    def reload_config(self):
        """Hot-reload configuration."""
        self._config_manager.reload()
        self._cfg = self._config_manager.to_dataclass()
        logger.info("ChipEngine config reloaded")

    # ── Public API ───────────────────────────────────────────────

    def analyze(
        self,
        ticker: str,
        period: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> ChipReport:
        """
        Run full chip analysis on a ticker.

        Args:
            ticker: Ticker symbol (e.g., 'AAPL', '0700.HK')
            period: yfinance period (default from config)
            df: Pre-fetched DataFrame (if provided, skips yfinance fetch)

        Returns:
            ChipReport with all analysis modules populated
        """
        period = period or self._cfg.default_period
        errors = []
        warnings = []

        # Fetch data if not provided
        if df is None:
            try:
                df = _fetch_data(ticker, period)
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                return ChipReport(
                    ticker=ticker,
                    timestamp=datetime.utcnow().isoformat(),
                    period=period,
                    data_points=0,
                    current_price=0.0,
                    errors=[str(e)],
                )

        data_points = len(df)
        if data_points < self._cfg.min_data_points:
            warnings.append(f"Only {data_points} data points (min {self._cfg.min_data_points})")

        high, low, close, open_, volume = self._extract_ohlcv(df)
        current_price = float(close[-1]) if len(close) > 0 else 0.0

        report = ChipReport(
            ticker=ticker,
            timestamp=datetime.utcnow().isoformat(),
            period=period,
            data_points=data_points,
            current_price=round(current_price, self._cfg.decimal_places),
            errors=errors,
            warnings=warnings,
        )

        # 1. Volume Distribution
        try:
            dist_result = analyze_distribution(df, ticker, period, self._cfg)
            report.distribution = _to_dict(dist_result)
            logger.info(f"[{ticker}] Distribution analysis complete")
        except Exception as e:
            logger.error(f"[{ticker}] Distribution analysis failed: {e}")
            report.errors.append(f"distribution: {e}")

        # 2. Concentration
        try:
            if report.distribution and report.distribution.get("buckets"):
                from engine.chip.distribution import VolumeProfileBucket
                buckets = [
                    VolumeProfileBucket(**{k: v for k, v in b.items()})
                    for b in report.distribution["buckets"]
                ]
                conc_result = compute_concentration(buckets, current_price, ticker, self._cfg)
                report.concentration = _to_dict(conc_result)
                logger.info(f"[{ticker}] Concentration analysis complete")
        except Exception as e:
            logger.error(f"[{ticker}] Concentration analysis failed: {e}")
            report.errors.append(f"concentration: {e}")

        # 3. Cost Basis
        try:
            cost_result = analyze_cost_basis(df, current_price, ticker, self._cfg)
            report.cost_basis = _to_dict(cost_result)
            logger.info(f"[{ticker}] Cost basis analysis complete")
        except Exception as e:
            logger.error(f"[{ticker}] Cost basis analysis failed: {e}")
            report.errors.append(f"cost_basis: {e}")

        # 4. Profitability (depends on cost basis for avg_cost)
        try:
            avg_cost = current_price
            if report.cost_basis:
                avg_cost = report.cost_basis.get("vwap", current_price)
            try:
                profit_result = compute_full_profitability(df, current_price, avg_cost, ticker, self._cfg)
                report.profitability = _to_dict(profit_result)
                logger.info(f"[{ticker}] Profitability analysis complete")
            except Exception as e:
                logger.error(f"[{ticker}] Profitability analysis failed (retry w/o VWAP): {e}")
                report.errors.append(f"profitability: {e}")
        except Exception as e:
            logger.error(f"[{ticker}] Profitability analysis failed: {e}")
            report.errors.append(f"profitability: {e}")

        # 5. Support/Resistance
        try:
            if report.distribution and report.distribution.get("buckets"):
                from engine.chip.distribution import VolumeProfileBucket
                buckets = [
                    VolumeProfileBucket(**{k: v for k, v in b.items()})
                    for b in report.distribution["buckets"]
                ]
                sr_result = detect_support_resistance(buckets, current_price, ticker, self._cfg)
                report.support_resistance = _to_dict(sr_result)
                logger.info(f"[{ticker}] S/R analysis complete")
        except Exception as e:
            logger.error(f"[{ticker}] S/R analysis failed: {e}")
            report.errors.append(f"support_resistance: {e}")

        return report

    def get_support_resistance(
        self,
        ticker: str,
        period: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Get key support/resistance levels for a ticker.

        Args:
            ticker: Ticker symbol
            period: Override period
            df: Pre-fetched DataFrame

        Returns:
            Dict with S/R levels
        """
        if df is None:
            period = period or self._cfg.default_period
            df = _fetch_data(ticker, period)

        high, low, close, open_, volume = self._extract_ohlcv(df)
        current_price = float(close[-1])

        dist_result = analyze_distribution(df, ticker, period or self._cfg.default_period, self._cfg)
        buckets = dist_result.buckets

        from engine.chip.distribution import VolumeProfileBucket
        if isinstance(buckets[0], dict) if buckets else False:
            bucket_objs = buckets
        else:
            bucket_objs = buckets

        sr = detect_support_resistance(bucket_objs, current_price, ticker, self._cfg)
        return _to_dict(sr)

    def get_cost_basis(
        self,
        ticker: str,
        period: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Get cost basis distribution for a ticker.

        Args:
            ticker: Ticker symbol
            period: Override period
            df: Pre-fetched DataFrame

        Returns:
            Dict with cost basis metrics
        """
        if df is None:
            period = period or self._cfg.default_period
            df = _fetch_data(ticker, period)

        high, low, close, open_, volume = self._extract_ohlcv(df)
        current_price = float(close[-1])

        cost = analyze_cost_basis(df, current_price, ticker, self._cfg)
        return _to_dict(cost)

    def get_money_flow(
        self,
        ticker: str,
        period: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Get money flow analysis (CMF, inflow/outflow).

        Args:
            ticker: Ticker symbol
            period: Override period
            df: Pre-fetched DataFrame

        Returns:
            Dict with money flow metrics
        """
        if df is None:
            period = period or self._cfg.default_period
            df = _fetch_data(ticker, period)

        from engine.chip.profitability import compute_money_flow
        mf = compute_money_flow(df, ticker, self._cfg)
        return _to_dict(mf)

    def get_distribution(
        self,
        ticker: str,
        period: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Get volume distribution (VP, VA, POC).

        Args:
            ticker: Ticker symbol
            period: Override period
            df: Pre-fetched DataFrame

        Returns:
            Dict with distribution metrics
        """
        if df is None:
            period = period or self._cfg.default_period
            df = _fetch_data(ticker, period)

        dist = analyze_distribution(df, ticker, period or self._cfg.default_period, self._cfg)
        return _to_dict(dist)

    # ── Helpers ──────────────────────────────────────────────────

    def _extract_ohlcv(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract OHLCV arrays from DataFrame."""
        from engine.chip.distribution import _extract_ohlcv
        return _extract_ohlcv(df)

    # ── JSON Output ─────────────────────────────────────────────

    def to_json(self, report: ChipReport, indent: int = 2) -> str:
        """Serialize a ChipReport to JSON string."""
        data = _to_dict(report)
        return json.dumps(data, indent=indent, default=str, ensure_ascii=False)

    def analyze_to_json(
        self,
        ticker: str,
        period: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        indent: int = 2,
    ) -> str:
        """Run analysis and return JSON string directly."""
        report = self.analyze(ticker, period, df)
        return self.to_json(report, indent)


# ═══════════════════════════════════════════════════════════════════
# Quick Functions (convenience)
# ═══════════════════════════════════════════════════════════════════

def quick_chip(ticker: str, period: str = "1y") -> ChipReport:
    """One-liner: run full chip analysis on a ticker."""
    engine = ChipEngine()
    return engine.analyze(ticker, period)


def quick_chip_json(ticker: str, period: str = "1y") -> str:
    """One-liner: full chip analysis as JSON."""
    engine = ChipEngine()
    return engine.analyze_to_json(ticker, period)


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_CHIP_ENGINE: Optional[ChipEngine] = None


def get_chip_engine() -> ChipEngine:
    """Get or create the singleton ChipEngine instance."""
    global _CHIP_ENGINE
    if _CHIP_ENGINE is None:
        _CHIP_ENGINE = ChipEngine()
    return _CHIP_ENGINE
