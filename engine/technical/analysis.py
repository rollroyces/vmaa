#!/usr/bin/env python3
"""
VMAA Technical Analysis Engine
===============================
Orchestrator that ties together indicators, registry, custom builder, and signals.

Usage:
    from engine.technical import TechnicalEngine

    engine = TechnicalEngine()
    df = yf.download("AAPL", ...)
    result = engine.compute(df)
    signals = engine.get_signals(df, ticker="AAPL")
    custom = engine.custom(df, "MA(CLOSE,5) - MA(CLOSE,20)")
    batch = engine.batch_analysis(["AAPL", "GOOGL", "MSFT"])
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.technical.config import TC, TechnicalConfig
from engine.technical.indicators import compute_all, _extract_columns
from engine.technical.registry import (
    IndicatorRegistry, IndicatorMeta, IndicatorCategory,
    get_registry, create_default_registry,
)
from engine.technical.custom import (
    custom_indicator, validate_formula, add_custom_indicator,
    remove_custom_indicator, load_custom_indicators, list_available_functions,
    CustomIndicator,
)
from engine.technical.signals import (
    generate_all_signals, SignalResult, SignalType, SignalTracker,
    IndicatorSignal,
)

logger = logging.getLogger("vmaa.engine.technical.analysis")


# ═══════════════════════════════════════════════════════════════════
# Technical Engine
# ═══════════════════════════════════════════════════════════════════

class TechnicalEngine:
    """Orchestrator for all technical analysis operations.

    Usage:
        >>> engine = TechnicalEngine()
        >>> engine.list_indicators()[:5]  # First 5 registered indicators
        >>> df = engine.compute(price_data)  # Compute all indicators
        >>> sig = engine.get_signals(df, "AAPL")  # Get trading signals
        >>> result = engine.custom(df, "MA(CLOSE,20) - MA(CLOSE,50)")  # Custom indicator
    """

    def __init__(self, config: Optional[TechnicalConfig] = None):
        """Initialize the technical engine.

        Args:
            config: Optional custom configuration. Uses defaults if not provided.
        """
        self.config = config or TC
        self.registry = get_registry()
        self.signal_tracker = SignalTracker()
        self._custom_cache: Dict[str, np.ndarray] = {}
        logger.info(f"TechnicalEngine initialized: {self.registry.count} indicators registered")

    # ── Indicator Registry ──

    def add_indicator(self, name: str, params: Optional[Dict[str, Any]] = None) -> IndicatorMeta:
        """Look up a registered indicator by name — returns its metadata.
        This is primarily for inspection. New built-in indicators are added via registry.register().

        Args:
            name: Indicator name (e.g., "rsi_14", "sma_20").
            params: Reserved for future use.

        Returns:
            IndicatorMeta if found.

        Raises:
            ValueError: If indicator not found.
        """
        meta = self.registry.get(name)
        if meta is None:
            available = self.registry.list_names()
            raise ValueError(f"Indicator '{name}' not found. Available: {available[:10]}{'...' if len(available) > 10 else ''}")
        return meta

    def list_indicators(
        self,
        category: Optional[str] = None,
        search: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[IndicatorMeta]:
        """List available indicators, optionally filtered.

        Args:
            category: Filter by category ("trend", "momentum", "volatility", "volume", "composite").
            search: Substring search on name/display_name.
            tags: Filter by tags.

        Returns:
            List of matching IndicatorMeta objects.
        """
        cat_enum = None
        if category:
            try:
                cat_enum = IndicatorCategory(category.lower())
            except ValueError:
                logger.warning(f"Unknown category '{category}'. Available: {[c.value for c in IndicatorCategory]}")
                return []
        return self.registry.search(query=search, category=cat_enum, tags=tags)

    def indicator_summary(self) -> str:
        """Return a summary string of the indicator registry."""
        return self.registry.summary()

    # ── Computation ──

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all standard technical indicators for a price DataFrame.

        This is a convenience wrapper around indicators.compute_all().

        Args:
            data: DataFrame with Open/High/Low/Close/Volume columns.
                  Can be from yfinance, CSV, or any source.

        Returns:
            DataFrame with original columns + all indicator columns appended.
        """
        logger.debug(f"Computing all indicators on DataFrame of shape {data.shape}")
        result = compute_all(data)
        return result

    def compute_single(
        self, data: pd.DataFrame, indicator_name: str
    ) -> np.ndarray:
        """Compute a single indicator from the registry.

        Args:
            data: DataFrame with OHLCV columns.
            indicator_name: Name of indicator in registry (e.g., "rsi_14").

        Returns:
            Numpy array with indicator values.

        Raises:
            ValueError: If indicator not found or missing required data.
        """
        meta = self.registry.get(indicator_name)
        if meta is None:
            raise ValueError(f"Indicator '{indicator_name}' not found in registry.")

        o, h, l, c, v = _extract_columns(data)
        field_map = {"open": o, "high": h, "low": l, "close": c, "volume": v}

        # Build args from required_fields
        args = []
        for field in meta.required_fields:
            if field not in field_map:
                raise ValueError(f"Indicator '{indicator_name}' requires '{field}' data, not available.")
            args.append(field_map[field])

        result = meta.compute(*args)
        return result

    # ── Signals ──

    def get_signals(self, data: pd.DataFrame, ticker: str = "") -> SignalResult:
        """Generate trading signals from all indicators.

        Args:
            data: DataFrame with OHLCV columns.
            ticker: Ticker symbol for identification.

        Returns:
            SignalResult with aggregated signal, strength, and individual indicator signals.
        """
        logger.info(f"Generating signals for {ticker or 'unknown'}")
        result = generate_all_signals(data, ticker)

        # Record to signal tracker
        if result.signal in (SignalType.BUY, SignalType.STRONG_BUY, SignalType.SELL, SignalType.STRONG_SELL):
            self.signal_tracker.record(
                ticker=ticker or "UNKNOWN",
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                signal=result.signal,
                price=result.latest_price,
            )

        return result

    def get_signal_history(self) -> Dict[str, Any]:
        """Get signal tracking statistics."""
        return self.signal_tracker.stats()

    # ── Custom Indicators ──

    def custom(self, data: pd.DataFrame, formula: str) -> np.ndarray:
        """Compute a custom indicator from a formula expression.

        Args:
            data: DataFrame with OHLCV columns.
            formula: Mathematical expression. Examples:
                - "MA(CLOSE, 20) - MA(CLOSE, 50)"
                - "(CLOSE - MA(CLOSE, 20)) / STD(CLOSE, 20) * 100"
                - "RSI(CLOSE, 14)"

        Returns:
            Numpy array with computed indicator values.
        """
        logger.debug(f"Computing custom indicator: {formula}")
        # Check cache
        cache_key = formula
        if cache_key in self._custom_cache and len(data) == len(self._custom_cache[cache_key]):
            return self._custom_cache[cache_key].copy()

        result = custom_indicator(data, formula)
        self._custom_cache[cache_key] = result
        return result

    def save_custom(self, name: str, formula: str, description: str = "", tags: Optional[List[str]] = None) -> CustomIndicator:
        """Save a custom indicator formula for later use.

        Args:
            name: Unique name.
            formula: Formula expression.
            description: Human-readable description.
            tags: Tags for organization.

        Returns:
            The created CustomIndicator.
        """
        return add_custom_indicator(name, formula, description, tags)

    def load_custom(self) -> List[CustomIndicator]:
        """Load all saved custom indicators."""
        return load_custom_indicators()

    def delete_custom(self, name: str) -> bool:
        """Delete a saved custom indicator."""
        return remove_custom_indicator(name)

    def validate_formula(self, formula: str) -> Tuple[bool, str]:
        """Validate a custom indicator formula.

        Returns:
            (is_valid, error_message).
        """
        return validate_formula(formula)

    def list_functions(self) -> Dict[str, str]:
        """Return available functions for custom indicators."""
        return list_available_functions()

    # ── Batch Analysis ──

    def batch_analysis(
        self,
        tickers: List[str],
        data_provider: Optional[Callable[[str], pd.DataFrame]] = None,
    ) -> List[Dict[str, Any]]:
        """Run technical analysis on multiple tickers.

        Each ticker result is a JSON-serializable dict with all indicators and signals.

        Args:
            tickers: List of ticker symbols.
            data_provider: Function that takes a ticker and returns a DataFrame.
                           If None, uses yfinance.

        Returns:
            List of dicts with ticker, indicators (summary), and signals.

        Raises:
            ImportError: If yfinance not available and no data_provider provided.
        """
        if data_provider is None:
            try:
                import yfinance as yf
                def _provider(ticker: str) -> pd.DataFrame:
                    return yf.download(ticker, period="6mo", progress=False, auto_adjust=True)
                data_provider = _provider
            except ImportError:
                raise ImportError("yfinance is required for batch_analysis. Install with: pip install yfinance")

        results = []
        for ticker in tickers:
            try:
                logger.info(f"Analyzing {ticker}...")
                df = data_provider(ticker)
                if df is None or df.empty:
                    logger.warning(f"No data for {ticker}")
                    results.append({"ticker": ticker, "error": "No data", "indicators": {}, "signals": None})
                    continue

                # Compute all indicators
                df_with_indicators = self.compute(df)

                # Build JSON-serializable summary
                last_row = df_with_indicators.iloc[-1]
                indicators_summary = {}
                for col in df_with_indicators.columns:
                    val = last_row[col]
                    if isinstance(val, (np.floating, float)):
                        indicators_summary[col] = round(float(val), 4) if not np.isnan(float(val)) else None
                    elif isinstance(val, (np.integer, int)):
                        indicators_summary[col] = int(val)
                    elif isinstance(val, np.bool_):
                        indicators_summary[col] = bool(val)
                    else:
                        try:
                            indicators_summary[col] = float(val) if not np.isnan(float(val)) else None
                        except (ValueError, TypeError):
                            indicators_summary[col] = str(val)

                # Generate signals
                signal_result = self.get_signals(df, ticker)

                results.append({
                    "ticker": ticker,
                    "indicators": indicators_summary,
                    "signals": signal_result.to_dict(),
                })

            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                results.append({"ticker": ticker, "error": str(e), "indicators": {}, "signals": None})

        return results

    # ── Export ──

    def to_json(self, data: pd.DataFrame, filepath: Optional[str] = None) -> str:
        """Export computed indicators as JSON.

        Args:
            data: DataFrame with indicators (output of compute()).
            filepath: If provided, saves to file.

        Returns:
            JSON string.
        """
        # Replace NaN with None for valid JSON
        clean = data.where(pd.notna(data), None)
        records = clean.to_dict(orient="records")
        json_str = json.dumps(records, indent=2, default=str)

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"Exported indicators to {filepath}")

        return json_str

    # ── Quick Stats ──

    def quick_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Return a quick summary of technical conditions for a stock.

        Args:
            data: DataFrame with OHLCV columns (raw, indicators computed on-the-fly).

        Returns:
            Dict with key technical readings.
        """
        df = self.compute(data)
        last = df.iloc[-1]

        def safe_val(col: str, default=None):
            v = last.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return round(float(v), 4)

        return {
            "price": safe_val("Close"),
            "trend": {
                "sma_20": safe_val("sma_20"),
                "sma_50": safe_val("sma_50"),
                "sma_200": safe_val("sma_200"),
                "above_sma_50": bool(safe_val("Close", 0) > safe_val("sma_50", float("inf"))),
                "above_sma_200": bool(safe_val("Close", 0) > safe_val("sma_200", float("inf"))),
            },
            "momentum": {
                "rsi_14": safe_val("rsi_14"),
                "macd_histogram": safe_val("macd_histogram"),
                "stoch_k": safe_val("stoch_k"),
            },
            "volatility": {
                "bb_pct_b": safe_val("bb_pct_b"),
                "bb_squeeze": bool(safe_val("bb_squeeze", 0)),
                "atr_14": safe_val("atr_14"),
            },
            "volume": {
                "vol_ratio": safe_val("vol_ratio"),
                "cmf_21": safe_val("cmf_21"),
            },
        }


# ═══════════════════════════════════════════════════════════════════
# Convenience re-exports
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    "TechnicalEngine",
    "TechnicalConfig",
    "TC",
    "IndicatorRegistry",
    "IndicatorMeta",
    "IndicatorCategory",
    "SignalResult",
    "SignalType",
    "SignalTracker",
    "IndicatorSignal",
    "CustomIndicator",
]
