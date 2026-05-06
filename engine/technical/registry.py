#!/usr/bin/env python3
"""
VMAA Indicator Registry
========================
Central registry of all available technical indicators with metadata.
Plugin architecture: easy to add new indicators without modifying existing code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("vmaa.engine.technical.registry")


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class IndicatorCategory(str, Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    COMPOSITE = "composite"
    CUSTOM = "custom"


class Complexity(str, Enum):
    CONSTANT = "O(1)"
    LINEAR = "O(n)"
    QUADRATIC = "O(n²)"


# ═══════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IndicatorMeta:
    """Metadata for a registered indicator."""

    name: str                          # unique identifier, e.g. "sma_20"
    display_name: str                  # human-readable, e.g. "SMA (20)"
    category: IndicatorCategory
    periods: List[int] = field(default_factory=list)  # default period(s)
    params: Dict[str, Any] = field(default_factory=dict)  # additional params
    description: str = ""
    required_fields: List[str] = field(default_factory=lambda: ["close"])
    # Which OHLCV fields does this indicator need?
    # Possible values: "open", "high", "low", "close", "volume"
    output_count: int = 1              # How many output arrays (1 for single, >1 for tuple)
    output_names: List[str] = field(default_factory=list)  # Names of output columns
    complexity: Complexity = Complexity.LINEAR
    tags: List[str] = field(default_factory=list)
    # The actual computation function (set at registration time)
    _func: Optional[Callable] = field(default=None, repr=False)

    def compute(self, *args, **kwargs):
        """Call the indicator function."""
        if self._func is None:
            raise ValueError(f"Indicator '{self.name}' has no compute function bound.")
        return self._func(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

class IndicatorRegistry:
    """Thread-safe registry of all indicators.

    Usage:
        reg = IndicatorRegistry()
        reg.register(sma_meta, sma_func)
        reg.register(ema_meta, ema_func)
        ...
        reg.list_all()
        reg.search("rsi", category="momentum")
    """

    def __init__(self):
        self._indicators: Dict[str, IndicatorMeta] = {}
        self._initialized = False

    # ── Registration ──

    def register(self, meta: IndicatorMeta, func: Callable) -> None:
        """Register an indicator with its compute function.

        Args:
            meta: IndicatorMeta with name, category, etc.
            func: The callable that computes this indicator.
        """
        if meta.name in self._indicators:
            logger.debug(f"Overwriting existing indicator: {meta.name}")
        meta._func = func
        self._indicators[meta.name] = meta
        logger.debug(f"Registered indicator: {meta.name} [{meta.category.value}]")

    def unregister(self, name: str) -> bool:
        """Remove an indicator from the registry.

        Returns:
            True if removed, False if not found.
        """
        if name in self._indicators:
            del self._indicators[name]
            return True
        return False

    # ── Lookup ──

    def get(self, name: str) -> Optional[IndicatorMeta]:
        """Get indicator metadata by name."""
        return self._indicators.get(name)

    def get_func(self, name: str) -> Optional[Callable]:
        """Get the indicator compute function by name."""
        meta = self._indicators.get(name)
        return meta._func if meta else None

    def list_all(self) -> List[IndicatorMeta]:
        """Return all registered indicators sorted by category then name."""
        return sorted(self._indicators.values(), key=lambda m: (m.category.value, m.name))

    def list_by_category(self, category: IndicatorCategory) -> List[IndicatorMeta]:
        """Return indicators filtered by category."""
        return [m for m in self._indicators.values() if m.category == category]

    def list_names(self) -> List[str]:
        """Return all indicator names."""
        return list(self._indicators.keys())

    def list_categories(self) -> List[str]:
        """Return all category names that have registered indicators."""
        cats = set(m.category.value for m in self._indicators.values())
        return sorted(cats)

    # ── Search ──

    def search(
        self,
        query: str = "",
        category: Optional[IndicatorCategory] = None,
        tags: Optional[List[str]] = None,
        required_fields: Optional[List[str]] = None,
    ) -> List[IndicatorMeta]:
        """Search indicators by name, category, tags, or required fields.

        Args:
            query: Substring match on name or display_name (case-insensitive).
            category: Filter by category.
            tags: Filter by tags (AND — must have all specified tags).
            required_fields: Filter by required data fields (must have all).

        Returns:
            List of matching IndicatorMeta.
        """
        results = list(self._indicators.values())

        if query:
            q = query.lower()
            results = [m for m in results if q in m.name.lower() or q in m.display_name.lower()]

        if category:
            results = [m for m in results if m.category == category]

        if tags:
            results = [m for m in results if all(t in m.tags for t in tags)]

        if required_fields:
            results = [m for m in results if all(f in m.required_fields for f in required_fields)]

        return sorted(results, key=lambda m: (m.category.value, m.name))

    def search_tag(self, tag: str) -> List[IndicatorMeta]:
        """Find all indicators with a given tag."""
        return [m for m in self._indicators.values() if tag in m.tags]

    # ── Info ──

    @property
    def count(self) -> int:
        return len(self._indicators)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def summary(self) -> str:
        """Return a human-readable summary of registered indicators."""
        lines = [f"Indicator Registry: {self.count} indicators"]
        by_cat: Dict[str, int] = {}
        for m in self._indicators.values():
            by_cat[m.category.value] = by_cat.get(m.category.value, 0) + 1
        for cat, count in sorted(by_cat.items()):
            lines.append(f"  {cat}: {count}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Default Registration (populates registry with all built-in indicators)
# ═══════════════════════════════════════════════════════════════════

def create_default_registry() -> IndicatorRegistry:
    """Create and populate a registry with all built-in technical indicators."""
    from engine.technical.indicators import (
        sma, ema, wma, dema, tema, trima, kama,
        rsi, macd, stochastic, kdj, williams_r, adx, cci, roc, mfi,
        bollinger_bands, bollinger_squeeze,
        obv, vwap, volume_profile, chaikin_money_flow, accumulation_distribution, volume_ratio,
        ichimoku, parabolic_sar, pivot_points, atr,
    )

    from engine.technical.config import TC

    reg = IndicatorRegistry()

    # ── Trend Indicators ──
    for p in TC.sma_periods:
        meta = IndicatorMeta(
            name=f"sma_{p}",
            display_name=f"SMA ({p})",
            category=IndicatorCategory.TREND,
            periods=[p],
            description=f"Simple Moving Average over {p} periods.",
            required_fields=["close"],
            output_names=[f"sma_{p}"],
            complexity=Complexity.LINEAR,
            tags=["trend", "moving_average", "simple"],
        )
        reg.register(meta, lambda data, period=p: sma(data, period))

    for p in TC.ema_periods:
        meta = IndicatorMeta(
            name=f"ema_{p}",
            display_name=f"EMA ({p})",
            category=IndicatorCategory.TREND,
            periods=[p],
            description=f"Exponential Moving Average over {p} periods.",
            required_fields=["close"],
            output_names=[f"ema_{p}"],
            complexity=Complexity.LINEAR,
            tags=["trend", "moving_average", "exponential"],
        )
        reg.register(meta, lambda data, period=p: ema(data, period))

    for p in TC.wma_periods:
        meta = IndicatorMeta(
            name=f"wma_{p}",
            display_name=f"WMA ({p})",
            category=IndicatorCategory.TREND,
            periods=[p],
            description=f"Weighted Moving Average over {p} periods.",
            required_fields=["close"],
            output_names=[f"wma_{p}"],
            complexity=Complexity.LINEAR,
            tags=["trend", "moving_average", "weighted"],
        )
        reg.register(meta, lambda data, period=p: wma(data, period))

    meta = IndicatorMeta(
        name="dema_20",
        display_name="DEMA (20)",
        category=IndicatorCategory.TREND,
        periods=[20],
        description="Double Exponential Moving Average over 20 periods.",
        required_fields=["close"],
        output_names=["dema_20"],
        complexity=Complexity.LINEAR,
        tags=["trend", "moving_average", "double_exponential", "low_lag"],
    )
    reg.register(meta, lambda data: dema(data, 20))

    meta = IndicatorMeta(
        name="tema_20",
        display_name="TEMA (20)",
        category=IndicatorCategory.TREND,
        periods=[20],
        description="Triple Exponential Moving Average over 20 periods.",
        required_fields=["close"],
        output_names=["tema_20"],
        complexity=Complexity.LINEAR,
        tags=["trend", "moving_average", "triple_exponential", "low_lag"],
    )
    reg.register(meta, lambda data: tema(data, 20))

    meta = IndicatorMeta(
        name="trima_20",
        display_name="TRIMA (20)",
        category=IndicatorCategory.TREND,
        periods=[20],
        description="Triangular Moving Average over 20 periods.",
        required_fields=["close"],
        output_names=["trima_20"],
        complexity=Complexity.LINEAR,
        tags=["trend", "moving_average", "triangular"],
    )
    reg.register(meta, lambda data: trima(data, 20))

    meta = IndicatorMeta(
        name="kama",
        display_name="KAMA (10, 2, 30)",
        category=IndicatorCategory.TREND,
        periods=[10],
        params={"fast": 2, "slow": 30},
        description="Kaufman Adaptive Moving Average (adaptive to volatility).",
        required_fields=["close"],
        output_names=["kama"],
        complexity=Complexity.LINEAR,
        tags=["trend", "moving_average", "adaptive", "volatility"],
    )
    reg.register(meta, lambda data: kama(data, 10, 2, 30))

    # ── Momentum Indicators ──
    meta = IndicatorMeta(
        name="rsi_14",
        display_name="RSI (14)",
        category=IndicatorCategory.MOMENTUM,
        periods=[14],
        description="Relative Strength Index — measures momentum and overbought/oversold conditions.",
        required_fields=["close"],
        output_names=["rsi_14"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "oscillator", "overbought", "oversold"],
    )
    reg.register(meta, lambda data: rsi(data, 14))

    meta = IndicatorMeta(
        name="macd",
        display_name="MACD (12, 26, 9)",
        category=IndicatorCategory.MOMENTUM,
        periods=[12, 26, 9],
        description="Moving Average Convergence Divergence with signal line and histogram.",
        required_fields=["close"],
        output_count=3,
        output_names=["macd", "macd_signal", "macd_histogram"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "trend_following", "crossover"],
    )
    reg.register(meta, lambda data: macd(data, 12, 26, 9))

    meta = IndicatorMeta(
        name="stochastic",
        display_name="Stochastic (14, 3, 3)",
        category=IndicatorCategory.MOMENTUM,
        periods=[14, 3],
        description="Stochastic Oscillator: %K and %D.",
        required_fields=["high", "low", "close"],
        output_count=2,
        output_names=["stoch_k", "stoch_d"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "oscillator", "overbought", "oversold"],
    )
    reg.register(meta, lambda h, l, c: stochastic(h, l, c, 14, 3))

    meta = IndicatorMeta(
        name="kdj",
        display_name="KDJ (14, 3, 3)",
        category=IndicatorCategory.MOMENTUM,
        periods=[14, 3],
        description="KDJ Indicator: Stochastic variant with J line.",
        required_fields=["high", "low", "close"],
        output_count=3,
        output_names=["kdj_k", "kdj_d", "kdj_j"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "oscillator", "kdj"],
    )
    reg.register(meta, lambda h, l, c: kdj(h, l, c, 14, 3))

    meta = IndicatorMeta(
        name="williams_r",
        display_name="Williams %R (14)",
        category=IndicatorCategory.MOMENTUM,
        periods=[14],
        description="Williams %R — momentum oscillator ranging from -100 to 0.",
        required_fields=["high", "low", "close"],
        output_names=["williams_r"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "oscillator", "overbought", "oversold"],
    )
    reg.register(meta, lambda h, l, c: williams_r(h, l, c, 14))

    meta = IndicatorMeta(
        name="adx",
        display_name="ADX (14)",
        category=IndicatorCategory.MOMENTUM,
        periods=[14],
        description="Average Directional Index — trend strength (not direction).",
        required_fields=["high", "low", "close"],
        output_count=4,
        output_names=["adx", "plus_di", "minus_di", "dx"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "trend_strength", "directional"],
    )
    reg.register(meta, lambda h, l, c: adx(h, l, c, 14))

    meta = IndicatorMeta(
        name="cci_20",
        display_name="CCI (20)",
        category=IndicatorCategory.MOMENTUM,
        periods=[20],
        description="Commodity Channel Index — identifies cyclical turns.",
        required_fields=["high", "low", "close"],
        output_names=["cci_20"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "oscillator", "cyclical"],
    )
    reg.register(meta, lambda h, l, c: cci(h, l, c, 20))

    meta = IndicatorMeta(
        name="roc_12",
        display_name="ROC (12)",
        category=IndicatorCategory.MOMENTUM,
        periods=[12],
        description="Rate of Change — pure momentum over 12 periods.",
        required_fields=["close"],
        output_names=["roc_12"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "rate_of_change"],
    )
    reg.register(meta, lambda data: roc(data, 12))

    meta = IndicatorMeta(
        name="mfi_14",
        display_name="MFI (14)",
        category=IndicatorCategory.MOMENTUM,
        periods=[14],
        description="Money Flow Index — volume-weighted RSI.",
        required_fields=["high", "low", "close", "volume"],
        output_names=["mfi_14"],
        complexity=Complexity.LINEAR,
        tags=["momentum", "volume_weighted", "overbought", "oversold"],
    )
    reg.register(meta, lambda h, l, c, v: mfi(h, l, c, v, 14))

    # ── Volatility / Bollinger ──
    meta = IndicatorMeta(
        name="bollinger_bands",
        display_name="Bollinger Bands (20, 2)",
        category=IndicatorCategory.VOLATILITY,
        periods=[20],
        params={"num_std": 2.0},
        description="Bollinger Bands: Middle, Upper, Lower, Bandwidth, %B.",
        required_fields=["close"],
        output_count=5,
        output_names=["bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_pct_b"],
        complexity=Complexity.LINEAR,
        tags=["volatility", "bands", "mean_reversion", "squeeze"],
    )
    reg.register(meta, lambda data: bollinger_bands(data, 20, 2.0))

    meta = IndicatorMeta(
        name="bb_squeeze",
        display_name="Bollinger Band Squeeze",
        category=IndicatorCategory.VOLATILITY,
        periods=[125],
        description="Detects Bollinger Band squeeze (narrowest bands in 6 months).",
        required_fields=["close"],
        output_names=["bb_squeeze"],
        complexity=Complexity.LINEAR,
        tags=["volatility", "squeeze", "breakout_signal"],
    )
    reg.register(meta, lambda bandwidth: bollinger_squeeze(bandwidth, 125))

    # ── Volume Indicators ──
    meta = IndicatorMeta(
        name="obv",
        display_name="On-Balance Volume",
        category=IndicatorCategory.VOLUME,
        periods=[],
        description="On-Balance Volume — cumulative volume flow indicator.",
        required_fields=["close", "volume"],
        output_names=["obv"],
        complexity=Complexity.LINEAR,
        tags=["volume", "accumulation", "distribution", "divergence"],
    )
    reg.register(meta, lambda c, v: obv(c, v))

    meta = IndicatorMeta(
        name="vwap",
        display_name="VWAP",
        category=IndicatorCategory.VOLUME,
        periods=[],
        description="Volume-Weighted Average Price.",
        required_fields=["high", "low", "close", "volume"],
        output_names=["vwap"],
        complexity=Complexity.LINEAR,
        tags=["volume", "fair_price", "institutional"],
    )
    reg.register(meta, lambda h, l, c, v: vwap(h, l, c, v))

    meta = IndicatorMeta(
        name="cmf_21",
        display_name="Chaikin Money Flow (21)",
        category=IndicatorCategory.VOLUME,
        periods=[21],
        description="Chaikin Money Flow — accumulation/distribution over 21 periods.",
        required_fields=["high", "low", "close", "volume"],
        output_names=["cmf_21"],
        complexity=Complexity.LINEAR,
        tags=["volume", "money_flow", "accumulation", "distribution"],
    )
    reg.register(meta, lambda h, l, c, v: chaikin_money_flow(h, l, c, v, 21))

    meta = IndicatorMeta(
        name="ad_line",
        display_name="Accumulation/Distribution Line",
        category=IndicatorCategory.VOLUME,
        periods=[],
        description="Accumulation/Distribution Line — cumulative indicator of money flow.",
        required_fields=["high", "low", "close", "volume"],
        output_names=["ad_line"],
        complexity=Complexity.LINEAR,
        tags=["volume", "accumulation", "distribution", "divergence"],
    )
    reg.register(meta, lambda h, l, c, v: accumulation_distribution(h, l, c, v))

    meta = IndicatorMeta(
        name="vol_ratio",
        display_name="Volume Ratio (20)",
        category=IndicatorCategory.VOLUME,
        periods=[20],
        description="Current volume / 20-day average volume ratio.",
        required_fields=["volume"],
        output_names=["vol_ratio"],
        complexity=Complexity.LINEAR,
        tags=["volume", "ratio", "liquidity"],
    )
    reg.register(meta, lambda v: volume_ratio(v, 20))

    # ── Composite Indicators ──
    meta = IndicatorMeta(
        name="ichimoku",
        display_name="Ichimoku Cloud",
        category=IndicatorCategory.COMPOSITE,
        periods=[9, 26, 52],
        params={"displacement": 26},
        description="Ichimoku Kinko Hyo — comprehensive equilibrium chart.",
        required_fields=["high", "low"],
        output_count=5,
        output_names=["ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_chikou"],
        complexity=Complexity.LINEAR,
        tags=["composite", "cloud", "trend", "support_resistance"],
    )
    reg.register(meta, lambda h, l: ichimoku(h, l))

    meta = IndicatorMeta(
        name="parabolic_sar",
        display_name="Parabolic SAR",
        category=IndicatorCategory.COMPOSITE,
        periods=[],
        params={"af_start": 0.02, "af_increment": 0.02, "af_max": 0.20},
        description="Parabolic Stop and Reverse — trailing stop indicator.",
        required_fields=["high", "low"],
        output_names=["parabolic_sar"],
        complexity=Complexity.LINEAR,
        tags=["composite", "stop_loss", "trend_reversal"],
    )
    reg.register(meta, lambda h, l: parabolic_sar(h, l))

    meta = IndicatorMeta(
        name="pivot_points",
        display_name="Pivot Points",
        category=IndicatorCategory.COMPOSITE,
        periods=[],
        description="Standard Pivot Points with R1-R3 and S1-S3.",
        required_fields=["high", "low", "close"],
        output_count=7,
        output_names=["pivot", "r1", "r2", "r3", "s1", "s2", "s3"],
        complexity=Complexity.LINEAR,
        tags=["composite", "support_resistance", "levels"],
    )
    reg.register(meta, lambda h, l, c: pivot_points(h, l, c))

    meta = IndicatorMeta(
        name="atr_14",
        display_name="ATR (14)",
        category=IndicatorCategory.VOLATILITY,
        periods=[14],
        description="Average True Range — volatility measure.",
        required_fields=["high", "low", "close"],
        output_names=["atr_14"],
        complexity=Complexity.LINEAR,
        tags=["volatility", "risk_management", "stop_loss"],
    )
    reg.register(meta, lambda h, l, c: atr(h, l, c, 14))

    # Volume Profile (special — returns dict, not array)
    meta = IndicatorMeta(
        name="volume_profile",
        display_name="Volume Profile",
        category=IndicatorCategory.VOLUME,
        periods=[],
        params={"price_levels": 10},
        description="Volume Profile by price level — POC, VWAP, volume distribution.",
        required_fields=["close", "volume"],
        output_names=["vwap", "poc_price", "poc_volume"],
        complexity=Complexity.LINEAR,
        tags=["volume", "profile", "poc", "market_structure"],
    )
    reg.register(meta, lambda c, v: volume_profile(c, v, 10))

    reg._initialized = True
    logger.info(f"Default registry created: {reg.count} indicators")
    return reg


# ═══════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════

_default_registry: Optional[IndicatorRegistry] = None


def get_registry() -> IndicatorRegistry:
    """Get (or create) the default indicator registry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = create_default_registry()
    return _default_registry
