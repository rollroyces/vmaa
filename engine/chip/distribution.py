#!/usr/bin/env python3
"""
VMAA Chip Engine — Volume Distribution
========================================
Volume Profile analysis: price-level volume accumulation, Value Area,
Point of Control (POC), and Relative Volume (RVOL).

All operations are vectorized via numpy/pandas — no TA-Lib.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.chip.config import ChipConfig, get_chip_config

logger = logging.getLogger("vmaa.engine.chip.distribution")


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VolumeProfileBucket:
    """A single bucket (price bin) in the volume profile."""
    price_low: float
    price_high: float
    price_mid: float
    volume: float        # Total volume traded in this range
    volume_pct: float    # % of total profile volume
    trades: int          # Number of bars touching this range


@dataclass
class ValueArea:
    """The Value Area (VA) — price zone containing ~70% of volume."""
    vah: float           # Value Area High (VAH)
    val: float           # Value Area Low (VAL)
    poc: float           # Point of Control (max volume price)
    poc_volume: float    # Volume at POC
    total_volume: float  # Total volume in the profile
    va_volume_pct: float # Actual % of volume inside VA


@dataclass
class RVOLResult:
    """Relative Volume analysis result."""
    current_volume: float
    avg_volume_20d: float
    avg_volume_50d: float
    rvol_20d: float      # current / avg_20d
    rvol_50d: float      # current / avg_50d
    volume_trend: str    # "high", "normal", "low"


@dataclass
class VolumeDistributionResult:
    """Complete volume distribution analysis for a ticker."""
    ticker: str
    period: str
    data_points: int
    current_price: float
    price_range: Tuple[float, float]
    num_bins: int

    # Volume profile
    value_area: ValueArea
    poc: float
    vah: float
    val: float
    buckets: List[VolumeProfileBucket]

    # Relative volume
    rvol: RVOLResult

    # Time-weighted (decay-weighted) profile
    tw_poc: float
    tw_vah: float
    tw_val: float

    # Summary
    profile_type: str    # 'single-distribution', 'bimodal', 'multi-modal'
    volume_skew: float   # positive = more volume at higher prices


# ═══════════════════════════════════════════════════════════════════
# Volume Profile Builder
# ═══════════════════════════════════════════════════════════════════

def _extract_ohlcv(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract OHLCV arrays from DataFrame (case-insensitive)."""
    col_map = {"high": ["High", "high"], "low": ["Low", "low"],
               "close": ["Close", "close", "Adj Close", "Adj Close"],
               "volume": ["Volume", "volume"]}
    results = {}
    for key, candidates in col_map.items():
        for c in candidates:
            if c in df.columns:
                results[key] = df[c].to_numpy(dtype=np.float64)
                break
        else:
            for c in df.columns:
                if c.lower() == key:
                    results[key] = df[c].to_numpy(dtype=np.float64)
                    break
            if key not in results:
                raise KeyError(f"Cannot find '{key}' column in DataFrame")
    return (results["high"], results["low"], results["close"],
            np.zeros(len(df), dtype=np.float64) if "open" not in {c.lower() for c in df.columns} else results.get("open", results["close"]),
            results["volume"])


def build_volume_profile(
    df: pd.DataFrame,
    cfg: Optional[ChipConfig] = None,
    price_bins: Optional[int] = None,
    value_area_pct: Optional[float] = None,
) -> Tuple[List[VolumeProfileBucket], ValueArea, float]:
    """
    Build a Volume Profile from OHLCV DataFrame.

    Each bar's volume is distributed across the high-low range using
    a simple proportional allocation: volume / (high - low) per unit price.

    Args:
        df: OHLCV DataFrame
        cfg: ChipConfig (optional, uses singleton if None)
        price_bins: Override number of bins
        value_area_pct: Override VA percentage threshold

    Returns:
        (buckets, value_area, volume_skew)
    """
    if cfg is None:
        cfg = get_chip_config()

    n_bins = price_bins or cfg.volume_profile_bins
    va_pct = value_area_pct or cfg.value_area_pct
    dec = cfg.decimal_places

    high, low, close, open_, volume = _extract_ohlcv(df)

    # Determine price range
    price_min = np.min(low)
    price_max = np.max(high)
    price_range = price_max - price_min
    if price_range <= 0:
        logger.warning("Zero price range — returning empty profile")
        return [], ValueArea(0, 0, 0, 0, 0, 0), 0.0

    # Create price bins
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volumes = np.zeros(n_bins, dtype=np.float64)
    bin_trades = np.zeros(n_bins, dtype=np.int64)

    # Distribute volume across the high-low range per bar
    # For each bar, we allocate volume proportionally to price bins
    for i in range(len(df)):
        h, l, v = high[i], low[i], volume[i]
        if v <= 0 or h <= l:
            continue

        bar_range = h - l
        if bar_range == 0:
            # Single price bar — assign to exact bin
            bin_idx = np.searchsorted(bin_edges, h, side="right") - 1
            bin_idx = max(0, min(n_bins - 1, bin_idx))
            bin_volumes[bin_idx] += v
            bin_trades[bin_idx] += 1
            continue

        # Find overlapping bins for this bar's range
        for j in range(n_bins):
            bin_low = bin_edges[j]
            bin_high = bin_edges[j + 1]
            # Overlap between [bin_low, bin_high] and [l, h]
            overlap_low = max(bin_low, l)
            overlap_high = min(bin_high, h)
            if overlap_high > overlap_low:
                overlap_pct = (overlap_high - overlap_low) / bar_range
                bin_volumes[j] += v * overlap_pct
                bin_trades[j] += 1

    total_volume = bin_volumes.sum()
    if total_volume <= 0:
        logger.warning("Zero total volume — returning empty profile")
        return [], ValueArea(0, 0, 0, 0, 0, 0), 0.0

    # Build buckets
    buckets = []
    for j in range(n_bins):
        buckets.append(VolumeProfileBucket(
            price_low=round(float(bin_edges[j]), dec),
            price_high=round(float(bin_edges[j + 1]), dec),
            price_mid=round(float(bin_mids[j]), dec),
            volume=round(float(bin_volumes[j]), 2),
            volume_pct=round(float(bin_volumes[j] / total_volume), dec),
            trades=int(bin_trades[j]),
        ))

    # Find POC
    poc_idx = int(np.argmax(bin_volumes))
    poc = float(bin_mids[poc_idx])
    poc_volume = float(bin_volumes[poc_idx])

    # Compute Value Area (VA) — the price range containing va_pct of total volume
    vah, val = _compute_value_area(bin_volumes, bin_mids, total_volume, va_pct)

    # Volume skew: positive = more volume at higher prices (relative to midpoint)
    mid_price = (price_min + price_max) / 2.0
    weighted_price = np.average(bin_mids, weights=bin_volumes) if total_volume > 0 else mid_price
    volume_skew = (weighted_price - mid_price) / price_range if price_range > 0 else 0.0

    va = ValueArea(
        vah=round(vah, dec),
        val=round(val, dec),
        poc=round(poc, dec),
        poc_volume=round(poc_volume, 2),
        total_volume=round(total_volume, 2),
        va_volume_pct=round(va_pct, dec),
    )

    return buckets, va, round(float(volume_skew), dec)


def _compute_value_area(
    bin_volumes: np.ndarray,
    bin_mids: np.ndarray,
    total_volume: float,
    target_pct: float,
) -> Tuple[float, float]:
    """Compute Value Area High and Low by expanding from POC."""
    poc_idx = int(np.argmax(bin_volumes))
    target_vol = total_volume * target_pct

    left = poc_idx
    right = poc_idx
    accumulated = float(bin_volumes[poc_idx])

    while accumulated < target_vol:
        # Expand toward the side with more volume
        vol_left = float(bin_volumes[left - 1]) if left > 0 else -1.0
        vol_right = float(bin_volumes[right + 1]) if right < len(bin_volumes) - 1 else -1.0

        if vol_left < 0 and vol_right < 0:
            break
        elif vol_left > vol_right:
            left -= 1
            accumulated += vol_left
        elif vol_right > 0:
            right += 1
            accumulated += vol_right
        elif vol_left > 0:
            left -= 1
            accumulated += vol_left
        else:
            break

    vah = float(bin_mids[right]) if right < len(bin_mids) else float(bin_mids[-1])
    val = float(bin_mids[left]) if left >= 0 else float(bin_mids[0])

    return vah, val


def compute_rvol(df: pd.DataFrame, cfg: Optional[ChipConfig] = None) -> RVOLResult:
    """
    Compute Relative Volume (RVOL) for the most recent bar.

    RVOL = current_volume / average_volume_over_period
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places

    _, _, _, _, volume = _extract_ohlcv(df)

    if len(volume) < 51:
        logger.warning(f"Only {len(volume)} data points — insufficient for RVOL")
        return RVOLResult(0.0, 0.0, 0.0, 0.0, 0.0, "normal")

    current_vol = float(volume[-1])
    avg_20 = float(np.mean(volume[-21:-1])) if len(volume) >= 21 else float(np.mean(volume[:-1]))
    avg_50 = float(np.mean(volume[-51:-1])) if len(volume) >= 51 else avg_20

    rvol_20 = current_vol / avg_20 if avg_20 > 0 else 1.0
    rvol_50 = current_vol / avg_50 if avg_50 > 0 else 1.0

    # Volume trend classification
    if rvol_20 > 1.5:
        trend = "high"
    elif rvol_20 < 0.5:
        trend = "low"
    else:
        trend = "normal"

    return RVOLResult(
        current_volume=round(current_vol, 2),
        avg_volume_20d=round(avg_20, 2),
        avg_volume_50d=round(avg_50, 2),
        rvol_20d=round(rvol_20, dec),
        rvol_50d=round(rvol_50, dec),
        volume_trend=trend,
    )


def build_time_weighted_profile(
    df: pd.DataFrame,
    cfg: Optional[ChipConfig] = None,
    half_life: int = 63,
) -> Tuple[List[VolumeProfileBucket], float, float, float]:
    """
    Build a time-weighted Volume Profile.

    Recent bars get higher weight using exponential decay:
        weight_i = exp(-λ * (N - i)), where λ = ln(2) / half_life

    Args:
        df: OHLCV DataFrame
        cfg: ChipConfig
        half_life: Half-life in trading days (default 63 = ~1 quarter)

    Returns:
        (buckets, tw_poc, tw_vah, tw_val)
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places
    va_pct = cfg.value_area_pct
    n_bins = cfg.volume_profile_bins

    high, low, close, open_, volume = _extract_ohlcv(df)
    N = len(df)

    # Exponential decay weights
    lam = np.log(2) / half_life
    time_weights = np.exp(-lam * np.arange(N)[::-1])
    time_weights = time_weights / time_weights.sum()  # normalize

    price_min = np.min(low)
    price_max = np.max(high)
    price_range = price_max - price_min
    if price_range <= 0:
        return [], 0.0, 0.0, 0.0

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volumes = np.zeros(n_bins, dtype=np.float64)

    for i in range(N):
        h, l, v = high[i], low[i], volume[i]
        w = time_weights[i]
        if v <= 0 or h <= l:
            continue

        weighted_vol = v * w
        bar_range = h - l
        if bar_range == 0:
            bin_idx = np.searchsorted(bin_edges, h, side="right") - 1
            bin_idx = max(0, min(n_bins - 1, bin_idx))
            bin_volumes[bin_idx] += weighted_vol
            continue

        for j in range(n_bins):
            overlap_low = max(bin_edges[j], l)
            overlap_high = min(bin_edges[j + 1], h)
            if overlap_high > overlap_low:
                overlap_pct = (overlap_high - overlap_low) / bar_range
                bin_volumes[j] += weighted_vol * overlap_pct

    total_volume = bin_volumes.sum()
    if total_volume <= 0:
        return [], 0.0, 0.0, 0.0

    buckets = []
    for j in range(n_bins):
        buckets.append(VolumeProfileBucket(
            price_low=round(float(bin_edges[j]), dec),
            price_high=round(float(bin_edges[j + 1]), dec),
            price_mid=round(float(bin_mids[j]), dec),
            volume=round(float(bin_volumes[j]), 2),
            volume_pct=round(float(bin_volumes[j] / total_volume), dec),
            trades=0,
        ))

    tw_poc = float(bin_mids[int(np.argmax(bin_volumes))])
    tw_vah, tw_val = _compute_value_area(bin_volumes, bin_mids, total_volume, va_pct)

    return buckets, round(tw_poc, dec), round(tw_vah, dec), round(tw_val, dec)


def classify_profile_type(buckets: List[VolumeProfileBucket], current_price: float) -> str:
    """
    Classify the volume profile shape.

    Heuristic:
    - 'single-distribution': POC in VA, VA spans <40% of total range
    - 'bimodal': Two distinct volume peaks with >15% of range apart
    - 'multi-modal': 3+ distinct peaks

    Falls back to 'single-distribution' for insufficient data.
    """
    if len(buckets) < 5:
        return "single-distribution"

    volumes = np.array([b.volume for b in buckets], dtype=np.float64)
    total_vol = volumes.sum()
    if total_vol <= 0:
        return "single-distribution"

    # Find peaks: local maxima in smoothed volume
    # Simple smoothing with 3-bin window
    from scipy.ndimage import maximum_filter
    try:
        smoothed = np.convolve(volumes, np.ones(3)/3, mode='same')
    except Exception:
        smoothed = volumes

    # Find peaks where smoothed[i] > neighbors
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            # Must have meaningful volume (>5% of max)
            if volumes[i] > 0.05 * volumes.max():
                peaks.append(i)

    num_significant_peaks = len(peaks)
    overall_range = buckets[-1].price_high - buckets[0].price_low

    if num_significant_peaks >= 3:
        # Check if peaks are spread out
        peak_prices = [buckets[p].price_mid for p in peaks]
        min_peak, max_peak = min(peak_prices), max(peak_prices)
        spread = (max_peak - min_peak) / overall_range if overall_range > 0 else 0
        if spread > 0.15:
            return "multi-modal"
        return "single-distribution"
    elif num_significant_peaks == 2:
        # Two peaks — check separation
        p1 = buckets[peaks[0]].price_mid
        p2 = buckets[peaks[1]].price_mid
        separation = abs(p2 - p1) / overall_range if overall_range > 0 else 0
        if separation > 0.15:
            return "bimodal"
        return "single-distribution"

    return "single-distribution"


def analyze_distribution(
    df: pd.DataFrame,
    ticker: str = "",
    period: str = "",
    cfg: Optional[ChipConfig] = None,
) -> VolumeDistributionResult:
    """
    Full volume distribution analysis for a ticker.

    Combines Volume Profile, RVOL, and time-weighted profile
    into a single comprehensive result.

    Args:
        df: OHLCV DataFrame
        ticker: Ticker symbol
        period: Data period description
        cfg: ChipConfig

    Returns:
        VolumeDistributionResult with all distribution metrics
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places

    if len(df) < cfg.min_data_points:
        logger.warning(f"Only {len(df)} data points for {ticker} — need ≥{cfg.min_data_points}")

    high, low, close, open_, volume = _extract_ohlcv(df)
    current_price = float(close[-1]) if len(close) > 0 else 0.0
    price_min = float(np.min(low))
    price_max = float(np.max(high))

    # Standard Volume Profile
    buckets, value_area, volume_skew = build_volume_profile(df, cfg)

    # RVOL
    rvol = compute_rvol(df, cfg)

    # Time-weighted profile
    tw_buckets, tw_poc, tw_vah, tw_val = build_time_weighted_profile(df, cfg)

    # Profile type
    profile_type = classify_profile_type(buckets, current_price)

    return VolumeDistributionResult(
        ticker=ticker,
        period=period,
        data_points=len(df),
        current_price=round(current_price, dec),
        price_range=(round(price_min, dec), round(price_max, dec)),
        num_bins=len(buckets),
        value_area=value_area,
        poc=value_area.poc,
        vah=value_area.vah,
        val=value_area.val,
        buckets=buckets,
        rvol=rvol,
        tw_poc=tw_poc,
        tw_vah=tw_vah,
        tw_val=tw_val,
        profile_type=profile_type,
        volume_skew=volume_skew,
    )
