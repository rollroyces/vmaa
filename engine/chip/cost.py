#!/usr/bin/env python3
"""
VMAA Chip Engine — Cost Basis Analysis
=======================================
Volume-weighted average price (VWAP), cost distribution across price levels,
unrealized P&L by cost bucket, marginal and impulse cost estimation.

All vectorized numpy/pandas — no TA-Lib.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.chip.config import ChipConfig, get_chip_config
from engine.chip.distribution import _extract_ohlcv

logger = logging.getLogger("vmaa.engine.chip.cost")


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CostBucket:
    """Cost basis for a price range."""
    price_low: float
    price_high: float
    price_mid: float
    volume: float              # Total volume in this range
    volume_pct: float          # % of total volume
    vwap: float                # VWAP within this bucket
    avg_cost: float            # Average cost (VWAP) within this bucket
    unrealized_pnl_pct: float  # (current_price - vwap) / vwap
    status: str                # "profitable", "unprofitable"


@dataclass
class CostBasisResult:
    """Complete cost basis analysis for a ticker."""
    ticker: str
    current_price: float

    # Overall VWAP
    vwap: float
    vwap_lookback_days: int

    # Cost distribution
    cost_buckets: List[CostBucket]
    num_buckets: int

    # Cost concentration
    cost_concentration_pct: float   # % of volume in top 3 cost zones
    primary_cost_zone: Tuple[float, float]  # (low, high) of densest zone

    # Marginal & Impulse cost
    marginal_cost: float        # VWAP of last N days (new volume)
    marginal_days: int
    impulse_cost: float         # VWAP of most recent days
    impulse_days: int

    # P&L summary
    profitable_volume_pct: float    # % volume below current price
    unprofitable_volume_pct: float  # % volume above current price
    breakeven_volume_pct: float     # % volume at current price (±1 bin)
    avg_unrealized_pnl_pct: float   # Average unrealized P&L across all volume

    # VWAP signal
    price_vs_vwap: float        # (price - vwap) / vwap
    vwap_signal: str            # "above" / "below" / "at"
    vwap_trend: str             # "rising", "falling", "flat"


# ═══════════════════════════════════════════════════════════════════
# Cost Basis Engine
# ═══════════════════════════════════════════════════════════════════

def compute_vwap(df: pd.DataFrame) -> float:
    """
    Compute Volume-Weighted Average Price (VWAP) over the entire period.

    VWAP = Σ(price_i * volume_i) / Σ(volume_i)
    Uses (high + low + close) / 3 as typical price per bar.

    Args:
        df: OHLCV DataFrame

    Returns:
        VWAP value
    """
    high, low, close, open_, volume = _extract_ohlcv(df)
    typical_price = (high + low + close) / 3.0
    total_vol = volume.sum()

    if total_vol <= 0:
        return 0.0

    return float(np.sum(typical_price * volume) / total_vol)


def build_cost_distribution(
    df: pd.DataFrame,
    current_price: float,
    cfg: Optional[ChipConfig] = None,
    num_bins: int = 20,
) -> Tuple[List[CostBucket], float, float, float, float]:
    """
    Build a cost basis distribution across price levels.

    Each bar's volume is allocated to price bins based on the bar's price range.
    Uses (high+low+close)/3 as typical price per bar.

    Args:
        df: OHLCV DataFrame
        current_price: Current market price
        cfg: ChipConfig
        num_bins: Number of price buckets for cost distribution

    Returns:
        (buckets, profitable_pct, unprofitable_pct, breakeven_pct, avg_pnl_pct)
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places

    high, low, close, open_, volume = _extract_ohlcv(df)

    price_min = np.min(low)
    price_max = np.max(high)
    price_range = price_max - price_min
    if price_range <= 0:
        return [], 0.0, 0.0, 0.0, 0.0

    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volumes = np.zeros(num_bins, dtype=np.float64)
    bin_cost_sum = np.zeros(num_bins, dtype=np.float64)  # Σ(price * vol) for each bin

    typical_price = (high + low + close) / 3.0

    for i in range(len(df)):
        h, l, v, tp = high[i], low[i], volume[i], typical_price[i]
        if v <= 0 or h <= l:
            continue

        bar_range = h - l
        for j in range(num_bins):
            overlap_low = max(bin_edges[j], l)
            overlap_high = min(bin_edges[j + 1], h)
            if overlap_high > overlap_low:
                overlap_pct = (overlap_high - overlap_low) / bar_range
                added_vol = v * overlap_pct
                bin_volumes[j] += added_vol
                bin_cost_sum[j] += tp * added_vol

    total_vol = bin_volumes.sum()
    if total_vol <= 0:
        return [], 0.0, 0.0, 0.0, 0.0

    # Build cost buckets
    cost_buckets = []
    profitable_vol = 0.0
    unprofitable_vol = 0.0
    breakeven_vol = 0.0
    total_pnl_weighted = 0.0

    for j in range(num_bins):
        v = bin_volumes[j]
        v_pct = v / total_vol
        vwap_in_bin = bin_cost_sum[j] / v if v > 0 else bin_mids[j]
        pnl_pct = (current_price - vwap_in_bin) / vwap_in_bin if vwap_in_bin > 0 else 0.0
        status = "profitable" if pnl_pct > 0 else "unprofitable"

        cost_buckets.append(CostBucket(
            price_low=round(float(bin_edges[j]), dec),
            price_high=round(float(bin_edges[j + 1]), dec),
            price_mid=round(float(bin_mids[j]), dec),
            volume=round(float(v), 2),
            volume_pct=round(float(v_pct), dec),
            vwap=round(float(vwap_in_bin), dec),
            avg_cost=round(float(vwap_in_bin), dec),
            unrealized_pnl_pct=round(float(pnl_pct), dec),
            status=status,
        ))

        # Track P&L categories
        if vwap_in_bin < current_price:
            profitable_vol += v
        elif vwap_in_bin > current_price:
            unprofitable_vol += v
        else:
            breakeven_vol += v

        total_pnl_weighted += pnl_pct * v_pct

    profitable_pct = profitable_vol / total_vol
    unprofitable_pct = unprofitable_vol / total_vol
    breakeven_pct = breakeven_vol / total_vol

    return (
        cost_buckets,
        round(float(profitable_pct), dec),
        round(float(unprofitable_pct), dec),
        round(float(breakeven_pct), dec),
        round(float(total_pnl_weighted), dec),
    )


def compute_cost_concentration(
    cost_buckets: List[CostBucket],
    top_n: int = 3,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute cost concentration — % of volume in top N cost zones.

    Args:
        cost_buckets: Cost distribution buckets
        top_n: Number of top zones

    Returns:
        (concentration_pct, (primary_zone_low, primary_zone_high))
    """
    if not cost_buckets:
        return 0.0, (0.0, 0.0)

    sorted_buckets = sorted(cost_buckets, key=lambda b: b.volume_pct, reverse=True)
    top_vol_pct = sum(b.volume_pct for b in sorted_buckets[:top_n])
    primary = sorted_buckets[0]
    return (
        round(top_vol_pct, 4),
        (primary.price_low, primary.price_high),
    )


def analyze_cost_basis(
    df: pd.DataFrame,
    current_price: float,
    ticker: str = "",
    cfg: Optional[ChipConfig] = None,
) -> CostBasisResult:
    """
    Full cost basis analysis.

    Computes VWAP, cost distribution, marginal/impulse cost,
    and unrealized P&L analysis.

    Args:
        df: OHLCV DataFrame
        current_price: Current market price
        ticker: Ticker symbol
        cfg: ChipConfig

    Returns:
        CostBasisResult
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places

    if len(df) < cfg.min_data_points:
        logger.warning(f"Only {len(df)} data points for {ticker} — need ≥{cfg.min_data_points}")

    high, low, close, open_, volume = _extract_ohlcv(df)
    typical_price = (high + low + close) / 3.0

    # Full-period VWAP
    vwap = compute_vwap(df)

    # Cost distribution using full lookback
    lookback_df = df.tail(cfg.cost_lookback_days)
    num_bins = min(30, max(10, len(lookback_df) // 5))
    cost_buckets, profitable_pct, unprofitable_pct, breakeven_pct, avg_pnl = \
        build_cost_distribution(lookback_df, current_price, cfg, num_bins)

    # Cost concentration
    conc_pct, primary_zone = compute_cost_concentration(cost_buckets)

    # Marginal cost — VWAP of last N days (where new volume enters)
    marginal_df = df.tail(cfg.cost_marginal_days)
    marginal_cost = compute_vwap(marginal_df)

    # Impulse cost — VWAP of most recent days
    impulse_df = df.tail(cfg.cost_impulse_days)
    impulse_cost = compute_vwap(impulse_df)

    # Price vs VWAP signal
    price_vs_vwap = (current_price - vwap) / vwap if vwap > 0 else 0.0
    if price_vs_vwap > 0.02:
        vwap_signal = "above"
    elif price_vs_vwap < -0.02:
        vwap_signal = "below"
    else:
        vwap_signal = "at"

    # VWAP trend (using 20-period rolling VWAP)
    if len(df) >= 40:
        roll_vwap_recent = np.array([
            np.average(typical_price[i-20:i], weights=volume[i-20:i])
            if volume[i-20:i].sum() > 0 else 0
            for i in range(max(20, len(df) - 20), len(df))
        ])
        if len(roll_vwap_recent) >= 2:
            vwap_slope = (roll_vwap_recent[-1] - roll_vwap_recent[0]) / roll_vwap_recent[0] if roll_vwap_recent[0] > 0 else 0
            if vwap_slope > 0.01:
                vwap_trend = "rising"
            elif vwap_slope < -0.01:
                vwap_trend = "falling"
            else:
                vwap_trend = "flat"
        else:
            vwap_trend = "flat"
    else:
        vwap_trend = "flat"

    return CostBasisResult(
        ticker=ticker,
        current_price=round(current_price, dec),
        vwap=round(vwap, dec),
        vwap_lookback_days=cfg.cost_lookback_days,
        cost_buckets=cost_buckets,
        num_buckets=len(cost_buckets),
        cost_concentration_pct=round(conc_pct, dec),
        primary_cost_zone=primary_zone,
        marginal_cost=round(marginal_cost, dec),
        marginal_days=cfg.cost_marginal_days,
        impulse_cost=round(impulse_cost, dec),
        impulse_days=cfg.cost_impulse_days,
        profitable_volume_pct=round(profitable_pct * 100, dec),
        unprofitable_volume_pct=round(unprofitable_pct * 100, dec),
        breakeven_volume_pct=round(breakeven_pct * 100, dec),
        avg_unrealized_pnl_pct=round(avg_pnl * 100, dec),
        price_vs_vwap=round(price_vs_vwap, dec),
        vwap_signal=vwap_signal,
        vwap_trend=vwap_trend,
    )
