#!/usr/bin/env python3
"""
VMAA Chip Engine — Concentration Analysis
==========================================
Concentration metrics: CR(N), HHI, volume skew, cost basis distribution,
support/resistance detection from volume data.

All vectorized numpy/pandas — no TA-Lib.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.chip.config import ChipConfig, get_chip_config
from engine.chip.distribution import VolumeProfileBucket, _extract_ohlcv

logger = logging.getLogger("vmaa.engine.chip.concentration")


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ConcentrationResult:
    """Concentration analysis for a ticker."""
    ticker: str

    # CR(N) — top N price levels
    cr_top_n: int
    cr_value: float             # 0-1, higher = more concentrated
    cr_buckets: List[Dict]      # Top N buckets with prices and volume %

    # HHI (Herfindahl Index)
    hhi: float                   # Raw HHI
    hhi_normalized: float        # Normalized HHI (0-1)
    hhi_interpretation: str      # "low", "moderate", "high" concentration

    # Volume skew
    volume_skewness: float       # Positive = volume at higher prices
    # Additional detail
    volume_above_poc_pct: float  # % of volume above POC
    volume_below_poc_pct: float  # % of volume below POC

    # Cost basis estimate
    avg_cost_estimate: float     # VWAP-based average holder cost
    median_cost_estimate: float   # Median of volume distribution

    # Profit ratio for current holders
    current_price: float
    profit_ratio: float          # (price - avg_cost) / avg_cost
    holder_status: str           # "in_profit", "at_cost", "underwater"


@dataclass
class SupportResistanceLevel:
    """A support or resistance zone derived from volume data."""
    price: float
    type: str                    # "support" or "resistance"
    volume: float                # Volume at this level
    volume_pct: float            # % of total volume
    strength: str                # "weak", "moderate", "strong"
    description: str


@dataclass
class SupportResistanceResult:
    """Complete S/R analysis from volume profile."""
    ticker: str
    current_price: float
    nearest_support: Optional[SupportResistanceLevel]
    nearest_resistance: Optional[SupportResistanceLevel]
    levels: List[SupportResistanceLevel]
    volume_weighted_support: float  # Regression-based estimate
    volume_weighted_resistance: float


# ═══════════════════════════════════════════════════════════════════
# Concentration Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_concentration(
    buckets: List[VolumeProfileBucket],
    current_price: float,
    ticker: str = "",
    cfg: Optional[ChipConfig] = None,
) -> ConcentrationResult:
    """
    Compute concentration metrics from Volume Profile buckets.

    Args:
        buckets: Volume profile buckets from distribution module
        current_price: Current price
        ticker: Ticker symbol
        cfg: ChipConfig

    Returns:
        ConcentrationResult
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places
    top_n = cfg.concentration_top_n

    if not buckets:
        logger.warning(f"No buckets for concentration analysis of {ticker}")
        return ConcentrationResult(
            ticker=ticker, cr_top_n=top_n, cr_value=0.0, cr_buckets=[],
            hhi=0.0, hhi_normalized=0.0, hhi_interpretation="low",
            volume_skewness=0.0, volume_above_poc_pct=0.0, volume_below_poc_pct=0.0,
            avg_cost_estimate=current_price, median_cost_estimate=current_price,
            current_price=current_price, profit_ratio=0.0, holder_status="at_cost",
        )

    volumes = np.array([b.volume for b in buckets], dtype=np.float64)
    prices = np.array([b.price_mid for b in buckets], dtype=np.float64)
    total_vol = volumes.sum()

    if total_vol <= 0:
        return ConcentrationResult(
            ticker=ticker, cr_top_n=top_n, cr_value=0.0, cr_buckets=[],
            hhi=0.0, hhi_normalized=0.0, hhi_interpretation="low",
            volume_skewness=0.0, volume_above_poc_pct=0.0, volume_below_poc_pct=0.0,
            avg_cost_estimate=current_price, median_cost_estimate=current_price,
            current_price=current_price, profit_ratio=0.0, holder_status="at_cost",
        )

    vol_pcts = volumes / total_vol

    # ── CR(N): top N price levels as % of total volume ──
    sorted_indices = np.argsort(volumes)[::-1]  # descending
    top_indices = sorted_indices[:top_n]
    cr_value = float(vol_pcts[top_indices].sum())
    cr_buckets = []
    for idx in top_indices:
        cr_buckets.append({
            "price_low": buckets[idx].price_low,
            "price_high": buckets[idx].price_high,
            "price_mid": buckets[idx].price_mid,
            "volume_pct": round(float(vol_pcts[idx]) * 100, dec),
        })

    # ── HHI: Σ(share_i)² ──
    hhi_raw = float(np.sum(vol_pcts ** 2))
    N = len(buckets)
    if cfg.hhi_normalization and N > 1:
        # Normalize: (HHI - 1/N) / (1 - 1/N) → 0-1 range
        hhi_norm = (hhi_raw - 1.0 / N) / (1.0 - 1.0 / N)
        hhi_norm = max(0.0, min(1.0, hhi_norm))
    else:
        hhi_norm = hhi_raw

    if hhi_norm < 0.15:
        hhi_interp = "low"
    elif hhi_norm < 0.25:
        hhi_interp = "moderate"
    else:
        hhi_interp = "high"

    # ── Volume skewness ──
    # Weighted skew: (price - weighted_mean)³ * volume
    weighted_mean = np.average(prices, weights=volumes)
    deviations = prices - weighted_mean
    vol_skew_unadj = np.average(deviations ** 3, weights=volumes)
    vol_std = np.sqrt(np.average(deviations ** 2, weights=volumes))
    if vol_std > 0:
        volume_skewness = vol_skew_unadj / (vol_std ** 3)
    else:
        volume_skewness = 0.0

    # ── Volume above/below POC ──
    poc_idx = int(np.argmax(volumes))
    poc_price = prices[poc_idx]
    vol_above = vol_pcts[prices > poc_price].sum()
    vol_below = vol_pcts[prices < poc_price].sum()

    # ── Cost basis estimate ──
    avg_cost = float(np.average(prices, weights=volumes))
    # Median: price at 50th percentile of cumulative volume
    cum_vol = np.cumsum(vol_pcts)
    median_idx = int(np.searchsorted(cum_vol, 0.5))
    median_cost = float(prices[min(median_idx, len(prices) - 1)])

    # ── Profit ratio ──
    profit_ratio = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
    if profit_ratio > 0.05:
        holder_status = "in_profit"
    elif profit_ratio < -0.05:
        holder_status = "underwater"
    else:
        holder_status = "at_cost"

    return ConcentrationResult(
        ticker=ticker,
        cr_top_n=top_n,
        cr_value=round(cr_value, dec),
        cr_buckets=cr_buckets,
        hhi=round(hhi_raw, dec + 2),
        hhi_normalized=round(hhi_norm, dec),
        hhi_interpretation=hhi_interp,
        volume_skewness=round(float(volume_skewness), dec),
        volume_above_poc_pct=round(float(vol_above) * 100, dec),
        volume_below_poc_pct=round(float(vol_below) * 100, dec),
        avg_cost_estimate=round(avg_cost, dec),
        median_cost_estimate=round(median_cost, dec),
        current_price=round(current_price, dec),
        profit_ratio=round(profit_ratio, dec),
        holder_status=holder_status,
    )


# ═══════════════════════════════════════════════════════════════════
# Support / Resistance Detection
# ═══════════════════════════════════════════════════════════════════

def _classify_sr(
    price: float,
    current_price: float,
    volume_pct: float,
) -> Tuple[str, str]:
    """Classify a level as support or resistance with strength."""
    if price <= current_price:
        sr_type = "support"
    else:
        sr_type = "resistance"

    # Strength based on volume concentration
    if volume_pct > 5.0:
        strength = "strong"
    elif volume_pct > 2.5:
        strength = "moderate"
    else:
        strength = "weak"

    return sr_type, strength


def _cluster_sr_levels(
    buckets: List[VolumeProfileBucket],
    min_gap_pct: float,
    current_price: float,
) -> List[Tuple[float, float, List[VolumeProfileBucket]]]:
    """
    Cluster nearby high-volume buckets into S/R zones.
    Returns list of (cluster_price, total_volume_pct, contributing_buckets).
    """
    if not buckets:
        return []

    # Filter to buckets with meaningful volume (>1% of total)
    vol_pcts = np.array([b.volume_pct * 100 for b in buckets])
    max_pct = vol_pcts.max()
    threshold = max(0.5, max_pct * 0.1)  # at least 0.5% or 10% of max

    significant = [(b, b.volume_pct * 100) for b in buckets if b.volume_pct * 100 >= threshold]
    if not significant:
        return []

    # Sort by volume descending
    significant.sort(key=lambda x: x[1], reverse=True)

    clusters = []  # list of (center_price, total_pct, [buckets])
    overall_range = buckets[-1].price_high - buckets[0].price_low
    if overall_range <= 0:
        overall_range = 1.0

    for bucket, pct in significant:
        placed = False
        for i, (c_price, c_vol, c_buckets) in enumerate(clusters):
            # Check if within gap threshold
            gap = abs(bucket.price_mid - c_price) / overall_range
            if gap < min_gap_pct:
                # Merge into cluster, weight by volume
                new_center = (c_price * c_vol + bucket.price_mid * pct) / (c_vol + pct)
                clusters[i] = (new_center, c_vol + pct, c_buckets + [bucket])
                placed = True
                break
        if not placed:
            clusters.append((bucket.price_mid, pct, [bucket]))

    return clusters


def detect_support_resistance(
    buckets: List[VolumeProfileBucket],
    current_price: float,
    ticker: str = "",
    cfg: Optional[ChipConfig] = None,
) -> SupportResistanceResult:
    """
    Detect support and resistance levels from volume profile.

    High-volume price zones = natural support/resistance levels
    because they represent prices where many shares changed hands.

    Args:
        buckets: Volume profile buckets
        current_price: Current price
        ticker: Ticker symbol
        cfg: ChipConfig

    Returns:
        SupportResistanceResult
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places

    if not buckets:
        return SupportResistanceResult(
            ticker=ticker, current_price=current_price,
            nearest_support=None, nearest_resistance=None, levels=[],
            volume_weighted_support=current_price,
            volume_weighted_resistance=current_price,
        )

    # Cluster high-volume levels into S/R zones
    clusters = _cluster_sr_levels(buckets, cfg.sr_min_cluster_gap_pct, current_price)

    # Build S/R levels
    levels = []
    for c_price, c_vol, c_buckets in sorted(clusters, key=lambda x: x[1], reverse=True)[:cfg.sr_top_clusters]:
        sr_type, strength = _classify_sr(c_price, current_price, c_vol)
        # Description for the zone
        contributing_prices = [b.price_mid for b in c_buckets]
        price_zone = f"{min(contributing_prices):.2f}-{max(contributing_prices):.2f}" if len(c_buckets) > 1 else f"{c_price:.2f}"

        levels.append(SupportResistanceLevel(
            price=round(c_price, dec),
            type=sr_type,
            volume=round(c_vol / 100 * sum(b.volume for b in buckets), 2),
            volume_pct=round(c_vol, dec),
            strength=strength,
            description=f"Volume {sr_type} zone at {price_zone} ({c_vol:.1f}% of volume)",
        ))

    # Find nearest support and resistance
    supports = [l for l in levels if l.type == "support"]
    resistances = [l for l in levels if l.type == "resistance"]

    nearest_support = max(supports, key=lambda l: l.price) if supports else None
    nearest_resistance = min(resistances, key=lambda l: l.price) if resistances else None

    # Volume-weighted support/resistance via regression
    volumes = np.array([b.volume for b in buckets], dtype=np.float64)
    prices = np.array([b.price_mid for b in buckets], dtype=np.float64)

    # Support: weighted average of prices below current
    below_mask = prices <= current_price
    if below_mask.any() and volumes[below_mask].sum() > 0:
        vw_support = float(np.average(prices[below_mask], weights=volumes[below_mask]))
    else:
        vw_support = current_price

    # Resistance: weighted average of prices above current
    above_mask = prices > current_price
    if above_mask.any() and volumes[above_mask].sum() > 0:
        vw_resistance = float(np.average(prices[above_mask], weights=volumes[above_mask]))
    else:
        vw_resistance = current_price

    return SupportResistanceResult(
        ticker=ticker,
        current_price=round(current_price, dec),
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        levels=levels,
        volume_weighted_support=round(vw_support, dec),
        volume_weighted_resistance=round(vw_resistance, dec),
    )
