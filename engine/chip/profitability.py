#!/usr/bin/env python3
"""
VMAA Chip Engine — Profitability Analysis
==========================================
Holder profitability: floating P&L, money flow analysis, Chaikin Money Flow (CMF),
and profitability distribution across cost levels.

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

logger = logging.getLogger("vmaa.engine.chip.profitability")


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MoneyFlowResult:
    """Money flow analysis for a ticker."""
    ticker: str

    # Chaikin Money Flow (CMF)
    cmf: float                      # 21-period CMF
    cmf_period: int
    cmf_signal: str                 # "accumulation", "distribution", "neutral"

    # Raw money flow metrics
    total_inflow: float             # Total positive money flow (raw)
    total_outflow: float            # Total negative money flow (raw)
    net_money_flow: float           # Net (in - out)
    money_flow_ratio: float         # inflow / outflow

    # Recent money flow (last 5d)
    recent_inflow: float
    recent_outflow: float
    recent_net_flow: float
    recent_flow_signal: str         # "buying", "selling", "neutral"

    # Money flow trend
    flow_trend: str                 # "increasing_inflow", "increasing_outflow", "balanced"


@dataclass
class ProfitabilitySummary:
    """Summary of holder profitability."""
    ticker: str
    current_price: float
    avg_cost: float

    # Profit ratios
    total_profit_ratio: float       # (price - avg_cost) / avg_cost
    profitable_volume_pct: float    # % of volume below current price
    underwater_volume_pct: float    # % of volume above current price

    # Floating P&L estimate
    floating_pnl: float             # Estimated unrealized P&L (price - avg_cost) * total volume
    floating_pnl_pct: float         # As percentage of total investment

    # Liquidity-weighted metrics
    liquidity_weighted_pnl: float   # P&L weighted by volume at each level
    effective_buying_pressure: float  # Positive → buying pressure, negative → selling

    # Holder status breakdown
    pct_in_profit: float            # % of holders in profit
    pct_at_breakeven: float         # % of holders at breakeven
    pct_underwater: float           # % of holders underwater

    # Overall assessment
    holder_health: str              # "healthy", "mixed", "distressed"


@dataclass
class ProfitabilityResult:
    """Complete profitability analysis."""
    ticker: str
    summary: ProfitabilitySummary
    money_flow: MoneyFlowResult


# ═══════════════════════════════════════════════════════════════════
# Money Flow Analysis
# ═══════════════════════════════════════════════════════════════════

def compute_money_flow(
    df: pd.DataFrame,
    ticker: str = "",
    cfg: Optional[ChipConfig] = None,
) -> MoneyFlowResult:
    """
    Compute money flow metrics from OHLCV data.

    Money Flow Multiplier = ((close - low) - (high - close)) / (high - low)
    = (2*close - low - high) / (high - low)

    Money Flow Volume = MFM * Volume

    Chaikin Money Flow (CMF) = Σ(MFV over N) / Σ(Volume over N)

    Args:
        df: OHLCV DataFrame
        ticker: Ticker symbol
        cfg: ChipConfig

    Returns:
        MoneyFlowResult
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places

    high, low, close, open_, volume = _extract_ohlcv(df)

    N = len(df)
    if N < 5:
        return MoneyFlowResult(
            ticker=ticker, cmf=0.0, cmf_period=cfg.cmf_period,
            cmf_signal="neutral", total_inflow=0.0, total_outflow=0.0,
            net_money_flow=0.0, money_flow_ratio=1.0,
            recent_inflow=0.0, recent_outflow=0.0, recent_net_flow=0.0,
            recent_flow_signal="neutral", flow_trend="balanced",
        )

    # Money Flow Multiplier
    hilo_diff = high - low
    # Avoid division by zero
    valid_mask = hilo_diff > 0
    mfm = np.zeros(N, dtype=np.float64)
    if valid_mask.any():
        mfm[valid_mask] = ((close[valid_mask] - low[valid_mask]) -
                           (high[valid_mask] - close[valid_mask])) / hilo_diff[valid_mask]

    # Money Flow Volume
    mfv = mfm * volume

    # Split into inflow (positive) and outflow (negative)
    positive_mask = mfv > 0
    negative_mask = mfv < 0

    total_inflow = float(mfv[positive_mask].sum()) if positive_mask.any() else 0.0
    total_outflow = float(abs(mfv[negative_mask].sum())) if negative_mask.any() else 0.0
    net_money_flow = total_inflow - total_outflow
    money_flow_ratio = total_inflow / total_outflow if total_outflow > 0 else float('inf')

    # Chaikin Money Flow (CMF)
    cmf_period = cfg.cmf_period
    if N >= cmf_period:
        cmf_window = cmf_period
    else:
        cmf_window = N

    # Rolling sum
    cmf_values = np.zeros(N)
    for i in range(cmf_window - 1, N):
        window_mfv = mfv[i - cmf_window + 1:i + 1].sum()
        window_vol = volume[i - cmf_window + 1:i + 1].sum()
        if window_vol > 0:
            cmf_values[i] = window_mfv / window_vol

    cmf = float(cmf_values[-1]) if N >= cmf_window else 0.0

    # CMF signal
    if cmf > 0.05:
        cmf_signal = "accumulation"
    elif cmf < -0.05:
        cmf_signal = "distribution"
    else:
        cmf_signal = "neutral"

    # Recent money flow (last 5 days)
    recent_n = min(5, N)
    recent_mfv = mfv[-recent_n:]
    recent_vol = volume[-recent_n:]

    recent_pos = recent_mfv[recent_mfv > 0].sum() if (recent_mfv > 0).any() else 0.0
    recent_neg = abs(recent_mfv[recent_mfv < 0].sum()) if (recent_mfv < 0).any() else 0.0
    recent_net = recent_pos - recent_neg

    if recent_net > 0 and abs(recent_net) / max(recent_vol.sum(), 1) > 0.02:
        recent_flow_signal = "buying"
    elif recent_net < 0 and abs(recent_net) / max(recent_vol.sum(), 1) > 0.02:
        recent_flow_signal = "selling"
    else:
        recent_flow_signal = "neutral"

    # Money flow trend — compare first half vs second half
    half = N // 2
    if half >= cmf_window:
        first_half_cmf = cmf_values[half - cmf_window: half].mean() if half >= cmf_window else 0
        second_half_cmf = cmf_values[half:].mean()
        cmf_diff = second_half_cmf - first_half_cmf
        if cmf_diff > 0.03:
            flow_trend = "increasing_inflow"
        elif cmf_diff < -0.03:
            flow_trend = "increasing_outflow"
        else:
            flow_trend = "balanced"
    else:
        flow_trend = "balanced"

    return MoneyFlowResult(
        ticker=ticker,
        cmf=round(cmf, dec),
        cmf_period=cfg.cmf_period,
        cmf_signal=cmf_signal,
        total_inflow=round(total_inflow, 2),
        total_outflow=round(total_outflow, 2),
        net_money_flow=round(net_money_flow, 2),
        money_flow_ratio=round(float(money_flow_ratio), dec) if money_flow_ratio != float('inf') else money_flow_ratio,
        recent_inflow=round(float(recent_pos), 2),
        recent_outflow=round(float(recent_neg), 2),
        recent_net_flow=round(float(recent_net), 2),
        recent_flow_signal=recent_flow_signal,
        flow_trend=flow_trend,
    )


# ═══════════════════════════════════════════════════════════════════
# Profitability Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_profitability(
    df: pd.DataFrame,
    current_price: float,
    avg_cost: float,
    ticker: str = "",
    cfg: Optional[ChipConfig] = None,
) -> ProfitabilitySummary:
    """
    Analyze holder profitability from volume distribution.

    Uses price-level volume allocation to estimate:
    - % of volume (shares) in profit
    - % of volume underwater
    - Floating P&L estimate
    - Effective buying/selling pressure

    Args:
        df: OHLCV DataFrame
        current_price: Current market price
        avg_cost: Volume-weighted average cost from cost analysis
        ticker: Ticker symbol
        cfg: ChipConfig

    Returns:
        ProfitabilitySummary
    """
    if cfg is None:
        cfg = get_chip_config()
    dec = cfg.decimal_places

    high, low, close, open_, volume = _extract_ohlcv(df)

    if len(df) < 5:
        return ProfitabilitySummary(
            ticker=ticker, current_price=current_price, avg_cost=avg_cost,
            total_profit_ratio=0.0, profitable_volume_pct=0.0, underwater_volume_pct=0.0,
            floating_pnl=0.0, floating_pnl_pct=0.0,
            liquidity_weighted_pnl=0.0, effective_buying_pressure=0.0,
            pct_in_profit=0.0, pct_at_breakeven=0.0, pct_underwater=0.0,
            holder_health="mixed",
        )

    # Build price-level volume allocation (similar to cost distribution)
    price_min = np.min(low)
    price_max = np.max(high)
    price_range = price_max - price_min
    num_bins = min(50, max(20, len(df) // 5))

    if price_range <= 0:
        return ProfitabilitySummary(
            ticker=ticker, current_price=current_price, avg_cost=avg_cost,
            total_profit_ratio=0.0, profitable_volume_pct=0.0, underwater_volume_pct=0.0,
            floating_pnl=0.0, floating_pnl_pct=0.0,
            liquidity_weighted_pnl=0.0, effective_buying_pressure=0.0,
            pct_in_profit=0.0, pct_at_breakeven=0.0, pct_underwater=0.0,
            holder_health="mixed",
        )

    bin_edges = np.linspace(price_min, price_max, num_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_volumes = np.zeros(num_bins, dtype=np.float64)

    for i in range(len(df)):
        h, l, v = high[i], low[i], volume[i]
        if v <= 0 or h <= l:
            continue
        bar_range = h - l
        for j in range(num_bins):
            overlap_low = max(bin_edges[j], l)
            overlap_high = min(bin_edges[j + 1], h)
            if overlap_high > overlap_low:
                overlap_pct = (overlap_high - overlap_low) / bar_range
                bin_volumes[j] += v * overlap_pct

    total_vol = bin_volumes.sum()
    if total_vol <= 0:
        return ProfitabilitySummary(
            ticker=ticker, current_price=current_price, avg_cost=avg_cost,
            total_profit_ratio=0.0, profitable_volume_pct=0.0, underwater_volume_pct=0.0,
            floating_pnl=0.0, floating_pnl_pct=0.0,
            liquidity_weighted_pnl=0.0, effective_buying_pressure=0.0,
            pct_in_profit=0.0, pct_at_breakeven=0.0, pct_underwater=0.0,
            holder_health="mixed",
        )

    # Classify volume by position relative to current price
    # Profitable: price < current_price (bought cheaper)
    # Underwater: price > current_price (bought higher)
    profitable_vol = bin_volumes[bin_mids < current_price].sum()
    underwater_vol = bin_volumes[bin_mids > current_price].sum()
    breakeven_vol = bin_volumes[np.abs(bin_mids - current_price) < price_range * 0.005].sum()

    profitable_pct = profitable_vol / total_vol
    underwater_pct = underwater_vol / total_vol
    breakeven_pct = breakeven_vol / total_vol

    # Profit ratio
    total_profit_ratio = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0

    # Floating P&L estimate
    floating_pnl = (current_price - avg_cost) * total_vol
    floating_pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0

    # Liquidity-weighted P&L (weighted by volume at each level)
    pnl_per_bin = (current_price - bin_mids) / bin_mids
    pnl_per_bin = np.where(bin_mids > 0, pnl_per_bin, 0)
    lw_pnl = np.average(pnl_per_bin, weights=bin_volumes)

    # Effective buying pressure
    # Positive → volume concentrated below current price (buying support)
    # Negative → volume concentrated above current price (selling pressure)
    buying_pressure = profitable_pct - underwater_pct

    # Holder health assessment
    if profitable_pct > 0.65:
        health = "healthy"
    elif profitable_pct < 0.35:
        health = "distressed"
    else:
        health = "mixed"

    return ProfitabilitySummary(
        ticker=ticker,
        current_price=round(current_price, dec),
        avg_cost=round(avg_cost, dec),
        total_profit_ratio=round(total_profit_ratio, dec),
        profitable_volume_pct=round(float(profitable_pct) * 100, dec),
        underwater_volume_pct=round(float(underwater_pct) * 100, dec),
        floating_pnl=round(float(floating_pnl), 2),
        floating_pnl_pct=round(float(floating_pnl_pct), dec),
        liquidity_weighted_pnl=round(float(lw_pnl), dec),
        effective_buying_pressure=round(float(buying_pressure), dec),
        pct_in_profit=round(float(profitable_pct) * 100, dec),
        pct_at_breakeven=round(float(breakeven_pct) * 100, dec),
        pct_underwater=round(float(underwater_pct) * 100, dec),
        holder_health=health,
    )


def compute_full_profitability(
    df: pd.DataFrame,
    current_price: float,
    avg_cost: float,
    ticker: str = "",
    cfg: Optional[ChipConfig] = None,
) -> ProfitabilityResult:
    """
    Full profitability analysis combining holder P&L and money flow.

    Args:
        df: OHLCV DataFrame
        current_price: Current market price
        avg_cost: VWAP from cost analysis
        ticker: Ticker symbol
        cfg: ChipConfig

    Returns:
        ProfitabilityResult
    """
    summary = analyze_profitability(df, current_price, avg_cost, ticker, cfg)
    money_flow = compute_money_flow(df, ticker, cfg)

    return ProfitabilityResult(
        ticker=ticker,
        summary=summary,
        money_flow=money_flow,
    )
