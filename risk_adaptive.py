#!/usr/bin/env python3
"""
VMAA 2.0 — Adaptive Stop Loss (Phase 1)
========================================
Implements the Adaptive Statistical Stop from RL research.

Key improvements over static median stop:
  - Dynamic ATR multiplier based on price level, volatility, and market regime
  - Dynamic hard stop % based on price (low-price stocks need wider stops)
  - Near 52w-low means more room for mean-reversion to work
  - All outputs are floats/strings — compatible with existing risk.py signature

Usage:
  from risk_adaptive import compute_stops_adaptive
  stop_price, stop_type = compute_stops_adaptive(entry_price, low_52w, hist, market)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import RC

try:
    from models import MarketRegime
except ImportError:
    MarketRegime = None  # Fallback if not available

logger = logging.getLogger("vmaa.risk.adaptive")


def compute_stops_adaptive(
    entry_price: float,
    low_52w: float,
    hist: pd.DataFrame,
    market: Optional[object] = None,
) -> Tuple[float, str]:
    """
    Adaptive statistical stop loss.

    Adjusts stop distance dynamically based on:
      1. Price level (low-price = more noise = wider stop)
      2. Stock volatility (high ATR% = wider stop)
      3. Proximity to 52w low (near low = more room for bounce)
      4. Market volatility regime (high vol = wider stop)

    Always picks the MEDIAN of 3 computed stops for balance.

    Returns: (stop_price, stop_type)
    """
    atr = compute_atr(hist, 14)
    atr_pct = atr / entry_price if entry_price > 0 else 0.03

    # ── 1. Dynamic ATR multiplier ──
    base_mult = float(RC.atr_stop_multiplier)  # 2.5

    # Price level: low-price stocks are noisier, need wider stops
    if entry_price < 10:
        base_mult += 1.0   # +40% wider
    elif entry_price < 30:
        base_mult += 0.5   # +20% wider

    # Volatility: high ATR% means more daily noise
    if atr_pct > 0.05:
        base_mult += 0.5
    elif atr_pct > 0.03:
        base_mult += 0.25

    # Proximity to 52w low: near bottom → need room for bounce
    ptl = entry_price / low_52w if low_52w > 0 else 1.0
    if ptl < 1.05:
        base_mult += 1.0  # Very close to 52w-low — max breathing room
    elif ptl < 1.10:
        base_mult += 0.5

    # Market regime: high volatility = wider stops
    if market is not None:
        vol_regime = getattr(market, 'vol_regime', 'NORMAL')
        if vol_regime == 'HIGH':
            base_mult += 0.5
        elif vol_regime == 'EXTREME':
            base_mult += 1.0

    atr_stop = round(entry_price - (atr * base_mult), 2) if atr > 0 else 0

    # ── 2. Dynamic hard stop % ──
    if entry_price < 10:
        hard_pct = 0.22    # 22% for penny stocks
    elif entry_price < 30:
        hard_pct = 0.18    # 18% for small caps
    else:
        hard_pct = float(RC.hard_stop_pct)  # 15% (config default)

    hard_stop = round(entry_price * (1 - hard_pct), 2)

    # ── 3. Structural stop (52w low) ──
    structural_stop = round(low_52w * 0.98, 2)

    # ── 4. Pick the MEDIAN stop ──
    candidates = [
        (atr_stop, "ATR_adaptive"),
        (hard_stop, "Hard_adaptive"),
        (structural_stop, "Structural"),
    ]
    candidates = [(s, n) for s, n in candidates if 0 < s < entry_price]

    if not candidates:
        return round(entry_price * 0.95, 2), "Fallback"

    candidates.sort(key=lambda x: x[0])  # ascending price
    median_idx = len(candidates) // 2
    return candidates[median_idx]


# ═══════════════════════════════════════════════════════════════════
# Shared Utility: ATR
# ═══════════════════════════════════════════════════════════════════


def compute_atr(hist: pd.DataFrame, period: int = 14) -> float:
    """Average True Range — measure of price volatility.
    
    Shared utility used by risk.py and risk_adaptive.py.
    Import: from risk_adaptive import compute_atr
    """
    if len(hist) < period + 1:
        return 0.0
    high = hist['High']
    low = hist['Low']
    close = hist['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.tail(period).mean())


# ── Test / Compare ──
if __name__ == "__main__":
    import yfinance as yf
    from risk import compute_stops, get_market_regime

    tickers = ['INMD', 'TMDX', 'CDRE', 'BLFS', 'MNDY']
    for t in tickers:
        yft = yf.Ticker(t)
        hist = yft.history(period='6mo')
        price = float(hist['Close'].iloc[-1])
        low_52w = float(hist['Low'].min())

        if hist is not None and len(hist) > 20:
            old_stop, old_type = compute_stops(price, low_52w, hist)
            new_stop, new_type = compute_stops_adaptive(price, low_52w, hist)

            old_dist = (price - old_stop) / price * 100
            new_dist = (price - new_stop) / price * 100

            print(f"{t:6s} @ ${price:<6.2f} | "
                  f"OLD: ${old_stop:<6.2f} ({old_dist:.0f}%) {old_type:15s} | "
                  f"NEW: ${new_stop:<6.2f} ({new_dist:.0f}%) {new_type}")
