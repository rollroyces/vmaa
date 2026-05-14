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

from vmaa.config import RC

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
    ATR-based adaptive stop loss (V3 redesign).

    V3 changes:
      - Hard stop = max(12%, 2 * ATR%). Dynamically adjusts to volatility.
      - No more static hard stop based on price level.
      - Trailing stop = 0.5 * ATR (from high watermark).
      - Structural stop (52w low) kept as absolute floor.

    Returns: (stop_price, stop_type)
    """
    atr = compute_atr(hist, 14)
    atr_pct = atr / entry_price if entry_price > 0 else 0.03

    # ── V3 Hard Stop: max(12%, 2 * ATR%) ──
    atr_based_stop_pct = max(0.12, 2.0 * atr_pct)  # min 12%, wider for volatile stocks

    # Cap at 30% max stop (no stock should have >30% hard stop)
    atr_based_stop_pct = min(atr_based_stop_pct, 0.30)

    # Market regime adjustment
    if market is not None:
        vol_regime = getattr(market, 'vol_regime', 'NORMAL')
        if vol_regime == 'HIGH':
            atr_based_stop_pct = min(atr_based_stop_pct * 1.15, 0.30)
        elif vol_regime == 'EXTREME':
            atr_based_stop_pct = min(atr_based_stop_pct * 1.25, 0.30)

    hard_stop = round(entry_price * (1 - atr_based_stop_pct), 2)

    # ── Structural stop (52w-low floor) ──
    structural_stop = round(low_52w * 0.98, 2)

    # ── Pick the HIGHER of hard_stop and structural (gives more breathing room) ──
    # We want the stop that's closer to entry (higher price) for tighter risk control
    # But structural_stop being below hard_stop means it's too far — use hard_stop
    if structural_stop > hard_stop:
        # Structural is tighter than ATR — use structural
        return structural_stop, "Structural_ATR"
    elif hard_stop < entry_price * 0.70:
        # Floor: never let stop go below 30% drawdown
        return round(entry_price * 0.70, 2), "ATR_Floor"
    else:
        return hard_stop, f"ATR_adaptive({atr_based_stop_pct:.0%})"


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
