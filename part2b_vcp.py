#!/usr/bin/env python3
"""
VMAA 2.0 — Part 2B: VCP (Volatility Contraction Pattern) Filter
=================================================================
Precision entry filter based on Mark Minervini's VCP methodology.

Sits between Stage 2 (MAGNA) and Stage 4 (Risk) as Stage 2.5.
Refines: entry_price, stop_loss, confidence_score.
VCP enhances — it never blocks entries.

Key outputs:
  - vcp_detected: bool (pattern present?)
  - vcp_quality: float 0.0-1.0 (how textbook is the pattern?)
  - vcp_contractions: int (number of contraction waves detected)
  - vcp_pivot_price: float (optimal entry at pivot breakout)
  - vcp_volatility_squeeze: float (current ATR vs base average ATR)
  - vcp_stop_suggestion: float (tightened stop based on pivot structure)

Based on: Mark Minervini — Trade Like a Stock Market Wizard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("vmaa.vcp")


# ═══════════════════════════════════════════════════════
# Data Model
# ═══════════════════════════════════════════════════════

@dataclass
class VCPResult:
    """Output of VCP analysis for a single candidate."""
    ticker: str
    vcp_detected: bool = False
    vcp_quality: float = 0.0            # 0.0–1.0
    contractions: int = 0               # Number of contraction waves
    pivot_price: float = 0.0            # Optimal entry at pivot
    pivot_volatility_pct: float = 0.0   # ATR% at pivot point
    volume_dry_up_ratio: float = 0.0    # Current vol / avg vol (lower = better)
    range_contraction_ratio: float = 0.0 # Latest wave range / first wave range
    stop_suggestion: float = 0.0        # VCP-based stop price
    stop_pct: float = 0.0               # VCP-based stop distance %
    signals: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "vcp_detected": self.vcp_detected,
            "vcp_quality": round(self.vcp_quality, 4),
            "contractions": self.contractions,
            "pivot_price": round(self.pivot_price, 2),
            "pivot_volatility_pct": round(self.pivot_volatility_pct, 4),
            "volume_dry_up_ratio": round(self.volume_dry_up_ratio, 4),
            "range_contraction_ratio": round(self.range_contraction_ratio, 4),
            "stop_suggestion": round(self.stop_suggestion, 2),
            "stop_pct": round(self.stop_pct, 4),
            "signals": self.signals,
            "rationale": self.rationale,
        }


# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════

@dataclass(frozen=True)
class VCPConfig:
    """Configuration for VCP detection and integration."""

    # ── Detection parameters ──
    min_history_days: int = 126              # Minimum ~6 months of data
    min_contractions: int = 2                # Minimum contraction waves
    ideal_contractions: int = 3              # Ideal number of waves for max quality
    max_pivot_atr_pct: float = 0.04          # ATR must be < 4% at pivot
    ideal_pivot_atr_pct: float = 0.02        # < 2% = ultra-tight (max quality)
    volume_dry_up_threshold: float = 0.50    # Current vol < 50% of base avg
    contraction_ratio_threshold: float = 0.80 # Each wave ≤ 80% of previous range
    vcp_quality_threshold: float = 0.50       # Minimum quality to flag vcp_detected
    max_2w_range_pct: float = 0.15           # 2-week range should be < 15% for tightness

    # ── Phase windows (trading days) ──
    phase_windows: Tuple[int, int, int] = (42, 42, 42)  # P1, P2, P3 size in days

    # ── Integration: VCP enhances, never blocks ──
    vcp_required_for_gap: bool = False       # If True, gap entries need VCP (NOT recommended)
    vcp_required_for_ma: bool = False        # If True, MA entries need VCP (NOT recommended)
    vcp_stop_tightening_pct: float = 0.40    # Tighten stop by 40% on VCP confirmation
    vcp_confidence_boost: float = 0.15       # Boost confidence by 0.15 on VCP confirmation
    vcp_position_size_boost: float = 0.30    # Boost position size by 30% on VCP (same dollar risk)

    # ── Quality scoring weights ──
    weight_contracting: float = 0.30         # Range contraction detection
    weight_volume: float = 0.20              # Volume dry-up
    weight_tightness: float = 0.25           # Recent pivot tightness
    weight_atr: float = 0.15                 # ATR compression
    weight_contractions: float = 0.10        # Number of contraction waves


# Singleton
VC = VCPConfig()


# ═══════════════════════════════════════════════════════
# Core Detection Algorithm
# ═══════════════════════════════════════════════════════

def analyze_vcp(
    ticker: str,
    hist: pd.DataFrame,
    current_price: float,
    config: VCPConfig = VC,
) -> Optional[VCPResult]:
    """
    Full VCP analysis on a single stock.

    Args:
        ticker: Stock symbol
        hist: Daily OHLCV DataFrame (at least 6 months of data)
        current_price: Current closing price
        config: VCP configuration

    Returns:
        VCPResult with full analysis, or None if insufficient data
    """
    if hist is None or len(hist) < config.min_history_days:
        logger.debug(f"{ticker}: insufficient history ({len(hist) if hist is not None else 0} days)")
        return None

    result = VCPResult(ticker=ticker)

    # ── Step 1: Identify contraction waves ──
    waves = _identify_contraction_waves(hist, config)
    if len(waves) < config.min_contractions:
        result.rationale = f"Insufficient contraction waves ({len(waves)} < {config.min_contractions})"
        return result

    result.contractions = len(waves)

    # ── Step 2: Verify contraction properties ──
    range_shrinking = _verify_range_contraction(waves, config)
    volume_declining = _verify_volume_decline(hist, waves)

    if not range_shrinking or not volume_declining:
        reasons = []
        if not range_shrinking:
            reasons.append("ranges not shrinking")
        if not volume_declining:
            reasons.append("volume not declining")
        result.rationale = f"VC pattern incomplete: {', '.join(reasons)}"
        return result

    # ── Step 3: Pivot tightness assessment ──
    pivot_data = _assess_pivot_tightness(hist, waves)
    result.pivot_price = pivot_data["pivot_price"]
    result.pivot_volatility_pct = pivot_data["atr_pct"]
    result.volume_dry_up_ratio = pivot_data["vol_ratio"]

    if result.pivot_volatility_pct > config.max_pivot_atr_pct:
        result.rationale = (
            f"Pivot too wide: ATR={result.pivot_volatility_pct:.1%} "
            f"(max {config.max_pivot_atr_pct:.0%})"
        )
        return result

    # ── Step 4: Compute VCP quality score ──
    quality = _compute_vcp_quality(result, waves, config)
    result.vcp_quality = quality
    result.vcp_detected = quality >= config.vcp_quality_threshold

    # ── Step 5: Compute contraction ratio ──
    if len(waves) >= 2:
        first_range = waves[0]["range_pct"]
        last_range = waves[-1]["range_pct"]
        result.range_contraction_ratio = last_range / first_range if first_range > 0 else 1.0

    # ── Step 6: Generate stop suggestion ──
    result.stop_suggestion, result.stop_pct = _compute_vcp_stop(
        current_price, waves, hist, config
    )

    # ── Step 7: Build signals and rationale ──
    if result.contractions >= config.ideal_contractions:
        result.signals.append(f"VCP_{result.contractions}c")
    else:
        result.signals.append(f"VCP_{result.contractions}c")

    if result.pivot_volatility_pct < config.ideal_pivot_atr_pct:
        result.signals.append("ULTRA_TIGHT")
    elif result.pivot_volatility_pct < config.max_pivot_atr_pct:
        result.signals.append("TIGHT_PIVOT")

    if result.volume_dry_up_ratio < config.volume_dry_up_threshold:
        result.signals.append("VOL_DRY_UP")

    if result.range_contraction_ratio < 0.50:
        result.signals.append("RANGE_HALVED")

    if result.vcp_detected:
        result.signals.append("VCP_CONFIRMED")

    result.rationale = (
        f"VCP {'✓' if result.vcp_detected else '✗'} "
        f"Q={result.vcp_quality:.0%} "
        f"waves={result.contractions} "
        f"pivot_ATR={result.pivot_volatility_pct:.1%} "
        f"vol_dry={result.volume_dry_up_ratio:.0%} "
        f"range_ratio={result.range_contraction_ratio:.0%}"
    )

    return result


# ═══════════════════════════════════════════════════════
# Contraction Wave Identification
# ═══════════════════════════════════════════════════════

def _identify_contraction_waves(
    hist: pd.DataFrame,
    config: VCPConfig = VC,
) -> List[dict]:
    """
    Identify contraction waves in price history.

    Each wave is a period where price oscillates within a range.
    Returns waves sorted chronologically, each with range_pct, volume, duration.
    """
    p1w, p2w, p3w = config.phase_windows
    total_days = p1w + p2w + p3w

    if len(hist) < total_days:
        # Scale windows proportionally
        scale = len(hist) / total_days
        p1w = max(21, int(p1w * scale))
        p2w = max(21, int(p2w * scale))
        p3w = max(21, int(p3w * scale))

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    vol = hist["Volume"]

    waves = []

    # Phase windows from oldest to newest
    for i, (name, size) in enumerate([
        ("P1", p1w), ("P2", p2w), ("P3", p3w)
    ]):
        start_idx = -(p1w + p2w + p3w) + (i * size) if i == 0 else \
                    -(p2w + p3w) + ((i - 1) * size) if i == 1 else -p3w
        end_idx = start_idx + size

        # Ensure valid indices
        start_idx = max(start_idx, -len(hist))
        end_idx = min(end_idx, len(hist)) if end_idx < 0 else None

        phase = hist.iloc[start_idx:end_idx]
        if len(phase) < 10:
            continue

        phase_high = phase["High"].max()
        phase_low = phase["Low"].min()
        phase_range_pct = (phase_high / phase_low - 1) * 100 if phase_low > 0 else 100
        phase_avg_vol = phase["Volume"].mean()
        phase_max_vol = phase["Volume"].max()
        phase_price_mean = phase["Close"].mean()

        waves.append({
            "name": name,
            "range_pct": phase_range_pct,
            "avg_vol": phase_avg_vol,
            "max_vol": phase_max_vol,
            "high": phase_high,
            "low": phase_low,
            "mean_price": phase_price_mean,
            "days": len(phase),
            "start_idx": start_idx,
            "end_idx": end_idx,
        })

    return waves


def _verify_range_contraction(waves: List[dict], config: VCPConfig = VC) -> bool:
    """Check if wave ranges are systematically contracting."""
    if len(waves) < 2:
        return False

    ranges = [w["range_pct"] for w in waves]
    contractions_found = 0

    for i in range(1, len(ranges)):
        if ranges[i] < ranges[i - 1] * config.contraction_ratio_threshold:
            contractions_found += 1
        elif ranges[i] >= ranges[i - 1]:
            # Range expanded — not contracting
            pass

    return contractions_found >= (len(waves) - 1)  # Need ALL waves to contract


def _verify_volume_decline(hist: pd.DataFrame, waves: List[dict]) -> bool:
    """Check for systematic volume dry-up across waves."""
    if len(waves) < 2:
        return False

    # Compare avg volume of last wave vs first wave
    first_vol = waves[0]["avg_vol"]
    last_vol = waves[-1]["avg_vol"]

    if first_vol <= 0:
        return False

    return last_vol < first_vol * 0.80  # At least 20% decline from wave 1


# ═══════════════════════════════════════════════════════
# Pivot Analysis
# ═══════════════════════════════════════════════════════

def _assess_pivot_tightness(hist: pd.DataFrame, waves: List[dict]) -> dict:
    """Assess tightness at the pivot (most recent contraction wave)."""
    if not waves:
        return {"pivot_price": 0, "atr_pct": 1.0, "vol_ratio": 2.0}

    # Focus on the last wave (current pivot zone)
    last_wave = waves[-1]
    start_idx = max(last_wave["start_idx"], -len(hist))

    # Use last half of the final wave for pivot analysis
    pivot_start = start_idx + len(hist.iloc[start_idx:]) // 2
    pivot_data = hist.iloc[pivot_start:]

    if len(pivot_data) < 5:
        pivot_data = hist.iloc[start_idx:]

    close = pivot_data["Close"]
    high = pivot_data["High"]
    low = pivot_data["Low"]
    vol = pivot_data["Volume"]

    # Pivot price: most recent close
    pivot_price = float(close.iloc[-1])

    # ATR at pivot
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = float(atr.iloc[-1] / pivot_price) if pivot_price > 0 else 1.0

    # Volume ratio: current vs 50-day average
    vol_50d_avg = hist["Volume"].rolling(50).mean().iloc[-1]
    current_vol = vol.iloc[-5:].mean()  # Last 5 days avg
    vol_ratio = float(current_vol / vol_50d_avg) if vol_50d_avg > 0 else 2.0

    return {"pivot_price": pivot_price, "atr_pct": atr_pct, "vol_ratio": vol_ratio}


def _compute_vcp_stop(
    current_price: float,
    waves: List[dict],
    hist: pd.DataFrame,
    config: VCPConfig = VC,
) -> Tuple[float, float]:
    """Compute VCP-based stop loss using pivot structure."""
    if not waves:
        return current_price * 0.85, 0.15

    # Base stop: below the lowest low of the last contraction wave
    last_wave = waves[-1]
    pivot_low = last_wave["low"]

    # Use the lower of: pivot low OR 2x ATR below current price
    atr = hist["High"].iloc[-14:].max() - hist["Low"].iloc[-14:].min()
    # Better: compute proper ATR
    tr = pd.concat(
        [
            hist["High"] - hist["Low"],
            (hist["High"] - hist["Close"].shift()).abs(),
            (hist["Low"] - hist["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = tr.rolling(14).mean().iloc[-1]

    atr_stop = current_price - (2.0 * atr_14)
    structural_stop = pivot_low * 0.98  # Just below pivot low

    # Use the tighter of the two (closer to entry = less risk)
    vcp_stop = max(atr_stop, structural_stop)

    # Floor: never tighter than 8% from entry (stop price ≥ 92% of current)
    tightest_stop = current_price * 0.92
    # Ceiling: never wider than 15% from entry (stop price ≥ 85% of current)
    widest_stop = current_price * 0.85

    # Clamp: stop price between widest_stop and tightest_stop
    vcp_stop = max(widest_stop, min(vcp_stop, tightest_stop))

    stop_pct = (current_price - vcp_stop) / current_price if current_price > 0 else 0.15

    return round(float(vcp_stop), 2), round(float(stop_pct), 4)


# ═══════════════════════════════════════════════════════
# Quality Scoring
# ═══════════════════════════════════════════════════════

def _compute_vcp_quality(
    result: VCPResult,
    waves: List[dict],
    config: VCPConfig = VC,
) -> float:
    """Compute composite VCP quality score (0.0–1.0)."""
    w = config  # shorthand

    # 1. Contraction count score (0-1)
    contraction_score = min(result.contractions / w.ideal_contractions, 1.0)

    # 2. Volume dry-up score (0-1): lower ratio = better
    if result.volume_dry_up_ratio >= 1.0:
        volume_score = 0.0
    elif result.volume_dry_up_ratio <= w.volume_dry_up_threshold:
        volume_score = 1.0
    else:
        volume_score = 1.0 - (result.volume_dry_up_ratio - w.volume_dry_up_threshold) / \
                       (1.0 - w.volume_dry_up_threshold)

    # 3. Tightness score (0-1): computed from 2-week range
    # We need to recompute 2w range from waves/last phase
    last_wave = waves[-1]
    tightness_score = max(0.0, 1.0 - (last_wave["range_pct"] / 30.0))

    # 4. ATR compression score (0-1)
    if result.pivot_volatility_pct <= w.ideal_pivot_atr_pct:
        atr_score = 1.0
    elif result.pivot_volatility_pct >= w.max_pivot_atr_pct:
        atr_score = 0.0
    else:
        atr_score = 1.0 - (result.pivot_volatility_pct - w.ideal_pivot_atr_pct) / \
                    (w.max_pivot_atr_pct - w.ideal_pivot_atr_pct)

    # 5. Range contraction ratio score (0-1)
    if result.range_contraction_ratio <= 0.30:
        range_score = 1.0
    elif result.range_contraction_ratio >= 1.0:
        range_score = 0.0
    else:
        range_score = 1.0 - (result.range_contraction_ratio - 0.30) / 0.70

    quality = (
        w.weight_contracting * range_score
        + w.weight_volume * volume_score
        + w.weight_tightness * tightness_score
        + w.weight_atr * atr_score
        + w.weight_contractions * contraction_score
    )

    return round(max(0.0, min(quality, 1.0)), 4)


# ═══════════════════════════════════════════════════════
# Batch Processing
# ═══════════════════════════════════════════════════════

def batch_vcp_filter(
    candidates: list,
    hist_cache: Optional[Dict[str, pd.DataFrame]] = None,
    config: VCPConfig = VC,
) -> Dict[str, VCPResult]:
    """
    Run VCP analysis on all entry-ready candidates.

    Args:
        candidates: List of VMAACandidate objects (need .ticker attribute)
        hist_cache: Optional dict of {ticker: hist_DataFrame} from Part 2
        config: VCP configuration

    Returns:
        Dict of {ticker: VCPResult}
    """
    import yfinance as yf
    import time

    results = {}

    for i, candidate in enumerate(candidates):
        ticker = getattr(candidate, "ticker", None)
        if not ticker:
            continue

        # Get price from candidate
        price = getattr(candidate, "current_price", 0)
        if not price and hasattr(candidate, "part1"):
            price = getattr(candidate.part1, "current_price", 0)

        # Try cached history first
        hist = None
        if hist_cache and ticker in hist_cache:
            hist = hist_cache[ticker]
        else:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                time.sleep(0.15)  # Rate limit
            except Exception as e:
                logger.warning(f"{ticker}: yfinance fetch failed: {e}")
                continue

        if hist is None or len(hist) < config.min_history_days:
            logger.debug(f"{ticker}: insufficient data for VCP")
            continue

        vcp_result = analyze_vcp(ticker, hist, price, config)
        if vcp_result:
            results[ticker] = vcp_result

    return results


# ═══════════════════════════════════════════════════════
# Integration Utilities
# ═══════════════════════════════════════════════════════

def apply_vcp_to_stop(
    base_stop_pct: float,
    vcp_result: Optional[VCPResult],
    config: VCPConfig = VC,
) -> float:
    """
    Adjust stop percentage based on VCP confirmation.

    VCP-confirmed entries get tightened stops.
    Non-VCP entries keep standard stops.
    """
    if vcp_result is None or not vcp_result.vcp_detected:
        return base_stop_pct

    tightened = base_stop_pct * (1.0 - config.vcp_stop_tightening_pct)
    tightened = max(tightened, 0.08)  # Never tighter than 8%
    return round(tightened, 4)


def apply_vcp_to_confidence(
    base_confidence: float,
    vcp_result: Optional[VCPResult],
    config: VCPConfig = VC,
) -> float:
    """
    Boost confidence score based on VCP confirmation.

    VCP adds to confidence because the technical setup is validated.
    """
    if vcp_result is None or not vcp_result.vcp_detected:
        return base_confidence

    boost = config.vcp_confidence_boost * vcp_result.vcp_quality
    return round(min(base_confidence + boost, 1.0), 4)


def apply_vcp_to_position_size(
    base_size: float,
    vcp_result: Optional[VCPResult],
    config: VCPConfig = VC,
) -> float:
    """
    Increase position size on VCP-confirmed entries.

    Same dollar risk at a tighter stop = larger position.
    """
    if vcp_result is None or not vcp_result.vcp_detected:
        return base_size

    boost = config.vcp_position_size_boost * vcp_result.vcp_quality
    return round(base_size * (1.0 + boost), 2)


def get_vcp_entry_quality(vcp_result: Optional[VCPResult]) -> str:
    """Human-readable VCP entry quality label."""
    if vcp_result is None:
        return "NO_DATA"
    if not vcp_result.vcp_detected:
        if vcp_result.vcp_quality >= 0.30:
            return "BORDERLINE"
        return "NO_VCP"
    if vcp_result.vcp_quality >= 0.80:
        return "PREMIUM_VCP"
    if vcp_result.vcp_quality >= 0.65:
        return "SOLID_VCP"
    return "BASIC_VCP"


# ═══════════════════════════════════════════════════════
# Quick Check (for CLI / rapid screening)
# ═══════════════════════════════════════════════════════

def quick_vcp_check(ticker: str) -> Optional[VCPResult]:
    """
    Quick VCP check for a single ticker — fetches data and analyzes.

    Useful for CLI usage and one-off checks.
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        price = float(hist["Close"].iloc[-1])
        return analyze_vcp(ticker, hist, price)
    except Exception as e:
        logger.error(f"quick_vcp_check({ticker}): {e}")
        return None


# ═══════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print("Usage: python3 part2b_vcp.py TICKER [TICKER ...]")
        print("Example: python3 part2b_vcp.py INMD TMDX CDRE")
        sys.exit(1)

    for ticker in sys.argv[1:]:
        result = quick_vcp_check(ticker)
        if result:
            print(f"\n{'='*60}")
            print(f"  {ticker} — VCP Analysis")
            print(f"{'='*60}")
            print(f"  VCP Detected:  {'✅ YES' if result.vcp_detected else '❌ NO'}")
            print(f"  VCP Quality:   {result.vcp_quality:.1%}")
            print(f"  Contractions:  {result.contractions}")
            print(f"  Pivot ATR:     {result.pivot_volatility_pct:.1%}")
            print(f"  Volume Dry-Up: {result.volume_dry_up_ratio:.0%}")
            print(f"  Range Ratio:   {result.range_contraction_ratio:.0%}")
            print(f"  VCP Stop:      ${result.stop_suggestion:.2f} ({result.stop_pct:.1%})")
            print(f"  Signals:       {', '.join(result.signals) if result.signals else '(none)'}")
            print(f"  Rationale:     {result.rationale}")
            print(f"  Quality:       {get_vcp_entry_quality(result)}")
        else:
            print(f"\n{ticker}: Insufficient data")
