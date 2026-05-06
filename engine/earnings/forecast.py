#!/usr/bin/env python3
"""
VMAA Earnings Prediction Engine — Earnings Forecast
======================================================
Fiscal quarter/year estimates from analyst consensus (yfinance).
Historical accuracy, surprise analysis, estimate revision tracking.

Usage:
  from engine.earnings.forecast import analyze_forecast
  result = analyze_forecast("AAPL", broker_data, config)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("vmaa.engine.earnings.forecast")


# ═══════════════════════════════════════════════════════════════════
# Surprise Analysis
# ═══════════════════════════════════════════════════════════════════

def compute_surprise_metrics(
    surprise_history: List[Dict[str, Any]],
    material_threshold: float = 0.02,
) -> Dict[str, Any]:
    """
    Compute beat/miss analysis from surprise history.

    Args:
        surprise_history: List of surprise dicts from broker_reports
        material_threshold: Minimum surprise pct to count as material beat/miss

    Returns:
        Dict with beat rate, streak, magnitude, momentum
    """
    if not surprise_history:
        return {
            "beat_rate_4q": None,
            "beat_rate_8q": None,
            "avg_surprise_pct": None,
            "max_surprise_pct": None,
            "min_surprise_pct": None,
            "consecutive_beats": 0,
            "consecutive_misses": 0,
            "surprise_momentum": "neutral",
            "total_quarters": 0,
            "beats": 0,
            "misses": 0,
            "in_lines": 0,
        }

    total = len(surprise_history)
    beats = sum(1 for s in surprise_history if s.get("beat") is True)
    misses = sum(1 for s in surprise_history if s.get("beat") is False)
    in_lines = total - beats - misses

    # Surprise magnitudes
    surprise_pcts = [s.get("surprise_pct", 0) or 0 for s in surprise_history]

    # Beat rates (last 4 and last 8)
    last_4 = surprise_history[:4]
    last_8 = surprise_history[:8]

    beat_rate_4q = (
        sum(1 for s in last_4 if s.get("beat") is True) / len(last_4)
        if last_4 else None
    )
    beat_rate_8q = (
        sum(1 for s in last_8 if s.get("beat") is True) / len(last_8)
        if last_8 else None
    )

    # Consecutive streak
    streak_type = None
    streak_count = 0
    for s in surprise_history:
        is_beat = s.get("beat")
        if is_beat is None:
            break  # in-line, stop streak
        if streak_type is None:
            streak_type = "beat" if is_beat else "miss"
            streak_count = 1
        elif (is_beat and streak_type == "beat") or (not is_beat and streak_type == "miss"):
            streak_count += 1
        else:
            break

    consecutive_beats = streak_count if streak_type == "beat" else 0
    consecutive_misses = streak_count if streak_type == "miss" else 0

    # Surprise momentum
    if beats >= total * 0.75:
        momentum = "strong_positive"
    elif beats >= total * 0.5:
        momentum = "positive"
    elif misses >= total * 0.75:
        momentum = "strong_negative"
    elif misses >= total * 0.5:
        momentum = "negative"
    else:
        momentum = "neutral"

    return {
        "beat_rate_4q": round(beat_rate_4q, 4) if beat_rate_4q is not None else None,
        "beat_rate_8q": round(beat_rate_8q, 4) if beat_rate_8q is not None else None,
        "avg_surprise_pct": round(float(np.mean(surprise_pcts)), 4) if surprise_pcts else None,
        "max_surprise_pct": round(float(np.max(surprise_pcts)), 4) if surprise_pcts else None,
        "min_surprise_pct": round(float(np.min(surprise_pcts)), 4) if surprise_pcts else None,
        "std_surprise_pct": round(float(np.std(surprise_pcts)), 4) if len(surprise_pcts) > 1 else None,
        "consecutive_beats": consecutive_beats,
        "consecutive_misses": consecutive_misses,
        "surprise_momentum": momentum,
        "total_quarters": total,
        "beats": beats,
        "misses": misses,
        "in_lines": in_lines,
        "material_beats_4q": sum(
            1 for s in last_4 if s.get("beat") and abs(s.get("surprise_pct", 0)) >= material_threshold
        ),
        "material_misses_4q": sum(
            1 for s in last_4 if s.get("beat") is False and abs(s.get("surprise_pct", 0)) >= material_threshold
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Historical Accuracy
# ═══════════════════════════════════════════════════════════════════

def compute_forecast_accuracy(
    surprise_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare analyst estimates vs actual reported EPS.

    Args:
        surprise_history: List of surprise dicts

    Returns:
        Dict with accuracy metrics
    """
    if not surprise_history:
        return {
            "mape": None,
            "accuracy_score": None,
            "avg_absolute_error": None,
            "directional_accuracy": None,
        }

    valid = [
        s for s in surprise_history
        if s.get("reported_eps") and s.get("estimated_eps") and s.get("reported_eps", 0) != 0
    ]

    if not valid:
        return {
            "mape": None,
            "accuracy_score": None,
            "avg_absolute_error": None,
            "directional_accuracy": None,
        }

    # Mean Absolute Percentage Error
    ape_values = [
        abs(s["reported_eps"] - s["estimated_eps"]) / abs(s["reported_eps"])
        for s in valid
    ]
    mape = float(np.mean(ape_values))

    # Mean Absolute Error
    mae_values = [abs(s["reported_eps"] - s["estimated_eps"]) for s in valid]
    avg_mae = float(np.mean(mae_values))

    # Directional accuracy: did estimates go in same direction as actual?
    # Use sign agreement
    direction_correct = sum(
        1 for s in valid
        if (s["reported_eps"] > 0 and s["estimated_eps"] > 0)
        or (s["reported_eps"] < 0 and s["estimated_eps"] < 0)
    )
    dir_accuracy = direction_correct / len(valid)

    # Accuracy score: 100 - MAPE (as percentage), capped
    accuracy_score = max(0, min(100, 100 * (1 - mape)))

    return {
        "mape": round(mape, 4),
        "accuracy_score": round(accuracy_score, 2),
        "avg_absolute_error": round(avg_mae, 4),
        "directional_accuracy": round(dir_accuracy, 4),
        "quarters_evaluated": len(valid),
    }


# ═══════════════════════════════════════════════════════════════════
# LTG and PEG Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_growth_metrics(
    estimates: Dict[str, Any],
    current_price: float = 0,
) -> Dict[str, Any]:
    """
    Analyze long-term growth and PEG metrics.

    Args:
        estimates: Estimate dict from broker_reports
        current_price: Current stock price

    Returns:
        Dict with LTG, PEG, and growth delta analysis
    """
    forward_eps = estimates.get("forward_eps", 0) or 0
    trailing_eps = estimates.get("trailing_eps", 0) or 0
    earnings_growth = estimates.get("earnings_growth", 0) or 0
    quarterly_growth = estimates.get("earnings_quarterly_growth", 0) or 0
    revenue_growth = estimates.get("revenue_growth", 0) or 0
    peg = estimates.get("peg_ratio", 0) or 0
    ltg = estimates.get("ltg_estimate", earnings_growth) or 0

    # PEG quality assessment
    peg_quality = "unknown"
    if peg > 0:
        if peg < 1:
            peg_quality = "undervalued"  # PEG < 1 suggests undervalued
        elif peg < 2:
            peg_quality = "fair_value"
        elif peg < 3:
            peg_quality = "overvalued"
        else:
            peg_quality = "highly_overvalued"

    # Earnings-revenue growth delta
    growth_delta = None
    if earnings_growth and revenue_growth:
        growth_delta = earnings_growth - revenue_growth
        # Positive delta means earnings growing faster than revenue (margin expansion)

    return {
        "ltg_estimate": round(ltg, 4),
        "peg_ratio": round(peg, 2),
        "peg_quality": peg_quality,
        "earnings_growth": round(earnings_growth, 4),
        "earnings_quarterly_growth": round(quarterly_growth, 4),
        "revenue_growth": round(revenue_growth, 4),
        "earnings_revenue_delta": round(growth_delta, 4) if growth_delta is not None else None,
        "margin_expansion_signal": growth_delta is not None and growth_delta > 0.05 if growth_delta is not None else None,
        "forward_eps": round(forward_eps, 4),
        "trailing_eps": round(trailing_eps, 4),
        "eps_growth_forward_vs_trailing": (
            round((forward_eps - trailing_eps) / trailing_eps, 4)
            if trailing_eps > 0 and forward_eps > 0 else None
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Estimate Revision Tracking
# ═══════════════════════════════════════════════════════════════════

def analyze_estimate_revisions(
    estimates: Dict[str, Any],
    recommendation_history: List[Dict[str, Any]],
    window_days_1w: int = 7,
    window_days_1m: int = 30,
    window_days_3m: int = 90,
) -> Dict[str, Any]:
    """
    Track upward/downward estimate revision momentum from rating changes.

    Uses yfinance recommendation history as proxy for estimate revision direction.

    Args:
        estimates: Estimate dict
        recommendation_history: Rating changes from broker_reports
        window_days_1w: Days for 1-week window
        window_days_1m: Days for 1-month window
        window_days_3m: Days for 3-month window

    Returns:
        Dict with revision momentum per window
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    upgrades = {"1w": 0, "1m": 0, "3m": 0}
    downgrades = {"1w": 0, "1m": 0, "3m": 0}

    for rec in recommendation_history:
        try:
            rec_date = datetime.strptime(rec["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_ago = (now - rec_date).days
            change = rec.get("change_type", "")

            if change == "upgrade":
                if days_ago <= window_days_1w:
                    upgrades["1w"] += 1
                if days_ago <= window_days_1m:
                    upgrades["1m"] += 1
                if days_ago <= window_days_3m:
                    upgrades["3m"] += 1
            elif change == "downgrade":
                if days_ago <= window_days_1w:
                    downgrades["1w"] += 1
                if days_ago <= window_days_1m:
                    downgrades["1m"] += 1
                if days_ago <= window_days_3m:
                    downgrades["3m"] += 1
        except (ValueError, KeyError):
            continue

    def _net(u, d):
        return u - d

    def _momentum(net_val, total):
        if total == 0:
            return "neutral"
        ratio = net_val / total
        if ratio > 0.3:
            return "positive"
        elif ratio < -0.3:
            return "negative"
        return "neutral"

    net_1w = _net(upgrades["1w"], downgrades["1w"])
    net_1m = _net(upgrades["1m"], downgrades["1m"])
    net_3m = _net(upgrades["3m"], downgrades["3m"])

    total_1w = upgrades["1w"] + downgrades["1w"]
    total_1m = upgrades["1m"] + downgrades["1m"]
    total_3m = upgrades["3m"] + downgrades["3m"]

    # EPS revision trend from estimate changes
    current_yr = estimates.get("eps_current_year", 0) or 0
    next_yr = estimates.get("eps_next_year", 0) or 0
    eps_growth_implied = ((next_yr - current_yr) / current_yr) if current_yr > 0 else None

    return {
        "revisions_1w": {
            "upgrades": upgrades["1w"],
            "downgrades": downgrades["1w"],
            "net": net_1w,
            "total_changes": total_1w,
            "momentum": _momentum(net_1w, total_1w),
        },
        "revisions_1m": {
            "upgrades": upgrades["1m"],
            "downgrades": downgrades["1m"],
            "net": net_1m,
            "total_changes": total_1m,
            "momentum": _momentum(net_1m, total_1m),
        },
        "revisions_3m": {
            "upgrades": upgrades["3m"],
            "downgrades": downgrades["3m"],
            "net": net_3m,
            "total_changes": total_3m,
            "momentum": _momentum(net_3m, total_3m),
        },
        "eps_estimated_growth_current_to_next": (
            round(eps_growth_implied, 4) if eps_growth_implied is not None else None
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Revenue vs EPS Comparison
# ═══════════════════════════════════════════════════════════════════

def analyze_revenue_vs_eps(estimates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare revenue and EPS growth trends for divergence.

    Args:
        estimates: Estimate dict

    Returns:
        Dict with revenue/EPS comparison
    """
    eps_growth = estimates.get("earnings_growth", 0) or 0
    rev_growth = estimates.get("revenue_growth", 0) or 0
    q_eps_growth = estimates.get("earnings_quarterly_growth", 0) or 0

    divergence = None
    if eps_growth and rev_growth:
        delta = eps_growth - rev_growth
        if abs(delta) > 0.05:
            if delta > 0:
                divergence = "eps_outpacing_revenue"  # Margin expansion
            else:
                divergence = "revenue_outpacing_eps"  # Margin compression

    return {
        "eps_growth_yoy": round(eps_growth, 4),
        "revenue_growth_yoy": round(rev_growth, 4),
        "eps_growth_quarterly": round(q_eps_growth, 4),
        "growth_divergence": divergence,
        "eps_to_revenue_growth_ratio": (
            round(eps_growth / rev_growth, 2) if rev_growth else None
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Main Forecast Analyzer
# ═══════════════════════════════════════════════════════════════════

def analyze_forecast(
    ticker: str,
    broker_data: Dict[str, Any],
    material_threshold: float = 0.02,
    window_days_1w: int = 7,
    window_days_1m: int = 30,
    window_days_3m: int = 90,
) -> Dict[str, Any]:
    """
    Run full forecast analysis for a ticker.

    Args:
        ticker: Ticker symbol
        broker_data: Dict from collect_broker_reports() for this ticker
        material_threshold: Material beat/miss threshold
        window_days_1w/1m/3m: Revision windows

    Returns:
        Dict with surprise, accuracy, growth, and revision metrics
    """
    if broker_data.get("error"):
        return {"ticker": ticker, "error": broker_data["error"]}

    estimates = broker_data.get("estimates", {})
    surprise_history = broker_data.get("surprise_history", [])
    rec_history = broker_data.get("recommendation_history", [])
    analyst_info = broker_data.get("analyst_info", {})
    current_price = analyst_info.get("current_price", 0)

    # Surprise analysis
    surprise_metrics = compute_surprise_metrics(surprise_history, material_threshold)

    # Historical accuracy
    accuracy = compute_forecast_accuracy(surprise_history)

    # Growth analysis
    growth = analyze_growth_metrics(estimates, current_price)

    # Estimate revisions from rating changes
    revisions = analyze_estimate_revisions(
        estimates, rec_history, window_days_1w, window_days_1m, window_days_3m
    )

    # Revenue vs EPS
    rev_vs_eps = analyze_revenue_vs_eps(estimates)

    return {
        "ticker": ticker,
        "surprise": surprise_metrics,
        "accuracy": accuracy,
        "growth": growth,
        "revisions": revisions,
        "revenue_vs_eps": rev_vs_eps,
    }
