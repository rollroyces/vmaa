#!/usr/bin/env python3
"""
VMAA Earnings Prediction Engine — Consensus Expectations
==========================================================
Computes analyst consensus: ratings, price targets, EPS/revenue estimates,
dispersion, revision trends, implied returns, and confidence scores.

Usage:
  from engine.earnings.consensus import build_consensus
  consensus = build_consensus("AAPL", broker_data, forecast_data, config)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("vmaa.engine.earnings.consensus")


# ═══════════════════════════════════════════════════════════════════
# Consensus Rating
# ═══════════════════════════════════════════════════════════════════

def compute_consensus_rating(analyst_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute consensus rating score from analyst info.

    Converts recommendation key to numeric 1-5 scale:
    5: Strong Buy, 4: Buy/Outperform, 3: Hold, 2: Underperform, 1: Sell

    Args:
        analyst_info: Analyst info dict from broker_reports

    Returns:
        Dict with consensus rating, score, and interpretation
    """
    score = analyst_info.get("recommendation_score", 3.0)
    recommendation = analyst_info.get("recommendation", "HOLD")
    raw_key = analyst_info.get("recommendation_key", "none")

    # Interpretation bands
    if score >= 4.5:
        interpretation = "strong_buy"
    elif score >= 3.7:
        interpretation = "buy"
    elif score >= 2.7:
        interpretation = "hold"
    elif score >= 2.0:
        interpretation = "underperform"
    else:
        interpretation = "sell"

    return {
        "rating_label": recommendation,
        "rating_raw": raw_key,
        "rating_score": score,  # 1-5 scale
        "rating_normalized": (score - 1) / 4.0,  # 0-1 scale
        "rating_interpretation": interpretation,
    }


# ═══════════════════════════════════════════════════════════════════
# Consensus Price Target
# ═══════════════════════════════════════════════════════════════════

def compute_consensus_target(
    analyst_info: Dict[str, Any],
    current_price: float = 0,
) -> Dict[str, Any]:
    """
    Compute consensus price target metrics.

    Args:
        analyst_info: Analyst info dict with target price fields
        current_price: Current stock price

    Returns:
        Dict with target mean/median/high/low and implied return
    """
    target_mean = analyst_info.get("target_mean", 0) or 0
    target_high = analyst_info.get("target_high", 0) or 0
    target_low = analyst_info.get("target_low", 0) or 0
    target_median = analyst_info.get("target_median", 0) or 0
    num_analysts = analyst_info.get("num_analysts", 0) or 0

    # If no median provided, estimate it (often mean ≈ median for large N)
    if not target_median and target_mean:
        target_median = target_mean

    # Implied return
    implied_return = None
    if target_mean > 0 and current_price > 0:
        implied_return = (target_mean - current_price) / current_price

    # High-low spread (wider = more disagreement)
    spread_pct = None
    if target_high > 0 and target_low > 0 and target_mean > 0:
        spread_pct = (target_high - target_low) / target_mean

    return {
        "target_mean": round(target_mean, 2),
        "target_median": round(target_median, 2),
        "target_high": round(target_high, 2),
        "target_low": round(target_low, 2),
        "target_spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "num_analysts": num_analysts,
        "implied_return": round(implied_return, 4) if implied_return is not None else None,
        "current_price": round(current_price, 2) if current_price else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# Consensus EPS & Revenue
# ═══════════════════════════════════════════════════════════════════

def compute_consensus_earnings(
    estimates: Dict[str, Any],
    surprise_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute consensus EPS and revenue with dispersion estimation.

    Args:
        estimates: Estimate dict from broker_reports
        surprise_history: Surprise history for dispersion estimation

    Returns:
        Dict with consensus EPS/revenue and dispersion
    """
    forward_eps = estimates.get("forward_eps", 0) or 0
    trailing_eps = estimates.get("trailing_eps", 0) or 0
    eps_curr_q = estimates.get("eps_current_quarter", 0) or 0
    eps_next_q = estimates.get("eps_next_quarter", 0) or 0
    eps_curr_yr = estimates.get("eps_current_year", 0) or 0
    eps_next_yr = estimates.get("eps_next_year", 0) or 0

    rev_curr_yr = estimates.get("revenue_estimate_current_year", 0) or 0
    rev_next_yr = estimates.get("revenue_estimate_next_year", 0) or 0

    # Dispersion estimated from historical surprise std
    # Higher surprise std → wider analyst disagreement
    surprise_std = None
    if len(surprise_history) > 1:
        pcts = [abs(s.get("surprise_pct", 0) or 0) for s in surprise_history if s.get("surprise_pct")]
        if pcts:
            surprise_std = float(np.std(pcts))

    # Estimate dispersion as percentage of EPS
    eps_dispersion = None
    if surprise_std and forward_eps > 0:
        eps_dispersion = surprise_std

    return {
        "eps_current_quarter": round(eps_curr_q, 4),
        "eps_next_quarter": round(eps_next_q, 4),
        "eps_current_year": round(eps_curr_yr, 4),
        "eps_next_year": round(eps_next_yr, 4),
        "eps_forward": round(forward_eps, 4),
        "eps_trailing": round(trailing_eps, 4),
        "eps_quarterly_growth_implied": (
            round((eps_next_q - eps_curr_q) / eps_curr_q, 4)
            if eps_curr_q > 0 and eps_next_q > 0 else None
        ),
        "eps_yearly_growth_implied": (
            round((eps_next_yr - eps_curr_yr) / eps_curr_yr, 4)
            if eps_curr_yr > 0 and eps_next_yr > 0 else None
        ),
        "revenue_current_year": round(rev_curr_yr, 4),
        "revenue_next_year": round(rev_next_yr, 4),
        "eps_dispersion_estimated": round(eps_dispersion, 4) if eps_dispersion is not None else None,
        "dispersion_score": (
            1.0 - min(1.0, eps_dispersion / 0.15)
            if eps_dispersion is not None else None
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Consensus Change Tracking
# ═══════════════════════════════════════════════════════════════════

def compute_consensus_changes(
    estimates: Dict[str, Any],
    revisions: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Track consensus estimate changes over 1-week, 1-month windows.

    Uses revision momentum as proxy since yfinance doesn't directly provide
    consensus change history. A strong net upgrade signal indicates upward
    estimate revisions.

    Args:
        estimates: Estimate dict
        revisions: Revision analysis from forecast module

    Returns:
        Dict with consensus change signals
    """
    rev_1w = revisions.get("revisions_1w", {})
    rev_1m = revisions.get("revisions_1m", {})

    # Convert net upgrades to consensus change signal
    def _signal(net, total):
        if total == 0:
            return {"direction": "stable", "strength": 0.0}
        strength = net / total
        if strength > 0.3:
            direction = "up"
        elif strength < -0.3:
            direction = "down"
        else:
            direction = "stable"
        return {"direction": direction, "strength": round(strength, 4)}

    return {
        "consensus_change_1w": _signal(rev_1w.get("net", 0), rev_1w.get("total_changes", 0)),
        "consensus_change_1m": _signal(rev_1m.get("net", 0), rev_1m.get("total_changes", 0)),
        "eps_revision_implied": revisions.get("eps_estimated_growth_current_to_next"),
    }


# ═══════════════════════════════════════════════════════════════════
# Confidence Score
# ═══════════════════════════════════════════════════════════════════

def compute_confidence_score(
    consensus_target: Dict[str, Any],
    consensus_earnings: Dict[str, Any],
    surprise_metrics: Dict[str, Any],
    accuracy: Dict[str, Any],
    min_analysts: int = 3,
    max_dispersion: float = 0.15,
) -> Dict[str, Any]:
    """
    Compute confidence score for consensus estimates.

    Factors:
    - Number of analysts (more = higher confidence)
    - Estimate dispersion (lower = higher confidence)
    - Historical accuracy (higher MAPE = lower confidence)
    - Surprise consistency (consistent direction = higher confidence)

    Args:
        consensus_target: Target consensus dict
        consensus_earnings: Earnings consensus dict
        surprise_metrics: Surprise analysis dict
        accuracy: Forecast accuracy dict
        min_analysts: Min analysts for full confidence
        max_dispersion: Max acceptable dispersion

    Returns:
        Dict with confidence score and breakdown
    """
    score = 0.0
    factors = {}

    # Factor 1: Analyst count (0-25)
    num = consensus_target.get("num_analysts", 0)
    analyst_score = min(25, (num / max(1, min_analysts)) * 25)
    factors["analyst_count_score"] = round(analyst_score, 1)
    score += analyst_score

    # Factor 2: Estimate dispersion (0-25)
    disp = consensus_earnings.get("eps_dispersion_estimated")
    if disp is not None:
        dispersion_score = max(0, min(25, (1 - disp / max_dispersion) * 25))
    else:
        dispersion_score = 12.5  # neutral if unknown
    factors["dispersion_score"] = round(dispersion_score, 1)
    score += dispersion_score

    # Factor 3: Historical accuracy (0-25)
    acc_score = accuracy.get("accuracy_score")
    if acc_score is not None:
        accuracy_factor = min(25, (acc_score / 100) * 25)
    else:
        accuracy_factor = 12.5
    factors["accuracy_score"] = round(accuracy_factor, 1)
    score += accuracy_factor

    # Factor 4: Surprise consistency (0-25)
    q = surprise_metrics.get("total_quarters", 0)
    beats = surprise_metrics.get("beats", 0)
    if q > 0:
        consistency = abs(beats / q - 0.5) * 2  # 0 = no consistency, 1 = all same
        consistency_score = consistency * 25
    else:
        consistency_score = 12.5
    factors["consistency_score"] = round(consistency_score, 1)
    score += consistency_score

    # Level
    if score >= 80:
        level = "high"
    elif score >= 60:
        level = "moderate"
    elif score >= 40:
        level = "low"
    else:
        level = "very_low"

    return {
        "confidence_score": round(score, 1),
        "confidence_level": level,
        "factors": factors,
    }


# ═══════════════════════════════════════════════════════════════════
# Main Consensus Builder
# ═══════════════════════════════════════════════════════════════════

def build_consensus(
    ticker: str,
    broker_data: Dict[str, Any],
    forecast_data: Dict[str, Any],
    min_analysts: int = 3,
    max_dispersion: float = 0.15,
) -> Dict[str, Any]:
    """
    Build full consensus expectations for a ticker.

    Args:
        ticker: Ticker symbol
        broker_data: Broker report dict for this ticker
        forecast_data: Forecast analysis dict for this ticker
        min_analysts: Minimum analysts for confidence threshold
        max_dispersion: Maximum acceptable dispersion

    Returns:
        Full consensus dict: rating, target, EPS, revenue, changes, confidence
    """
    if broker_data.get("error"):
        return {"ticker": ticker, "error": broker_data["error"]}

    analyst_info = broker_data.get("analyst_info", {})
    estimates = broker_data.get("estimates", {})
    surprise_history = broker_data.get("surprise_history", [])

    surprise_metrics = forecast_data.get("surprise", {})
    accuracy = forecast_data.get("accuracy", {})
    revisions = forecast_data.get("revisions", {})

    # Compute sub-components
    rating = compute_consensus_rating(analyst_info)
    target = compute_consensus_target(analyst_info, analyst_info.get("current_price", 0))
    earnings = compute_consensus_earnings(estimates, surprise_history)
    changes = compute_consensus_changes(estimates, revisions)
    confidence = compute_confidence_score(
        target, earnings, surprise_metrics, accuracy,
        min_analysts, max_dispersion,
    )

    return {
        "ticker": ticker,
        "consensus_rating": rating,
        "consensus_target": target,
        "consensus_earnings": earnings,
        "consensus_changes": changes,
        "confidence": confidence,
    }
