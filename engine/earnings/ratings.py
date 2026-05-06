#!/usr/bin/env python3
"""
VMAA Earnings Prediction Engine — Rating Changes
===================================================
Tracks analyst upgrade/downgrade activity, target price changes,
rating momentum, earnings calendar, and pre-earnings sentiment.

Usage:
  from engine.earnings.ratings import analyze_ratings
  result = analyze_ratings("AAPL", broker_data, t)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("vmaa.engine.earnings.ratings")


# ═══════════════════════════════════════════════════════════════════
# Upgrade/Downgrade Detection
# ═══════════════════════════════════════════════════════════════════

def detect_upgrade_downgrade_activity(
    recommendation_history: List[Dict[str, Any]],
    window_days_1w: int = 7,
    window_days_1m: int = 30,
    window_days_3m: int = 90,
) -> Dict[str, Any]:
    """
    Detect and summarize upgrade/downgrade activity across time windows.

    Args:
        recommendation_history: Rating changes from broker_reports
        window_days_1w/1m/3m: Analysis windows

    Returns:
        Dict with upgrade/downgrade counts and net momentum per window
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    windows = {
        "1w": {"upgrades": 0, "downgrades": 0, "maintains": 0, "details": []},
        "1m": {"upgrades": 0, "downgrades": 0, "maintains": 0, "details": []},
        "3m": {"upgrades": 0, "downgrades": 0, "maintains": 0, "details": []},
    }

    for rec in recommendation_history:
        try:
            rec_date = datetime.strptime(rec["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_ago = (now - rec_date).days
            change = rec.get("change_type", "maintain")

            entry = {
                "date": rec["date"],
                "firm": rec.get("firm", "Unknown"),
                "from": rec.get("from_normalized", "N/A"),
                "to": rec.get("to_normalized", "HOLD"),
                "change": change,
            }

            if days_ago <= window_days_1w:
                if change == "upgrade":
                    windows["1w"]["upgrades"] += 1
                elif change == "downgrade":
                    windows["1w"]["downgrades"] += 1
                else:
                    windows["1w"]["maintains"] += 1
                windows["1w"]["details"].append(entry)

            if days_ago <= window_days_1m:
                if change == "upgrade":
                    windows["1m"]["upgrades"] += 1
                elif change == "downgrade":
                    windows["1m"]["downgrades"] += 1
                else:
                    windows["1m"]["maintains"] += 1
                windows["1m"]["details"].append(entry)

            if days_ago <= window_days_3m:
                if change == "upgrade":
                    windows["3m"]["upgrades"] += 1
                elif change == "downgrade":
                    windows["3m"]["downgrades"] += 1
                else:
                    windows["3m"]["maintains"] += 1
                windows["3m"]["details"].append(entry)

        except (ValueError, KeyError):
            continue

    result = {}
    for window_name, data in windows.items():
        net = data["upgrades"] - data["downgrades"]
        total = data["upgrades"] + data["downgrades"] + data["maintains"]

        if total > 0:
            if net > 0:
                momentum = "positive"
            elif net < 0:
                momentum = "negative"
            else:
                momentum = "neutral"
        else:
            momentum = "no_activity"

        # Limit details to most recent 10
        details = sorted(data["details"], key=lambda x: x["date"], reverse=True)[:10]

        result[f"rating_activity_{window_name}"] = {
            "window_days": (
                window_days_1w if window_name == "1w"
                else window_days_1m if window_name == "1m"
                else window_days_3m
            ),
            "upgrades": data["upgrades"],
            "downgrades": data["downgrades"],
            "maintains": data["maintains"],
            "net": net,
            "momentum": momentum,
            "upgrade_ratio": (
                round(data["upgrades"] / (data["upgrades"] + data["downgrades"]), 2)
                if (data["upgrades"] + data["downgrades"]) > 0 else None
            ),
            "recent_changes": details,
        }

    return result


# ═══════════════════════════════════════════════════════════════════
# Target Price Change Detection
# ═══════════════════════════════════════════════════════════════════

def detect_target_price_changes(
    recommendation_history: List[Dict[str, Any]],
    window_days: int = 30,
) -> Dict[str, Any]:
    """
    Detect price target increases/decreases from recommendation history.
    Target changes are extracted from the 'action' field in yfinance recommendations.

    Args:
        recommendation_history: Rating changes
        window_days: Analysis window

    Returns:
        Dict with target increases/decreases summary
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    increases = 0
    decreases = 0
    target_details = []

    for rec in recommendation_history:
        try:
            rec_date = datetime.strptime(rec["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_ago = (now - rec_date).days
            if days_ago > window_days:
                continue

            action = rec.get("action", "").lower()
            if not action:
                continue

            if any(word in action for word in ["raise", "raised", "increase", "boost", "upgrade target"]):
                increases += 1
                target_details.append({
                    "date": rec["date"],
                    "firm": rec.get("firm", "Unknown"),
                    "direction": "increase",
                    "action": action,
                })
            elif any(word in action for word in ["lower", "cut", "reduce", "decrease", "downgrade target"]):
                decreases += 1
                target_details.append({
                    "date": rec["date"],
                    "firm": rec.get("firm", "Unknown"),
                    "direction": "decrease",
                    "action": action,
                })

        except (ValueError, KeyError):
            continue

    net = increases - decreases
    if increases + decreases > 0:
        if net > 0:
            momentum = "targets_rising"
        elif net < 0:
            momentum = "targets_falling"
        else:
            momentum = "targets_stable"
    else:
        momentum = "no_recent_target_changes"

    return {
        "window_days": window_days,
        "target_increases": increases,
        "target_decreases": decreases,
        "net_target_changes": net,
        "target_momentum": momentum,
        "recent_target_changes": target_details[:10],
    }


# ═══════════════════════════════════════════════════════════════════
# Rating Change Momentum
# ═══════════════════════════════════════════════════════════════════

def compute_rating_momentum(
    rating_activity: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute aggregate rating momentum across time windows.

    Args:
        rating_activity: Output from detect_upgrade_downgrade_activity

    Returns:
        Dict with composite momentum score
    """
    windows = ["1w", "1m", "3m"]
    scores = []

    for w in windows:
        key = f"rating_activity_{w}"
        data = rating_activity.get(key, {})
        net = data.get("net", 0)
        total = (data.get("upgrades", 0) + data.get("downgrades", 0))

        if total > 0:
            scores.append(net / total)
        else:
            scores.append(0)

    # Weighted composite: recent windows weighted more
    weights = {"1w": 0.5, "1m": 0.3, "3m": 0.2}
    composite = 0.0
    for i, w in enumerate(windows):
        composite += scores[i] * weights[w]

    # Convert to signal
    if composite > 0.3:
        signal = "strong_upgrade_momentum"
    elif composite > 0.1:
        signal = "upgrade_momentum"
    elif composite > -0.1:
        signal = "neutral"
    elif composite > -0.3:
        signal = "downgrade_momentum"
    else:
        signal = "strong_downgrade_momentum"

    return {
        "composite_momentum_score": round(composite, 4),
        "momentum_signal": signal,
        "window_scores": {
            w: round(scores[i], 4) for i, w in enumerate(windows)
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Earnings Calendar
# ═══════════════════════════════════════════════════════════════════

def get_earnings_calendar(
    ticker: str,
    t: Any,
) -> Dict[str, Any]:
    """
    Get upcoming earnings date for a ticker from yfinance.

    Args:
        ticker: Ticker symbol
        t: yfinance Ticker object

    Returns:
        Dict with earnings date info
    """
    result: Dict[str, Any] = {
        "ticker": ticker,
        "earnings_date": None,
        "next_earnings_date": None,
        "days_to_earnings": None,
        "earnings_quarterly_growth": None,
        "revenue_quarterly_growth": None,
        "is_earnings_soon": False,
    }

    try:
        info = t.info

        # Next earnings date from calendar
        try:
            calendar = t.calendar
            if calendar and not (hasattr(calendar, 'empty') and calendar.empty):
                if isinstance(calendar, dict):
                    earnings_dates = calendar.get("Earnings Date", [])
                    if earnings_dates:
                        earliest = min(earnings_dates) if isinstance(earnings_dates, list) else earnings_dates
                        if hasattr(earliest, 'strftime'):
                            result["next_earnings_date"] = earliest.strftime("%Y-%m-%d")
                elif hasattr(calendar, 'iloc'):
                    ed = calendar.get("Earnings Date", calendar.get("Earnings Date", None))
                    if ed is not None:
                        result["next_earnings_date"] = str(ed)
        except Exception:
            pass

        # Fallback: use earnings dates from info
        if not result["next_earnings_date"]:
            ed_raw = info.get("earningsDate")
            if ed_raw:
                if isinstance(ed_raw, list):
                    # Sort and take the earliest future date
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    future_dates = [
                        d for d in ed_raw
                        if hasattr(d, 'timestamp') and d.timestamp() > now.timestamp()
                    ]
                    if future_dates:
                        result["next_earnings_date"] = min(future_dates).strftime("%Y-%m-%d")
                elif hasattr(ed_raw, 'strftime'):
                    result["next_earnings_date"] = ed_raw.strftime("%Y-%m-%d")

        # Days to next earnings
        if result["next_earnings_date"]:
            from datetime import datetime, timezone
            try:
                next_date = datetime.strptime(result["next_earnings_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                days = (next_date - now).days
                result["days_to_earnings"] = max(0, days)
                result["is_earnings_soon"] = 0 <= days <= 14
            except (ValueError, TypeError):
                pass

        result["earnings_quarterly_growth"] = info.get("earningsQuarterlyGrowth")
        result["revenue_quarterly_growth"] = info.get("revenueQuarterlyGrowth")

    except Exception as e:
        logger.debug(f"[{ticker}] Failed to get earnings calendar: {e}")

    return result


# ═══════════════════════════════════════════════════════════════════
# Pre-Earnings Sentiment
# ═══════════════════════════════════════════════════════════════════

def compute_pre_earnings_sentiment(
    revisions: Dict[str, Any],
    rating_activity: Dict[str, Any],
    target_changes: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute pre-earnings sentiment based on estimate revisions and rating activity.

    Args:
        revisions: Revision analysis from forecast
        rating_activity: Rating activity analysis
        target_changes: Target price change analysis

    Returns:
        Dict with pre-earnings sentiment assessment
    """
    signals = []

    # Estimate revision direction
    rev_1m = revisions.get("revisions_1m", {})
    rev_momentum = rev_1m.get("momentum", "neutral")
    if rev_momentum == "positive":
        signals.append(("estimate_revisions", 1))
    elif rev_momentum == "negative":
        signals.append(("estimate_revisions", -1))

    # Rating momentum
    rating_1m = rating_activity.get("rating_activity_1m", {})
    rating_momentum = rating_1m.get("momentum", "no_activity")
    if rating_momentum == "positive":
        signals.append(("rating_momentum", 1))
    elif rating_momentum == "negative":
        signals.append(("rating_momentum", -1))

    # Target price momentum
    target_momentum = target_changes.get("target_momentum", "no_recent_target_changes")
    if target_momentum == "targets_rising":
        signals.append(("target_momentum", 1))
    elif target_momentum == "targets_falling":
        signals.append(("target_momentum", -1))

    # Composite sentiment
    total = sum(s[1] for s in signals) if signals else 0
    count = len(signals)

    if count == 0:
        sentiment = "neutral"
    elif total >= 2:
        sentiment = "bullish"
    elif total >= 1:
        sentiment = "slightly_bullish"
    elif total <= -2:
        sentiment = "bearish"
    elif total <= -1:
        sentiment = "slightly_bearish"
    else:
        sentiment = "neutral"

    return {
        "pre_earnings_sentiment": sentiment,
        "sentiment_score": round(total / max(count, 1), 2),
        "signal_breakdown": signals,
        "active_signals": count,
        "total_signals_checked": 3,
    }


# ═══════════════════════════════════════════════════════════════════
# Main Rating Analyzer
# ═══════════════════════════════════════════════════════════════════

def analyze_ratings(
    ticker: str,
    broker_data: Dict[str, Any],
    t: Any = None,
    forecast_data: Optional[Dict[str, Any]] = None,
    window_days_1w: int = 7,
    window_days_1m: int = 30,
    window_days_3m: int = 90,
) -> Dict[str, Any]:
    """
    Run full ratings analysis for a ticker.

    Args:
        ticker: Ticker symbol
        broker_data: Broker report dict
        t: yfinance Ticker object (for earnings calendar)
        forecast_data: Forecast analysis dict (for revisions)
        window_days_1w/1m/3m: Analysis windows

    Returns:
        Full ratings report dict
    """
    if broker_data.get("error"):
        return {"ticker": ticker, "error": broker_data["error"]}

    rec_history = broker_data.get("recommendation_history", [])

    # Upgrade/downgrade detection
    rating_activity = detect_upgrade_downgrade_activity(
        rec_history, window_days_1w, window_days_1m, window_days_3m
    )

    # Target price changes (1-month window)
    target_changes = detect_target_price_changes(rec_history, window_days=window_days_1m)

    # Rating momentum
    momentum = compute_rating_momentum(rating_activity)

    # Earnings calendar
    calendar = {}
    if t is not None:
        calendar = get_earnings_calendar(ticker, t)

    # Pre-earnings sentiment
    revisions = forecast_data.get("revisions", {}) if forecast_data else {}
    sentiment = compute_pre_earnings_sentiment(revisions, rating_activity, target_changes)

    # Coverage summary
    coverage = broker_data.get("broker_coverage", [])
    buy_count = sum(1 for c in coverage if c.get("latest_rating") == "BUY")
    hold_count = sum(1 for c in coverage if c.get("latest_rating") == "HOLD")
    sell_count = sum(1 for c in coverage if c.get("latest_rating") == "SELL")
    total = len(coverage)

    coverage_summary = {
        "total_firms": total,
        "buy_pct": round(buy_count / total, 4) if total > 0 else None,
        "hold_pct": round(hold_count / total, 4) if total > 0 else None,
        "sell_pct": round(sell_count / total, 4) if total > 0 else None,
        "bullish_pct": round((buy_count) / total, 4) if total > 0 else None,
        "bearish_pct": round((sell_count) / total, 4) if total > 0 else None,
    }

    return {
        "ticker": ticker,
        "rating_activity": rating_activity,
        "target_price_changes": target_changes,
        "rating_momentum": momentum,
        "earnings_calendar": calendar,
        "pre_earnings_sentiment": sentiment,
        "coverage_summary": coverage_summary,
    }
