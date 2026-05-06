#!/usr/bin/env python3
"""
VMAA Earnings Prediction Engine — Broker Report Aggregation
=============================================================
Collects analyst ratings, price targets, earnings estimates, and
recommendation history from yfinance.

Data sources:
  - yfinance `info` dict: analyst ratings, price targets, earnings estimates
  - yfinance `recommendations`: upgrade/downgrade history
  - yfinance `earnings_dates`: surprise history

Output format:
  {ticker: [(date, broker, rating_from, rating_to, target_from, target_to)]}

Usage:
  from engine.earnings.broker_reports import collect_broker_reports
  reports = collect_broker_reports(["AAPL", "MSFT"])
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("vmaa.engine.earnings.broker_reports")

# ═══════════════════════════════════════════════════════════════════
# Rating normalization
# ═══════════════════════════════════════════════════════════════════

RATING_MAP: Dict[str, str] = {
    "strong_buy": "BUY",
    "buy": "BUY",
    "outperform": "BUY",
    "overweight": "BUY",
    "positive": "BUY",
    "sector_outperform": "BUY",
    "market_outperform": "BUY",
    "long-term_buy": "BUY",
    "neutral": "HOLD",
    "hold": "HOLD",
    "equal-weight": "HOLD",
    "sector_perform": "HOLD",
    "market_perform": "HOLD",
    "peer_perform": "HOLD",
    "in-line": "HOLD",
    "underperform": "SELL",
    "underweight": "SELL",
    "sell": "SELL",
    "negative": "SELL",
    "sector_underperform": "SELL",
    "reduce": "SELL",
}


def normalize_rating(rating: Optional[str]) -> str:
    """
    Normalize a free-form analyst rating to HOLD/BUY/SELL.

    Args:
        rating: Raw rating string from yfinance

    Returns:
        Normalized rating: HOLD, BUY, or SELL
    """
    if not rating:
        return "HOLD"
    key = rating.lower().strip().replace(" ", "_").replace("-", "_")
    return RATING_MAP.get(key, "HOLD")


def rating_to_score(rating: str) -> float:
    """
    Convert normalized rating to numeric score 1-5 (higher = more bullish).

    5.0 = Strong Buy, 4.0 = Buy/Outperform, 3.0 = Hold, 2.0 = Underperform, 1.0 = Sell
    """
    scores = {"BUY": 4.0, "HOLD": 3.0, "SELL": 2.0}
    return scores.get(rating, 3.0)


# ═══════════════════════════════════════════════════════════════════
# Analyst Info from yfinance info dict
# ═══════════════════════════════════════════════════════════════════

@pd.api.extensions.register_dataframe_accessor("_earnings_ext")
class _EarningsExtension:
    """Internal accessor for pandas DataFrame earnings extensions."""
    def __init__(self, pandas_obj):
        self._obj = pandas_obj


def collect_analyst_info(ticker: str, info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract analyst rating and target data from yfinance info dict.

    Args:
        ticker: Ticker symbol
        info: yfinance Ticker.info dict

    Returns:
        Dict with normalized analyst data
    """
    raw_rec = info.get("recommendationKey", "")
    recommendation = normalize_rating(raw_rec)

    return {
        "ticker": ticker,
        "recommendation_key": raw_rec or "none",
        "recommendation": recommendation,
        "recommendation_score": rating_to_score(recommendation),
        "target_mean": float(info.get("targetMeanPrice", 0) or 0),
        "target_high": float(info.get("targetHighPrice", 0) or 0),
        "target_low": float(info.get("targetLowPrice", 0) or 0),
        "target_median": float(info.get("targetMedianPrice", 0) or 0),
        "num_analysts": int(info.get("numberOfAnalystOpinions", 0) or 0),
        "current_price": float(
            info.get("currentPrice", 0)
            or info.get("regularMarketPrice", 0)
            or info.get("previousClose", 0)
            or 0
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Upgrade/Downgrade History from recommendations
# ═══════════════════════════════════════════════════════════════════

def collect_recommendation_history(
    ticker: str,
    t: Any,  # yfinance Ticker object
) -> List[Dict[str, Any]]:
    """
    Collect rating change history (upgrades/downgrades) from yfinance.

    Args:
        ticker: Ticker symbol
        t: yfinance Ticker object

    Returns:
        List of rating change dicts with date, firm, from/to grades
    """
    changes: List[Dict[str, Any]] = []

    try:
        recs = t.recommendations
        if recs is None or recs.empty:
            return changes

        # Sort by date descending
        recs = recs.sort_index(ascending=False)

        prev_grade = None
        prev_firm = None

        for idx, row in recs.iterrows():
            date = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
            firm = str(row.get("Firm", row.get("To Grade", "")) if "Firm" in row.index else "Unknown")
            to_grade = str(row.get("To Grade", ""))
            from_grade = str(row.get("From Grade", "")) if "From Grade" in row.index else ""

            # If no "From Grade" column, infer from previous row
            if not from_grade and prev_grade:
                from_grade = prev_grade

            to_norm = normalize_rating(to_grade)
            from_norm = normalize_rating(from_grade) if from_grade else None

            # Detect upgrade/downgrade
            change_type = "maintain"
            if from_norm:
                to_score = rating_to_score(to_norm)
                from_score = rating_to_score(from_norm)
                if to_score > from_score:
                    change_type = "upgrade"
                elif to_score < from_score:
                    change_type = "downgrade"

            changes.append({
                "date": date.strftime("%Y-%m-%d"),
                "firm": firm,
                "from_grade": from_grade if from_grade else "N/A",
                "to_grade": to_grade,
                "from_normalized": from_norm or "N/A",
                "to_normalized": to_norm,
                "change_type": change_type,
            })

            prev_grade = to_grade
            prev_firm = firm

    except Exception as e:
        logger.debug(f"[{ticker}] Failed to collect recommendation history: {e}")

    return changes


# ═══════════════════════════════════════════════════════════════════
# Earnings Surprise History from earnings_dates
# ═══════════════════════════════════════════════════════════════════

def collect_earnings_surprise_history(
    ticker: str,
    t: Any,  # yfinance Ticker object
    lookback_quarters: int = 8,
) -> List[Dict[str, Any]]:
    """
    Collect earnings surprise history from yfinance earnings_dates.

    Args:
        ticker: Ticker symbol
        t: yfinance Ticker object
        lookback_quarters: Number of quarters to retrieve

    Returns:
        List of surprise dicts with date, reported/estimated EPS, surprise
    """
    surprises: List[Dict[str, Any]] = []

    try:
        ed = t.earnings_dates
        if ed is None or ed.empty:
            return surprises

        # Sort by date descending
        ed = ed.sort_index(ascending=False)

        count = 0
        for idx, row in ed.iterrows():
            if count >= lookback_quarters:
                break

            date = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
            reported = float(row.get("Reported EPS", row.get("Earnings Actual", 0)) or 0)
            estimated = float(row.get("Surprise(%)", 0) or 0)

            # If surprise is stored as percentage, compute actual surprise
            surprise_pct_raw = float(row.get("Surprise(%)", 0) or 0)

            # Some yfinance versions store absolute surprise, others store %
            if abs(surprise_pct_raw) <= 2.0 and reported > 0 and estimated > 0:
                # Likely absolute surprise
                surprise_pct = (surprise_pct_raw / reported) if reported else 0
                surprise_abs = surprise_pct_raw
            else:
                # Likely percentage already
                surprise_pct = surprise_pct_raw / 100.0
                # Estimate actual surprise
                surprise_abs = reported * surprise_pct if reported > 0 else 0
                estimated = reported - surprise_abs

            beat = surprise_pct > 0.001 if abs(surprise_pct) > 0.001 else None

            surprises.append({
                "date": date.strftime("%Y-%m-%d"),
                "reported_eps": round(reported, 4),
                "estimated_eps": round(estimated, 4) if estimated else None,
                "surprise_pct": round(surprise_pct, 4),
                "surprise_abs": round(surprise_abs, 4),
                "beat": beat,
            })

            count += 1

    except Exception as e:
        logger.debug(f"[{ticker}] Failed to collect earnings surprise history: {e}")

    return surprises


# ═══════════════════════════════════════════════════════════════════
# EPS/Revenue Estimates from info dict
# ═══════════════════════════════════════════════════════════════════

def collect_earnings_estimates(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract EPS and revenue estimates from yfinance info dict.

    Args:
        info: yfinance Ticker.info dict

    Returns:
        Dict with forward/revenue/growth estimates
    """
    return {
        "forward_eps": float(info.get("forwardEps", 0) or 0),
        "trailing_eps": float(info.get("trailingEps", 0) or 0),
        "revenue_per_share": float(info.get("revenuePerShare", 0) or 0),
        "earnings_growth": float(info.get("earningsGrowth", 0) or 0),
        "earnings_quarterly_growth": float(info.get("earningsQuarterlyGrowth", 0) or 0),
        "revenue_growth": float(info.get("revenueGrowth", 0) or 0),
        "forward_pe": float(info.get("forwardPE", 0) or 0),
        "trailing_pe": float(info.get("trailingPE", 0) or 0),
        "peg_ratio": float(info.get("pegRatio", 0) or 0),
        "ltg_estimate": float(info.get("longTermGrowthEstimate", info.get("earningsGrowth", 0)) or 0),
        "eps_current_year": float(info.get("epsCurrentYear", 0) or 0),
        "eps_next_year": float(info.get("epsNextYear", 0) or 0),
        "eps_next_quarter": float(info.get("epsNextQuarter", 0) or 0),
        "eps_current_quarter": float(info.get("epsCurrentQuarter", 0) or 0),
        "revenue_estimate_current_year": float(info.get("revenueEstimateCurrentYear", 0) or 0),
        "revenue_estimate_next_year": float(info.get("revenueEstimateNextYear", 0) or 0),
    }


# ═══════════════════════════════════════════════════════════════════
# Broker Coverage Tracking
# ═══════════════════════════════════════════════════════════════════

def track_broker_coverage(ticker: str, t: Any) -> List[Dict[str, Any]]:
    """
    Track which brokers/analysts cover a stock and their latest ratings.

    Args:
        ticker: Ticker symbol
        t: yfinance Ticker object

    Returns:
        List of dicts per firm: {firm, latest_rating, latest_date}
    """
    coverage: List[Dict[str, Any]] = []
    seen_firms: set = set()

    try:
        recs = t.recommendations
        if recs is None or recs.empty:
            return coverage

        recs = recs.sort_index(ascending=False)

        for idx, row in recs.iterrows():
            firm = str(row.get("Firm", ""))
            if not firm or firm in seen_firms:
                continue

            date = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
            to_grade = str(row.get("To Grade", ""))

            coverage.append({
                "firm": firm,
                "latest_rating": normalize_rating(to_grade),
                "latest_raw_rating": to_grade,
                "latest_date": date.strftime("%Y-%m-%d"),
            })
            seen_firms.add(firm)

    except Exception as e:
        logger.debug(f"[{ticker}] Failed to track broker coverage: {e}")

    return coverage


# ═══════════════════════════════════════════════════════════════════
# Main collector
# ═══════════════════════════════════════════════════════════════════

def collect_broker_reports(
    tickers: List[str],
    surprise_lookback: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """
    Collect comprehensive broker/analyst reports for a list of tickers.

    Args:
        tickers: List of ticker symbols
        surprise_lookback: Number of quarters for surprise history

    Returns:
        Dict keyed by ticker with info, recommendations, surprises, coverage
    """
    result: Dict[str, Dict[str, Any]] = {}

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not available")
        return result

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info

            if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
                logger.debug(f"[{ticker}] No market data in info")
                continue

            analyst_info = collect_analyst_info(ticker, info)
            rec_history = collect_recommendation_history(ticker, t)
            surprise_history = collect_earnings_surprise_history(ticker, t, surprise_lookback)
            estimates = collect_earnings_estimates(info)
            coverage = track_broker_coverage(ticker, t)

            result[ticker] = {
                "ticker": ticker,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "analyst_info": analyst_info,
                "recommendation_history": rec_history,
                "surprise_history": surprise_history,
                "estimates": estimates,
                "broker_coverage": coverage,
                "coverage_count": len(coverage),
            }

            logger.info(
                f"[{ticker}] Broker reports collected: "
                f"{len(rec_history)} rating changes, "
                f"{len(surprise_history)} surprises, "
                f"{len(coverage)} firms covering"
            )

        except Exception as e:
            logger.warning(f"[{ticker}] Failed to collect broker reports: {e}")
            result[ticker] = {
                "ticker": ticker,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    logger.info(f"Collected broker reports for {len(result)}/{len(tickers)} tickers")
    return result


# ═══════════════════════════════════════════════════════════════════
# Timeline tracking: rating changes over time
# ═══════════════════════════════════════════════════════════════════

def build_rating_timeline(
    ticker: str,
    t: Any,
    days_back: int = 365,
) -> List[Dict[str, Any]]:
    """
    Build a timeline of rating changes for a stock.

    Returns list of (date, broker, rating_from, rating_to, target_from, target_to)
    with price target changes when available.

    Args:
        ticker: Ticker symbol
        t: yfinance Ticker object
        days_back: Days of history to include

    Returns:
        List of timeline entries
    """
    timeline: List[Dict[str, Any]] = []
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)

    try:
        recs = t.recommendations
        if recs is None or recs.empty:
            return timeline

        recs = recs.sort_index(ascending=False)

        prev_grade = None
        prev_target = None

        for idx, row in recs.iterrows():
            date = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
            if date.tz_localize(None) < cutoff.tz_localize(None):
                continue

            firm = str(row.get("Firm", "Unknown"))
            to_grade = str(row.get("To Grade", ""))
            from_grade = str(row.get("From Grade", "")) if "From Grade" in row.index else (prev_grade or "")
            action = str(row.get("Action", ""))

            # Extract target price if available
            target_to = None
            if action and ("target" in action.lower() or "pt" in action.lower()):
                try:
                    # Target is often in the action field like "target raised to $200"
                    import re
                    match = re.search(r'\$?([\d,.]+)', action)
                    if match:
                        target_to = float(match.group(1).replace(",", ""))
                except (ValueError, AttributeError):
                    pass

            entry = {
                "date": date.strftime("%Y-%m-%d"),
                "firm": firm,
                "rating_from": normalize_rating(from_grade) if from_grade else "N/A",
                "rating_to": normalize_rating(to_grade),
                "target_from": prev_target,
                "target_to": target_to,
                "action": action,
            }
            timeline.append(entry)

            prev_grade = to_grade
            if target_to:
                prev_target = target_to

    except Exception as e:
        logger.debug(f"[{ticker}] Failed to build rating timeline: {e}")

    return timeline
