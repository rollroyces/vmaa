#!/usr/bin/env python3
"""
VMAA 2.0 — Risk Management Layer
=================================
Position sizing, stop management, correlation checks, sector limits.
Works with Part1 + Part2 results to generate TradeDecisions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from vmaa.config import RC
from vmaa.models import (
    MarketRegime, Part1Result, Part2Signal, VMAACandidate, TradeDecision
)

# Adaptive stop (Phase 1 — 2026-05-06)
# Dynamically adjusts stop distance based on price level, volatility, market regime
from vmaa.risk_adaptive import compute_stops_adaptive as compute_stops

# Lazy import compute_atr to avoid circular dependency
_compute_atr = None
def _get_atr():
    global _compute_atr
    if _compute_atr is None:
        from vmaa.risk_adaptive import compute_atr as ca
        _compute_atr = ca
    return _compute_atr

logger = logging.getLogger("vmaa.risk")


# ═══════════════════════════════════════════════════════════════════
# Market Regime Detection
# ═══════════════════════════════════════════════════════════════════

def get_market_regime() -> MarketRegime:
    """Assess overall market conditions for sizing and risk calibration."""
    import time as _time
    for attempt in range(3):
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(period="3mo")
            if hist is None or len(hist) < 20:
                raise ValueError("Insufficient data")
            info = spy.info

            current = float(hist['Close'].iloc[-1]) if len(hist) > 0 else 0
            ma50 = float(hist['Close'].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else 0
            above_ma = current > ma50 if ma50 > 0 else True

            returns = hist['Close'].pct_change().dropna()
            vol_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.15

            if vol_20d < 0.12:
                vol_regime = "LOW"
                scalar = 1.0
            elif vol_20d < 0.22:
                vol_regime = "NORMAL"
                scalar = 0.80
            else:
                vol_regime = "HIGH"
                scalar = 0.50

            high_3mo = float(hist['High'].max())
            dd_from_high = (current - high_3mo) / high_3mo if high_3mo > 0 else 0
            market_ok = above_ma and (dd_from_high > -0.12)

            return MarketRegime(
                spy_price=round(current, 2),
                spy_ma50=round(ma50, 2),
                above_ma50=above_ma,
                volatility_20d=round(vol_20d, 4),
                vol_regime=vol_regime,
                dd_from_3mo_high=round(dd_from_high, 4),
                market_ok=market_ok,
                position_scalar=scalar,
                vix_proxy=vol_20d,
            )
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 2
                logger.warning(f"Market regime check attempt {attempt+1} failed ({e}), retrying in {wait}s")
                _time.sleep(wait)
            else:
                logger.warning(f"Market regime check failed after 3 attempts: {e}")

    return MarketRegime(
        spy_price=0, spy_ma50=0, above_ma50=True,
        volatility_20d=0.15, vol_regime="UNKNOWN",
        dd_from_3mo_high=0, market_ok=True,
        position_scalar=0.75,
    )


# ═══════════════════════════════════════════════════════════════════
# Per-Stock Adaptive Trailing Stop (v3.2.1)
# ═══════════════════════════════════════════════════════════════════

def compute_trailing_stop(candidate: VMAACandidate, entry_price: float,
                          hist: pd.DataFrame) -> tuple:
    """
    V3.4 ATR-based trailing stop — S3 from backtest (Sharpe 3.11, +517%).
    
    0.5×ATR trailing from high watermark, activated immediately.
    Lets winners run while capping drawdowns.
    
    Returns: (trail_pct, activate_pct)
      trail_pct = 0.5 × ATR% (floor 6%, ceiling 15%)
      activate_pct = 0.01 (essentially immediate)
    """
    atr_pct = 0.03
    try:
        if hist is not None and len(hist) >= 14:
            ca = _get_atr()
            hist_df = hist if isinstance(hist, pd.DataFrame) else hist
            atr_val = ca(hist_df, 14)
            atr_pct = float(atr_val / entry_price) if entry_price > 0 else 0.03
    except Exception:
        pass
    
    # V3.4: 0.5×ATR trailing, floor 6%, ceiling 15%
    trail = max(0.06, min(0.15, atr_pct * 0.5))
    activate = 0.01  # Immediate trailing from near entry
    
    logger.info(f"  {candidate.ticker}: V3.4 trail={trail:.0%} (ATR={atr_pct:.1%})")
    return round(trail, 3), round(activate, 3)


# ═══════════════════════════════════════════════════════════════════
# Position Sizing (Fixed Fractional)
# ═══════════════════════════════════════════════════════════════════

def compute_position_size(
    ticker: str,
    entry_price: float,
    stop_loss: float,
    portfolio_value: float,
    confidence: float,
    market: MarketRegime,
) -> Tuple[int, float, float]:
    """
    Compute position size using Fixed Fractional sizing.
    
    Changed from Quarter-Kelly to Fixed Fractional because:
    - Kelly was double-penalizing (win_prob=0.5*confidence + quarter-kelly)
    - Result: confidence < 67% always produced 0 shares 
    - Fixed fractional: 1.5% base risk * confidence multiplier
    - Confidence >= 35% guarantees a position
    
    Returns: (quantity, position_pct, risk_capital)
    """
    # Risk per share
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        risk_per_share = entry_price * 0.05  # Fallback 5%

    # Fixed Fractional: base_risk% * confidence scalar
    base_risk_pct = 0.015  # 1.5% of portfolio per trade
    confidence_scalar = max(0.35, min(confidence, 1.0))  # 0.35 - 1.0x
    risk_pct = base_risk_pct * confidence_scalar
    
    risk_capital = portfolio_value * risk_pct
    risk_shares = int(risk_capital / risk_per_share) if risk_per_share > 0 else 0

    # Allocation-based sizing (cap at max_position_pct)
    max_alloc = portfolio_value * RC.max_position_pct
    alloc_shares = int(max_alloc / entry_price) if entry_price > 0 else 0

    # Market-adjusted sizing
    market_shares = int(risk_shares * market.position_scalar)

    # Median selection: balances risk-driven, allocation-cap, and market-adjusted views
    quantities = [q for q in [risk_shares, alloc_shares, market_shares] if q > 0]
    quantity = int(np.median(quantities)) if quantities else 0
    quantity = max(1, min(quantity, 10000))

    # Min/max position checks
    position_value = quantity * entry_price
    if position_value < RC.min_position_size:
        quantity = 0
    elif position_value > RC.max_position_size:
        quantity = int(RC.max_position_size / entry_price)

    position_pct = (quantity * entry_price / portfolio_value * 100) if portfolio_value > 0 else 0

    return quantity, round(position_pct, 2), round(risk_capital, 2)


# ═══════════════════════════════════════════════════════════════════
# Take Profit Levels
# ═══════════════════════════════════════════════════════════════════

def compute_take_profits(
    entry_price: float,
    market: MarketRegime,
) -> List[Dict[str, Any]]:
    """
    Compute take profit levels — FIXED R:R strategy.
    
    ATR-based trailing TP — V3.4 (S3 Trailing TP from backtest).
    
    Backtest result: Sharpe 2.30→3.11, Return +194%→+517%
    
    Instead of fixed TP levels, use 0.5×ATR trailing from peak price.
    This lets winners run while capping drawdown on pullbacks.
    
    Strategy:
      - TP1-A: +15% sell 50% (early lock-in)
      - TP1-B: remaining 50% trails at 0.5×ATR from peak
        (max trail = 25% gain from entry, min trail = 0.5×ATR)
      - Floor: never let TP1-B level go below entry_price (breakeven)
    """
    ca = _get_atr()
    
    tp1a_pct = RC.tp1_a_level_pct  # 0.15
    tp1a = round(entry_price * (1 + tp1a_pct), 2)
    
    # TP1-B is a marker level for trailing — the actual exit is managed
    # by the trailing stop (0.5×ATR from high watermark).
    # Set it high enough that trailing fires first for most trades.
    tp1b = round(entry_price * 1.25, 2)  # +25% ceiling for trailing
    
    return [
        {"level": tp1a, "sell_pct": 50,  "label": "TP1-A"},       # +15%: sell 50%
        {"level": tp1b, "sell_pct": 50,  "label": "TP1-B_TRAIL"}, # 50% trails via trailing stop
    ]


# ═══════════════════════════════════════════════════════════════════
# Entry Price Calculation
# ═══════════════════════════════════════════════════════════════════

def compute_entry(
    ticker: str,
    current_price: float,
    low_52w: float,
    hist: pd.DataFrame,
    gap_detected: bool,
    gap_pct: float,
) -> Tuple[float, str]:
    """
    Compute optimal entry price based on signal type.
    
    Entry methods:
      - gap_entry: Buy 5% below gap day high (pullback after gap)
      - 52w_low_bounce: Buy at 52w low + 0.5%
      - base_breakout: Buy at base high + 1% breakout
      - current_price: Fallback
    """
    if gap_detected and gap_pct >= 0.04:
        # Find gap day and compute entry
        entry = _compute_gap_entry(hist, low_52w)
        return entry, "gap_entry"

    # Near 52w low — bounce entry
    if current_price <= low_52w * 1.05:
        entry = round(low_52w * 1.005, 2)
        return max(entry, current_price), "52w_low_bounce"

    # In base — breakout entry (approximate)
    if _is_in_base(hist):
        base_high = float(hist['High'].tail(60).max())
        entry = round(base_high * 1.01, 2)
        if entry > current_price * 1.03:  # Don't chase too far
            entry = round(current_price * 1.01, 2)
        return entry, "base_breakout"

    # Fallback: current price
    return round(current_price, 2), "current_price"


# ═══════════════════════════════════════════════════════════════════
# Confidence Scoring
# ═══════════════════════════════════════════════════════════════════

def compute_confidence(
    candidate: VMAACandidate,
    market: MarketRegime,
) -> float:
    """
    Compute overall trade confidence from Part 1 quality + Part 2 signal +
    Part 3 sentiment + market.
    Returns: 0.0–1.0 confidence score.
    """
    confidence = 0.0
    p1 = candidate.part1
    p2 = candidate.part2
    sentiment = candidate.sentiment  # Part 3

    # Part 1 quality contribution (35% weight — was 40%)
    confidence += p1.quality_score * 0.35

    # Part 2 signal strength (30% weight — was 35%)
    confidence += (p2.magna_score / 10) * 0.30

    # Part 3 sentiment (15% weight — NEW)
    sentiment_notes = []
    if sentiment is not None:
        # Base sentiment contribution
        # For value mean-reversion: contrarian sentiment is GOOD
        base_confidence = confidence
        confidence, sentiment_notes = _apply_sentiment_to_confidence(
            sentiment, confidence
        )
        candidate._sentiment_notes = sentiment_notes
    else:
        # No sentiment data — redistribute to market/technical
        pass

    # Market regime (10% weight — was 15%, reduced for sentiment)
    if market.market_ok:
        confidence += 0.10
    elif market.vol_regime == "LOW":
        confidence += 0.05

    # Technical edge (10% weight — unchanged)
    if p1.ptl_ratio < 1.05:
        confidence += 0.05
    if p2.entry_ready:
        confidence += 0.05
    if p2.g_gap_up:
        confidence += 0.02
    if p2.short_interest_score >= 2:
        confidence += 0.03

    return round(min(confidence, 1.0), 3)


def _apply_sentiment_to_confidence(
    sentiment,
    base_confidence: float,
) -> tuple:
    """
    Apply sentiment analysis to confidence scoring.
    
    Value mean-reversion logic:
      CONTRARIAN_BUY (extreme negative sentiment) → boost confidence
        (fear creates opportunity — buy when others are fearful)
      CROWDED_TRADE (extreme positive sentiment) → reduce confidence
        (euphoria means limited upside, everyone is already in)
      NEWS_EXTREME_NEGATIVE → slight boost (overreaction opportunity)
      NEWS_EXTREME_POSITIVE → slight cut (euphoria risk)
      HIGH_ANALYST_UPSIDE → boost
      SENTIMENT_DIVERGENCE → directional adjustment
    """
    from part3_sentiment import sentiment_confidence_adjustment
    return sentiment_confidence_adjustment(sentiment, base_confidence)


# ═══════════════════════════════════════════════════════════════════
# Volatility Filter (v3.3 — MAGNA)
# ═══════════════════════════════════════════════════════════════════

def check_volatility_filter(ticker: str, hist: pd.DataFrame = None) -> Tuple[bool, float, str]:
    """
    Check if a stock passes the avg daily range filter.
    Exclude stocks with >5% average daily range (too volatile for strategy).
    
    Returns: (passes_filter, avg_daily_range_pct, reason)
    """
    max_range = RC.max_avg_daily_range_pct
    try:
        if hist is not None and len(hist) >= 20:
            daily_range = (hist['High'] - hist['Low']) / hist['Close']
            avg_range = float(daily_range.tail(20).mean())
            if avg_range > max_range:
                return False, round(avg_range, 4), \
                    f"Avg daily range {avg_range:.1%} > {max_range:.0%} (too volatile)"
            return True, round(avg_range, 4), "OK"
        t = yf.Ticker(ticker)
        h = t.history(period="1mo")
        if h is not None and len(h) >= 20:
            daily_range = (h['High'] - h['Low']) / h['Close']
            avg_range = float(daily_range.tail(20).mean())
            if avg_range > max_range:
                return False, round(avg_range, 4), \
                    f"Avg daily range {avg_range:.1%} > {max_range:.0%} (too volatile)"
            return True, round(avg_range, 4), "OK"
    except Exception as e:
        logger.warning(f"Volatility filter failed for {ticker}: {e}")
    # Default: pass (no data available)
    return True, 0.0, "No data — passing"


# ═══════════════════════════════════════════════════════════════════
# Correlation & Sector Checks
# ═══════════════════════════════════════════════════════════════════

def check_correlation(new_ticker: str, existing_tickers: List[str]) -> float:
    """Max correlation between new ticker and existing positions.
    
    Handles delisted/invalid tickers gracefully: if yf.download fails
    (e.g. one ticker is delisted), falls back to individual yf.Ticker()
    fetches, skipping failed ones.
    """
    if not existing_tickers:
        return 0.0
    all_tickers = [new_ticker] + existing_tickers

    # Primary: batch download (fast but fragile to a single bad ticker)
    try:
        data = yf.download(all_tickers, period="3mo", progress=False)['Close']
        returns = data.pct_change().dropna()
        if returns.shape[1] >= 2:
            corr_matrix = returns.corr()
            if new_ticker in corr_matrix.columns:
                new_corr = corr_matrix[new_ticker].drop(new_ticker, errors='ignore')
                return float(new_corr.max())
    except Exception:
        pass

    # Fallback: individual ticker fetch, skip failures (delisted/invalid)
    try:
        data = pd.DataFrame()
        for t in all_tickers:
            try:
                tkr = yf.Ticker(t)
                hist = tkr.history(period="3mo")
                if not hist.empty and 'Close' in hist.columns and len(hist) >= 20:
                    data[t] = hist['Close']
            except Exception:
                pass  # Skip delisted/invalid tickers
        if data.shape[1] >= 2:
            returns = data.pct_change().dropna()
            if returns.shape[1] >= 2:
                corr_matrix = returns.corr()
                if new_ticker in corr_matrix.columns:
                    new_corr = corr_matrix[new_ticker].drop(new_ticker, errors='ignore')
                    if len(new_corr) > 0:
                        return float(new_corr.max())
    except Exception:
        pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════
# Full Trade Decision
# ═══════════════════════════════════════════════════════════════════

def generate_trade_decision(
    candidate: VMAACandidate,
    portfolio_value: float,
    existing_tickers: List[str],
    market: MarketRegime,
) -> TradeDecision:
    """
    Generate complete trade decision with entry, stops, sizing, and risk.
    This is the final output before execution.
    """
    ticker = candidate.ticker
    p1 = candidate.part1
    p2 = candidate.part2
    risk_flags: List[str] = []

    # ── Blacklist check (v3.3) ──
    if ticker.upper() in [t.upper() for t in RC.blacklisted_tickers]:
        return TradeDecision(
            ticker=ticker, action='HOLD', quantity=0,
            entry_price=0, entry_method='n/a',
            stop_loss=0, stop_type='n/a', take_profits=[],
            trailing_stop_pct=0, time_stop_days=0,
            position_pct=0, risk_amount=0, reward_ratio=0,
            confidence_score=0, risk_flags=['blacklisted'],
            rationale=f"{ticker} is blacklisted"
        )

    # Fetch fresh price data
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="6mo")
    except Exception:
        return TradeDecision(
            ticker=ticker, action='HOLD', quantity=0,
            entry_price=0, entry_method='n/a',
            stop_loss=0, stop_type='n/a', take_profits=[],
            trailing_stop_pct=0, time_stop_days=0,
            position_pct=0, risk_amount=0, reward_ratio=0,
            confidence_score=0, risk_flags=['data_fetch_failed'],
            rationale=f"Data fetch failed for {ticker}"
        )

    # ── Volatility filter (v3.3) — reject stocks with >5% avg daily range ──
    vol_ok, avg_range, vol_reason = check_volatility_filter(ticker, hist)
    if not vol_ok:
        return TradeDecision(
            ticker=ticker, action='HOLD', quantity=0,
            entry_price=0, entry_method='n/a',
            stop_loss=0, stop_type='n/a', take_profits=[],
            trailing_stop_pct=0, time_stop_days=0,
            position_pct=0, risk_amount=0, reward_ratio=0,
            confidence_score=0, risk_flags=['volatility_filter'],
            rationale=f"{ticker}: {vol_reason}"
        )

    current = p1.current_price

    # ── Entry Price ──
    entry_price, entry_method = compute_entry(
        ticker, current, p1.low_52w, hist,
        p2.g_gap_up, p2.gap_pct
    )

    # ── Stop Loss ──
    # Use adaptive stop (Phase 1) — adjusts for price level, volatility, regime
    stop_loss, stop_type = compute_stops(entry_price, p1.low_52w, hist, market)

    # ── Take Profits ──
    take_profits = compute_take_profits(entry_price, market)

    # ── Confidence ──
    confidence = compute_confidence(candidate, market)

    # ── VCP Adjustment (Stage 2.5) ──
    vcp = getattr(candidate, '_vcp', None)
    if vcp is not None and vcp.vcp_detected:
        vcp_stop_pct = vcp.stop_pct if vcp.stop_pct > 0 else (entry_price - stop_loss) / entry_price
        orig_stop_pct = (entry_price - stop_loss) / entry_price if entry_price > 0 else 0
        # Tighten stop by VCP suggestion (with floor)
        new_stop_pct = min(orig_stop_pct, vcp_stop_pct * 1.2)  # Give 20% breathing room from VCP stop
        stop_loss = round(entry_price * (1 - new_stop_pct), 2)
        stop_type = f"VCP_{stop_type}"
        confidence = round(min(confidence + 0.10 * vcp.vcp_quality, 1.0), 4)
        risk_flags.append(f"VCP_confirmed_Q={vcp.vcp_quality:.0%}")

    # ── Per-Stock Adaptive Trailing Stop (v3.2.1) ──
    trailing_stop_pct, trailing_activate_pct = compute_trailing_stop(
        candidate, entry_price, hist
    )

    # ── Position Sizing ──
    quantity, position_pct, risk_capital = compute_position_size(
        ticker, entry_price, stop_loss, portfolio_value, confidence, market
    )

    # ── Risk Flags ──
    if not market.market_ok:
        risk_flags.append("Market below 50MA or deep drawdown")
    if p1.interest_rate_sensitive:
        risk_flags.append("IR_sensitive")
    if p1.debt_to_equity > 100 and p1.debt_to_equity < 9999:
        risk_flags.append(f"High_D/E={p1.debt_to_equity:.0f}")
    if confidence < 0.40:
        risk_flags.append("Low_confidence")
    if quantity == 0:
        risk_flags.append("Position_too_small")

    # ── Risk/Reward ──
    risk_per_share = entry_price - stop_loss
    avg_reward = sum(
        (tp['level'] - entry_price) * (tp['sell_pct'] / 100)
        for tp in take_profits
    )
    rr = avg_reward / risk_per_share if risk_per_share > 0 else 0

    # ── Decision ──
    if quantity <= 0 or confidence < 0.30:
        action = 'HOLD'
        if candidate.sentiment is not None and candidate.sentiment.composite_score < -0.20:
            risk_flags.append("Sentiment_rejected")
    elif confidence >= 0.50 and p2.entry_ready:
        action = 'BUY'
    elif p2.entry_ready:
        action = 'BUY_WEAK'
    else:
        action = 'MONITOR'

    # ── Sentiment risk flags ──
    sentiment = candidate.sentiment
    if sentiment is not None:
        if "CONTRARIAN_BUY" in sentiment.signals:
            risk_flags.append("Contrarian_entry")
        if "CROWDED_TRADE" in sentiment.signals:
            risk_flags.append("Crowded_trade")

    # ── Rationale ──
    parts = [
        f"{ticker}: {action} {quantity}sh @ ${entry_price:.2f}",
        f"Stop: ${stop_loss:.2f}({stop_type})",
        f"R:R=1:{rr:.1f}",
        f"Q={p1.quality_score:.0%}",
        f"MAGNA={p2.magna_score}/10",
        f"Conf={confidence:.0%}",
    ]
    if p2.entry_ready:
        parts.append(f"⚡{'G' if p2.g_gap_up else 'MA'}")
    if p2.trigger_signals:
        parts.append(f"Sig={','.join(p2.trigger_signals)}")
    if sentiment is not None:
        parts.append(f"😐{sentiment.sentiment_label[:4]}")
        if sentiment.signals:
            parts.append(f"Sent={','.join(sentiment.signals[:2])}")
    if risk_flags:
        parts.append(f"⚠️ {';'.join(risk_flags)}")
    rationale = " | ".join(parts)

    return TradeDecision(
        ticker=ticker,
        action=action,
        quantity=quantity,
        entry_price=entry_price,
        entry_method=entry_method,
        stop_loss=stop_loss,
        stop_type=stop_type,
        take_profits=take_profits,
        trailing_stop_pct=trailing_stop_pct,
        time_stop_days=RC.time_stop_days,
        position_pct=position_pct,
        risk_amount=risk_capital,
        reward_ratio=round(rr, 2),
        confidence_score=confidence,
        risk_flags=risk_flags,
        rationale=rationale,
        trailing_activate_pct=trailing_activate_pct,
    )


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _compute_gap_entry(hist: pd.DataFrame, low_52w: float) -> float:
    """Entry 5% below gap day high (pullback entry after gap)."""
    if len(hist) < 20:
        return low_52w * 1.005
    recent = hist.tail(20)
    for i in range(1, len(recent)):
        prev_close = float(recent['Close'].iloc[i-1])
        curr_open = float(recent['Open'].iloc[i])
        if prev_close > 0 and (curr_open - prev_close) / prev_close >= 0.04:
            return round(float(recent['High'].iloc[i]) * 0.95, 2)
    return low_52w * 1.005


def _is_in_base(hist: pd.DataFrame) -> bool:
    """Quick check if stock appears to be in a base/consolidation."""
    if len(hist) < 60:
        return False
    recent = hist.tail(60)
    high = float(recent['High'].max())
    low = float(recent['Low'].min())
    if high <= 0:
        return False
    return (high - low) / high <= 0.25
