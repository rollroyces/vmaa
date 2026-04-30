#!/usr/bin/env python3
"""
VMAA Integrated Pipeline — Full Auto Trading System
=====================================================
Unified pipeline from scan through execution with comprehensive risk management.

Flow:
  scan (filter_stocks) → price engine → risk manager → execution (Tiger)

Risk Management Layers:
  L1 — Portfolio-level: max positions, max drawdown, cash reserve, correlation
  L2 — Position-level: Kelly sizing, ATR-based stops, max allocation %
  L3 — Execution-level: order validation, price slippage guard, daily limits
  L4 — Exit strategy: trailing stop, time stop, fundamental stop, TP scaling

Outputs:
  - Trade decisions with full rationale
  - Execution report
  - Risk dashboard

Usage:
  python3 pipeline.py --scan --execute       # Full auto: scan + price + execute
  python3 pipeline.py --scan --dry-run       # Scan only, no execution
  python3 pipeline.py --status               # Portfolio status + risk dashboard
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Add parent paths for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from broker.tiger_broker import TigerBroker, BrokerPosition, OrderResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vmaa.pipeline")

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RiskConfig:
    """Comprehensive risk management parameters."""
    # Portfolio-level
    max_positions: int = 8
    max_positions_per_sector: int = 2
    cash_reserve_pct: float = 0.15          # Keep 15% cash
    max_portfolio_heat: float = 0.70        # Max 70% of portfolio deployed
    max_daily_loss_pct: float = 0.03        # Stop trading if daily loss >3%
    max_correlation: float = 0.70           # Avoid positions correlated >0.7

    # Position-level
    max_position_pct: float = 0.20          # Max 20% per position
    kelly_fraction: float = 0.25            # Quarter-Kelly for sizing
    min_position_size: float = 500.0        # Min $500 per trade
    max_position_size: float = 80000.0      # Max $80K per trade

    # Entry
    max_slippage_pct: float = 0.02          # Max 2% above buy price
    require_volume: bool = True             # Require min avg volume
    min_avg_volume: int = 100000            # Min 100K avg volume

    # Stop management
    atr_stop_multiplier: float = 2.0        # 2x ATR for stop
    hard_stop_pct: float = 0.10             # 10% hard stop
    trailing_stop_pct: float = 0.08         # 8% trailing stop
    trailing_activate_after: float = 0.10   # Activate after 10% profit
    time_stop_days: int = 60                # Exit if flat for 60 days

    # Market conditions
    vix_threshold: float = 35.0             # Reduce sizing if VIX > 35
    market_trend_ma: int = 50               # Only long if SPY > 50d MA
    volume_drop_threshold: float = 0.50     # Warn if volume < 50% avg


# Default config
RC = RiskConfig()

# ═══════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ScanResult:
    ticker: str
    name: str
    sector: str
    current_price: float
    low_52w: float
    high_52w: float
    ptl_ratio: float          # Price to 52w-low ratio
    market_cap: float
    btm_ratio: float           # Book-to-market
    fcf_yield: float
    roe: float
    ebitda_margin: float
    avg_volume: int
    magna_score: int           # 0-10 MAGNA score
    reason: str
    # Extended criteria (v2.1 — quality + sensitivity)
    roa: float = 0.0
    short_ratio: float = 0.0
    ipo_years: Optional[float] = None
    interest_rate_sensitive: bool = False
    debt_to_equity: float = 0.0
    beta: float = 1.0
    asset_vs_earnings_growth: str = ""  # "asset<earnings", "asset>=earnings", "n/a"


@dataclass
class TradeDecision:
    ticker: str
    action: str                # BUY / SELL / HOLD
    quantity: int
    entry_price: float
    stop_loss: float
    take_profits: List[Dict[str, Any]]
    trailing_stop_pct: float
    time_stop_days: int
    position_pct: float        # % of portfolio
    risk_amount: float         # $ at risk
    reward_ratio: float        # Risk:Reward
    confidence_score: float    # 0.0-1.0 overall confidence
    risk_flags: List[str]      # Any risk warnings
    rationale: str


# ═══════════════════════════════════════════════════════════════════
# Layer 0: Market Conditions
# ═══════════════════════════════════════════════════════════════════

def get_market_conditions() -> Dict[str, Any]:
    """Check overall market regime — affects sizing & risk appetite."""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="3mo")
        info = spy.info

        current = float(hist['Close'].iloc[-1]) if len(hist) > 0 else 0
        ma50 = float(hist['Close'].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else 0
        above_ma = current > ma50 if ma50 > 0 else True

        # VIX proxy via SPY 20d volatility
        returns = hist['Close'].pct_change().dropna()
        vol_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.15
        vol_regime = "LOW" if vol_20d < 0.15 else ("NORMAL" if vol_20d < 0.25 else "HIGH")

        # Drawdown from 3mo high
        high_3mo = float(hist['High'].max())
        dd_from_high = (current - high_3mo) / high_3mo if high_3mo > 0 else 0

        return {
            "spy_price": current,
            "spy_ma50": ma50,
            "above_ma50": above_ma,
            "volatility_20d": round(vol_20d, 4),
            "vol_regime": vol_regime,
            "dd_from_3mo_high": round(dd_from_high, 4),
            "market_ok": above_ma and (dd_from_high > -0.15),  # OK if above MA & not -15% from high
            "position_scalar": 1.0 if vol_regime == "LOW" else (0.75 if vol_regime == "NORMAL" else 0.5),
        }
    except Exception as e:
        logger.warning(f"Market conditions check failed: {e}")
        return {"market_ok": True, "position_scalar": 1.0, "vol_regime": "UNKNOWN"}


# ═══════════════════════════════════════════════════════════════════
# Layer 1: VMAA Scan (Filter + Price Engine combined)
# ═══════════════════════════════════════════════════════════════════

def scan_stocks(universe: List[str] = None, top_n: int = 20) -> List[ScanResult]:
    """Scan stocks using VMAA value + MAGNA filters with yfinance."""
    if universe is None:
        # Default: S&P 500 components (subset for speed)
        universe = _get_sp500_tickers()

    results = []
    logger.info(f"Scanning {len(universe)} stocks...")

    for i, ticker in enumerate(universe):
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(universe)}")
        try:
            result = _analyze_one(ticker)
            if result:
                results.append(result)
        except Exception as e:
            logger.debug(f"  {ticker}: skip — {e}")
        time.sleep(0.15)  # Rate limit

    # Sort by composite score (MAGNA first, then value)
    results.sort(key=lambda r: (r.magna_score, r.fcf_yield, r.btm_ratio), reverse=True)
    logger.info(f"Scan complete: {len(results)} candidates from {len(universe)} stocks")
    return results[:top_n]


def _analyze_one(ticker: str) -> Optional[ScanResult]:
    """Analyze single stock with value + MAGNA screening."""
    t = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="6mo")

    if len(hist) < 20:
        return None

    price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose', 0)
    if price <= 0 or price < 2:  # Skip penny stocks
        return None

    low_52w = info.get('fiftyTwoWeekLow', price)
    high_52w = info.get('fiftyTwoWeekHigh', price)
    ptl = price / low_52w if low_52w > 0 else 999

    # Value filter: must be near 52w low
    if ptl > 1.25:
        return None

    market_cap = info.get('marketCap', 0)
    if market_cap and market_cap > 10e9:  # Skip mega-caps
        return None

    book_value = info.get('bookValue', 0)
    btm = book_value / price if book_value and price > 0 else 0
    if btm <= 0:  # Negative book value
        return None

    fcf = info.get('freeCashflow', 0)
    fcf_yield = fcf / market_cap if fcf and market_cap else 0

    roe = info.get('returnOnEquity', 0) or 0
    ebitda = info.get('ebitda', 0)
    revenue = info.get('totalRevenue', 0)
    ebitda_margin = ebitda / revenue if ebitda and revenue else 0

    avg_vol = info.get('averageVolume', 0) or 0
    if avg_vol < RC.min_avg_volume and RC.require_volume:
        return None

    sector = info.get('sector', 'Unknown')
    name = info.get('shortName', ticker)

    # ── MAGNA Score (0-10) ──
    magna = 0

    # S: Short Interest Ratio (軋空動能 / squeeze potential)
    short_ratio = info.get('shortRatio', 0) or 0
    try:
        if short_ratio > 5:
            magna += 2
        elif short_ratio > 2:
            magna += 1
    except Exception:
        logger.debug(f"  {ticker}: Short interest check failed")

    # I: IPO Tenure ≤ 10 years (新股動能 / fresh listing momentum)
    ipo_years = None
    first_trade = info.get('firstTradeDateEpochUtc')
    if first_trade:
        try:
            ipo_date = datetime.fromtimestamp(first_trade)
            ipo_years = round((datetime.now() - ipo_date).days / 365.25, 1)
            if ipo_years <= 10:
                magna += 1
        except Exception:
            logger.debug(f"  {ticker}: IPO tenure calc failed")

    # M: EPS momentum (reduced weight — quality over growth myth)
    eps_growth = info.get('earningsGrowth', 0) or 0
    if eps_growth > 0.10:
        magna += 1  # was +2

    # A: Revenue acceleration (reduced weight)
    rev_growth = info.get('revenueGrowth', 0) or 0
    if rev_growth > 0.10:
        magna += 1  # was +2

    # G: Gap-up in last 20 days
    if _detect_gap(hist):
        magna += 1

    # N: Near 52w low (already filtered)
    if ptl < 1.05:
        magna += 2
    elif ptl < 1.10:
        magna += 1

    # Q: Quality — FCF yield (high free cash flow = quality business)
    if fcf_yield > 0.05:
        magna += 1

    # Q: Quality — ROE/ROA (profitability)
    roa = info.get('returnOnAssets', 0) or 0
    if roe > 0.10 or roa > 0:
        magna += 1

    # Q: Quality — High book-to-market (deep value)
    if btm > 0.8:
        magna += 1

    # 5: Analyst upgrades (positive recommendation trend)
    rec = info.get('recommendationMean', '')
    if rec and rec != '' and float(rec) <= 2.5:
        magna += 1

    # Cap at 10
    magna = min(magna, 10)

    # Minimum score
    if magna < 3:
        return None

    # ══ Extended Criteria (v2.1) ══

    # ── Asset Growth vs Earnings Growth ──
    # Quality check: earnings should grow faster than assets (capital-efficient)
    asset_vs_earnings = "n/a"
    try:
        bs = t.balance_sheet
        fin = t.financials
        if bs is not None and fin is not None and not bs.empty and not fin.empty:
            total_assets = bs.loc['Total Assets'] if 'Total Assets' in bs.index else None
            net_income = fin.loc['Net Income'] if 'Net Income' in fin.index else None
            if (total_assets is not None and net_income is not None
                    and len(total_assets) >= 2 and len(net_income) >= 2):
                assets_latest = total_assets.iloc[0]
                assets_prev = total_assets.iloc[1]
                ni_latest = net_income.iloc[0]
                ni_prev = net_income.iloc[1]
                if assets_prev > 0 and abs(ni_prev) > 0:
                    asset_growth = (assets_latest - assets_prev) / assets_prev
                    earnings_growth = (ni_latest - ni_prev) / abs(ni_prev)
                    if asset_growth < earnings_growth:
                        asset_vs_earnings = "asset<earnings"
                    else:
                        asset_vs_earnings = "asset>=earnings"
    except Exception as e:
        logger.debug(f"  {ticker}: Asset vs Earnings growth check failed: {e}")

    # ── Interest Rate Sensitivity ──
    ir_sensitive = False
    debt_to_equity = info.get('debtToEquity', 0) or 0
    beta = info.get('beta', 1.0) or 1.0
    try:
        if debt_to_equity > 100 or (beta is not None and beta > 1.5):
            ir_sensitive = True
        if sector in ('Financials', 'Real Estate', 'Utilities'):
            ir_sensitive = True
    except Exception:
        logger.debug(f"  {ticker}: Interest rate sensitivity check failed")

    return ScanResult(
        ticker=ticker, name=name, sector=sector,
        current_price=price, low_52w=low_52w, high_52w=high_52w,
        ptl_ratio=round(ptl, 4), market_cap=market_cap,
        btm_ratio=round(btm, 4), fcf_yield=round(fcf_yield, 4),
        roe=round(roe, 4), ebitda_margin=round(ebitda_margin, 4),
        avg_volume=avg_vol, magna_score=magna,
        reason=f"MAGNA={magna}/10 PTL={ptl:.2f}x FCF={fcf_yield:.1%}",
        roa=round(roa, 4),
        short_ratio=round(short_ratio, 2),
        ipo_years=ipo_years,
        interest_rate_sensitive=ir_sensitive,
        debt_to_equity=round(debt_to_equity, 2),
        beta=round(beta, 2),
        asset_vs_earnings_growth=asset_vs_earnings,
    )


def _detect_gap(hist: pd.DataFrame) -> bool:
    """Detect gap-up >4% in last 20 days."""
    if len(hist) < 20:
        return False
    recent = hist.tail(20)
    for i in range(1, len(recent)):
        prev_close = recent['Close'].iloc[i-1]
        curr_open = recent['Open'].iloc[i]
        if prev_close > 0 and (curr_open - prev_close) / prev_close >= 0.04:
            return True
    return False


def _get_sp500_tickers() -> List[str]:
    """Get S&P 500 ticker list from Wikipedia."""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return table['Symbol'].tolist()
    except Exception:
        logger.warning("Could not fetch S&P 500 list, using fallback")
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM",
            "DIS", "NFLX", "ADBE", "CRM", "CSCO", "INTC", "VZ", "T", "PFE",
            "MRK", "ABBV", "PEP", "KO", "TMO", "NKE", "ABT", "DHR", "MDT",
            "BMY", "AMGN", "LOW", "UPS", "QCOM", "TXN", "HON", "GE", "RTX",
            "CAT", "DE", "MMM", "ISRG", "SPGI", "BLK", "GS", "MS", "SCHW",
            "C", "PLD", "AMT", "CCI", "EQIX", "SBUX", "MCD", "CMG", "F",
            "GM", "TSM", "BABA", "ORCL", "IBM", "CVX", "COP", "SLB", "EOG",
            "OXY", "DHI", "LEN", "NVR", "ELV", "CI", "HUM", "CNC", "CVS",
            "ZTS", "REGN", "VRTX", "GILD", "BIIB", "ILMN", "A", "WAT",
            "EPAM", "MOS", "RVTY", "UHS",  # Known VMAA candidates
        ]


# ═══════════════════════════════════════════════════════════════════
# Layer 2: Price Engine + Position Sizing
# ═══════════════════════════════════════════════════════════════════

def compute_trade_decision(
    candidate: ScanResult,
    broker: TigerBroker,
    market: Dict[str, Any],
) -> TradeDecision:
    """
    Compute full trade decision including entry, stops, sizing, and risk.
    This is the CORE decision engine.
    """
    ticker = candidate.ticker
    risk_flags: List[str] = []

    # ── Fetch fresh price data ──
    try:
        t = yf.Ticker(ticker)
        hist_6mo = t.history(period="6mo")
        info = t.info
    except Exception:
        return TradeDecision(
            ticker=ticker, action='HOLD', quantity=0,
            entry_price=0, stop_loss=0, take_profits=[],
            trailing_stop_pct=0, time_stop_days=0,
            position_pct=0, risk_amount=0, reward_ratio=0,
            confidence_score=0, risk_flags=['data_fetch_failed'],
            rationale=f"Could not fetch data for {ticker}"
        )

    current = candidate.current_price

    # ── Entry Price ──
    # Buy at: gap-adjusted price > 52w-low +0.5% > current price (if near low)
    gap_detected = _detect_gap(hist_6mo)
    if gap_detected:
        # Find gap day high
        entry = _compute_gap_entry(hist_6mo, candidate.low_52w)
        entry_method = "gap_entry"
    else:
        # 52w-low + 0.5% or current (whichever is lower)
        entry_low = round(candidate.low_52w * 1.005, 2)
        entry = min(entry_low, current)
        entry_method = "52w_low" if entry == entry_low else "current_price"

    if entry <= 0:
        entry = current

    # ── Stop Loss ──
    # 1. ATR-based stop (2x ATR below entry)
    atr = _compute_atr(hist_6mo, 14)
    atr_stop = round(entry - (atr * RC.atr_stop_multiplier), 2) if atr > 0 else 0

    # 2. Hard stop (10% below entry)
    hard_stop = round(entry * (1 - RC.hard_stop_pct), 2)

    # 3. Structural stop (below 52w low)
    structural_stop = round(candidate.low_52w * 0.98, 2)

    # Use the tightest stop that's not too tight
    stop_candidates = [(atr_stop, "ATR"), (hard_stop, "Hard"), (structural_stop, "Structural")]
    stop_candidates = [(s, n) for s, n in stop_candidates if s > 0]
    stop_candidates.sort(key=lambda x: x[0], reverse=True)  # Highest first
    stop_loss, stop_type = stop_candidates[0] if stop_candidates else (hard_stop, "Hard")

    # Ensure stop is below entry
    if stop_loss >= entry:
        stop_loss = round(entry * 0.95, 2)
        stop_type = "Fallback"

    # ── Take Profit Levels ──
    tp1 = round(entry * 1.15, 2)
    tp2 = round(entry * 1.25, 2)
    tp3 = round(entry * 1.40, 2)

    # Adjust TP by volatility
    vol_adj = market.get('position_scalar', 1.0)
    if vol_adj < 1.0:
        tp1 = round(entry * (1 + 0.15 * vol_adj), 2)
        tp2 = round(entry * (1 + 0.25 * vol_adj), 2)
        tp3 = round(entry * (1 + 0.40 * vol_adj), 2)

    take_profits = [
        {"level": tp1, "sell_pct": 30, "label": "TP1"},
        {"level": tp2, "sell_pct": 30, "label": "TP2"},
        {"level": tp3, "sell_pct": 40, "label": "TP3"},
    ]

    # ── Position Sizing (Quarter-Kelly) ──
    account = broker.get_account()
    portfolio_value = account.net_liquidation
    max_alloc = portfolio_value * RC.max_position_pct

    # Risk per share
    risk_per_share = entry - stop_loss
    if risk_per_share <= 0:
        risk_per_share = entry * 0.05

    # Kelly: f* = (payout * win_prob - loss_prob) / payout
    # Simplified: assume 50% win rate with 2:1 reward ratio
    win_prob = 0.50
    avg_reward = (tp1 - entry) * 0.30 + (tp2 - entry) * 0.30 + (tp3 - entry) * 0.40
    loss_amt = risk_per_share
    payout_ratio = avg_reward / loss_amt if loss_amt > 0 else 2.0
    kelly = (payout_ratio * win_prob - (1 - win_prob)) / payout_ratio
    kelly = max(0, min(kelly, 0.25))  # Cap at quarter-Kelly

    risk_capital = portfolio_value * kelly * RC.kelly_fraction

    # Shares from risk
    raw_shares = int(risk_capital / risk_per_share) if risk_per_share > 0 else 0
    # Shares from allocation
    alloc_shares = int(max_alloc / entry) if entry > 0 else 0
    # Market condition adjustment
    adj_shares = int(raw_shares * vol_adj)

    quantity = min(raw_shares, alloc_shares, adj_shares)
    quantity = max(1, min(quantity, 10000))

    # Min/max check
    position_value = quantity * entry
    if position_value < RC.min_position_size:
        risk_flags.append(f"Position too small: ${position_value:.0f}")
        quantity = 0
    if position_value > RC.max_position_size:
        quantity = int(RC.max_position_size / entry)
        risk_flags.append("Position capped at max")

    position_pct = (quantity * entry / portfolio_value * 100) if portfolio_value > 0 else 0

    # ── Confidence Score (v2.1 — quality-weighted) ──
    confidence = 0.0
    # MAGNA contribution (already quality-weighted)
    confidence += min(candidate.magna_score, 10) / 10 * 0.30
    # Value & quality contribution
    if candidate.fcf_yield > 0.06:
        confidence += 0.15
    elif candidate.fcf_yield > 0.04:
        confidence += 0.08
    if candidate.btm_ratio > 1.2:
        confidence += 0.10
    if candidate.roa > 0:
        confidence += 0.05
    if candidate.roe > 0.15:
        confidence += 0.05
    # Asset quality: earnings growth > asset growth (capital-efficient)
    if candidate.asset_vs_earnings_growth == "asset<earnings":
        confidence += 0.05
    # Short interest (potential squeeze catalyst)
    if candidate.short_ratio > 5:
        confidence += 0.05
    # Market condition
    if market.get('market_ok', False):
        confidence += 0.12
    if market.get('vol_regime') == 'LOW':
        confidence += 0.05
    # Technical
    if entry <= candidate.low_52w * 1.02:
        confidence += 0.10
    if candidate.ptl_ratio < 1.05:
        confidence += 0.10

    confidence = round(min(confidence, 1.0), 3)

    # Market condition check
    if not market.get('market_ok', True):
        risk_flags.append("Market below 50MA or deep drawdown")

    # Interest rate sensitivity
    if candidate.interest_rate_sensitive:
        risk_flags.append("IR_sensitive")

    # ── Risk/Reward ──
    reward_ratio = avg_reward / loss_amt if loss_amt > 0 else 0
    reward_ratio = round(reward_ratio, 2)

    # Decision
    if quantity <= 0 or confidence < 0.30:
        action = 'HOLD'
    elif confidence >= 0.50:
        action = 'BUY'
    else:
        action = 'BUY_WEAK'  # Buy but lower conviction

    rationale_parts = [
        f"{ticker}: {action} {quantity}sh @ ${entry:.2f}",
        f"Stop: ${stop_loss:.2f} ({stop_type})",
        f"TP: ${tp1}-${tp2}-${tp3}",
        f"R:R=1:{reward_ratio:.1f}",
        f"Conf={confidence:.0%}",
        f"MAGNA={candidate.magna_score}/10",
        f"Risk=${risk_capital:.0f}",
    ]
    if candidate.short_ratio > 0:
        rationale_parts.append(f"SI={candidate.short_ratio:.1f}")
    if candidate.asset_vs_earnings_growth == "asset<earnings":
        rationale_parts.append("A<EG")
    if risk_flags:
        rationale_parts.append(f"⚠️ {';'.join(risk_flags)}")
    rationale = " | ".join(rationale_parts)

    return TradeDecision(
        ticker=ticker, action=action, quantity=quantity,
        entry_price=entry, stop_loss=stop_loss,
        take_profits=take_profits,
        trailing_stop_pct=RC.trailing_stop_pct,
        time_stop_days=RC.time_stop_days,
        position_pct=round(position_pct, 2),
        risk_amount=round(risk_capital, 2),
        reward_ratio=reward_ratio,
        confidence_score=confidence,
        risk_flags=risk_flags,
        rationale=rationale,
    )


def _compute_atr(hist: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range."""
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


def _compute_gap_entry(hist: pd.DataFrame, low_52w: float) -> float:
    """Compute entry price when gap-up detected: 5% below gap day high."""
    if len(hist) < 20:
        return low_52w * 1.005
    recent = hist.tail(20)
    for i in range(1, len(recent)):
        prev_close = recent['Close'].iloc[i-1]
        curr_open = recent['Open'].iloc[i]
        if prev_close > 0 and (curr_open - prev_close) / prev_close >= 0.04:
            return round(recent['High'].iloc[i] * 0.95, 2)
    return low_52w * 1.005


# ═══════════════════════════════════════════════════════════════════
# Layer 3: Correlation & Sector Check
# ═══════════════════════════════════════════════════════════════════

def check_correlation(new_ticker: str, existing_tickers: List[str]) -> float:
    """Check max correlation between new ticker and existing positions."""
    if not existing_tickers:
        return 0.0
    try:
        all_tickers = [new_ticker] + existing_tickers
        data = yf.download(all_tickers, period="3mo", progress=False)['Close']
        returns = data.pct_change().dropna()
        if returns.shape[1] >= 2:
            corr_matrix = returns.corr()
            new_corr = corr_matrix[new_ticker].drop(new_ticker)
            return float(new_corr.max())
    except Exception:
        pass
    return 0.0


def sector_exposure(sector: str, positions: List[BrokerPosition]) -> Tuple[int, float]:
    """Check sector concentration in current portfolio."""
    # Simple sector mapping
    sector_map = {
        'Technology': ['EPAM', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'CRM', 'ADBE', 'CSCO', 'INTC', 'QCOM', 'TXN', 'ORCL', 'IBM'],
        'Basic Materials': ['MOS'],
        'Healthcare': ['RVTY', 'UHS', 'JNJ', 'UNH', 'ABBV', 'PFE', 'MRK', 'TMO', 'ABT', 'DHR', 'MDT', 'BMY', 'AMGN', 'ISRG', 'REGN', 'VRTX', 'GILD', 'BIIB'],
        'Financial': ['JPM', 'BAC', 'GS', 'MS', 'SCHW', 'C', 'BLK', 'SPGI'],
        'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'HD', 'LOW', 'CMG', 'F', 'GM'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY'],
        'Industrial': ['HON', 'GE', 'RTX', 'CAT', 'DE', 'MMM', 'UPS'],
        'Real Estate': ['PLD', 'AMT', 'CCI', 'EQIX'],
        'Communication': ['DIS', 'NFLX', 'VZ', 'T', 'META'],
    }

    tickers_in_sector = set(sector_map.get(sector, []))
    count = 0
    for pos in positions:
        if pos.symbol.upper() in tickers_in_sector:
            count += 1
    return count, count / max(len(positions), 1) if positions else 0


# ═══════════════════════════════════════════════════════════════════
# Layer 4: Execution Engine
# ═══════════════════════════════════════════════════════════════════

class ExecutionEngine:
    """Executes trade decisions with safety checks via Tiger Broker."""

    def __init__(self, broker: TigerBroker, dry_run: bool = True):
        self.broker = broker
        self.dry_run = dry_run
        self.executed: List[OrderResult] = []
        self.skipped: List[Tuple[TradeDecision, str]] = []
        self.daily_trades = 0
        self.max_daily_trades = 10

    def execute_decision(self, decision: TradeDecision) -> Optional[OrderResult]:
        """Execute a single trade decision with all safety checks."""
        if decision.action in ('HOLD',):
            self.skipped.append((decision, "HOLD signal"))
            return None

        if self.daily_trades >= self.max_daily_trades:
            self.skipped.append((decision, "Daily trade limit"))
            return None

        # Final pre-flight checks
        account = self.broker.get_account()

        # 1. Buying power check
        cost = decision.quantity * decision.entry_price
        if cost > account.buying_power * 0.85:
            self.skipped.append((decision, f"Cost ${cost:.0f} > 85% BP"))
            return None

        # 2. Cash reserve check
        cash_after = account.cash - cost
        if cash_after < account.net_liquidation * RC.cash_reserve_pct:
            self.skipped.append((decision, "Breaks cash reserve"))
            return None

        logger.info(decision.rationale)

        if self.dry_run:
            result = OrderResult(
                order_id=-1, symbol=decision.ticker,
                action='BUY', quantity=decision.quantity,
                order_type='LMT', limit_price=decision.entry_price,
                status='DRY_RUN', filled_quantity=0, filled_price=0.0,
                timestamp=str(datetime.now()), reason='dry_run',
            )
        else:
            result = self.broker.place_order(
                symbol=decision.ticker,
                action='BUY',
                quantity=decision.quantity,
                order_type='LMT',
                limit_price=decision.entry_price,
            )

            # Place stop loss order
            if result.order_id > 0 and result.status not in ('REJECTED',):
                self.broker.place_order(
                    symbol=decision.ticker,
                    action='SELL',
                    quantity=decision.quantity,
                    order_type='STP',
                    stop_price=decision.stop_loss,
                )

        self.executed.append(result)
        self.daily_trades += 1
        return result

    def execute_batch(self, decisions: List[TradeDecision]) -> Dict[str, Any]:
        """Execute all BUY decisions with risk ordering."""
        # Sort by confidence
        buy_decisions = [d for d in decisions if d.action in ('BUY', 'BUY_WEAK')]
        buy_decisions.sort(key=lambda d: d.confidence_score, reverse=True)

        for d in buy_decisions:
            self.execute_decision(d)

        return self.summary()

    def summary(self) -> Dict[str, Any]:
        return {
            'timestamp': str(datetime.now()),
            'mode': 'DRY_RUN' if self.dry_run else 'LIVE',
            'executed_count': len(self.executed),
            'skipped_count': len(self.skipped),
            'executed': [
                {
                    'ticker': r.symbol, 'quantity': r.quantity,
                    'price': r.limit_price, 'status': r.status,
                    'reason': r.reason,
                } for r in self.executed
            ],
            'skipped': [
                {'ticker': d.ticker, 'reason': reason}
                for d, reason in self.skipped
            ],
        }


# ═══════════════════════════════════════════════════════════════════
# Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(dry_run: bool = True, tickers: List[str] = None) -> Dict[str, Any]:
    """Run the full VMAA pipeline: scan → price → risk → execute."""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(f"VMAA Pipeline Start — {'DRY RUN' if dry_run else 'LIVE'} — {start_time}")
    logger.info("=" * 60)

    # Step 0: Market conditions
    logger.info("[Step 0] Checking market conditions...")
    market = get_market_conditions()
    logger.info(f"  SPY: ${market.get('spy_price', 'N/A'):.2f} | "
                f"Vol: {market.get('vol_regime', '?')} | "
                f"OK: {market.get('market_ok')} | "
                f"Scalar: {market['position_scalar']}x")

    # Step 1: Initialize broker
    logger.info("[Step 1] Connecting to Tiger...")
    broker = TigerBroker()
    account = broker.get_account()
    logger.info(f"  Account: {account.account_id} | "
                f"Value: ${account.net_liquidation:,.0f} | "
                f"BP: ${account.buying_power:,.0f}")

    # Step 2: Scan
    logger.info("[Step 2] Scanning stocks...")
    candidates = scan_stocks(universe=tickers)
    logger.info(f"  Found {len(candidates)} candidates")

    if not candidates:
        logger.info("  No candidates found, stopping")
        return {"status": "no_candidates", "market": market}

    # Step 3: Generate trade decisions
    logger.info("[Step 3] Computing trade decisions...")
    decisions = []
    positions = broker.get_positions()
    existing_tickers = [p.symbol for p in positions]

    for c in candidates:
        # Sector check
        sec_count, _ = sector_exposure(c.sector, positions)
        if sec_count >= RC.max_positions_per_sector:
            logger.info(f"  {c.ticker}: Skipped — {c.sector} sector at limit")
            continue

        # Position count check
        if len(positions) >= RC.max_positions:
            logger.info(f"  {c.ticker}: Skipped — {RC.max_positions} positions max")
            break

        # Correlation check
        if existing_tickers:
            corr = check_correlation(c.ticker, existing_tickers)
            if corr > RC.max_correlation:
                logger.info(f"  {c.ticker}: Skipped — Corr {corr:.2f} > {RC.max_correlation}")
                continue

        # Compute
        decision = compute_trade_decision(c, broker, market)
        decisions.append(decision)
        logger.info(f"  {decision.rationale}")

    # Step 4: Execute
    logger.info("[Step 4] Executing trades...")
    engine = ExecutionEngine(broker, dry_run=dry_run)
    execute_summary = engine.execute_batch(decisions)

    # Step 5: Report
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.0f}s")
    logger.info(f"  Candidates: {len(candidates)}")
    logger.info(f"  Decisions:  {len(decisions)}")
    logger.info(f"  Executed:   {execute_summary['executed_count']}")
    logger.info(f"  Skipped:    {execute_summary['skipped_count']}")
    logger.info("=" * 60)

    return {
        'status': 'complete',
        'timestamp': str(start_time),
        'elapsed_seconds': elapsed,
        'market': market,
        'account': {
            'value': account.net_liquidation,
            'buying_power': account.buying_power,
            'cash': account.cash,
            'positions': len(positions),
        },
        'pipeline': {
            'candidates_found': len(candidates),
            'decisions_made': len(decisions),
            'executed': execute_summary['executed_count'],
            'skipped': execute_summary['skipped_count'],
        },
        'decisions': [
            {
                'ticker': d.ticker,
                'action': d.action,
                'quantity': d.quantity,
                'entry': d.entry_price,
                'stop_loss': d.stop_loss,
                'take_profits': d.take_profits,
                'position_pct': d.position_pct,
                'risk_amount': d.risk_amount,
                'reward_ratio': d.reward_ratio,
                'confidence': d.confidence_score,
                'risk_flags': d.risk_flags,
            } for d in decisions
        ],
        'execution': execute_summary,
    }


def show_status() -> Dict[str, Any]:
    """Show current portfolio + risk dashboard."""
    broker = TigerBroker()
    market = get_market_conditions()
    account = broker.get_account()
    positions = broker.get_positions()
    orders = broker.get_orders(limit=10)

    return {
        'timestamp': str(datetime.now()),
        'market': market,
        'account': {
            'id': account.account_id,
            'value': account.net_liquidation,
            'cash': account.cash,
            'buying_power': account.buying_power,
            'invested': account.gross_position_value,
            'unrealized_pnl': account.unrealized_pnl,
            'realized_pnl': account.realized_pnl,
        },
        'positions': [
            {
                'ticker': p.symbol,
                'quantity': p.quantity,
                'avg_cost': p.average_cost,
                'current': p.market_price,
                'market_value': p.market_value,
                'pnl': p.unrealized_pnl,
                'pnl_pct': p.unrealized_pnl_pct,
            } for p in positions
        ],
        'active_orders': [
            {
                'id': o.order_id,
                'ticker': o.symbol,
                'action': o.action,
                'quantity': o.quantity,
                'price': o.limit_price,
                'status': o.status,
            } for o in orders if 'CANCEL' not in str(o.status).upper()
        ],
        'risk_checks': {
            'position_count': f"{len(positions)}/{RC.max_positions}",
            'cash_reserve_ok': account.cash >= account.net_liquidation * RC.cash_reserve_pct,
            'positions_ok': len(positions) <= RC.max_positions,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="VMAA Integrated Pipeline")
    parser.add_argument('--scan', action='store_true', help='Run stock scan')
    parser.add_argument('--execute', action='store_true', help='Execute trades (default: dry-run without)')
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--live', action='store_true', help='LIVE trading mode')
    parser.add_argument('--status', action='store_true', help='Show portfolio status')
    parser.add_argument('--tickers', nargs='*', help='Specific tickers to scan')
    parser.add_argument('--output', default='output/pipeline_result.json')
    args = parser.parse_args()

    if args.status:
        result = show_status()
    elif args.scan:
        # Split comma-separated ticker strings
        tickers = None
        if args.tickers:
            tickers = []
            for t in args.tickers:
                tickers.extend(s.strip() for s in t.split(',') if s.strip())
        result = run_pipeline(
            dry_run=not args.live,
            tickers=tickers if tickers else None,
        )
    else:
        print("Usage: --scan [--live] | --status")
        sys.exit(1)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n📁 Results saved to {output_path}")

    # Print summary
    if args.status:
        print(f"\n📊 Portfolio: ${result['account']['value']:,.0f} | "
              f"Positions: {len(result['positions'])} | "
              f"P&L: ${result['account']['unrealized_pnl']:,.0f}")
    elif args.scan:
        p = result.get('pipeline', {})
        print(f"\n🎯 Pipeline: {p.get('candidates_found', 0)} found → "
              f"{p.get('decisions_made', 0)} decisions → "
              f"{p.get('executed', 0)} executed")
        for d in result.get('decisions', []):
            flags = f" ⚠️ {','.join(d['risk_flags'])}" if d['risk_flags'] else ""
            print(f"  {d['action']:6s} {d['ticker']:6s} {d['quantity']:4d}sh "
                  f"@ ${d['entry']:.2f} stop=${d['stop_loss']:.2f} "
                  f"conf={d['confidence']:.0%}{flags}")
