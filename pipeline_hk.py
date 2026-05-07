#!/usr/bin/env python3
"""
VMAA-HK — Hong Kong Stock Pipeline
====================================
Two-stage VMAA adapted for HK stocks using yfinance (primary) + Tushare (supplementary).

Universe: Hang Seng Index constituents (88 stocks)
Market benchmark: HSI (^HSI)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse VMAA MAGNA Part 2
from part2_magna import screen_magna as part2_screen
from part2_magna import check_earnings_accel, check_sales_accel
from models import Part1Result
from config import RC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vmaa.hk")


def np_safe(obj):
    """Convert numpy scalar/array types to native Python types for JSON serialization.

    Handles: np.integer → int, np.floating → float, np.ndarray → list, np.bool_ → bool.
    Returns the object unchanged if it is not a numpy type.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════
# HSI Constituents
# ═══════════════════════════════════════════════════════════════════

HSI_TICKERS = [
    "0005.HK", "0011.HK", "0012.HK", "0016.HK", "0017.HK", "0027.HK",
    "0066.HK", "0083.HK", "0101.HK", "0148.HK", "0151.HK", "0175.HK",
    "0241.HK", "0267.HK", "0268.HK", "0270.HK", "0285.HK", "0288.HK",
    "0291.HK", "0316.HK", "0322.HK", "0386.HK", "0388.HK", "0669.HK",
    "0688.HK", "0700.HK", "0762.HK", "0823.HK", "0856.HK", "0857.HK",
    "0868.HK", "0881.HK", "0883.HK", "0939.HK", "0941.HK", "0960.HK",
    "0968.HK", "0981.HK", "0992.HK", "0998.HK", "1038.HK", "1044.HK",
    "1060.HK", "1093.HK", "1109.HK", "1113.HK", "1177.HK", "1199.HK",
    "1209.HK", "1211.HK", "1288.HK", "1299.HK", "1378.HK", "1398.HK",
    "1810.HK", "1833.HK", "1876.HK", "1928.HK", "1929.HK", "1997.HK",
    "2007.HK", "2015.HK", "2018.HK", "2020.HK", "2026.HK", "2096.HK",
    "2269.HK", "2291.HK", "2313.HK", "2318.HK", "2319.HK", "2331.HK",
    "2359.HK", "2388.HK", "2601.HK", "2628.HK", "2688.HK", "3690.HK",
    "3968.HK", "3988.HK", "6098.HK", "6186.HK", "6618.HK", "6690.HK",
    "6862.HK", "9618.HK", "9626.HK", "9633.HK", "9888.HK", "9988.HK",
    "9999.HK",
]

# Financial sector stocks (different screening criteria)
FINANCIAL_SECTORS = {
    "Financial Services", "Financial", "Banks", "Insurance",
    "Banks - Diversified", "Banks - Regional", "Insurance - Life",
    "Insurance - Diversified", "Asset Management", "Capital Markets",
}

FINANCIAL_INDUSTRIES = {
    "Banks - Diversified", "Banks - Regional", "Insurance - Life",
    "Insurance - Diversified", "Insurance - Property & Casualty",
    "Asset Management", "Financial Data & Stock Exchanges",
    "Financial Conglomerates", "Credit Services", "Capital Markets",
}


# ═══════════════════════════════════════════════════════════════════
# Market Regime (HSI)
# ═══════════════════════════════════════════════════════════════════

def get_hk_market_regime() -> dict:
    """Assess HK market conditions using HSI."""
    try:
        hsi = yf.Ticker("^HSI")
        hist = hsi.history(period="3mo")
        current = float(hist['Close'].iloc[-1]) if len(hist) > 0 else 0
        ma50 = float(hist['Close'].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else 0
        above_ma = current > ma50 if ma50 > 0 else True
        returns = hist['Close'].pct_change().dropna()
        vol_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.15

        if vol_20d < 0.12:
            vol_regime, scalar = "LOW", 1.0
        elif vol_20d < 0.22:
            vol_regime, scalar = "NORMAL", 0.80
        else:
            vol_regime, scalar = "HIGH", 0.50

        high_3mo = float(hist['High'].max())
        dd_from_high = (current - high_3mo) / high_3mo if high_3mo > 0 else 0
        market_ok = above_ma and (dd_from_high > -0.12)

        return {
            "hsi_price": round(current, 2),
            "hsi_ma50": round(ma50, 2),
            "above_ma50": above_ma,
            "vol_regime": vol_regime,
            "volatility_20d": round(vol_20d, 4),
            "dd_from_high": round(dd_from_high, 4),
            "market_ok": market_ok,
            "position_scalar": scalar,
        }
    except Exception as e:
        logger.warning(f"Market regime check failed: {e}")
        return {"market_ok": True, "position_scalar": 0.70, "vol_regime": "UNKNOWN"}


# ═══════════════════════════════════════════════════════════════════
# HK-Adapted Part 1: Core Fundamentals (using yfinance)
# ═══════════════════════════════════════════════════════════════════

def screen_hk_fundamentals(ticker: str) -> Optional[Dict]:
    """
    HK-adapted fundamental screening using yfinance info.
    Returns dict with all metrics or None if insufficient data.
    """
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        if not info or info.get('trailingPE') is None:
            return None

        name = info.get('shortName') or info.get('longName') or ticker
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        is_financial = (sector in FINANCIAL_SECTORS or industry in FINANCIAL_INDUSTRIES)

        # ── Price & Market Cap ──
        current_price = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)
        market_cap = float(info.get('marketCap') or 0)
        low_52w = float(info.get('fiftyTwoWeekLow') or 0)
        high_52w = float(info.get('fiftyTwoWeekHigh') or 0)

        if current_price <= 1 or market_cap <= 0:
            return None

        # ── Quality Metrics ──
        bv = float(info.get('bookValue') or 0)
        bm_ratio = bv / current_price if bv > 0 and current_price > 0 else 0
        roa = float(info.get('returnOnAssets') or 0)
        roe = float(info.get('returnOnEquity') or 0)
        ebitda_margin = float(info.get('ebitdaMargins') or 0)

        # ── Cash Flow ──
        fcf = float(info.get('freeCashflow') or 0)
        ocf = float(info.get('operatingCashflow') or 0)
        ni = float(info.get('netIncomeToCommon') or 0)
        revenue = float(info.get('totalRevenue') or 0)
        ev = float(info.get('enterpriseValue') or 0)

        fcf_yield = fcf / ev if fcf > 0 and ev > 0 else (
            fcf / market_cap if fcf > 0 and market_cap > 0 else 0
        )
        fcf_conversion = fcf / ni if fcf > 0 and ni > 0 else 0

        # ── Safety Margin ──
        ptl_ratio = current_price / low_52w if low_52w > 0 else 999

        # ── Growth ──
        rev_growth = float(info.get('revenueGrowth') or 0)
        earn_growth = float(info.get('earningsGrowth') or 0)

        # ── Risk ──
        debt_to_equity = float(info.get('debtToEquity') or 0)
        beta_val = float(info.get('beta') or 1.0)
        total_debt = float(info.get('totalDebt') or 0)
        total_cash = float(info.get('totalCash') or 0)

        # Interest rate sensitivity
        ir_sensitive = (
            sector in {'Real Estate', 'Utilities', 'Financial Services', 'Technology'} or
            debt_to_equity > 80 or beta_val > 1.5
        )

        # ── Screening Criteria (HK-adjusted) ──
        passed = []
        failed = []
        warnings = []

        # 1. Market Cap (HK special: accept >= 10B HKD ≈ 1.3B USD)
        if market_cap >= 1.3e9:
            passed.append("market_cap")
        else:
            failed.append("market_cap")

        # 2. B/M Ratio (looser for HK: >= 0.2)
        if bm_ratio >= 0.2:
            passed.append("bm_ratio")
        elif is_financial:
            warnings.append("bm_ratio_financial_skip")
            passed.append("bm_ratio")
        else:
            failed.append("bm_ratio")

        # 3. ROA (banks/insurers often have low ROA, use ROE instead)
        if is_financial:
            if roe >= 0.05:
                passed.append("roe_financial")
            else:
                failed.append("roe_financial")
        else:
            if roa >= 0.01:  # Looser for HK: 1%+
                passed.append("roa")
            else:
                failed.append("roa")

        # 4. EBITDA Margin (skip for financials)
        if is_financial:
            passed.append("ebitda_skip")
        elif ebitda_margin >= 0.03:  # Looser: 3%+
            passed.append("ebitda_margin")
        else:
            failed.append("ebitda_margin")

        # 5. FCF Yield
        if fcf_yield >= 0.01:  # Looser: 1%+ (was 3% for US)
            passed.append("fcf_yield")
        elif fcf <= 0:
            warnings.append("fcf_negative")
            failed.append("fcf_yield")
        else:
            failed.append("fcf_yield")

        # 6. Safety Margin (Price near 52w-low)
        if ptl_ratio <= 1.35:  # Slightly looser for HK
            passed.append("safety_margin")
        else:
            failed.append("safety_margin")

        # 7. FCF Conversion
        if is_financial:
            passed.append("fcf_conv_skip")
        elif fcf_conversion >= 0.30:  # Looser for HK
            passed.append("fcf_conversion")
        else:
            failed.append("fcf_conversion")

        # 8. Revenue/Earnings Growth check
        if rev_growth >= -0.15:  # Not declining >15%
            passed.append("rev_stability")
        else:
            warnings.append("rev_declining")

        # ── Quality Score (0-100) ──
        # Weight different dimensions
        score = 0.0
        if bm_ratio >= 0.2:
            score += min(bm_ratio, 1.5) / 1.5 * 20  # Up to 20 pts
        if is_financial:
            score += min(roe, 0.25) / 0.25 * 15
        else:
            score += min(roa, 0.15) / 0.15 * 15

        if not is_financial:
            score += min(ebitda_margin, 0.30) / 0.30 * 10

        score += min(fcf_yield, 0.10) / 0.10 * 20
        score += max(0, (1.35 - ptl_ratio)) / 0.35 * 15  # Safety margin
        if not is_financial:
            score += min(fcf_conversion, 1.0) * 10

        # Bonus for growth
        if rev_growth > 0:
            score += min(rev_growth, 0.3) / 0.3 * 5
        if earn_growth > 0:
            score += min(earn_growth, 0.5) / 0.5 * 5

        quality_score = min(score / 100, 1.0)

        # Only pass if >= 5 core criteria met (excluding skips)
        core_passed = sum(1 for p in passed if '_skip' not in p)
        passed_screen = core_passed >= 4 and quality_score >= 0.25

        if not passed_screen:
            return None

        # Build Part1Result-compatible dict
        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "market_cap_type": "turnaround" if market_cap < 10e9 else "large",
            "current_price": round(current_price, 2),
            "low_52w": round(low_52w, 2),
            "high_52w": round(high_52w, 2),
            "ptl_ratio": round(ptl_ratio, 3),
            "bm_ratio": round(bm_ratio, 3),
            "roa": round(roa, 4),
            "roe": round(roe, 4),
            "ebitda_margin": round(ebitda_margin, 4),
            "fcf_yield": round(fcf_yield, 4),
            "fcf_conversion": round(fcf_conversion, 4),
            "asset_growth": 0,
            "earnings_growth": round(earn_growth, 4),
            "asset_vs_earnings": "n/a_hk",
            "debt_to_equity": round(debt_to_equity, 1),
            "beta": round(beta_val, 3),
            "interest_rate_sensitive": ir_sensitive,
            "quality_score": round(quality_score, 4),
            "passed_criteria": passed,
            "failed_criteria": failed,
            "warnings": warnings,
            "is_financial": is_financial,
        }

    except Exception as e:
        logger.debug(f"Screen failed for {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# HK-Adapted Part 2: MAGNA Momentum (HK market compatible)
# ═══════════════════════════════════════════════════════════════════

class HKMagnaSignal:
    """HK-adapted MAGNA signal (no US-specific data needed)."""
    def __init__(self, ticker, magna_score=0, entry_ready=False,
                 gap_trigger=False, eps_accel=False, sales_accel=False,
                 neglect_base=False, short_score=0, analyst_ok=False,
                 triggers=None, details=None):
        self.ticker = ticker
        self.magna_score = magna_score
        self.entry_ready = entry_ready
        self.gap_trigger = gap_trigger
        self.eps_accel = eps_accel
        self.sales_accel = sales_accel
        self.neglect_base = neglect_base
        self.short_interest_score = short_score
        self.analyst_ok = analyst_ok
        self.triggers = triggers or []
        self.details = details or {}


def screen_hk_magna(ticker, quality):
    """
    HK-adapted MAGNA screening using yfinance.
    Uses price action + growth metrics; skips US-specific data.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="6mo")

        if hist is None or len(hist) < 40:
            return None

        score = 0
        triggers = []
        details = {}

        # ── M: Earnings Acceleration (from quarterly financials, same as US path) ──
        m_pass, eps_curr, eps_prev, eps_accel_val = check_earnings_accel(t)
        eps_accel = m_pass
        if eps_accel:
            score += 2
            triggers.append("M")

        # ── A: Sales Acceleration (from quarterly financials, same as US path) ──
        a_pass, rev_curr, rev_prev, rev_accel_val = check_sales_accel(t)
        sales_accel = a_pass
        if sales_accel:
            score += 2
            triggers.append("A")

        # ── G: Breakout Proxy (recent price surge + volume) ──
        close = hist['Close'].values
        vol = hist['Volume'].values if 'Volume' in hist.columns else np.ones(len(close))

        recent_ret = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
        recent_vol_ratio = np.mean(vol[-3:]) / np.mean(vol[-20:]) if len(vol) >= 20 else 1

        gap_up = recent_ret >= 0.04 and recent_vol_ratio >= 1.5
        if gap_up:
            score += 2
            triggers.append("G")

        details['recent_return'] = round(float(recent_ret), 4)
        details['vol_ratio'] = round(float(recent_vol_ratio), 2)

        # ── N: Neglect / Base Pattern ──
        if len(close) >= 60:
            close_3m = close[-60:]
            range_3m = (np.max(close_3m) - np.min(close_3m)) / np.mean(close_3m)
            vol_trend = np.mean(vol[-20:]) < np.mean(vol[-60:-20]) if len(vol) >= 60 else False
            is_base = range_3m <= 0.30 and vol_trend
        else:
            range_3m = 0
            is_base = False

        details['range_3m'] = round(float(range_3m), 3)
        if is_base:
            score += 1
            triggers.append("N")

        # ── 5: Short Interest — N/A for HK via yfinance ──
        short_score = 0

        # ── 3: Analyst Coverage ──
        analyst_count = int(info.get('numberOfAnalystOpinions', 0) or 0)
        target_mean = float(info.get('targetMeanPrice', 0) or 0)
        current = float(close[-1])
        analyst_ok = analyst_count >= 3 and target_mean >= current * 1.10
        if analyst_ok:
            score += 1
            triggers.append("Alyst")

        details['analyst_count'] = analyst_count
        details['target_mean'] = round(target_mean, 2)
        details['eps_growth_qoq'] = round(eps_curr, 4)
        details['eps_accel'] = round(eps_accel_val, 4)
        details['rev_growth_qoq'] = round(rev_curr, 4)
        details['rev_accel'] = round(rev_accel_val, 4)
        details['gap'] = gap_up
        details['base'] = is_base

        # ── Entry Trigger Logic ──
        entry_ready = gap_up or (eps_accel and sales_accel)

        return HKMagnaSignal(
            ticker=ticker,
            magna_score=score,
            entry_ready=entry_ready,
            gap_trigger=gap_up,
            eps_accel=eps_accel,
            sales_accel=sales_accel,
            neglect_base=is_base,
            short_score=short_score,
            analyst_ok=analyst_ok,
            triggers=triggers,
            details=details,
        )

    except Exception as e:
        logger.debug(f"MAGNA-HK failed for {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Stage 2 Wrapper: Adapt HK Part1Result for existing MAGNA module
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════

def hk_result_to_part1(r: dict) -> Part1Result:
    """Convert HK screen result dict to Part1Result for MAGNA consumption."""
    rationale = (
        f"Q={r['quality_score']:.0%} | Cap={r['market_cap_type']}(${r['market_cap']/1e9:.1f}B) | "
        f"B/M={r['bm_ratio']:.2f} | ROA={r['roa']:.1%} | FCF/Y={r['fcf_yield']:.1%} | "
        f"PTL={r['ptl_ratio']:.2f}x | {'⚠️IR' if r['interest_rate_sensitive'] else ''}"
    )
    return Part1Result(
        ticker=r["ticker"],
        name=r["name"],
        sector=r["sector"],
        industry=r["industry"],
        market_cap=r["market_cap"],
        market_cap_type=r["market_cap_type"],
        current_price=r["current_price"],
        low_52w=r["low_52w"],
        high_52w=r["high_52w"],
        ptl_ratio=r["ptl_ratio"],
        bm_ratio=r["bm_ratio"],
        roa=r["roa"],
        roe=r["roe"],
        ebitda_margin=r["ebitda_margin"],
        fcf_yield=r["fcf_yield"],
        fcf_conversion=r["fcf_conversion"],
        asset_growth=r["asset_growth"],
        earnings_growth=r["earnings_growth"],
        asset_vs_earnings=r["asset_vs_earnings"],
        debt_to_equity=r["debt_to_equity"],
        beta=r["beta"],
        interest_rate_sensitive=r["interest_rate_sensitive"],
        quality_score=r["quality_score"],
        passed_criteria=r["passed_criteria"],
        failed_criteria=r["failed_criteria"],
        warnings=r["warnings"],
        rationale=rationale,
        data_date=datetime.now().strftime("%Y-%m-%d"),
    )


# ═══════════════════════════════════════════════════════════════════
# Risk & Trade Decisions
# ═══════════════════════════════════════════════════════════════════

def compute_hk_trade_decision(
    ticker: str,
    quality: dict,
    magna_signal,  # HKMagnaSignal
    price: float,
    market: dict,
    sentiment=None,  # SentimentResult from part3_sentiment
) -> dict:
    """Generate HK trade decision with sentiment integration."""
    scalar = market.get("position_scalar", 0.8)
    atr = price * 0.025

    if magna_signal and magna_signal.entry_ready:
        action = "BUY"
        position_size = min(int(RC.max_position_size_hkd * scalar / price), RC.max_shares_hk)
        stop_loss = round(price * (1 - max(atr / price, 0.08)), 2)
        magna_score = magna_signal.magna_score
        # Base confidence: Q(45%) + MAGNA(25%) + Market(10%) + Sentiment(20%)
        base_conf = quality["quality_score"] * 0.45 + (magna_score / 10) * 0.25 + 0.10
        conf = base_conf
    elif magna_signal and magna_signal.magna_score >= 4:
        action = "WATCH"
        position_size = 0
        stop_loss = round(price * 0.92, 2)
        magna_score = magna_signal.magna_score
        base_conf = quality["quality_score"] * 0.45 + (magna_score / 10) * 0.20
        conf = base_conf
    else:
        action = "SKIP"
        position_size = 0
        stop_loss = 0
        magna_score = magna_signal.magna_score if magna_signal else 0
        conf = 0
        base_conf = 0

    # ── Sentiment Adjustment ──
    sent_label = ""
    sent_signals = []
    if sentiment is not None and base_conf > 0:
        from part3_sentiment import sentiment_confidence_adjustment
        conf, sent_notes = sentiment_confidence_adjustment(sentiment, base_conf)
        sent_label = sentiment.sentiment_label
        sent_signals = sentiment.signals[:2]

    conf = round(min(conf, 1.0), 2)

    triggers = ",".join(magna_signal.triggers) if magna_signal and magna_signal.triggers else ""

    rationale_parts = [
        f"{action} @ HKD{price:.2f}",
        f"Stop:{stop_loss:.2f}",
        f"Q={quality['quality_score']:.0%}",
        f"MAGNA={magna_score}/10",
    ]
    if sent_label:
        rationale_parts.append(f"😐{sent_label[:4]}")

    return {
        "ticker": ticker,
        "name": quality.get("name", ticker),
        "action": action,
        "quantity": position_size,
        "entry": round(price, 2),
        "stop_loss": stop_loss,
        "confidence": conf,
        "magna_score": magna_score,
        "quality_score": quality["quality_score"],
        "triggers": triggers,
        "sentiment_label": sent_label,
        "sentiment_signals": sent_signals,
        "rationale": " | ".join(rationale_parts),
    }


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_hk_pipeline(tickers: List[str] = None,
                    dry_run: bool = True,
                    output_path: str = None) -> dict:
    """Run VMAA-HK two-stage pipeline."""
    start_time = time.time()
    tickers = [t for t in (tickers or HSI_TICKERS) if not t.startswith("0011.HK")]  # skip delisted

    print(f"\n{'='*70}")
    print(f"🇭🇰 VMAA-HK Pipeline — {'DRY RUN' if dry_run else 'LIVE'} — "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # ── Market Regime ──
    print(f"\n[Market] Assessing HK regime...")
    market = get_hk_market_regime()
    hsi_p = market.get('hsi_price', 0)
    print(f"  HSI: {hsi_p:,.0f} | >MA50: {'✓' if market.get('above_ma50') else '✗'} | "
          f"Vol: {market.get('vol_regime', 'N/A')} | "
          f"Scalar: {market.get('position_scalar', 0.8)}x | "
          f"OK: {'✓' if market.get('market_ok') else '⚠️'}")

    print(f"\n[Universe] {len(tickers)} HSI constituents")
    print(f"{'='*70}")
    print(f"STAGE 1: Core Financial Fundamentals (HK-Adapted)")
    print(f"{'='*70}")

    # ── Stage 1: Fundamentals ──
    quality_pool: List[dict] = []
    for i, ticker in enumerate(tickers):
        try:
            result = screen_hk_fundamentals(ticker)
            if result:
                quality_pool.append(result)
        except Exception:
            pass

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(tickers)} ({len(quality_pool)} passed)")

    pass_rate = len(quality_pool) / len(tickers) * 100 if tickers else 0
    print(f"\nPart 1 complete: {len(quality_pool)}/{len(tickers)} passed ({pass_rate:.1f}%)")

    # Sort by quality
    quality_pool.sort(key=lambda x: x["quality_score"], reverse=True)

    print(f"\n🏆 Quality Pool ({len(quality_pool)} total):")
    for i, q in enumerate(quality_pool[:20]):
        fin = "🏦" if q.get("is_financial") else "  "
        print(f"  {i+1:2d}. {fin} {q['ticker']:<10s} {q['name'][:25]:25s} "
              f"Q={q['quality_score']:.0%} | "
              f"B/M={q['bm_ratio']:.2f} | {'ROE' if q.get('is_financial') else 'ROA'}="
              f"{q['roe'] if q.get('is_financial') else q['roa']:.1%} | "
              f"FCF/Y={q['fcf_yield']:.1%} | PTL={q['ptl_ratio']:.2f}")

    # ── Stage 2: MAGNA ──
    print(f"\n{'='*70}")
    print(f"STAGE 2: MAGNA 53/10 Momentum Signals")
    print(f"{'='*70}")
    print(f"Quality pool: {len(quality_pool)} stocks")

    signals: List[tuple] = []  # (quality_dict, HKMagnaSignal)
    for q in quality_pool:
        try:
            sig = screen_hk_magna(q["ticker"], q)
            if sig and sig.magna_score >= 1:
                signals.append((q, sig))
        except Exception as e:
            logger.debug(f"MAGNA failed for {q['ticker']}: {e}")

    signals.sort(key=lambda x: (x[1].magna_score, x[1].entry_ready), reverse=True)
    entry_ready = [(q, s) for q, s in signals if s.entry_ready]

    print(f"\nPart 2 complete: {len(signals)} have MAGNA signals "
          f"({len(entry_ready)} entry-ready)")

    if signals:
        print(f"\n🎯 MAGNA Signals:")
        for q, s in signals[:20]:
            marker = "⚡ENTRY" if s.entry_ready else "📊     "
            details = s.details
            extra = ""
            if details:
                ret = details.get('recent_return', 0)
                vr = details.get('vol_ratio', 0)
                ac = details.get('analyst_count', 0)
                extra = f" | Ret: {ret:+.1%}" if ret else ""
                extra += f" | VolRatio: {vr:.1f}x" if vr else ""
                extra += f" | Analysts: {ac}" if ac > 0 else ""
            print(f"  {marker} {s.ticker:<10s} MAGNA={s.magna_score}/10 "
                  f"G={'✓' if s.gap_trigger else '✗'} "
                  f"M={'✓' if s.eps_accel else '✗'} "
                  f"A={'✓' if s.sales_accel else '✗'} "
                  f"| {','.join(s.triggers) if s.triggers else '—'}{extra}")

    # ── Stage 3: Sentiment Analysis ──
    print(f"\n{'='*70}")
    print(f"STAGE 3: Sentiment Analysis")
    print(f"{'='*70}")

    from part3_sentiment import batch_sentiment
    signal_tickers = [s.ticker for _, s in signals[:30]]  # Top 30 MAGNA for sentiment
    if signal_tickers:
        print(f"Analyzing sentiment for {len(signal_tickers)} candidates...")
        sentiment_results = batch_sentiment(signal_tickers, delay=0.15)

        # Print sentiment summary
        print(f"\n😐 Sentiment Summary:")
        for tkr in signal_tickers[:15]:
            sr = sentiment_results.get(tkr)
            if sr:
                sig_str = f" [{','.join(sr.signals[:2])}]" if sr.signals else ""
                print(f"  {sr.sentiment_label:20s} {tkr:<10s} "
                      f"comp={sr.composite_score:+.2f} "
                      f"A={sr.analyst_score:+.2f} N={sr.news_score:+.2f} "
                      f"S={sr.social_score:+.2f} T={sr.technical_score:+.2f}"
                      f"{sig_str}")

        contrarians = [tkr for tkr in signal_tickers
                       if sentiment_results.get(tkr) and
                       "CONTRARIAN_BUY" in sentiment_results[tkr].signals]
        if contrarians:
            print(f"\n  🔥 Contrarian Opportunities ({len(contrarians)}):")
            for tkr in contrarians[:5]:
                sr = sentiment_results[tkr]
                # Find quality data
                q_data = next((q for q, _ in signals if q['ticker'] == tkr), None)
                q_score = q_data['quality_score'] if q_data else 0
                print(f"    {tkr:<10s} Q={q_score:.0%} Sent={sr.composite_score:+.2f}")
    else:
        sentiment_results = {}

    # ── Trade Decisions ──
    print(f"\n{'='*70}")
    print(f"💼 Trade Decisions — HKD")
    print(f"{'='*70}")

    decisions = []
    # Get prices
    price_cache = {}
    for q, _ in signals[:20]:
        try:
            hist = yf.Ticker(q["ticker"]).history(period="5d")
            if len(hist) > 0:
                price_cache[q["ticker"]] = float(hist['Close'].iloc[-1])
        except Exception:
            pass

    for q, s in signals[:15]:
        price = price_cache.get(q["ticker"], q.get("current_price", 0))
        if price <= 0:
            continue
        sent = sentiment_results.get(q["ticker"]) if sentiment_results else None
        dec = compute_hk_trade_decision(q["ticker"], q, s, price, market, sentiment=sent)
        decisions.append(dec)
        emoji = {"BUY": "🟢", "WATCH": "🟡", "SKIP": "⚪"}.get(dec["action"], "❓")
        sent_str = f"😐{dec.get('sentiment_label', '')[:4]}" if dec.get('sentiment_label') else ""
        print(f"  {emoji} {dec['action']:6s} {dec['ticker']:<10s} "
              f"{dec.get('name',''):28s} "
              f"HKD {dec['entry']:>8.2f} "
              f"Stop: {dec['stop_loss']:>8.2f} "
              f"Q={dec['quality_score']:.0%} "
              f"Conf={dec['confidence']:.0%} {sent_str}")

    # ── Save ──
    elapsed = time.time() - start_time
    result = {
        "status": "complete",
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "market": market,
        "pipeline": {
            "universe_size": len(tickers),
            "part1_passed": len(quality_pool),
            "part1_pass_rate": f"{pass_rate:.1f}%",
            "part2_signals": len(signals),
            "part2_entry_ready": len(entry_ready),
            "decisions": len(decisions),
            "buys": sum(1 for d in decisions if d["action"] == "BUY"),
            "watches": sum(1 for d in decisions if d["action"] == "WATCH"),
        },
        "quality_pool": [
            {k: v for k, v in q.items() if k not in ('passed_criteria', 'failed_criteria', 'warnings')}
            for q in quality_pool
        ],
        "signals": [
            {
                "ticker": s.ticker,
                "magna_score": s.magna_score,
                "entry_ready": s.entry_ready,
                "gap_trigger": s.gap_trigger,
                "eps_accel": s.eps_accel,
                "sales_accel": s.sales_accel,
                "triggers": s.triggers,
                "details": s.details,
            }
            for _, s in signals
        ],
        "decisions": decisions,
    }

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    out_path = output_path or str(output_dir / "hk_pipeline_result.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=np_safe)

    # ── Summary ──
    buy_count = sum(1 for d in decisions if d["action"] == "BUY")
    watch_count = sum(1 for d in decisions if d["action"] == "WATCH")

    print(f"\n{'='*70}")
    print(f"🇭🇰 VMAA-HK COMPLETE — {elapsed:.0f}s")
    print(f"  HSI: {hsi_p:,.0f} | Market: {'🟢 OK' if market.get('market_ok') else '⚠️ CAUTION'}")
    print(f"  Scanned:    {len(tickers)} HSI stocks")
    print(f"  Quality:    {len(quality_pool)} passed ({pass_rate:.1f}%)")
    print(f"  MAGNA:      {len(signals)} signals ({len(entry_ready)} entry-ready)")
    print(f"  Decisions:  {len(decisions)} ({buy_count} BUY, {watch_count} WATCH)")
    print(f"  Currency:   HKD")
    print(f"{'='*70}")
    print(f"\n📁 Results: {out_path}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="VMAA-HK — Hong Kong Stock Pipeline")
    parser.add_argument("--full-scan", action="store_true", default=True)
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Specific HK tickers (e.g., 0700.HK 0005.HK)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run_hk_pipeline(tickers=args.tickers, dry_run=not args.live, output_path=args.output)

if __name__ == "__main__":
    main()
