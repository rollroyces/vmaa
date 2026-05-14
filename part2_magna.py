#!/usr/bin/env python3
"""
VMAA 2.0 — Part 2: MAGNA 53/10 Momentum Screener
==================================================
Stage 2 of the two-stage pipeline. Purpose: capture momentum and breakout
signals on stocks that already passed Part 1 quality screening.

MAGNA 53/10 Components:
  M — Massive Earnings Acceleration
  A — Acceleration of Sales
  G — Gap Up (>4% gap + pre-market volume >100K)
  N — Neglect / Base pattern (months of sideways consolidation)
  5 — Short Interest Ratio (elevated = squeeze potential)
  3 — Analyst Target Upgrades (≥3 analysts, target above current)
  Cap 10 — Market Cap strictly under $10B
  10 — IPO within 10 years

Entry Trigger Logic:
  - G (Gap Up) fires → immediate entry signal
  - M + A both fire → entry signal (fundamental acceleration)
  - Otherwise → MONITOR (stock is in quality pool but no trigger yet)
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from vmaa.config import P2C
from vmaa.models import Part1Result, Part2Signal
from vmaa.part1_fundamentals import get_price_from_info

logger = logging.getLogger("vmaa.part2")


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def screen_magna(ticker: str, part1: Optional[Part1Result] = None,
                 prefetched: dict = None) -> Optional[Part2Signal]:
    """
    Run full MAGNA 53/10 screening on a single stock.
    If part1 is provided, skips market cap and IPO checks (already done).
    If prefetched is provided, uses cached yfinance objects to avoid re-fetch.
        prefetched = {'info': dict, 'hist': DataFrame, 'ticker': yf.Ticker}
    Returns Part2Signal with scores and trigger assessment.
    """
    try:
        if prefetched:
            info = prefetched.get('info', {})
            hist = prefetched.get('hist')
            t = prefetched.get('ticker')
            if t is None:
                t = yf.Ticker(ticker)
        else:
            t = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="1y")

        if hist is None or len(hist) < 60:
            logger.debug(f"  {ticker}: Insufficient price history for Part 2")
            return None

        return _evaluate_magna(ticker, info, hist, t, part1)
    except Exception as e:
        logger.debug(f"  {ticker}: Part 2 error — {e}")
        return None


def batch_screen_magna(quality_pool: List[Part1Result], max_workers: int = 12) -> List[Part2Signal]:
    """
    Screen all stocks in the quality pool for MAGNA signals.
    Returns only stocks with valid signals, sorted by MAGNA score.
    
    Uses ThreadPoolExecutor for parallel yfinance I/O.
    """
    signals = []
    total = len(quality_pool)
    logger.info(f"Part 2: Screening {total} quality-pool stocks for MAGNA signals (workers={max_workers})...")

    completed = 0
    batch_start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(screen_magna, p1.ticker, part1=p1): p1
            for p1 in quality_pool
        }
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - batch_start
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(f"  Part 2 progress: {completed}/{total} "
                            f"({len(signals)} signals, {rate:.0f} stocks/s)")
            try:
                signal = future.result(timeout=45)
                if signal:
                    signals.append(signal)
            except Exception:
                pass

    elapsed = time.time() - batch_start
    signals.sort(key=lambda s: (s.magna_score, s.entry_ready), reverse=True)
    entry_ready = [s for s in signals if s.entry_ready]
    logger.info(f"Part 2 complete: {len(signals)}/{total} have MAGNA signals "
                f"({len(entry_ready)} entry-ready) in {elapsed:.0f}s")
    return signals


# ═══════════════════════════════════════════════════════════════════
# M: Massive Earnings Acceleration
# ═══════════════════════════════════════════════════════════════════

def check_earnings_accel(t: yf.Ticker, ticker: str = None) -> Tuple[bool, float, float, float]:
    """
    Check for massive earnings acceleration.
    Compares latest quarter EPS growth vs previous quarter.
    Falls back to SEC EDGAR quarterly EPS when yfinance fails.
    Returns: (passed, current_growth, prev_growth, acceleration)
    """
    yf_failed = False
    try:
        fin = t.quarterly_financials
        if fin is None or fin.empty:
            yf_failed = True
        else:
            # Try to get Diluted EPS
            eps = None
            for label in ['Diluted EPS', 'DilutedEPS', 'Basic EPS', 'BasicEPS']:
                if label in fin.index:
                    eps = fin.loc[label]
                    break
            if eps is None:
                for idx in fin.index:
                    if 'eps' in str(idx).lower() or 'earnings per share' in str(idx).lower():
                        eps = fin.loc[idx]
                        break

            if eps is None or len(eps) < 3:
                yf_failed = True
            else:
                # Latest Q, Previous Q, Same Q last year
                eps_q0 = float(eps.iloc[0])
                eps_q1 = float(eps.iloc[1])
                eps_q4 = float(eps.iloc[4]) if len(eps) > 4 else (float(eps.iloc[3]) if len(eps) > 3 else eps_q1)

                if abs(eps_q1) < 1e-6 or abs(eps_q4) < 1e-6:
                    yf_failed = True
                else:
                    current_growth = (eps_q0 - eps_q4) / abs(eps_q4)
                    prev_growth = (eps_q1 - eps_q4) / abs(eps_q4) if len(eps) > 3 else current_growth
                    acceleration = current_growth - prev_growth

                    passed = (current_growth >= P2C.eps_growth_min and
                              acceleration >= P2C.eps_accel_min)
                    return passed, round(current_growth, 4), round(prev_growth, 4), round(acceleration, 4)

    except Exception as e:
        logger.debug(f"  Earnings acceleration check failed (yfinance): {e}")
        yf_failed = True

    # ── Fallback: SEC EDGAR quarterly EPS ──
    if yf_failed and ticker:
        try:
            from data.sec_edgar import get_sec_quarterly
            eps_data = get_sec_quarterly(ticker, 'EarningsPerShareDiluted')
            if not eps_data:
                eps_data = get_sec_quarterly(ticker, 'EarningsPerShareBasic')

            if eps_data and len(eps_data) >= 3:
                # Latest Q vs Same Q last year (QoQ same quarter)
                eps_q0 = eps_data[0].get('val', 0)
                eps_q1 = eps_data[1].get('val', 0)
                # Same quarter last year = offset 4 quarters
                eps_q4 = eps_data[4].get('val', 0) if len(eps_data) > 4 else eps_q0

                if abs(eps_q4) > 1e-6:
                    current_growth = (eps_q0 - eps_q4) / abs(eps_q4)
                    prev_growth = (eps_q1 - eps_q4) / abs(eps_q4) if len(eps_data) > 3 else current_growth
                    acceleration = current_growth - prev_growth

                    passed = (current_growth >= P2C.eps_growth_min and
                              acceleration >= P2C.eps_accel_min)
                    logger.debug(f"  {ticker}: SEC earnings accel OK "
                                f"(growth={current_growth:.2%}, accel={acceleration:.2%})")
                    return passed, round(current_growth, 4), round(prev_growth, 4), round(acceleration, 4)
        except Exception as sec_e:
            logger.debug(f"  {ticker}: SEC earnings fallback failed: {sec_e}")

    return False, 0.0, 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════
# A: Acceleration of Sales
# ═══════════════════════════════════════════════════════════════════

def check_sales_accel(t: yf.Ticker, ticker: str = None) -> Tuple[bool, float, float, float]:
    """
    Check for revenue acceleration.
    Compares latest quarter revenue growth vs previous quarter.
    Falls back to SEC EDGAR quarterly revenue when yfinance fails.
    Returns: (passed, current_growth, prev_growth, acceleration)
    """
    yf_failed = False
    try:
        fin = t.quarterly_financials
        if fin is None or fin.empty:
            yf_failed = True
        else:
            # Get Total Revenue
            revenue = None
            for label in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                if label in fin.index:
                    revenue = fin.loc[label]
                    break
            if revenue is None:
                for idx in fin.index:
                    if 'revenue' in str(idx).lower():
                        revenue = fin.loc[idx]
                        break

            if revenue is None or len(revenue) < 3:
                yf_failed = True
            else:
                rev_q0 = float(revenue.iloc[0])
                rev_q1 = float(revenue.iloc[1])
                rev_q4 = float(revenue.iloc[4]) if len(revenue) > 4 else (float(revenue.iloc[3]) if len(revenue) > 3 else rev_q1)

                if rev_q4 <= 0:
                    yf_failed = True
                else:
                    current_growth = (rev_q0 - rev_q4) / rev_q4
                    prev_growth = (rev_q1 - rev_q4) / rev_q4
                    acceleration = current_growth - prev_growth

                    passed = (current_growth >= P2C.revenue_growth_min and
                              acceleration >= P2C.revenue_accel_min)
                    return passed, round(current_growth, 4), round(prev_growth, 4), round(acceleration, 4)

    except Exception as e:
        logger.debug(f"  Sales acceleration check failed (yfinance): {e}")
        yf_failed = True

    # ── Fallback: SEC EDGAR quarterly revenue ──
    if yf_failed and ticker:
        try:
            from data.sec_edgar import get_sec_quarterly
            rev_data = get_sec_quarterly(ticker, 'Revenues')
            if not rev_data:
                rev_data = get_sec_quarterly(ticker, 'RevenueFromContractWithCustomerExcludingAssessedTax')

            if rev_data and len(rev_data) >= 3:
                rev_q0 = rev_data[0].get('val', 0)
                rev_q1 = rev_data[1].get('val', 0)
                rev_q4 = rev_data[4].get('val', 0) if len(rev_data) > 4 else rev_q0

                if rev_q4 > 0:
                    current_growth = (rev_q0 - rev_q4) / rev_q4
                    prev_growth = (rev_q1 - rev_q4) / rev_q4 if len(rev_data) > 3 else current_growth
                    acceleration = current_growth - prev_growth

                    passed = (current_growth >= P2C.revenue_growth_min and
                              acceleration >= P2C.revenue_accel_min)
                    logger.debug(f"  {ticker}: SEC sales accel OK "
                                f"(growth={current_growth:.2%}, accel={acceleration:.2%})")
                    return passed, round(current_growth, 4), round(prev_growth, 4), round(acceleration, 4)
        except Exception as sec_e:
            logger.debug(f"  {ticker}: SEC sales fallback failed: {sec_e}")

    return False, 0.0, 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════
# G: Gap Up Detection
# ═══════════════════════════════════════════════════════════════════

def _check_gap_up(hist: pd.DataFrame, info: dict) -> Tuple[bool, bool, float, int]:
    """
    Detect gap-up events:
      Condition 1: Gap > 4% (open > prev close by 4%+)
      Condition 2: Gap day volume ≥ 1.5x the 20-day average volume

    FIXED (2026-05): Previously checked info.get('preMarketVolume')
    which yfinance NEVER returns (always 0/None), making G trigger
    effectively broken. Now uses the gap day's actual total volume
    vs its 20-day average — a genuine gap-up should have elevated volume.
    
    Returns: (gap_detected, volume_confirmed, gap_pct, gap_day_volume)
    """
    gap_detected = False
    gap_pct = 0.0
    gap_day_volume = 0
    volume_confirmed = False
    lookback = min(P2C.gap_lookback_days, len(hist) - 1)

    recent = hist.tail(lookback + 1)
    best_gap = 0.0
    best_gap_day_volume = 0
    best_gap_idx = -1

    for i in range(1, len(recent)):
        prev_close = float(recent['Close'].iloc[i-1])
        curr_open = float(recent['Open'].iloc[i])
        if prev_close > 0:
            gap = (curr_open - prev_close) / prev_close
            if gap >= P2C.gap_min_pct:
                gap_day_vol = int(recent['Volume'].iloc[i])
                # Track best (gap × volume) combination instead of first match
                if gap * gap_day_vol > best_gap * best_gap_day_volume:
                    best_gap = gap
                    best_gap_day_volume = gap_day_vol
                    best_gap_idx = i
                    gap_detected = True

    # After scanning all bars, verify volume for the best gap
    volume_confirmed = False
    if gap_detected:
        gap_pct = best_gap
        gap_day_volume = best_gap_day_volume
        # Check if best gap day volume ≥ multiplier × 20-day average
        # AND absolute volume ≥ 100K (gap_min_volume)
        vol_window = recent['Volume'].iloc[max(0, best_gap_idx - 20):best_gap_idx]
        if len(vol_window) >= 5:
            avg_vol_20d = float(vol_window.mean())
            if avg_vol_20d > 0:
                volume_confirmed = (gap_day_volume >= avg_vol_20d * P2C.gap_volume_multiplier
                                    and gap_day_volume >= P2C.gap_min_volume)
        elif gap_day_volume > 0:
            avg_vol = info.get('averageVolume', 0) or 0
            if avg_vol > 0:
                volume_confirmed = (gap_day_volume >= avg_vol * P2C.gap_volume_multiplier
                                    and gap_day_volume >= P2C.gap_min_volume)
        else:
            # Not enough volume history — flag but don't block gap signal
            logger.warning(f"  Gap up detected but only {len(vol_window)} bars of volume "
                          f"history (need ≥5) — volume confirmation skipped")

    return gap_detected, volume_confirmed, round(gap_pct, 4), gap_day_volume


# ═══════════════════════════════════════════════════════════════════
# N: Neglect / Base Pattern Detection
# ═══════════════════════════════════════════════════════════════════

def _check_neglect_base(hist: pd.DataFrame) -> Tuple[bool, float, float]:
    """
    Detect if stock is in a neglect/base consolidation pattern.
    Characteristics:
      - Sideways price action within 20-30% range
      - Duration of at least 3 months
      - Declining volume over the base period
    
    Returns: (base_detected, duration_months, range_pct)
    """
    if len(hist) < 63:  # Need at least ~3 months of data
        return False, 0.0, 0.0

    # Split into two halves: early base vs late base
    midpoint = len(hist) // 2
    early = hist.iloc[:midpoint]
    late = hist.iloc[midpoint:]

    # Overall range
    high_all = float(hist['High'].max())
    low_all = float(hist['Low'].min())
    if high_all <= 0:
        return False, 0.0, 0.0

    range_pct = (high_all - low_all) / high_all

    # Duration in months
    days = (hist.index[-1] - hist.index[0]).days
    duration_months = round(days / 30.44, 1)

    # Volume decline: is late volume lower than early volume?
    early_vol = float(early['Volume'].mean()) if len(early) > 0 else 0
    late_vol = float(late['Volume'].mean()) if len(late) > 0 else 0
    vol_decline = (early_vol - late_vol) / early_vol if early_vol > 0 else 0

    # Check base criteria
    is_base = (
        duration_months >= P2C.base_min_months and
        range_pct <= P2C.base_max_range_pct and
        vol_decline >= P2C.base_vol_decline_pct
    )

    return is_base, duration_months, round(range_pct, 4)


# ═══════════════════════════════════════════════════════════════════
# 5: Short Interest
# ═══════════════════════════════════════════════════════════════════

def _check_short_interest(info: dict) -> Tuple[int, float, float]:
    """
    Evaluate short interest for squeeze potential.
    Returns: (score 0-2, short_ratio, short_pct_float)
    """
    short_ratio = info.get('shortRatio', 0) or 0
    short_pct = info.get('shortPercentOfFloat', 0) or 0

    score = 0
    if short_ratio >= P2C.short_ratio_high and short_pct >= P2C.short_pct_float_min:
        score = 2
    elif short_ratio >= P2C.short_ratio_moderate:
        score = 1

    return score, round(float(short_ratio), 2), round(float(short_pct), 4)


# ═══════════════════════════════════════════════════════════════════
# 3: Analyst Target Upgrades
# ═══════════════════════════════════════════════════════════════════

def _check_analyst_upgrades(info: dict, price: float) -> Tuple[bool, int, float]:
    """
    Check if ≥3 analysts cover the stock with target above current price.
    
    Note: yfinance doesn't provide analyst revision history, so we use:
      - Number of analysts covering
      - Mean target price vs current price (≥15% upside)
    For full revision tracking, would need Bloomberg/Refinitiv data.
    
    Returns: (passed, analyst_count, target_mean)
    """
    analyst_count = info.get('numberOfAnalystOpinions', 0) or 0
    target_mean = info.get('targetMeanPrice', 0) or 0

    if analyst_count < P2C.analyst_count_min:
        return False, analyst_count, 0.0

    if price <= 0 or target_mean <= 0:
        return False, analyst_count, target_mean

    premium = (target_mean - price) / price
    passed = premium >= P2C.analyst_target_premium_pct

    return passed, analyst_count, round(float(target_mean), 2)


# ═══════════════════════════════════════════════════════════════════
# Cap 10 & 10 (prerequisites, not scored)
# ═══════════════════════════════════════════════════════════════════

def _check_cap_and_ipo(info: dict) -> Tuple[bool, bool, Optional[float]]:
    """
    Prerequisite checks (not scored):
      - Market cap < $10B
      - IPO within 10 years
    
    Returns: (cap_ok, ipo_ok, ipo_years)
    """
    market_cap = info.get('marketCap', 0) or 0
    if market_cap > 0 and market_cap <= P2C.max_market_cap:
        cap_ok = True
    elif market_cap > P2C.max_market_cap and P2C.large_cap_enabled:
        cap_ok = True  # Large cap accepted when enabled
    else:
        cap_ok = False

    ipo_years = None
    ipo_ok = False
    first_trade = info.get('firstTradeDateEpochUtc')
    if first_trade:
        try:
            ipo_date = datetime.fromtimestamp(first_trade)
            ipo_years = round((datetime.now() - ipo_date).days / 365.25, 1)
            ipo_ok = ipo_years <= P2C.max_ipo_years
        except Exception:
            pass

    return cap_ok, ipo_ok, ipo_years


# ═══════════════════════════════════════════════════════════════════
# Full MAGNA Evaluation
# ═══════════════════════════════════════════════════════════════════

def _evaluate_magna(ticker: str, info: dict, hist: pd.DataFrame,
                     t: yf.Ticker, part1: Optional[Part1Result] = None) -> Optional[Part2Signal]:
    """
    Run all MAGNA 53/10 checks and compute composite score.
    Entry is triggered when G fires OR M+A both fire.
    """
    price = get_price_from_info(info, ticker)

    magna_score = 0.0
    trigger_signals: List[str] = []
    details: Dict[str, Any] = {}

    # ── Cap 10 & 10 (prerequisites) ──
    if part1 and part1.market_cap > 0:
        # Use Part1's market cap (already vetted, more current)
        cap_ok = (part1.market_cap <= P2C.max_market_cap or
                  (part1.market_cap > P2C.max_market_cap and P2C.large_cap_enabled))
        # Still need IPO check from yfinance info
        _, ipo_ok, ipo_years = _check_cap_and_ipo(info)
    else:
        cap_ok, ipo_ok, ipo_years = _check_cap_and_ipo(info)

    # Market cap check (configurable — large_cap_enabled bypasses MAGNA's <$10B requirement)
    if not cap_ok:
        logger.debug(f"  {ticker}: Cap > $10B, rejected by MAGNA Cap 10")
        return None
    # IPO 10: hard requirement per spec ("需在10年以內")
    if not ipo_ok and ipo_years is not None:
        if P2C.ipo_hard_requirement:
            logger.debug(f"  {ticker}: IPO > 10yr ({ipo_years:.0f}yr), REJECTED")
            return None
        else:
            logger.debug(f"  {ticker}: IPO > 10yr ({ipo_years:.0f}yr), soft flag (not rejected)")

    # ── M: Earnings Acceleration (graduated scoring) ──
    m_pass, eps_curr, eps_prev, eps_accel = check_earnings_accel(t, ticker)
    # Graduated EPS score: 20%+=2pt, 15%+=1.5pt, 10%+=1pt, 5%+=0.5pt
    eps_growth_rate = eps_curr if eps_curr > 0 else 0
    eps_score = 0.0
    if eps_growth_rate >= 0.20: eps_score = 2.0
    elif eps_growth_rate >= 0.15: eps_score = 1.5
    elif eps_growth_rate >= 0.10: eps_score = 1.0
    elif eps_growth_rate >= 0.05: eps_score = 0.5
    if m_pass:
        magna_score += max(P2C.magna_points['m_earnings_accel'], eps_score)
    elif eps_score > 0:
        magna_score += eps_score
    if eps_score >= 0.5:
        trigger_signals.append(f'M({eps_score:.1f})')

    # ── A: Sales Acceleration (graduated scoring) ──
    a_pass, rev_curr, rev_prev, rev_accel = check_sales_accel(t, ticker)
    # Graduated sales score: 10%+=2pt, 8%+=1.5pt, 5%+=1pt
    sales_score = 0.0
    if rev_curr >= 0.10: sales_score = 2.0
    elif rev_curr >= 0.08: sales_score = 1.5
    elif rev_curr >= 0.05: sales_score = 1.0
    if a_pass:
        magna_score += max(P2C.magna_points['a_sales_accel'], sales_score)
    elif sales_score > 0:
        magna_score += sales_score
    if sales_score >= 0.5:
        trigger_signals.append(f'A({sales_score:.1f})')

    # ── Price Momentum Bonus ──
    # 3-month return vs universe = quality signal for growth candidates
    price_momentum = 0.0
    if hist is not None and len(hist) >= 60:
        price_3m_ago = float(hist['Close'].iloc[-63]) if len(hist) >= 63 else float(hist['Close'].iloc[0])
        price_now = float(hist['Close'].iloc[-1])
        if price_3m_ago > 0:
            mom_3m = (price_now - price_3m_ago) / price_3m_ago
            if mom_3m >= 0.10: price_momentum = 1.0
            elif mom_3m >= 0.05: price_momentum = 0.5
            details['price_momentum_3m'] = round(mom_3m, 4)
    if price_momentum > 0:
        magna_score += price_momentum
        trigger_signals.append(f'PM({price_momentum:.1f})')

    # ── V3 Momentum Filter: 1-month return (don't catch falling knives) ──
    momentum_filter_pass = True
    mom_1m = 0.0
    if hist is not None and len(hist) >= 21:
        price_1m_ago = float(hist['Close'].iloc[-21])
        price_now_mom = float(hist['Close'].iloc[-1])
        if price_1m_ago > 0:
            mom_1m = (price_now_mom - price_1m_ago) / price_1m_ago
            min_1m = P2C.magna_entry_min_1m_return
            momentum_filter_pass = mom_1m > min_1m
            details['mom_1m'] = round(mom_1m, 4)
            if not momentum_filter_pass:
                trigger_signals.append(f'⚠MOM({mom_1m:.1%})')

    # ── G: Gap Up ──
    g_pass, g_vol_ok, gap_pct, gap_vol = _check_gap_up(hist, info)
    # Only award points and trigger if BOTH gap AND volume conditions met
    g_full_pass = g_pass and g_vol_ok
    if g_pass:
        trigger_signals.append('G')
    if g_full_pass:
        magna_score += P2C.magna_points['g_gap_up']
        trigger_signals.append('G_vol')

    # ── N: Neglect / Base ──
    n_pass, base_duration, base_range = _check_neglect_base(hist)
    if n_pass:
        magna_score += P2C.magna_points['n_neglect_base']
        trigger_signals.append('N')

    # ── 5: Short Interest ──
    si_score, short_ratio, short_pct = _check_short_interest(info)
    magna_score += si_score
    if si_score > 0:
        trigger_signals.append(f'SI({si_score})')

    # ── 3: Analyst Target Upgrades ──
    analyst_pass, analyst_count, analyst_target = _check_analyst_upgrades(info, price)
    # Check for RECENT upgrade via cached analyst data (heuristic for spec requirement)
    recently_upgraded = False
    if analyst_pass:
        try:
            from analyst_tracker import check_recent_upgrade
            recently_upgraded = check_recent_upgrade(ticker, analyst_target, analyst_count)
        except ImportError:
            pass  # Tracker not available, fall through to basic check
    upgraded_pass = analyst_pass and recently_upgraded
    if upgraded_pass:
        magna_score += P2C.magna_points['analyst_upgrades']
        trigger_signals.append(f'A[{analyst_count}]↑')
    elif analyst_pass:
        # Analysts favorable but no recent upgrade detected — flag as informational
        trigger_signals.append(f'A[{analyst_count}]·')

    # ── Composite ──
    magna_score = round(min(magna_score, 10), 1)

    # ── Entry Trigger Logic (V3 rewritten) ──
    # V3 rules:
    #   - G + volume confirmed → instant entry
    #   - M+A + at least 1 bonus (N, SI≥1, or analyst upgraded) → entry
    #   - Must pass momentum filter (1m return > -5%)
    growth_composite = eps_score + sales_score + price_momentum
    details['growth_composite'] = round(growth_composite, 1)

    has_bonus = (n_pass or si_score >= 1 or upgraded_pass)
    entry_ready = False

    if not momentum_filter_pass:
        entry_ready = False  # V3: momentum filter blocks all entries
    elif g_full_pass:
        entry_ready = True   # Gap-up WITH volume confirmation
    elif m_pass and a_pass and has_bonus:
        entry_ready = True   # V3: M+A PLUS at least 1 bonus signal required
    elif growth_composite >= 2.5:
        entry_ready = True   # Strong graduated growth signal (legacy fallback)

    if magna_score < P2C.magna_pass_threshold and not entry_ready:
        return None

    return Part2Signal(
        ticker=ticker,
        m_earnings_accel=m_pass,
        a_sales_accel=a_pass,
        g_gap_up=g_pass,
        g_volume_confirmed=g_vol_ok,
        n_neglect_base=n_pass,
        short_interest_high=(si_score >= 2),
        short_interest_score=si_score,
        analyst_upgrades=upgraded_pass,
        analyst_recently_upgraded=recently_upgraded,
        cap_under_10b=cap_ok,
        ipo_within_10yr=ipo_ok,
        eps_growth_qoq=eps_curr,
        eps_growth_prev_qoq=eps_prev,
        eps_acceleration=eps_accel,
        revenue_growth_qoq=rev_curr,
        revenue_growth_prev_qoq=rev_prev,
        revenue_acceleration=rev_accel,
        gap_pct=gap_pct,
        gap_day_volume=gap_vol,
        short_ratio=short_ratio,
        short_pct_float=short_pct,
        base_duration_months=base_duration,
        base_range_pct=base_range,
        analyst_count=analyst_count,
        analyst_target_mean=analyst_target,
        ipo_years=ipo_years,
        magna_score=magna_score,
        trigger_signals=trigger_signals,
        entry_ready=entry_ready,
        signal_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        magna_momentum_filter=momentum_filter_pass,
    )
