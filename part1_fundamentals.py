#!/usr/bin/env python3
"""
VMAA 2.0 — Part 1: Core Financial Fundamentals Screener
========================================================
Stage 1 of the two-stage pipeline. Purpose: ensure the company has genuine
cash-generation ability and asset efficiency, eliminating value traps.

Criteria (7 dimensions):
  1. Market Cap Positioning — deep value (<$250M) or turnaround (<$10B)
  2. High Profitability Quality — B/M + ROA + EBITDA vs sector
  3. Cash Flow Yield — Strong FCF/EV yield
  4. Safety Margin — Price near 52-week low
  5. Asset Expansion Constraint — ΔAssets < ΔEarnings
  6. Interest Rate Sensitivity — Identify IR-sensitive, high-leverage names
  7. Earnings Authenticity — FCF/NI conversion, quality over quantity

Output: Part1Result with quality_score and pass/fail breakdown.
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

from config import P1C
from models import Part1Result

logger = logging.getLogger("vmaa.part1")


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def screen_fundamentals(ticker: str, sector_medians: dict = None,
                        prefetched: dict = None) -> Optional[Part1Result]:
    """
    Screen a single stock through all Part 1 criteria.
    Optionally uses sector_medians for relative quality comparison (C2).
    If prefetched is provided, uses cached yfinance objects to avoid re-fetch.
        prefetched = {'info': dict, 'hist': DataFrame, 'ticker': yf.Ticker}
    Returns Part1Result if it passes, None if rejected.
    """
    try:
        info = {}
        hist = None
        t = None
        yf_ok = True
        used_fallback = False

        if prefetched:
            info = prefetched.get('info', {})
            hist = prefetched.get('hist')
            t = prefetched.get('ticker')
            if not info:
                yf_ok = False

        # Try yfinance first (if no prefetched or prefetched info empty)
        if not info:
            try:
                t = yf.Ticker(ticker)
                info = t.info
                hist = t.history(period="1y")
            except Exception as yf_e:
                logger.debug(f"  {ticker}: yfinance failed ({yf_e})")
                yf_ok = False

        # If yfinance failed (rate-limited / empty info), use Finnhub/SEC fallback
        if not yf_ok or not info or not info.get('regularMarketPrice'):
            # Try Finnhub first (fast, comprehensive)
            from data.hybrid import get_finnhub_fundamentals, get_bars_hybrid, get_fundamentals_tiger, get_finnhub_recommendation
            fund = get_finnhub_fundamentals(ticker)
            bars = get_bars_hybrid(ticker, "1y")

            if fund and len(fund) > 3:
                info.update(fund)
                # Get sector from Tiger if Finnhub didn't provide it
                if 'sector' not in info or info.get('sector') == 'Unknown':
                    try:
                        from data.hybrid import _get_tiger_qc
                        qc = _get_tiger_qc()
                        if qc:
                            ind = qc.get_stock_industry(ticker)
                            if ind:
                                for item in ind:
                                    if item.get('industry_level') == 'GSECTOR':
                                        info['sector'] = item.get('name_en', 'Unknown')
                                        break
                    except Exception:
                        pass
                # Populate essential fields from bars (average volume, price metadata)
                if bars is not None and len(bars) >= 1:
                    vol_col = 'Volume' if 'Volume' in bars.columns else 'volume'
                    if len(bars) >= 20:
                        info['averageVolume'] = int(bars[vol_col].tail(20).mean())
                    else:
                        info['averageVolume'] = int(bars[vol_col].mean())
                hist = bars
                if not yf_ok or t is None:
                    t = yf.Ticker(ticker)
                used_fallback = True
                logger.debug(f"  {ticker}: Using Finnhub fallback data")
            elif not fund or len(fund) < 3:
                # Fallback to SEC EDGAR
                if bars is not None and len(bars) >= 20:
                    sec_fund = get_fundamentals_tiger(ticker)
                    if sec_fund:
                        info = _build_info_from_sec(ticker, sec_fund, bars)
                        hist = bars
                        if not yf_ok or t is None:
                            t = yf.Ticker(ticker)
                        used_fallback = True
                        logger.debug(f"  {ticker}: Using Tiger/SEC fallback data")
                    else:
                        logger.debug(f"  {ticker}: No SEC fundamentals for fallback")
                else:
                    logger.debug(f"  {ticker}: Insufficient bar data ({len(bars) if bars is not None else 0} rows), skipping SEC")
            
            if not info and not used_fallback:
                logger.debug(f"  {ticker}: No data from any source")
                return None

        if not _basic_checks(info, hist, ticker):
            return None

        return _evaluate_part1(ticker, info, hist, t, sector_medians)
    except Exception as e:
        logger.debug(f"  {ticker}: Part 1 error — {e}")
        return None


def batch_screen(tickers: List[str], max_workers: int = 15) -> List[Part1Result]:
    """
    Screen a batch of tickers. Returns only passing candidates sorted by quality_score.
    Pre-computes sector medians for relative quality comparison (C2 fix).
    
    Uses ThreadPoolExecutor for parallel yfinance I/O — scales to 2000+ stocks.
    """
    results = []
    total = len(tickers)
    logger.info(f"Part 1: Screening {total} stocks for fundamental quality (workers={max_workers})...")

    # Pre-compute sector medians for relative comparison (fix: previously dead code)
    sector_medians = None
    try:
        logger.info("  Computing sector medians for relative quality comparison...")
        sector_medians = compute_sector_medians(tickers, sample_size=min(150, total))
        n_sectors = len(sector_medians)
        logger.info(f"  Sector medians computed for {n_sectors} sectors")
    except Exception as e:
        logger.warning(f"  Sector median computation failed ({e}), using absolute thresholds")

    # Parallel screen via ThreadPoolExecutor (I/O-bound: yfinance HTTP calls)
    completed = 0
    batch_start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(screen_fundamentals, t, sector_medians): t for t in tickers}
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0 or completed == total:
                elapsed = time.time() - batch_start
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(f"  Part 1 progress: {completed}/{total} "
                            f"({len(results)} passed, {rate:.0f} stocks/s)")
            try:
                result = future.result(timeout=45)
                if result:
                    results.append(result)
            except Exception:
                pass

    elapsed = time.time() - batch_start
    results.sort(key=lambda r: r.quality_score, reverse=True)
    logger.info(f"Part 1 complete: {len(results)}/{total} passed quality screening "
                f"({len(results)/max(total,1)*100:.1f}%) in {elapsed:.0f}s")
    return results


# ═══════════════════════════════════════════════════════════════════
# Basic Pre-checks
# ═══════════════════════════════════════════════════════════════════

def _basic_checks(info: dict, hist: pd.DataFrame, ticker: str = None) -> bool:
    """Fast-reject stocks that don't meet basic data requirements."""
    if hist is None or len(hist) < 20:
        return False

    price = get_price_from_info(info, ticker)
    if price <= 0 or price < P1C.min_price:
        return False

    avg_vol = info.get('averageVolume', 0) or 0
    if avg_vol < P1C.min_avg_volume:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════
# Criterion 1: Market Cap Positioning
# ═══════════════════════════════════════════════════════════════════

def _check_market_cap(info: dict) -> Tuple[bool, str, float]:
    """
    Classify market cap and reject mega-caps (unless large_cap_enabled).
    Returns: (passed, cap_type, market_cap_value)
    """
    market_cap = info.get('marketCap', 0) or 0
    if market_cap <= 0:
        return False, "unknown", 0

    if market_cap <= P1C.deep_value_max_cap:
        return True, "deep_value", market_cap
    elif market_cap <= P1C.turnaround_max_cap:
        return True, "turnaround", market_cap
    elif P1C.large_cap_enabled:
        return True, "large_cap", market_cap
    else:
        return False, "mega_cap", market_cap


# ═══════════════════════════════════════════════════════════════════
# Criterion 2: Quality — B/M, ROA, EBITDA
# ═══════════════════════════════════════════════════════════════════

def _check_quality(info: dict, price: float, sector_medians: dict = None) -> Dict[str, Any]:
    """
    Check B/M ratio, ROA, and EBITDA margin.
    When sector_medians is provided, uses relative-to-sector comparison
    ("顯著高於同業平均"). Falls back to absolute thresholds otherwise.
    Returns dict with pass/fail, values, and partial score.
    """
    result = {'bm_pass': False, 'roa_pass': False, 'ebitda_pass': False,
              'bm': 0.0, 'roa': 0.0, 'ebitda_margin': 0.0, 'score': 0.0}

    sector = info.get('sector', 'Unknown')
    sm = sector_medians.get(sector) if sector_medians else None

    # B/M Ratio — compare to sector median if available, else absolute
    book_value = info.get('bookValue', 0)
    if book_value and book_value > 0 and price > 0:
        bm = book_value / price
        result['bm'] = round(bm, 4)
        if sm and sm.get('bm'):
            # Must be significantly above sector median (≥10% premium for 顯著高於)
            result['bm_pass'] = bm >= sm['bm'] * 1.1
        else:
            result['bm_pass'] = bm >= P1C.min_bm_ratio

    # ROA — compare to sector median if available, else absolute
    roa = info.get('returnOnAssets', None)
    if roa is not None:
        result['roa'] = round(float(roa), 4)
        if sm and sm.get('roa') is not None:
            result['roa_pass'] = result['roa'] >= sm['roa'] * 1.1
        else:
            result['roa_pass'] = result['roa'] >= P1C.min_roa

    # EBITDA Margin — compare to sector median if available, else absolute
    ebitda = info.get('ebitda', 0)
    revenue = info.get('totalRevenue', 0)
    if ebitda and revenue and revenue > 0:
        ebitda_margin = ebitda / revenue
        result['ebitda_margin'] = round(ebitda_margin, 4)
        if sm and sm.get('ebitda_margin') is not None:
            result['ebitda_pass'] = ebitda_margin >= sm['ebitda_margin'] * 1.1
        else:
            result['ebitda_pass'] = ebitda_margin >= P1C.min_ebitda_margin

    # Quality scoring: when sector medians are available AND used for
    # pass/fail, normalise to sector-adjusted targets to avoid a stock
    # passing the sector-relative check but scoring 0 on absolute targets.
    q_score = 0.0
    if result['bm_pass']:
        if sm and sm.get('bm'):
            target = sm['bm'] * 1.5
        else:
            target = P1C.target_bm_ratio
        bm_score = min(result['bm'] / target, 1.0) if target > 0 else 0
        q_score += bm_score * P1C.weight_bm
    if result['roa_pass']:
        if sm and sm.get('roa') is not None:
            target = max(sm['roa'] * 1.5, 0.005)
        else:
            target = P1C.target_roa
        roa_score = min(result['roa'] / target, 1.0) if target > 0 else 0
        q_score += roa_score * P1C.weight_roa
    if result['ebitda_pass']:
        if sm and sm.get('ebitda_margin') is not None:
            target = sm['ebitda_margin'] * 1.5
        else:
            target = P1C.target_ebitda_margin
        ebitda_score = min(result['ebitda_margin'] / target, 1.0) if target > 0 else 0
        q_score += ebitda_score * P1C.weight_ebitda

    result['score'] = round(q_score, 4)
    return result


# ═══════════════════════════════════════════════════════════════════
# Criterion 3: FCF Yield — Cash Flow Quality
# ═══════════════════════════════════════════════════════════════════

def _check_fcf_yield(info: dict, market_cap: float) -> Tuple[bool, float, float]:
    """
    Check Free Cash Flow Yield.
    FCF Yield = FCF / Market Cap
    Returns: (passed, fcf_yield, score_contribution)
    """
    fcf = info.get('freeCashflow', 0)
    if not fcf or market_cap <= 0:
        return False, 0.0, 0.0

    fcf_yield = fcf / market_cap
    passed = fcf_yield >= P1C.min_fcf_yield
    score = min(fcf_yield / P1C.target_fcf_yield, 1.0) * P1C.weight_fcf_yield if passed else 0.0

    return passed, round(fcf_yield, 4), round(score, 4)


# ═══════════════════════════════════════════════════════════════════
# Criterion 4: Safety Margin — Near 52-week Low
# ═══════════════════════════════════════════════════════════════════

def _check_safety_margin(info: dict, price: float) -> Tuple[bool, float, float]:
    """
    Check price proximity to 52-week low.
    PTL ratio = Price / 52w-low
    Returns: (passed, ptl_ratio, score_contribution)
    """
    low_52w = info.get('fiftyTwoWeekLow', price)
    if low_52w <= 0:
        return False, 999.0, 0.0

    ptl = price / low_52w
    passed = ptl <= P1C.max_ptl_ratio

    # Score: closer to 1.0 = better
    if ptl <= 1.0:
        score = P1C.weight_ptl  # Full points at or below 52w low
    elif passed:
        # Linear decay from 1.0 (full) to max_ptl_ratio (0)
        score = P1C.weight_ptl * (1 - (ptl - 1.0) / (P1C.max_ptl_ratio - 1.0))
    else:
        score = 0.0

    return passed, round(ptl, 4), round(score, 4)


# ═══════════════════════════════════════════════════════════════════
# Criterion 5: Asset Expansion Constraint
# ═══════════════════════════════════════════════════════════════════

def _check_asset_efficiency(t: yf.Ticker, ticker: str = None) -> Tuple[bool, float, float, float, str]:
    """
    Check ΔAssets < ΔEarnings — capital efficiency constraint.
    Prevents companies growing through reckless acquisitions or over-investment.
    
    Returns: (passed, asset_growth, earnings_growth, score, status_str)
    """
    try:
        bs = t.balance_sheet
        fin = t.financials
        if bs is None or fin is None or bs.empty or fin.empty:
            return False, 0.0, 0.0, 0.0, "n/a"

        # Get Total Assets
        total_assets = None
        for label in ['Total Assets', 'TotalAssets', 'totalAssets']:
            if label in bs.index:
                total_assets = bs.loc[label]
                break
        if total_assets is None:
            # Try case-insensitive
            for idx in bs.index:
                if 'total asset' in str(idx).lower():
                    total_assets = bs.loc[idx]
                    break

        # Get Net Income
        net_income = None
        for label in ['Net Income', 'NetIncome', 'netIncome']:
            if label in fin.index:
                net_income = fin.loc[label]
                break
        if net_income is None:
            for idx in fin.index:
                if 'net income' in str(idx).lower():
                    net_income = fin.loc[idx]
                    break

        if total_assets is None or net_income is None:
            return False, 0.0, 0.0, 0.0, "n/a"

        # Prefer YoY (iloc[0] vs iloc[4]=same quarter prior year) to avoid
        # seasonal distortion.  Fall back to QoQ (iloc[0] vs iloc[1]) when
        # fewer than 5 quarters are available.
        use_yoy = len(total_assets) >= 5 and len(net_income) >= 5
        if not use_yoy and (len(total_assets) < 2 or len(net_income) < 2):
            return False, 0.0, 0.0, 0.0, "n/a"

        # yfinance quarterly data: .iloc[0] = most recent quarter
        assets_latest = float(total_assets.iloc[0])
        assets_prev = float(total_assets.iloc[4] if use_yoy else total_assets.iloc[1])
        ni_latest = float(net_income.iloc[0])
        ni_prev = float(net_income.iloc[4] if use_yoy else net_income.iloc[1])

        if assets_prev <= 0:
            return False, 0.0, 0.0, 0.0, "n/a"
        if abs(ni_prev) < 1e-6:
            return False, 0.0, 0.0, 0.0, "n/a"

        asset_growth = (assets_latest - assets_prev) / assets_prev
        earnings_growth = (ni_latest - ni_prev) / abs(ni_prev)

        # Core check: asset growth must be less than earnings growth
        passed = asset_growth < earnings_growth
        status = "asset<earnings" if passed else "asset>=earnings"
        score = P1C.weight_asset_efficiency if passed else 0.0

        return passed, round(asset_growth, 4), round(earnings_growth, 4), score, status

    except Exception as e:
        logger.debug(f"  Asset efficiency check failed: {e}")

    # ── Fallback: SEC EDGAR when yfinance rate-limited ──
    if ticker:
        try:
            from data.sec_edgar import get_sec_quarterly
            assets = get_sec_quarterly(ticker, 'Assets')
            incomes = get_sec_quarterly(ticker, 'NetIncomeLoss')

            if assets and incomes and len(assets) >= 2 and len(incomes) >= 2:
                # Use most recent two quarters
                use_yoy = len(assets) >= 5 and len(incomes) >= 5
                assets_latest = assets[0]['val']
                assets_prev = assets[4]['val'] if (use_yoy and len(assets) > 4) else assets[1]['val']
                ni_latest = incomes[0]['val']
                ni_prev = incomes[4]['val'] if (use_yoy and len(incomes) > 4) else incomes[1]['val']

                if assets_prev <= 0 or abs(ni_prev) < 1e-6:
                    return False, 0.0, 0.0, 0.0, "n/a"

                asset_growth = (assets_latest - assets_prev) / assets_prev
                earnings_growth = (ni_latest - ni_prev) / abs(ni_prev)

                passed = asset_growth < earnings_growth
                status = "asset<earnings" if passed else "asset>=earnings"
                score = P1C.weight_asset_efficiency if passed else 0.0
                logger.debug(f"  {ticker}: SEC asset efficiency OK (status={status})")
                return passed, round(asset_growth, 4), round(earnings_growth, 4), score, status
        except Exception as sec_e:
            logger.debug(f"  {ticker}: SEC asset efficiency fallback failed: {sec_e}")

    return False, 0.0, 0.0, 0.0, "n/a"


# ═══════════════════════════════════════════════════════════════════
# Criterion 6: Interest Rate Sensitivity
# ═══════════════════════════════════════════════════════════════════

def _check_ir_sensitivity(info: dict) -> Tuple[bool, float, float]:
    """
    Detect interest rate sensitivity.
    High debt/equity, high beta, or IR-sensitive sectors.
    Returns: (is_sensitive, debt_to_equity, beta)
    """
    de = info.get('debtToEquity', 0) or 0
    # Check for negative equity (meaningless D/E ratio)
    total_equity = info.get('totalEquity') or info.get('bookValue', 0) or 0
    if isinstance(total_equity, (int, float)) and total_equity <= 0:
        de = 9999  # Sentinel for "negative equity — D/E meaningless"
    elif de > 1000:
        de = 1000  # Cap at 1000x
    debt_to_equity = de
    beta = info.get('beta', 1.0) or 1.0
    sector = info.get('sector', '')

    sensitive = False
    if debt_to_equity > P1C.ir_debt_to_equity_threshold:
        sensitive = True
    if beta and beta > P1C.ir_beta_threshold:
        sensitive = True
    if sector in P1C.interest_sensitive_sectors:
        sensitive = True

    return sensitive, round(float(debt_to_equity), 2), round(float(beta), 2)


# ═══════════════════════════════════════════════════════════════════
# Criterion 7: Earnings Authenticity — FCF / Net Income
# ═══════════════════════════════════════════════════════════════════

def _check_earnings_authenticity(info: dict) -> Tuple[bool, float, float]:
    """
    FCF / Net Income ratio — measures how much earnings convert to actual cash.
    High ratio (>0.8) = genuine earnings. Low ratio = accounting profits, no cash.
    This replaces simple EPS growth which has low statistical significance.
    
    Returns: (passed, fcf_ni_ratio, score)
    """
    fcf = info.get('freeCashflow', 0)
    ni = info.get('netIncomeToCommon', 0) or info.get('netIncome', 0)

    if ni is None or abs(ni) < 1e-6:
        # Can't compute ratio, check if FCF is positive as fallback
        passed = fcf > 0
        return passed, 0.0, P1C.weight_fcf_conversion * 0.5 if passed else 0.0

    if ni < 0:
        # Negative net income: earnings are not authentic regardless of FCF.
        # Positive FCF with negative NI can happen temporarily (working cap
        # changes, delayed capex) but does not indicate sustainable earnings
        # quality. A money-losing company should NEVER get a high earnings
        # authenticity score.
        fcf_ni = 0.0
        if fcf > 0:
            # For informational purposes only: show FCF/abs(NI) capped at 0.2
            fcf_ni = min(fcf / abs(ni), 0.2)
        return False, round(fcf_ni, 4), 0.0

    fcf_ni = fcf / ni
    # Cap at 2.0 (more than 2x NI in FCF is unusual but positive)
    fcf_ni = min(fcf_ni, 2.0)

    passed = fcf_ni >= P1C.min_fcf_conversion
    score = min(fcf_ni / P1C.target_fcf_conversion, 1.0) * P1C.weight_fcf_conversion if passed else 0.0

    return passed, round(fcf_ni, 4), round(score, 4)


# ═══════════════════════════════════════════════════════════════════
# SEC EDGAR → yfinance info dict builder (fallback when yf rate-limited)
# ═══════════════════════════════════════════════════════════════════

def _build_info_from_sec(ticker: str, fund: dict, bars: pd.DataFrame) -> dict:
    """
    Build a yfinance-compatible info dict from SEC EDGAR fundamental data
    and Tiger/yfinance hybrid price bars.

    Provides enough fields for all 7 Part 1 criteria to work without yfinance.
    """
    info = {}

    # Basic metadata
    info['shortName'] = ticker
    info['symbol'] = ticker
    info['sector'] = 'Unknown'  # SEC doesn't provide sector
    info['industry'] = 'Unknown'

    # Price from bars
    if bars is not None and len(bars) >= 1:
        close_col = 'Close' if 'Close' in bars.columns else 'close'
        high_col = 'High' if 'High' in bars.columns else 'high'
        low_col = 'Low' if 'Low' in bars.columns else 'low'
        vol_col = 'Volume' if 'Volume' in bars.columns else 'volume'

        price = float(bars[close_col].iloc[-1])
        info['regularMarketPrice'] = price
        info['currentPrice'] = price
        info['previousClose'] = float(bars[close_col].iloc[-2]) if len(bars) >= 2 else price
        info['fiftyTwoWeekHigh'] = float(bars[high_col].max())
        info['fiftyTwoWeekLow'] = float(bars[low_col].min())
        # Average volume
        if len(bars) >= 20:
            info['averageVolume'] = int(bars[vol_col].tail(20).mean())
        else:
            info['averageVolume'] = int(bars[vol_col].mean())
    else:
        info['regularMarketPrice'] = 0
        info['currentPrice'] = 0
        info['averageVolume'] = 0

    # Revenue
    info['totalRevenue'] = fund.get('totalRevenue', 0) or fund.get('revenue_ttm', 0)
    info['revenueGrowth'] = 0  # Not available from single-company SEC data

    # Net Income
    info['netIncomeToCommon'] = fund.get('netIncomeToCommon', 0)
    info['netIncome'] = fund.get('netIncomeToCommon', 0)

    # EPS
    info['trailingEps'] = fund.get('trailingEps', 0)
    info['eps_ttm'] = fund.get('eps_ttm', 0)

    # Assets & Equity
    info['totalAssets'] = fund.get('totalAssets', 0)
    info['totalEquity'] = fund.get('totalEquity', 0)

    # Book Value — estimate from total equity if we have shares outstanding
    book_value_raw = fund.get('bookValue_raw', 0)
    if book_value_raw > 0:
        # We don't have shares outstanding from SEC, use a typical range
        # Try to get shares from Tiger briefs if price is available
        try:
            from data.hybrid import _get_tiger_qc
            qc = _get_tiger_qc()
            if qc:
                briefs = qc.get_stock_delay_briefs([ticker])
                if briefs is not None and not briefs.empty:
                    mcap = float(briefs.iloc[0].get('market_cap', 0) or 0)
                    price = info.get('regularMarketPrice', 0)
                    if mcap > 0 and price > 0:
                        shares = mcap / price
                        if shares > 0:
                            info['sharesOutstanding'] = int(shares)
                            info['bookValue'] = book_value_raw / shares
                            info['marketCap'] = mcap
        except Exception:
            pass
        # If market cap still not set, estimate from price and typical shares
        if not info.get('marketCap', 0):
            price = info.get('regularMarketPrice', 0)
            if price > 0:
                # Estimate shares from total equity and book value
                # Typical P/B range: 0.5-5x → shares = equity/(P/B * price)
                # Use conservative P/B=1.5 as default
                est_shares = book_value_raw / price  # if P/B=1
                info['marketCap'] = price * est_shares
                info['bookValue'] = price  # rough estimate
    else:
        info['bookValue'] = 0
        info['marketCap'] = 0

    # If market cap still 0, set a small positive value to avoid division errors
    if info.get('marketCap', 0) <= 0 and info.get('regularMarketPrice', 0) > 0:
        info['marketCap'] = info['regularMarketPrice'] * 10_000_000  # $10M min

    # EBITDA
    info['ebitda'] = fund.get('ebitda', 0)

    # Free Cash Flow
    info['freeCashflow'] = fund.get('freeCashflow', 0)

    # ROA
    info['returnOnAssets'] = fund.get('returnOnAssets', 0)

    # ROE
    info['returnOnEquity'] = fund.get('returnOnEquity', 0)

    # Debt to Equity
    info['debtToEquity'] = fund.get('debtToEquity', 0)

    # Other yfinance fields with sensible defaults
    info['beta'] = 1.0
    info['shortRatio'] = 0
    info['shortPercentOfFloat'] = 0
    info['numberOfAnalystOpinions'] = 0
    info['targetMeanPrice'] = 0
    info['trailingPE'] = 0
    info['forwardPE'] = 0
    info['profitMargins'] = 0
    info['earningsGrowth'] = 0
    info['grossMargins'] = fund.get('grossMargins', 0)

    # Mark data source
    info['vmaa_data_source'] = 'tiger_sec_fallback'

    return info


# ═══════════════════════════════════════════════════════════════════
# Comprehensive Part 1 Evaluation
# ═══════════════════════════════════════════════════════════════════

def _evaluate_part1(ticker: str, info: dict, hist: pd.DataFrame,
                    t: yf.Ticker, sector_medians: dict = None) -> Optional[Part1Result]:
    """
    Run all 7 criteria and compute composite quality score.
    Uses sector_medians for relative comparison when provided (C2).
    Returns Part1Result if quality_score >= min_quality_score, else None.
    """
    price = get_price_from_info(info, ticker)
    passed_criteria: List[str] = []
    failed_criteria: List[str] = []
    warnings: List[str] = []
    quality_score = 0.0

    # ── C1: Market Cap Positioning ──
    cap_pass, cap_type, market_cap = _check_market_cap(info)
    if not cap_pass:
        return None  # Hard reject mega-caps
    if cap_type == "deep_value":
        passed_criteria.append("cap_deep_value")
    else:
        passed_criteria.append("cap_turnaround")

    # ── C2: Quality (B/M, ROA, EBITDA) — relative to sector when available
    quality = _check_quality(info, price, sector_medians)
    for key in ['bm_pass', 'roa_pass', 'ebitda_pass']:
        if quality[key]:
            passed_criteria.append(key.replace('_pass', ''))
        else:
            failed_criteria.append(key.replace('_pass', ''))
    quality_score += quality['score']

    # ── C3: FCF Yield ──
    fcf_pass, fcf_yield, fcf_score = _check_fcf_yield(info, market_cap)
    if fcf_pass:
        passed_criteria.append("fcf_yield")
    else:
        failed_criteria.append("fcf_yield")
    quality_score += fcf_score

    # ── C4: Safety Margin (52-week low proximity) ──
    safety_pass, ptl_ratio, safety_score = _check_safety_margin(info, price)
    if safety_pass:
        passed_criteria.append("safety_margin")
    else:
        failed_criteria.append("safety_margin")
    quality_score += safety_score

    # ── C5: Asset Expansion Constraint ──
    asset_pass, asset_growth, earnings_growth, asset_score, asset_status = \
        _check_asset_efficiency(t, ticker)
    if asset_pass:
        quality_score += asset_score
        passed_criteria.append("asset_efficiency")
    elif asset_status == "n/a":
        warnings.append("asset_efficiency_n/a")
        quality_score += P1C.weight_asset_efficiency * 0.5  # partial credit for unavailable
    else:
        failed_criteria.append("asset_efficiency")

    # ── C6: Interest Rate Sensitivity ──
    ir_sensitive, debt_to_equity, beta = _check_ir_sensitivity(info)
    if ir_sensitive:
        passed_criteria.append("ir_sensitive")
        warnings.append(f"IR_sensitive(D/E={debt_to_equity},β={beta})")
    # Note: Being IR sensitive is a feature, not a bug — we WANT these names
    # because they benefit most from rate cuts.

    # ── C7: Earnings Authenticity (FCF Conversion) ──
    auth_pass, fcf_conversion, auth_score = _check_earnings_authenticity(info)
    if auth_pass:
        passed_criteria.append("earnings_authenticity")
    else:
        failed_criteria.append("earnings_authenticity")
    quality_score += auth_score

    # ── Round quality score ──
    quality_score = round(min(quality_score, 1.0), 4)

    # ── Pass/Fail Decision ──
    if quality_score < P1C.min_quality_score:
        logger.debug(f"  {ticker}: Quality {quality_score:.2f} < {P1C.min_quality_score}")
        return None

    # ── Build result ──
    low_52w = info.get('fiftyTwoWeekLow', price)
    high_52w = info.get('fiftyTwoWeekHigh', price)
    roe = info.get('returnOnEquity', 0) or 0

    rationale_parts = [
        f"Q={quality_score:.0%}",
        f"Cap={cap_type}({_fmt_cap(market_cap)})",
        f"B/M={quality['bm']:.2f}",
        f"ROA={quality['roa']:.1%}",
        f"FCF/Y={fcf_yield:.1%}",
        f"PTL={ptl_ratio:.2f}x",
    ]
    if asset_status == "asset<earnings":
        rationale_parts.append("A<EG✓")
    elif asset_status == "asset>=earnings":
        rationale_parts.append("A≥EG✗")
    if ir_sensitive:
        rationale_parts.append("IR+")
    if auth_pass:
        rationale_parts.append(f"FCF/NI={fcf_conversion:.1f}")

    return Part1Result(
        ticker=ticker,
        name=info.get('shortName', ticker),
        sector=info.get('sector', 'Unknown'),
        industry=info.get('industry', 'Unknown'),
        market_cap=market_cap,
        market_cap_type=cap_type,
        current_price=round(price, 2),
        low_52w=round(low_52w, 2),
        high_52w=round(high_52w, 2),
        ptl_ratio=ptl_ratio,
        bm_ratio=quality['bm'],
        roa=quality['roa'],
        roe=round(float(roe), 4),
        ebitda_margin=quality['ebitda_margin'],
        fcf_yield=fcf_yield,
        fcf_conversion=fcf_conversion,
        asset_growth=asset_growth,
        earnings_growth=earnings_growth,
        asset_vs_earnings=asset_status,
        debt_to_equity=debt_to_equity,
        beta=beta,
        interest_rate_sensitive=ir_sensitive,
        quality_score=quality_score,
        passed_criteria=passed_criteria,
        failed_criteria=failed_criteria,
        warnings=warnings,
        rationale=" | ".join(rationale_parts),
        data_date=datetime.now().strftime("%Y-%m-%d"),
    )


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def get_price_from_info(info: dict, ticker: str = None) -> float:
    """Extract current price from yfinance info dict.

    Works across multiple yfinance field names. Use this canonical
    implementation instead of inlining price extraction elsewhere.
    If ticker is provided and yfinance returns 0, falls back to
    data.hybrid.get_price() (Tiger delayed → yfinance history).
    """
    price = float(
        info.get('regularMarketPrice') or
        info.get('currentPrice') or
        info.get('previousClose', 0)
    )
    if price <= 0 and ticker:
        try:
            from data.hybrid import get_price as hybrid_get_price
            price, _, source, _ = hybrid_get_price(ticker)
            if price > 0:
                logger.debug(f"  {ticker}: Using Tiger/yfinance hybrid price (source={source})")
        except Exception:
            pass
    return price


def _fmt_cap(cap: float) -> str:
    """Format market cap for display."""
    if cap >= 1e9:
        return f"${cap/1e9:.1f}B"
    elif cap >= 1e6:
        return f"${cap/1e6:.0f}M"
    else:
        return f"${cap:,.0f}"


# ═══════════════════════════════════════════════════════════════════
# Sector Median Computation (for enhanced B/M, ROA, EBITDA comparison)
# ═══════════════════════════════════════════════════════════════════

def _fetch_sector_info(ticker: str):
    """Fetch a single ticker's sector metrics. Returns (sector, data_dict) or None."""
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', 'Other')
        price = get_price_from_info(info, ticker)
        bv = info.get('bookValue', 0)
        roa = info.get('returnOnAssets')
        ebitda = info.get('ebitda', 0)
        rev = info.get('totalRevenue', 0)
        fcf = info.get('freeCashflow', 0)
        mcap = info.get('marketCap', 0)
        return sector, {
            'bm': bv / price if bv and price > 0 else None,
            'roa': float(roa) if roa is not None else None,
            'ebitda_margin': ebitda / rev if ebitda and rev else None,
            'fcf_yield': fcf / mcap if fcf and mcap else None,
        }
    except Exception:
        return None


def compute_sector_medians(tickers: List[str],
                           sample_size: int = 150) -> Dict[str, Dict[str, float]]:
    """
    Compute sector-level median metrics for relative comparison.
    Sample a subset, fetch in parallel via ThreadPoolExecutor.
    Returns: {sector: {bm_median, roa_median, ebitda_median, fcf_yield_median}}
    """
    import random
    if not tickers or sample_size <= 0:
        return {}
    sample = random.sample(tickers, min(sample_size, len(tickers)))

    sector_data: Dict[str, List[Dict]] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_sector_info, t): t for t in sample}
        for future in as_completed(futures):
            result = future.result()
            if result:
                sector, data = result
                if sector not in sector_data:
                    sector_data[sector] = []
                sector_data[sector].append(data)

    medians = {}
    for sector, values in sector_data.items():
        medians[sector] = {}
        for metric in ['bm', 'roa', 'ebitda_margin', 'fcf_yield']:
            vals = [v[metric] for v in values if v[metric] is not None]
            medians[sector][metric] = round(float(np.median(vals)), 4) if vals else None

    return medians
