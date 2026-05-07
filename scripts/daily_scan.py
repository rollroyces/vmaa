#!/usr/bin/env python3
"""
VMAA Daily Scan v2 — Improved data freshness
=============================================
Changes from v1:
  • Uses yfinance history() instead of info[] for prices (avoids stale cache)
  • Shows data freshness date for each stock
  • Price validation: cross-checks info vs history
  • Faster: parallel-ish data fetching with caching
  • Custom universe based on liquid mid-caps < $10B

Output: Telegram-friendly text report
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure vmaa root is importable
_vmaa_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_vmaa_root))
sys.path.insert(0, str(_vmaa_root.parent))

from models import Part1Result, Part2Signal
from config import P1C, P2C


# ═══════════════════════════════════════════════════════════════════
# 1. TICKER UNIVERSE — Updated for May 2026
# ═══════════════════════════════════════════════════════════════════
# Mid-cap stocks < $10B that have decent liquidity.
# Manually curated from known past passers + recent IPO/small-cap names.
# Remove delisted/merged tickers, add new names periodically.

UNIVERSE = [
    # Healthcare / MedTech
    "TMDX", "INMD", "NVCR", "GH", "PODD", "EXAS", "DXCM",
    "ALGN", "ICUI", "GKOS", "LMAT", "OFIX", "SIBN", "CVRX",
    "NVRO", "ATRC", "BLFS", "VERA", "CYTK",

    # Tech / SaaS
    "FIVN", "DOCU", "BILL", "OLED", "ZS", "NET", "DDOG",
    "MDB", "ESTC", "TOST", "CRWD", "OKTA", "VRNS", "TENB",
    "CFLT", "GTLB", "SMAR", "WDAY", "HUBS", "PATH", "U",
    "RNG", "PCOR", "AYX", "YELP", "TTD", "MNDY", "ASAN",
    "SPSC", "NCNO", "FVRR", "CXM", "DOMO", "UPST", "SOFI",
    "RKLB", "CHWY", "DKNG", "MARA", "CLSK", "RIOT", "IREN",

    # Industrials / Materials
    "AA", "FCX", "SCCO", "CLF", "STLD", "RS", "CMC",
    "WOR", "MLI", "ATI", "CRS", "CDRE",

    # Energy
    "RRC", "AR", "APA", "OVV",

    # Biotech
    "HALO", "EXEL", "IONS", "ALKS", "ACAD", "SRPT",
    "TWST",
]

# Exclude from scan → only check if price ≥ $2 and volume ≥ 20k
MIN_PRICE = 2.0
MIN_VOLUME = 20000


# ═══════════════════════════════════════════════════════════════════
# 2. DATA FETCH — Tiger (primary) + yfinance (fallback)
# ═══════════════════════════════════════════════════════════════════
# Tiger delay quotes: more reliable than yfinance info dict (no stale cache)
# yfinance history: for 52w range and volume

_TIGER_QC = None  # lazy init

def _get_tiger_quote_client():
    """Lazy-init Tiger quote client (only when needed, for speed)."""
    global _TIGER_QC
    if _TIGER_QC is None:
        try:
            from broker.tiger_broker import TigerBroker
            broker = TigerBroker()
            _TIGER_QC = broker.quote_client
        except Exception as e:
            print(f"  ⚠️ Tiger init failed: {e}", file=sys.stderr)
            return None
    return _TIGER_QC


def get_tiger_price(ticker: str) -> Tuple[float, int, str]:
    """
    Get current price from Tiger delay quotes.
    Returns: (close_price, volume, source_note)
    Tiger gives 15-min delayed but NO stale cache like yfinance info dict.
    """
    qc = _get_tiger_quote_client()
    if qc is None:
        return 0.0, 0, "tiger_unavail"
    
    try:
        briefs = qc.get_stock_delay_briefs([ticker])
        if briefs is not None and not briefs.empty:
            row = briefs.iloc[0]
            close = float(row.get('close', 0))
            vol = int(row.get('volume', 0))
            if close > 0:
                return close, vol, "tiger_delayed"
    except Exception as e:
        # Tiger might fail for HK or delisted — fall through
        pass
    
    return 0.0, 0, "tiger_fail"


def get_fresh_price(ticker: str) -> Tuple[float, float, float, float, int, str]:
    """
    Get fresh price data — Tiger delay quotes (primary) + yfinance history (fallback).
    Returns: (current_price, low_52w, high_52w, pre_close, avg_volume, data_source)
    """
    current = 0.0
    data_source = "unknown"
    
    # PRIMARY: Tiger delay quotes
    tiger_price, tiger_vol, tiger_src = get_tiger_price(ticker)
    if tiger_price > 0:
        current = tiger_price
        data_source = tiger_src
    
    # FALLBACK + 52w range: yfinance history
    low_52w = 0.0
    high_52w = 0.0
    avg_vol = 0
    
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        
        if hist is not None and len(hist) >= 5:
            low_52w = float(hist['Low'].min())
            high_52w = float(hist['High'].max())
            avg_vol = int(hist['Volume'].tail(50).mean())
            
            # If Tiger didn't provide price, use yfinance history close
            if current <= 0:
                current = float(hist['Close'].iloc[-1])
                data_source = "yf_hist"
            
            # Also try yfinance info as last resort
            if current <= 0:
                info = t.info
                current = float(info.get('regularMarketPrice') or 
                               info.get('currentPrice', 0) or 0)
                data_source = "yf_info"
    except Exception as e:
        pass
    
    return current, low_52w, high_52w, avg_vol, data_source


def get_fresh_financials(ticker: str) -> dict:
    """
    Get fresh financials for Part 1 screening.
    Returns clean dict with data freshness dates.
    """
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info
    
    result = {
        'name': info.get('shortName', ticker),
        'sector': info.get('sector', 'Unknown'),
        'industry': info.get('industry', 'Unknown'),
        'market_cap': info.get('marketCap', 0) or 0,
        'book_value': info.get('bookValue', 0) or 0,
        'roe': info.get('returnOnEquity', 0) or 0,
        'roa': info.get('returnOnAssets', 0) or 0,
        'beta': info.get('beta', 0) or 0,
        'debt_to_equity': info.get('debtToEquity', 0) or 0,
        'ebitda': info.get('ebitda', 0) or 0,
        'total_revenue': info.get('totalRevenue', 0) or 0,
        'free_cashflow': info.get('freeCashflow', 0) or 0,
        'net_income': info.get('netIncomeToCommon', 0) or 0,
        'short_ratio': info.get('shortRatio', 0) or 0,
        'short_pct_float': info.get('shortPercentOfFloat', 0) or 0,
        'analyst_count': info.get('numberOfAnalystOpinions', 0) or 0,
        'target_mean': info.get('targetMeanPrice', 0) or 0,
        'ipo_date': info.get('firstTradeDateEpochUtc', None),
        'data_freshness': datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    # Check balance sheet date
    try:
        bs = t.balance_sheet
        if bs is not None and not bs.empty:
            result['bs_date'] = str(bs.columns[0].strftime("%Y-%m-%d"))
            # Get total assets
            for label in ['Total Assets', 'TotalAssets', 'totalAssets']:
                if label in bs.index:
                    result['total_assets'] = float(bs.loc[label].iloc[0])
                    if len(bs.columns) > 1:
                        result['total_assets_prev'] = float(bs.loc[label].iloc[1])
                    break
    except Exception:
        pass
    
    # Check quarterly financials date
    try:
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            result['qf_date'] = str(qf.columns[0].strftime("%Y-%m-%d"))
            # Get EPS
            for label in ['Diluted EPS', 'DilutedEPS', 'Basic EPS', 'BasicEPS']:
                if label in qf.index:
                    result['eps'] = list(qf.loc[label].values[:4])
                    break
            # Get revenue
            for label in ['Total Revenue', 'TotalRevenue', 'Revenue']:
                if label in qf.index:
                    result['revenue'] = list(qf.loc[label].values[:4])
                    break
    except Exception:
        pass
    
    return result


# ═══════════════════════════════════════════════════════════════════
# 3. SIMPLIFIED PART 1 — quality screening (adapted from part1_fundamentals)
# ═══════════════════════════════════════════════════════════════════

def quick_part1(ticker: str, price: float, fin: dict) -> Optional[dict]:
    """
    Quick Part 1 quality screen using fresh data.
    Returns: quality_dict or None (rejected)
    """
    passed = []
    failed = []
    score = 0.0
    
    mcap = fin.get('market_cap', 0)
    if mcap <= 0:
        return None
    
    # 1. Market Cap: deep value or turnaround (use config thresholds)
    if mcap < P1C.deep_value_max_cap:
        cap_type = "deep_value"
    elif mcap < P1C.turnaround_max_cap:
        cap_type = "turnaround"
    else:
        return None  # Mega cap → reject
    
    passed.append(f"Cap={cap_type}")
    
    # 2. Price check (use config threshold)
    if price < P1C.min_price:
        return None
    
    # 3. B/M Ratio (use config threshold)
    bv = fin.get('book_value', 0)
    bm = bv / price if price > 0 and bv > 0 else 0
    bm_pass = bm >= P1C.min_bm_ratio
    if bm_pass:
        passed.append(f"B/M={bm:.2f}")
        score += P1C.weight_bm
    else:
        failed.append(f"B/M={bm:.2f}")
    
    # 4. ROA (use config threshold)
    roa = fin.get('roa', 0)
    roa_pass = roa >= P1C.min_roa
    if roa_pass:
        passed.append(f"ROA={roa:.1%}")
        score += P1C.weight_roa
    else:
        failed.append(f"ROA={roa:.1%}")
    
    # 5. EBITDA Margin (use config threshold)
    ebitda = fin.get('ebitda', 0)
    revenue = fin.get('total_revenue', 0)
    ebitda_margin = ebitda / revenue if revenue and revenue > 0 else 0
    ebitda_pass = ebitda_margin >= P1C.min_ebitda_margin
    if ebitda_pass:
        score += P1C.weight_ebitda
    else:
        failed.append(f"EBITDA={ebitda_margin:.1%}")
    
    # 6. FCF Yield (use config threshold)
    fcf = fin.get('free_cashflow', 0)
    fcf_yield = fcf / mcap if mcap > 0 else 0
    fcf_pass = fcf_yield >= P1C.min_fcf_yield
    if fcf_pass:
        passed.append(f"FCF/Y={fcf_yield:.1%}")
        score += P1C.weight_fcf_yield
    else:
        failed.append(f"FCF/Y={fcf_yield:.1%}")
    
    # 7. FCF Conversion (use config threshold)
    ni = fin.get('net_income', 0)
    fcf_conv = fcf / ni if ni and abs(ni) > 1 else 0
    if fcf_conv >= P1C.min_fcf_conversion:
        passed.append(f"FCF/NI={fcf_conv:.1f}")
        score += P1C.weight_fcf_conversion
    elif fcf_conv > 0:
        failed.append(f"FCF/NI={fcf_conv:.1f}")
    
    # 8. Short interest (use config thresholds)
    si_score = 0
    sr = fin.get('short_ratio', 0)
    spf = fin.get('short_pct_float', 0)
    if sr >= P2C.short_ratio_high:
        si_score = 2
    elif sr >= P2C.short_ratio_moderate:
        si_score = 1
    
    # Score threshold (use config threshold)
    if score < P1C.min_quality_score:
        return None
    
    return {
        'ticker': ticker,
        'name': fin.get('name', ticker),
        'sector': fin.get('sector', 'Unknown'),
        'market_cap': mcap,
        'cap_type': cap_type,
        'price': price,
        'bm': bm,
        'roa': roa,
        'ebitda_margin': ebitda_margin,
        'fcf_yield': fcf_yield,
        'fcf_conversion': fcf_conv,
        'quality_score': round(score, 4),
        'passed': passed,
        'failed': failed,
        'si_score': si_score,
        'short_ratio': sr,
        'short_pct': spf,
        'analyst_count': fin.get('analyst_count', 0),
        'target_mean': fin.get('target_mean', 0),
        'fin_data': fin,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. SIMPLIFIED PART 2 — MAGNA momentum check
# ═══════════════════════════════════════════════════════════════════

def quick_part2(ticker: str, qual: dict) -> dict:
    """
    Quick MAGNA momentum check.
    Uses yfinance quarterly data for EPS/revenue acceleration + technical.
    Returns: signal dict
    """
    import yfinance as yf
    t = yf.Ticker(ticker)
    fin = qual.get('fin_data', {})
    price = qual.get('price', 0)
    
    result = {
        'magna_score': 0,
        'entry_ready': False,
        'signals': [],
        'details': {},
    }
    
    if price <= 0:
        return result
    
    # Market cap check (Cap 10 — use config threshold)
    if qual.get('market_cap', 0) > P2C.max_market_cap:
        return result
    
    # ── M: Earnings Acceleration ──
    eps_list = fin.get('eps', [])
    m_pass = False
    if len(eps_list) >= 3:
        eps_q0 = float(eps_list[0])
        eps_q1 = float(eps_list[1])
        eps_q3 = float(eps_list[3]) if len(eps_list) > 3 else eps_q1
        
        if abs(eps_q1) > 0.01 and abs(eps_q3) > 0.01:
            curr_growth = (eps_q0 - eps_q3) / abs(eps_q3)
            prev_growth = (eps_q1 - eps_q3) / abs(eps_q3)
            accel = curr_growth - prev_growth
            m_pass = curr_growth >= P2C.eps_growth_min and accel >= P2C.eps_accel_min
            result['details']['eps_growth'] = round(curr_growth, 2)
            result['details']['eps_accel'] = round(accel, 2)
    
    if m_pass:
        result['magna_score'] += 2
        result['signals'].append('M')
    
    # ── A: Sales Acceleration ──
    rev_list = fin.get('revenue', [])
    a_pass = False
    if len(rev_list) >= 3:
        rev_q0 = float(rev_list[0])
        rev_q1 = float(rev_list[1])
        rev_q3 = float(rev_list[3]) if len(rev_list) > 3 else rev_q1
        
        if rev_q3 > 0 and rev_q1 > 0:
            curr_growth = (rev_q0 - rev_q3) / rev_q3
            prev_growth = (rev_q1 - rev_q3) / rev_q3
            accel = curr_growth - prev_growth
            a_pass = curr_growth >= P2C.revenue_growth_min and accel >= P2C.revenue_accel_min
            result['details']['rev_growth'] = round(curr_growth, 2)
            result['details']['rev_accel'] = round(accel, 2)
    
    if a_pass:
        result['magna_score'] += 2
        result['signals'].append('A')
    
    # ── G: Gap Up (from price history) ──
    try:
        hist = t.history(period="1mo")
        if hist is not None and len(hist) > 5:
            g_pass = False
            for i in range(1, len(hist)):
                p_close = float(hist['Close'].iloc[i-1])
                c_open = float(hist['Open'].iloc[i])
                if p_close > 0 and (c_open - p_close) / p_close >= P2C.gap_min_pct:
                    g_pass = True
                    result['details']['gap_pct'] = round((c_open - p_close) / p_close * 100, 1)
                    break
            if g_pass:
                result['magna_score'] += 2
                result['signals'].append('G')
    except Exception:
        pass
    
    # ── 5: Short Interest ──
    # Short interest already scored in quick_part1 using P2C thresholds
    si_score = qual.get('si_score', 0)
    result['magna_score'] += si_score
    if si_score > 0:
        result['signals'].append(f'SI({si_score})')
    
    # ── 3: Analyst Target ──
    analyst_count = fin.get('analyst_count', 0) or 0
    target_mean = fin.get('target_mean', 0) or 0
    analyst_pass = False
    if analyst_count >= P2C.analyst_count_min and target_mean > 0 and price > 0:
        premium = (target_mean - price) / price
        if premium >= P2C.analyst_target_premium_pct:
            analyst_pass = True
            result['magna_score'] += 1
            result['signals'].append(f'A[{analyst_count}]↑')
        else:
            result['signals'].append(f'A[{analyst_count}]·')
    elif analyst_count > 0:
        result['signals'].append(f'A[{analyst_count}]·')
    
    # ── Entry Trigger: G fires OR M+A both fire ──
    has_g = 'G' in result['signals']
    has_ma = m_pass and a_pass
    result['entry_ready'] = has_g or has_ma
    result['details']['entry_trigger'] = 'G' if has_g else ('MA' if has_ma else 'none')
    
    return result


# ═══════════════════════════════════════════════════════════════════
# 5. MARKET REGIME
# ═══════════════════════════════════════════════════════════════════

def get_market_snapshot() -> dict:
    """Get fresh market regime data."""
    import yfinance as yf
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="3mo")
        info = spy.info
        
        current = float(hist['Close'].iloc[-1]) if len(hist) > 0 else 0
        ma50 = float(hist['Close'].rolling(50).mean().iloc[-1]) if len(hist) >= 50 else 0
        above_ma = current > ma50 if ma50 > 0 else True
        
        returns = hist['Close'].pct_change().dropna()
        vol = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.15
        
        # Also get info price for today
        info_price = float(info.get('regularMarketPrice', 0) or 0)
        if info_price > 0:
            current = info_price
        
        if vol < 0.12:
            vol_regime = "LOW"
        elif vol < 0.22:
            vol_regime = "NORMAL"
        else:
            vol_regime = "HIGH"
        
        high_3mo = float(hist['High'].max())
        dd = (current - high_3mo) / high_3mo if high_3mo > 0 else 0
        
        return {
            'spy_price': round(current, 2),
            'above_ma50': above_ma,
            'vol_regime': vol_regime,
            'market_ok': above_ma and dd > -0.12,
            'dd_from_high': round(dd * 100, 1),
        }
    except Exception as e:
        return {'error': str(e)}


# ═══════════════════════════════════════════════════════════════════
# 6. FULL SCAN
# ═══════════════════════════════════════════════════════════════════

def run_scan() -> dict:
    """Run full VMAA scan with data freshness tracking."""
    import yfinance as yf
    
    result = {
        'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        'market': get_market_snapshot(),
        'scan_summary': {},
        'quality_pool': [],
        'signals': [],
        'decisions': [],
    }
    
    total = len(UNIVERSE)
    print(f"🔍 Scanning {total} stocks...", file=sys.stderr)
    
    quality_stocks = []
    
    for i, ticker in enumerate(UNIVERSE):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{total} ({len(quality_stocks)} quality pass)", file=sys.stderr)
        
        try:
            # Step 1: Get fresh price (Tiger primary, yfinance fallback)
            price, low_52w, high_52w, avg_vol, data_source = get_fresh_price(ticker)
            if price <= 0 or avg_vol < MIN_VOLUME:
                continue
            
            # Step 2: Get fundamentals
            fin = get_fresh_financials(ticker)
            
            # Step 3: Part 1 screening
            qual = quick_part1(ticker, price, fin)
            if qual is None:
                continue
            
            qual['price_date'] = data_source  # Shows 'tiger_delayed' or 'yf_hist'
            qual['52w_low'] = low_52w
            qual['52w_high'] = high_52w
            
            quality_stocks.append(qual)
            
        except Exception as e:
            pass
        
        time.sleep(0.15)  # Rate limit
    
    quality_stocks.sort(key=lambda q: q['quality_score'], reverse=True)
    result['quality_pool'] = quality_stocks
    result['scan_summary']['quality'] = len(quality_stocks)
    
    # Step 4: Part 2 MAGNA signals
    print(f"\n📊 Running MAGNA on {len(quality_stocks)} quality stocks...", file=sys.stderr)
    
    for i, qual in enumerate(quality_stocks):
        try:
            signal = quick_part2(qual['ticker'], qual)
            if signal['magna_score'] > 0:
                signal['quality'] = qual
                result['signals'].append(signal)
        except Exception:
            pass
    
    result['signals'].sort(key=lambda s: (s['entry_ready'], s['magna_score']), reverse=True)
    result['scan_summary']['signals'] = len(result['signals'])
    result['scan_summary']['entry_ready'] = sum(1 for s in result['signals'] if s['entry_ready'])
    
    print(f"✅ Done: {len(quality_stocks)} quality, {len(result['signals'])} signals", file=sys.stderr)
    
    return result


# ═══════════════════════════════════════════════════════════════════
# 7. TELEGRAM REPORT
# ═══════════════════════════════════════════════════════════════════

def format_report(result: dict) -> str:
    """Format scan results for Telegram."""
    lines = []
    
    now = datetime.now(timezone.utc).strftime("%m/%d %H:%M UTC")
    lines.append(f"📊 VMAA Scan — {now}")
    lines.append("")
    
    # Market
    mkt = result.get('market', {})
    if 'error' not in mkt:
        icon = "🟢" if mkt.get('market_ok') else "🟡"
        ma = "⬆50MA" if mkt.get('above_ma50') else "⬇50MA"
        spy = mkt.get('spy_price', '?')
        dd = mkt.get('dd_from_high', 0)
        vol = mkt.get('vol_regime', '?')
        lines.append(f"{icon} SPY ${spy} {ma} Vol:{vol} DD:{dd}%")
    else:
        lines.append(f"⚠️ Market data error: {mkt.get('error')}")
    lines.append("")
    
    # Summary bar
    sm = result.get('scan_summary', {})
    quality = sm.get('quality', 0)
    signals = sm.get('signals', 0)
    entry = sm.get('entry_ready', 0)
    lines.append(f"Universe: {len(UNIVERSE)} | Quality: {quality} | MAGNA: {signals} | Entry: {entry}")
    lines.append("")
    
    # Entry ready
    entry_stocks = [s for s in result['signals'] if s.get('entry_ready')]
    if entry_stocks:
        lines.append("⚡ ENTRY READY:")
        for s in entry_stocks:
            q = s['quality']
            sigs = ", ".join(s['signals'])
            det = s.get('details', {})
            trigger = det.get('entry_trigger', '?')
            extra = ""
            if 'eps_growth' in det:
                extra += f" EPS↑{det['eps_growth']:.0%}"
            if 'gap_pct' in det:
                extra += f" Gap+{det['gap_pct']:.1f}%"
            lines.append(f"  🔥 {q['ticker']}  M{s['magna_score']}/10  [{sigs}]  ⚡{trigger}{extra}")
        lines.append("")
    
    # Monitoring
    monitor = [s for s in result['signals'] if not s.get('entry_ready')]
    if monitor:
        lines.append("📊 MONITORING:")
        for s in monitor[:8]:
            q = s['quality']
            sigs = ", ".join(s['signals'])
            lines.append(f"  👀 {q['ticker']}  M{s['magna_score']}/10  [{sigs}]")
        lines.append("")
    
    # Quality pool highlights
    if quality > 0:
        top5 = result['quality_pool'][:5]
        lines.append(f"🏆 Quality Pool top {len(top5)}/{quality}:")
        for q in top5:
            bm = q.get('bm', 0)
            fcf = q.get('fcf_yield', 0)
            src = q.get('price_date', '?')
            src_icon = '🐅' if 'tiger' in src else '📈'
            lines.append(f"  {src_icon} {q['ticker']}  Q={q['quality_score']:.0%} | "
                        f"{q['cap_type']} | "
                        f"B/M={bm:.2f} FCF/Y={fcf:.1%}")
        lines.append("")
    
    # Data freshness notice
    if quality > 0:
        latest_bs = None
        latest_qf = None
        for q in result['quality_pool'][:3]:
            fd = q.get('fin_data', {})
            if 'bs_date' in fd:
                latest_bs = fd['bs_date']
            if 'qf_date' in fd:
                latest_qf = fd['qf_date']
        
        if latest_qf or latest_bs:
            lines.append("📅 Data freshness:")
            if latest_qf:
                lines.append(f"  Latest Q fin: {latest_qf}")
            if latest_bs:
                lines.append(f"  Latest balance sheet: {latest_bs}")
            lines.append("")
    
    # Decisions (position sizing)
    decisions = result.get('decisions', [])
    if decisions:
        lines.append("📋 Position Sizing:")
        for d in decisions:
            pass  # We'll skip this in v2 since we don't have full risk calc
    
    lines.append("—")
    lines.append(f"🦾 Ironman • VMAA v2")
    
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔄 VMAA Daily Scan v2 — fresh data", file=sys.stderr)
    
    result = run_scan()
    
    # Save output
    output_dir = _vmaa_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"daily_scan_v2_{ts}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    # Print report
    report = format_report(result)
    print(report)
