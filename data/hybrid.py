#!/usr/bin/env python3
"""
VMAA Hybrid Data Layer
======================
Multi-source data with intelligent fallback:

  Price         → Tiger delay quotes (primary) → yfinance history (fallback)
  US stocks     → SEC EDGAR official (primary) → yfinance (fallback/cross-check)
  Foreign cos   → yfinance (primary) → SEC EDGAR annual (annual verify)
  Cross-verify  → Compare SEC vs yfinance, flag discrepancies

Usage:
  from data.hybrid import HybridData
  hd = HybridData()
  data = hd.get_fundamentals("TMDX")  # Returns unified dict
  price = hd.get_price("INMD")        # Returns (price, source)
"""
from __future__ import annotations

import logging
# Suppress tiger_openapi SDK debug noise
import logging as _logging
_logging.getLogger("tiger_openapi").setLevel(_logging.WARNING)
_logging.getLogger("getmac").setLevel(_logging.WARNING)
_logging.getLogger("urllib3.connectionpool").setLevel(_logging.WARNING)
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
import numpy as np

# Finnhub rate limit tracking (free tier: 60 calls/min)
_FINNHUB_CALL_TIMES = []

# Increase Tiger API connection pool (default size=1 causes bottleneck)
try:
    from urllib3 import PoolManager, HTTPConnectionPool
    # Patch: increase pool size for Tiger API host
    _tiger_adapter = HTTPConnectionPool('openapi.tigerfintech.com', maxsize=20, block=False)
except Exception:
    pass

from data.sec_edgar import (
    get_fundamentals_sec,
    is_foreign_issuer,
    has_sec_data,
    clear_cache as sec_clear_cache,
)

# SQLite data cache (720-day retention, once-per-day fetch)
try:
    from data.cache import cache_get, cache_set
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    def cache_get(*a, **kw): return None
    def cache_set(*a, **kw): return False

# Tiger Trade broker import (module-level for reuse)
try:
    from broker.tiger_broker import TigerBroker
    _TIGER_AVAILABLE = True
except ImportError:
    TigerBroker = None
    _TIGER_AVAILABLE = False

logger = logging.getLogger("vmaa.data.hybrid")


# ═══════════════════════════════════════════════════════════════════
# 1. PRICE — Tiger (primary) + yfinance (fallback)
# ═══════════════════════════════════════════════════════════════════

_TIGER_QC = None
_TIGER_LOCK = __import__('threading').Lock()
_TIGER_BARS_EXHAUSTED = True  # Skip Tiger bars (pool issues), use delay briefs instead

def _get_tiger_qc():
    global _TIGER_QC
    if _TIGER_QC is None:
        with _TIGER_LOCK:
            if _TIGER_QC is not None:
                return _TIGER_QC
            try:
                if TigerBroker is None:
                    return None
                broker = TigerBroker()
                _TIGER_QC = broker.quote_client
                # Increase Tiger API connection pool for concurrent workers
                try:
                    for client in [broker.trade_client, broker.quote_client]:
                        session = getattr(client, '_session', None)
                        if session:
                            from requests.adapters import HTTPAdapter
                            adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=3)
                            session.mount('https://', adapter)
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"Tiger init failed: {e}")
    return _TIGER_QC
def get_price(ticker: str) -> Tuple[float, int, str, str]:
    """
    Get current price from Tiger (primary) → yfinance (fallback) → Finnhub quote (last resort).
    Returns: (price, volume, source_note, data_date)
    """
    # Try Tiger first
    qc = _get_tiger_qc()
    if qc:
        try:
            briefs = qc.get_stock_delay_briefs([ticker])
            if briefs is not None and not briefs.empty:
                row = briefs.iloc[0]
                close = float(row.get("close", 0))
                vol = int(row.get("volume", 0))
                if close > 0:
                    return close, vol, "tiger_delayed", datetime.now().strftime("%Y-%m-%d")
        except Exception:
            pass
    
    # Fallback: yfinance history
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo")
        if hist is not None and len(hist) >= 2:
            close = float(hist["Close"].iloc[-1])
            vol = int(hist["Volume"].iloc[-1])
            date = hist.index[-1].strftime("%Y-%m-%d")
            return close, vol, "yf_hist", date
    except Exception:
        pass
    
    # Last resort: yfinance info
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = float(info.get("regularMarketPrice") or info.get("currentPrice", 0) or 0)
        if price > 0:
            return price, 0, "yf_info", "?"
    except Exception:
        pass

    # Final fallback: Finnhub quote (works when yfinance 401s)
    try:
        import requests as _req
        resp = _req.get(
            f'https://finnhub.io/api/v1/quote?symbol={ticker}&token={_FINNHUB_KEY}',
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            close = float(data.get('c', 0))
            if close > 0:
                return close, 0, "finnhub_quote", datetime.now().strftime("%Y-%m-%d")
    except Exception:
        pass
    
    return 0.0, 0, "none", ""


def get_prices_batch(tickers: List[str]) -> Dict[str, Tuple[float, int, str, str]]:
    """
    Batch price fetch via Tiger delay quotes (1 API call for all tickers).
    Falls back to individual yfinance calls for tickers Tiger fails on.
    
    Returns: {ticker: (price, volume, source, date)}
    """
    if not tickers:
        return {}
    
    results: Dict[str, Tuple[float, int, str, str]] = {}
    
    # Try Tiger batch first
    qc = _get_tiger_qc()
    if qc:
        try:
            briefs = qc.get_stock_delay_briefs(tickers)
            if briefs is not None and not briefs.empty:
                for _, row in briefs.iterrows():
                    sym = str(row.get("symbol", ""))
                    close = float(row.get("close", 0))
                    vol = int(row.get("volume", 0))
                    if close > 0 and sym:
                        results[sym] = (close, vol, "tiger_delayed", datetime.now().strftime("%Y-%m-%d"))
        except Exception:
            pass
    
    # Fallback for tickers Tiger didn't cover
    for t in tickers:
        if t not in results:
            price, vol, src, dt = get_price(t)  # will use yfinance fallback
            if price > 0:
                results[t] = (price, vol, src, dt)
    
    return results


def get_52w_range(ticker: str) -> Tuple[float, float]:
    """
    Get 52-week high/low from yfinance history.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if hist is not None and len(hist) >= 20:
            return float(hist["Low"].min()), float(hist["High"].max())
    except Exception:
        pass
    return 0.0, 0.0


# ═══════════════════════════════════════════════════════════════════
# 2. FUNDAMENTALS — SEC (primary US) + yfinance (fallback + foreign)
# ═══════════════════════════════════════════════════════════════════

def get_fundamentals(ticker: str) -> dict:
    """
    Get fundamental data from best available source.
    - US stocks: SEC EDGAR (primary) → yfinance (supplement)
    - Foreign: yfinance (primary) → SEC (annual verify)
    
    Returns unified dict with all key metrics + source tracking.
    """
    result = {
        "ticker": ticker,
        "sources": [],
        "name": "",
        "sector": "",
        "industry": "",
        "price": 0.0,
        "market_cap": 0,
        "book_value": 0,
        "bm_ratio": 0,
        "roa": 0,
        "roe": 0,
        "ebitda_margin": 0,
        "fcf_yield": 0,
        "fcf_conversion": 0,
        "debt_to_equity": 0,
        "beta": 0,
        "short_ratio": 0,
        "short_pct_float": 0,
        "analyst_count": 0,
        "target_mean": 0,
        "trailing_pe": 0,
        "forward_pe": 0,
        "profit_margin": 0,
        "revenue_growth": 0,
        "earnings_growth": 0,
        "latest_eps": 0,
        "latest_revenue": 0,
        "eps_trend": [],
        "revenue_trend": [],
        "freshness": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "data_source": "",
    }
    
    # Determine source strategy
    foreign = is_foreign_issuer(ticker)
    
    if foreign == False:
        # US stock → SEC primary
        result["data_source"] = "sec"
        result["sources"].append("sec_edgar")
        _extract_sec_data(ticker, result)
        # Supplement with yfinance for sector/industry/beta
        _supplement_yfinance(ticker, result)
    elif foreign == True:
        # Foreign stock → yfinance primary
        result["data_source"] = "yfinance"
        result["sources"].append("yfinance")
        _extract_yfinance_full(ticker, result)
        # Supplement B/M from SEC annual data if yfinance missing it
        if result["bm_ratio"] <= 0:
            _supplement_bm_sec(ticker, result)
    else:
        # Unknown → try both
        result["data_source"] = "yfinance"
        result["sources"].append("yfinance")
        _extract_yfinance_full(ticker, result)
        # Supplement B/M from SEC data
        if result["bm_ratio"] <= 0:
            _supplement_bm_sec(ticker, result)
        # Try SEC as supplement
        try:
            sec = get_fundamentals_sec(ticker)
            if sec.get("annual", {}).get("revenue"):
                result["sources"].append("sec_edgar")
        except Exception:
            pass
    
    return result


def _extract_sec_data(ticker: str, result: dict):
    """Extract data from SEC EDGAR into unified result dict."""
    sec = get_fundamentals_sec(ticker)
    
    # Quarterly EPS trend
    eps_q = sec.get("quarterly", {}).get("eps_diluted", [])
    if eps_q:
        result["eps_trend"] = [{"fy": e["fy"], "fp": e["fp"], "val": e["val"]} for e in eps_q[:6]]
        result["latest_eps"] = eps_q[0]["val"] if eps_q else 0
    
    # Quarterly revenue trend
    rev_q = sec.get("quarterly", {}).get("revenue", []) or sec.get("quarterly", {}).get("revenue_alt", [])
    if rev_q:
        result["revenue_trend"] = [{"fy": r["fy"], "fp": r["fp"], "val": r["val"]} for r in rev_q[:6]]
        result["latest_revenue"] = rev_q[0]["val"] if rev_q else 0
    else:
        # Try annual
        rev_a = sec.get("annual", {}).get("revenue", [])
        if rev_a:
            result["latest_revenue"] = rev_a[0]["val"]
    
    # Annual total equity → book value per share
    equity = sec.get("annual", {}).get("total_equity", [])
    if equity:
        result["book_value"] = equity[0]["val"]
    
    # Net income
    ni = sec.get("annual", {}).get("net_income", [])
    if ni:
        result["net_income"] = ni[0]["val"]


def _extract_yfinance_full(ticker: str, result: dict):
    """Extract all data from yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        result["name"] = info.get("shortName", ticker)
        result["sector"] = info.get("sector", "Unknown")
        result["industry"] = info.get("industry", "Unknown")
        result["market_cap"] = info.get("marketCap", 0) or 0
        result["beta"] = info.get("beta", 0) or 0
        result["roa"] = info.get("returnOnAssets", 0) or 0
        result["roe"] = info.get("returnOnEquity", 0) or 0
        result["debt_to_equity"] = info.get("debtToEquity", 0) or 0
        result["short_ratio"] = info.get("shortRatio", 0) or 0
        result["short_pct_float"] = info.get("shortPercentOfFloat", 0) or 0
        result["analyst_count"] = info.get("numberOfAnalystOpinions", 0) or 0
        result["target_mean"] = info.get("targetMeanPrice", 0) or 0
        result["trailing_pe"] = info.get("trailingPE", 0) or 0
        result["forward_pe"] = info.get("forwardPE", 0) or 0
        result["profit_margin"] = info.get("profitMargins", 0) or 0
        result["revenue_growth"] = info.get("revenueGrowth", 0) or 0
        result["earnings_growth"] = info.get("earningsGrowth", 0) or 0
        
        # Book value → B/M (use local price from info dict, NOT result.get)
        info_price = float(info.get('regularMarketPrice') or info.get('currentPrice', 0) or 0)
        bv = info.get("bookValue", 0) or 0
        
        # Fallback: calculate from balance sheet if bookValue missing
        if bv <= 0:
            try:
                bs = t.balance_sheet
                if bs is not None and 'StockholdersEquity' in bs.index:
                    total_eq = float(bs.loc['StockholdersEquity'].iloc[0])
                    shares_out = info.get('sharesOutstanding', 0) or info.get('impliedSharesOutstanding', 0) or 0
                    if shares_out > 0:
                        bv = total_eq / shares_out
            except Exception:
                pass
        
        # Also try SEC annual data (for foreign filers with SEC data)
        if bv <= 0:
            try:
                from data.sec_edgar import get_sec_annual
                eq = get_sec_annual(ticker, 'StockholdersEquity')
                if eq:
                    shares = info.get('sharesOutstanding', 0) or info.get('impliedSharesOutstanding', 0) or 0
                    if shares > 0:
                        bv = eq[0]['val'] / shares
            except Exception:
                pass
        
        result["book_value"] = bv
        local_price = info_price if info_price > 0 else result.get("price", 0)
        if local_price > 0 and bv > 0:
            result["bm_ratio"] = bv / local_price
        
        # FCF metrics
        fcf = info.get("freeCashflow", 0) or 0
        mcap = result["market_cap"]
        result["fcf_yield"] = fcf / mcap if mcap > 0 and fcf else 0
        ni = info.get("netIncomeToCommon", 0) or 0
        result["fcf_conversion"] = fcf / ni if ni and abs(ni) > 1 else 0
        
        # EBITDA margin
        ebitda = info.get("ebitda", 0) or 0
        rev = info.get("totalRevenue", 0) or 0
        result["ebitda_margin"] = ebitda / rev if rev > 0 else 0
        
        # Quarterly EPS/revenue trend
        try:
            qf = t.quarterly_financials
            if qf is not None:
                # EPS
                for label in ["Diluted EPS", "Basic EPS"]:
                    if label in qf.index:
                        eps_series = qf.loc[label]
                        result["eps_trend"] = [
                            {"fy": eps_series.index[i].year, 
                             "fp": f"Q{(eps_series.index[i].month-1)//3+1}",
                             "val": float(eps_series.iloc[i])}
                            for i in range(min(6, len(eps_series)))
                        ]
                        result["latest_eps"] = float(eps_series.iloc[0]) if len(eps_series) > 0 else 0
                        break
                # Revenue
                for label in ["Total Revenue", "Revenues"]:
                    if label in qf.index:
                        rev_series = qf.loc[label]
                        result["revenue_trend"] = [
                            {"fy": rev_series.index[i].year,
                             "fp": f"Q{(rev_series.index[i].month-1)//3+1}",
                             "val": float(rev_series.iloc[i])}
                            for i in range(min(6, len(rev_series)))
                        ]
                        result["latest_revenue"] = float(rev_series.iloc[0]) if len(rev_series) > 0 else 0
                        break
        except Exception:
            pass
    
    except Exception as e:
        logger.debug(f"yfinance extract failed for {ticker}: {e}")


def _supplement_bm_sec(ticker: str, result: dict):
    """Supplement B/M from SEC annual data when yfinance missing it."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info
        shares = info.get('sharesOutstanding', 0) or info.get('impliedSharesOutstanding', 0) or 0
        
        sec = get_fundamentals_sec(ticker)
        equity = sec.get("annual", {}).get("total_equity", [])
        
        if equity and shares > 0:
            bv_per_share = equity[0]["val"] / shares
            price = result.get("price", 0)
            if price > 0 and bv_per_share > 0:
                result["book_value"] = bv_per_share
                result["bm_ratio"] = bv_per_share / price
                result["sources"].append("sec_edgar_bm")
    except Exception:
        pass


def _supplement_yfinance(ticker: str, result: dict):
    """Supplement SEC data with yfinance info dict fields."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        result["name"] = info.get("shortName", result.get("name", ticker))
        result["sector"] = info.get("sector", result.get("sector", "Unknown"))
        result["industry"] = info.get("industry", result.get("industry", "Unknown"))
        result["market_cap"] = info.get("marketCap", 0) or result.get("market_cap", 0)
        result["beta"] = info.get("beta", 0) or 0
        result["short_ratio"] = info.get("shortRatio", 0) or 0
        result["short_pct_float"] = info.get("shortPercentOfFloat", 0) or 0
        result["analyst_count"] = info.get("numberOfAnalystOpinions", 0) or 0
        result["target_mean"] = info.get("targetMeanPrice", 0) or 0
        result["trailing_pe"] = info.get("trailingPE", 0) or 0
        result["forward_pe"] = info.get("forwardPE", 0) or 0
        result["profit_margin"] = info.get("profitMargins", 0) or 0
        result["revenue_growth"] = info.get("revenueGrowth", 0) or 0
        result["earnings_growth"] = info.get("earningsGrowth", 0) or 0
        result["debt_to_equity"] = info.get("debtToEquity", 0) or 0
        result["roa"] = info.get("returnOnAssets", 0) or 0
        result["roe"] = info.get("returnOnEquity", 0) or 0
        
        # Book value
        bv = info.get("bookValue", 0) or 0
        result["book_value"] = bv or result.get("book_value", 0)
        
        # FCF
        fcf = info.get("freeCashflow", 0) or 0
        mcap = result.get("market_cap", 0)
        result["fcf_yield"] = fcf / mcap if mcap > 0 and fcf else 0
        ni = info.get("netIncomeToCommon", 0) or 0
        result["fcf_conversion"] = fcf / ni if ni and abs(ni) > 1 else 0
        
        # EBITDA margin
        ebitda = info.get("ebitda", 0) or 0
        rev = info.get("totalRevenue", 0) or 0
        result["ebitda_margin"] = ebitda / rev if rev > 0 else 0
        
        # B/M (use local info price, not result.get which may be 0)
        local_price = float(info.get("regularMarketPrice") or info.get("currentPrice", 0) or 0)
        if local_price > 0:
            result["price"] = local_price
            if bv > 0:
                result["bm_ratio"] = bv / local_price
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# 3. CROSS-VERIFY — Compare SEC vs yfinance for discrepancy
# ═══════════════════════════════════════════════════════════════════

def verify_cross_source(ticker: str) -> dict:
    """
    Compare SEC vs yfinance data to detect discrepancies.
    Returns dict with comparison results.
    """
    result = {
        "ticker": ticker,
        "verdict": "consistent",
        "discrepancies": [],
        "sec": {},
        "yfinance": {},
    }
    
    # Get from both sources
    if is_foreign_issuer(ticker) == False:
        sec = get_fundamentals_sec(ticker)
        result["sec"] = {
            "revenue": sec.get("annual", {}).get("revenue", [{}])[0].get("val", 0) if sec.get("annual", {}).get("revenue") else 0,
            "net_income": sec.get("annual", {}).get("net_income", [{}])[0].get("val", 0) if sec.get("annual", {}).get("net_income") else 0,
        }
    
    # yfinance
    try:
        t = yf.Ticker(ticker)
        info = t.info
        mcap = info.get("marketCap", 1)
        result["yfinance"] = {
            "revenue": info.get("totalRevenue", 0) or 0,
            "net_income": info.get("netIncomeToCommon", 0) or 0,
        }
    except Exception:
        pass
    
    # Compare
    if result["sec"].get("revenue") and result["yfinance"].get("revenue"):
        sec_rev = result["sec"]["revenue"]
        yf_rev = result["yfinance"]["revenue"]
        if yf_rev > 0:
            diff = abs(sec_rev - yf_rev) / yf_rev
            if diff > 0.10:
                result["discrepancies"].append(
                    f"Revenue: SEC=${sec_rev/1e6:.1f}M vs yf=${yf_rev/1e6:.1f}M ({diff:.0%} diff)"
                )
                result["verdict"] = "discrepancy"
    
    return result


# ═══════════════════════════════════════════════════════════════════
# 4. CONVENIENCE — One-call snapshot
# ═══════════════════════════════════════════════════════════════════

def get_snapshot(ticker: str) -> dict:
    """
    Complete one-call snapshot: price + fundamentals + cross-verify.
    Returns a single dict with everything VMAA needs.
    """
    # Price
    price, vol, price_src, price_date = get_price(ticker)
    
    # 52w range
    low_52w, high_52w = get_52w_range(ticker)
    
    # Fundamentals
    fund = get_fundamentals(ticker)
    fund["price"] = price
    fund["price_source"] = price_src
    fund["price_date"] = price_date
    fund["52w_low"] = low_52w
    fund["52w_high"] = high_52w
    fund["volume"] = vol
    
    # PTL ratio
    fund["ptl_ratio"] = price / low_52w if low_52w > 0 else 999
    
    return fund


# ═══════════════════════════════════════════════════════════════════
# 5. YFINANCE FALLBACK — Bars + Fundamentals from Tiger/SEC
# ═══════════════════════════════════════════════════════════════════

def get_bars_hybrid(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Get OHLCV bars from Tiger (primary) -> yfinance (fallback).
    Tiger get_bars() is limited to 20 symbols/day for the current account
    tier.  Once the limit is exhausted we skip Tiger for the rest of the
    run and go straight to yfinance.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    (renamed to yfinance-compatible format). Or None if both sources fail.
    """
    global _TIGER_BARS_EXHAUSTED

    # Check cache first (same-day reuse)
    if _CACHE_AVAILABLE:
        cached = cache_get(ticker, 'bars')
        if cached is not None:
            try:
                df = pd.DataFrame(cached)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], format='mixed')
                    df = df.set_index('time')
                    # Rename columns if needed
                    rename = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
                    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
                logger.debug(f"  {ticker}: bars loaded from cache ({len(df)} rows)")
                return df
            except Exception:
                pass

    # Try Tiger get_bars first (skip if 20-symbol/day limit already hit)
    qc = _get_tiger_qc()
    if qc and not _TIGER_BARS_EXHAUSTED:
        try:
            # Map period strings to Tiger parameters
            period_map = {
                "1y": ("day", 252),
                "6mo": ("day", 126),
                "3mo": ("day", 63),
                "1mo": ("day", 22),
                "1wk": ("week", 52),
            }
            bar_period, bar_limit = period_map.get(period, ("day", 252))
            bars = qc.get_bars(ticker, period=bar_period, limit=bar_limit)
            if bars is not None and not bars.empty:
                # Convert time from epoch ms to datetime index
                df = bars[['time','open','high','low','close','volume']].copy()
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df = df.set_index('time')
                # Rename to yfinance-compatible column names
                df = df.rename(columns={
                    'open': 'Open', 'high': 'High',
                    'low': 'Low', 'close': 'Close', 'volume': 'Volume'
                })
                logger.debug(f"  {ticker}: Tiger bars OK ({len(df)} rows)")
                if _CACHE_AVAILABLE:
                    try:
                        cache_data = df.reset_index().to_dict('records')
                        cache_set(ticker, 'bars', cache_data)
                    except Exception:
                        pass
                return df
        except Exception as e:
            msg = str(e)
            if 'permission denied' in msg.lower() or '20 symbols' in msg.lower():
                _TIGER_BARS_EXHAUSTED = True
                logger.debug(f"  {ticker}: Tiger bar limit reached ({e}), falling back to yfinance for remaining stocks")
            else:
                logger.debug(f"  {ticker}: Tiger bars failed ({e}), trying yfinance")

    # Fallback: yfinance history
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist is not None and len(hist) >= 5:
            logger.debug(f"  {ticker}: yfinance bars OK ({len(hist)} rows)")
            result_df = hist[['Open','High','Low','Close','Volume']]
            if _CACHE_AVAILABLE:
                try:
                    cache_data = result_df.reset_index().to_dict('records')
                    cache_set(ticker, 'bars', cache_data)
                except Exception:
                    pass
            return result_df
    except Exception as e:
        logger.debug(f"  {ticker}: yfinance bars failed ({e})")

    # Final fallback: try Finnhub for a minimal price bar (when yfinance 401s)
    try:
        import requests as _req
        resp = _req.get(
            f'https://finnhub.io/api/v1/quote?symbol={ticker}&token={_FINNHUB_KEY}',
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            close = float(data.get('c', 0))
            high = float(data.get('h', 0))
            low = float(data.get('l', 0))
            open_ = float(data.get('o', 0))
            vol = data.get('v', 0) or int(data.get('volume', 0))
            if close > 0:
                # Build minimal single-row DataFrame
                df = pd.DataFrame({
                    'Open': [open_ or close], 'High': [high or close],
                    'Low': [low or close], 'Close': [close], 'Volume': [vol]
                }, index=pd.DatetimeIndex([datetime.now()]))
                logger.debug(f"  {ticker}: Finnhub quote fallback")
                return df
    except Exception:
        pass

    return None


def get_fundamentals_tiger(ticker: str) -> dict:
    """
    Get fundamental data from SEC EDGAR when yfinance is rate-limited.
    Uses SEC EDGAR as primary source for fundamentals.

    Returns dict with keys matching a subset of yfinance info fields:
    - totalRevenue, netIncomeToCommon, freeCashflow
    - bookValue, totalAssets, currentAssets, currentLiabilities
    - returnOnAssets, returnOnEquity, debtToEquity
    - ebitda (estimated from operating_income + depreciation)
    Returns empty dict if nothing works.
    """
    result = {}
    try:
        from data.sec_edgar import get_fundamentals_sec as sec_fund
        sec = sec_fund(ticker)
        if not sec:
            return result

        # Annual data
        annual = sec.get('annual', {})
        quarterly = sec.get('quarterly', {})

        rev = annual.get('revenue', []) or annual.get('revenue_alt', [])
        ni = annual.get('net_income', [])
        eps = annual.get('eps_diluted', []) or annual.get('eps_basic', [])
        ta = annual.get('total_assets', [])
        ca = annual.get('current_assets', [])
        cl = annual.get('current_liabilities', [])
        te = annual.get('total_equity', [])
        oi = annual.get('operating_income', [])
        dep = annual.get('depreciation', [])
        debt_lt = annual.get('debt_longterm', [])
        gp = annual.get('gross_profit', [])

        # Revenue
        if rev and rev[0]:
            result['totalRevenue'] = rev[0].get('val', 0)
        if ni and ni[0]:
            result['netIncomeToCommon'] = ni[0].get('val', 0)
        if eps and eps[0]:
            result['trailingEps'] = eps[0].get('val', 0)
        if ta and ta[0]:
            result['totalAssets'] = ta[0].get('val', 0)
        if ca and ca[0]:
            result['currentAssets'] = ca[0].get('val', 0)
        if cl and cl[0]:
            result['currentLiabilities'] = cl[0].get('val', 0)
        if te and te[0]:
            result['totalEquity'] = te[0].get('val', 0)

        # ROA = net_income / total_assets
        ni_val = result.get('netIncomeToCommon', 0)
        ta_val = result.get('totalAssets', 1)
        if ta_val > 0:
            result['returnOnAssets'] = ni_val / ta_val

        # ROE = net_income / total_equity
        te_val = result.get('totalEquity', 0)
        if te_val > 0:
            result['returnOnEquity'] = ni_val / te_val

        # Debt to Equity
        if te_val > 0:
            total_debt = (cl[0].get('val', 0) if cl and cl[0] else 0) + \
                         (debt_lt[0].get('val', 0) if debt_lt and debt_lt[0] else 0)
            if total_debt > 0:
                result['debtToEquity'] = total_debt / te_val
            else:
                result['debtToEquity'] = 0.0

        # Book value per share (we'll calculate in _build_info_from_sec with shares)
        if te_val > 0:
            result['bookValue_raw'] = te_val

        # EBITDA estimate = operating_income + depreciation
        oi_val = oi[0].get('val', 0) if oi and oi[0] else 0
        dep_val = dep[0].get('val', 0) if dep and dep[0] else 0
        result['ebitda'] = oi_val + dep_val

        # Free cash flow estimate = operating_income + depreciation (EBITDA proxy) - capex
        # We don't have CAPEX from SEC in our current concepts, use EBITDA as proxy
        result['freeCashflow'] = result['ebitda']

        # Try to get EPS from quarterly (more recent TTM)
        q_eps = quarterly.get('eps_diluted', []) or quarterly.get('eps_basic', [])
        if q_eps and len(q_eps) >= 4:
            result['eps_ttm'] = sum(e.get('val', 0) for e in q_eps[:4])

        # Revenue TTM from quarterly
        q_rev = quarterly.get('revenue', []) or quarterly.get('revenue_alt', [])
        if q_rev and len(q_rev) >= 4:
            result['revenue_ttm'] = sum(r.get('val', 0) for r in q_rev[:4])

        # Gross profit margin
        gp_val = gp[0].get('val', 0) if gp and gp[0] else 0
        if rev and rev[0]:
            rev_val = rev[0].get('val', 0)
            if rev_val > 0:
                result['grossMargins'] = gp_val / rev_val

        result['data_source'] = 'sec_edgar'
        logger.debug(f"  {ticker}: SEC fundamentals OK")

    except Exception as e:
        logger.debug(f"  SEC EDGAR fundamentals failed for {ticker}: {e}")

    return result


def yfinance_available() -> bool:
    """Quick check if yfinance is currently working (not rate-limited).
    Uses SPY as a proxy — the most liquid and reliable ticker."""
    try:
        t = yf.Ticker('SPY')
        info = t.info
        price = info.get('regularMarketPrice', 0)
        if price and float(price) > 0:
            return True
        # Try history as backup check
        hist = t.history(period="5d")
        return hist is not None and len(hist) >= 2
    except Exception as e:
        logger.debug(f"yfinance availability check failed: {e}")
        return False


def _is_yf_rate_limited(error: Exception) -> bool:
    """Detect if an exception is caused by yfinance rate-limiting (HTTP 401)."""
    msg = str(error)
    return '401' in msg or 'Invalid Crumb' in msg or 'Unauthorized' in msg


# ═══════════════════════════════════════════════════════════════════
# 6. FINNHUB — Fundamentals + Earnings + Recommendations
# ═══════════════════════════════════════════════════════════════════

_FINNHUB_KEY = "d2ebgbhr01qr1ro95mrgd2ebgbhr01qr1ro95ms0"


def get_finnhub_fundamentals(ticker: str) -> dict:
    """
    Get fundamental metrics from Finnhub (all 133 metrics).
    Fast API, no rate limit issues for small batches.
    Returns dict with: marketCap, sector, industry, beta, roa, roe, 
    bookValuePerShare, epsTTM, revenuePerShareTTM, fcfPerShare, 
    debtToEquity, 52wHigh, 52wLow, ebitdaPerShare, dividendYield, etc.
    """
    # Check cache first
    if _CACHE_AVAILABLE:
        cached = cache_get(ticker, 'fundamentals')
        if cached is not None:
            return cached
    import requests as _requests
    import time as _time
    
    # Rate limit: ensure max 50 calls per 60 seconds (under free tier 60/min)
    global _FINNHUB_CALL_TIMES
    now = _time.time()
    _FINNHUB_CALL_TIMES = [t for t in _FINNHUB_CALL_TIMES if t > now - 60]
    if len(_FINNHUB_CALL_TIMES) >= 55:
        sleep_time = 60 - (now - _FINNHUB_CALL_TIMES[0])
        sleep_time += __import__('random').uniform(0, 5)  # 0-5s stagger
        if sleep_time > 0:
            _time.sleep(sleep_time)
    _FINNHUB_CALL_TIMES.append(_time.time())
    
    result = {}
    try:
        resp = _requests.get(
            f'https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={_FINNHUB_KEY}',
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            metrics = data.get('metric', {})
            if metrics:
                # Map Finnhub field names to yfinance-compatible names
                # Actual Finnhub metric field names (verified 2026-05-08)
                # All entries as (field_name, is_percentage) tuples
                mapping = {
                    'marketCapitalization': ('marketCap', False),
                    'bookValuePerShareAnnual': ('bookValue', False),
                    'epsTTM': ('trailingEps', False),
                    'revenuePerShareTTM': ('revenuePerShare', False),
                    'beta': ('beta', False),
                    'roaTTM': ('returnOnAssets', True),       # percentage, auto-/100
                    'roeTTM': ('returnOnEquity', True),        # percentage, auto-/100
                    'ebitdPerShareTTM': ('ebitda', False),
                    'totalDebt/totalEquityAnnual': ('debtToEquity', False),
                    '52WeekHigh': ('fiftyTwoWeekHigh', False),
                    '52WeekLow': ('fiftyTwoWeekLow', False),
                    'dividendYieldIndicatedAnnual': ('dividendYield', True),
                    'epsBasicExclExtraItemsTTM': ('trailingEps', False),
                    'cashFlowPerShareTTM': ('cashFlowPerShare', False),
                    'revenueGrowthTTMYoy': ('revenueGrowth', True),
                    'epsGrowthTTMYoy': ('earningsGrowth', True),
                    'operatingMarginTTM': ('operatingMargins', True),
                    'grossMarginTTM': ('grossMargins', True),
                    'netProfitMarginTTM': ('profitMargins', True),
                    'currentRatioQuarterly': ('currentRatio', False),
                }
                for fin_key, (yf_key, is_pct) in mapping.items():
                    if fin_key in metrics and metrics[fin_key] is not None:
                        val = metrics[fin_key]
                        # Don't overwrite trailingEps from epsTTM with the backup source
                        if yf_key == 'trailingEps' and yf_key in result:
                            continue
                        if is_pct:
                            result[yf_key] = val / 100.0
                        else:
                            result[yf_key] = val
                
                # Also fetch company profile for sector, name, shares outstanding
                try:
                    prof_resp = _requests.get(
                        f'https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={_FINNHUB_KEY}',
                        timeout=10
                    )
                    if prof_resp.status_code == 200:
                        profile = prof_resp.json()
                        if profile.get('finnhubIndustry'):
                            result['sector'] = profile['finnhubIndustry']
                        if profile.get('name'):
                            result['longName'] = profile['name']
                        if profile.get('shareOutstanding'):
                            result['sharesOutstanding'] = float(profile['shareOutstanding']) * 1e6  # Finnhub returns in millions
                except Exception:
                    pass
                
                # Convert market cap from millions to actual dollars
                if 'marketCap' in result:
                    result['marketCap'] = int(result['marketCap'] * 1e6)
                
                # If we have shares outstanding, calculate total free cash flow
                # Note: bookValue stays as PER-SHARE (yfinance convention for B/M calc)
                shares = result.get('sharesOutstanding', 0)
                if shares > 0:
                    # Total free cash flow = cashFlowPerShare * shares
                    fcf_ps = result.get('cashFlowPerShare', 0)
                    if fcf_ps > 0:
                        result['freeCashflow'] = fcf_ps * shares
                
                # Calculate FCF yield = freeCashflow / marketCap
                fcf = result.get('freeCashflow', 0)
                mcap = result.get('marketCap', 0)
                if fcf > 0 and mcap > 0:
                    result['fcfYield'] = fcf / mcap
                
                result['_source'] = 'finnhub'
    except Exception as e:
        pass
    
    # Store in cache
    if _CACHE_AVAILABLE and result and len(result) > 3:
        cache_set(ticker, 'fundamentals', result)
    
    return result


def get_finnhub_earnings(ticker: str) -> list:
    """
    Get quarterly earnings from Finnhub.
    Returns list of {period, actual, estimate, surprise, quarter, year}
    """
    # Check cache first
    if _CACHE_AVAILABLE:
        cached = cache_get(ticker, 'earnings')
        if cached is not None:
            return cached
    import requests as _requests
    try:
        resp = _requests.get(
            f'https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={_FINNHUB_KEY}',
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if _CACHE_AVAILABLE and data:
                cache_set(ticker, 'earnings', data)
            return data
    except Exception:
        pass
    return []


def get_finnhub_recommendation(ticker: str) -> list:
    """
    Get analyst recommendations from Finnhub.
    Returns list of {period, buy, hold, sell, strongBuy, strongSell}
    """
    # Check cache first
    if _CACHE_AVAILABLE:
        cached = cache_get(ticker, 'analyst')
        if cached is not None:
            return cached
    import requests as _requests
    try:
        resp = _requests.get(
            f'https://finnhub.io/api/v1/stock/recommendation?symbol={ticker}&token={_FINNHUB_KEY}',
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if _CACHE_AVAILABLE and data:
                cache_set(ticker, 'analyst', data)
            return data
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    for ticker in ["AAPL", "TMDX", "INMD", "MNDY"]:
        print(f"\n{'='*60}")
        print(f"  {ticker}")
        print(f"{'='*60}")
        
        snap = get_snapshot(ticker)
        print(f"  Price: \${snap['price']:.2f} ({snap['price_source']})")
        print(f"  Source: {snap['data_source']} + {snap['sources']}")
        print(f"  Sector: {snap['sector']}")
        print(f"  Market Cap: \${snap['market_cap']/1e9:.2f}B")
        print(f"  B/M: {snap['bm_ratio']:.2f}")
        print(f"  FCF Yield: {snap['fcf_yield']:.1%}")
        print(f"  Latest EPS: \${snap['latest_eps']:.2f}")
        print(f"  Latest Rev: \${snap['latest_revenue']/1e6:.1f}M")
        print(f"  EPS Trend:")
        for e in snap.get("eps_trend", [])[:4]:
            print(f"    FY{e['fy']} {e['fp']}: \${e['val']:.2f}")
        print(f"  52w: \${snap['52w_low']:.2f} - \${snap['52w_high']:.2f}")
        print(f"  PTL: {snap['ptl_ratio']:.2f}x")
