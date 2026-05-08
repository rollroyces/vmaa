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
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf
import numpy as np

from data.sec_edgar import (
    get_fundamentals_sec,
    is_foreign_issuer,
    has_sec_data,
    clear_cache as sec_clear_cache,
)

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

def _get_tiger_qc():
    global _TIGER_QC
    if _TIGER_QC is None:
        try:
            if TigerBroker is None:
                return None
            broker = TigerBroker()
            _TIGER_QC = broker.quote_client
        except Exception as e:
            logger.debug(f"Tiger init failed: {e}")
    return _TIGER_QC


def get_price(ticker: str) -> Tuple[float, int, str, str]:
    """
    Get current price from Tiger (primary) → yfinance (fallback).
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
