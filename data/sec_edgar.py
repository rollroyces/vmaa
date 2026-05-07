#!/usr/bin/env python3
"""
SEC EDGAR API Client — Free, no API key required
==================================================
Access official SEC filings data (10-Q, 10-K, 20-F) via the SEC's
public XBRL/JSON API: https://data.sec.gov/api/xbrl/

Requirements: User-Agent header with your email (SEC policy)
Rate limits: ~10 req/sec should be safe; we cache aggressively.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("vmaa.data.sec")

# ── SEC API Base ──
_SEC_BASE = "https://data.sec.gov/api/xbrl"
_contact = os.environ.get("SEC_USER_AGENT_CONTACT", "research@vmaa.local")
_SEC_HEADERS = {
    "User-Agent": f"VMAA Research ({_contact})",
    "Accept-Encoding": "gzip, deflate",
}

# ── Cache ──
_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL = 86400 * 7  # 7 days (SEC data doesn't change often)


# ═══════════════════════════════════════════════════════════════════
# Ticker → CIK lookup
# ═══════════════════════════════════════════════════════════════════

_TICKER_CIK_CACHE: Dict[str, str] = {}

def get_cik(ticker: str) -> Optional[str]:
    """
    Look up CIK (Central Index Key) for a ticker from SEC.
    Returns 10-digit CIK string (e.g. '0000320193' for AAPL).
    """
    ticker = ticker.upper().strip()
    if ticker in _TICKER_CIK_CACHE:
        return _TICKER_CIK_CACHE[ticker]
    
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_SEC_HEADERS,
            timeout=15
        )
        if resp.status_code == 200:
            companies = resp.json()
            for key, val in companies.items():
                if val.get("ticker") == ticker:
                    cik = str(val["cik_str"]).zfill(10)
                    _TICKER_CIK_CACHE[ticker] = cik
                    return cik
    except Exception as e:
        logger.debug(f"Ticker lookup failed for {ticker}: {e}")
    
    return None


# ═══════════════════════════════════════════════════════════════════
# Company Facts — Full XBRL data
# ═══════════════════════════════════════════════════════════════════

def get_company_facts(ticker: str) -> Optional[dict]:
    """
    Fetch all company facts from SEC EDGAR.
    Returns dict with us-gaap, dei, and other namespaces.
    """
    cik = get_cik(ticker)
    if not cik:
        return None
    
    # Check cache
    cache_file = _CACHE_DIR / f"sec_{cik}.json"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < _CACHE_TTL:
            with open(cache_file) as f:
                return json.load(f)
    
    try:
        url = f"{_SEC_BASE}/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=20)
        
        if resp.status_code == 200:
            data = resp.json()
            # Cache it
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return data
        else:
            logger.warning(f"SEC API returned {resp.status_code} for {ticker}")
            return None
    except Exception as e:
        logger.error(f"SEC fetch failed for {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Filing Type Detection: 10-Q (quarterly) vs 20-F (annual only)
# ═══════════════════════════════════════════════════════════════════

def is_foreign_issuer(ticker: str) -> Optional[bool]:
    """
    Detect if a company files 20-F (foreign) instead of 10-Q.
    Returns True if foreign (annual-only), False if US (quarterly), None if unknown.
    """
    facts = get_company_facts(ticker)
    if not facts:
        return None
    
    # Check the form types used in filings
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept, data in us_gaap.items():
        units = data.get("units", {})
        for currency, entries in units.items():
            for entry in entries[:5]:
                form = entry.get("form", "")
                if form == "10-Q":
                    return False  # Definitely US quarterly filer
                elif form == "20-F":
                    return True   # Foreign issuer
            break
        break
    
    return None  # Can't determine


# ═══════════════════════════════════════════════════════════════════
# Get Quarterly Data from SEC (10-Q filers only)
# ═══════════════════════════════════════════════════════════════════

def get_sec_quarterly(ticker: str, concept: str) -> List[Dict]:
    """
    Get quarterly data for a US-GAAP concept from SEC.
    Works for US-domiciled companies (10-Q/10-K filers).
    
    De-duplicates by (fy, fp), keeping the most recent filing.
    Filters out amended filings (frame=CYxxxxQx) and 8-K entries.
    
    Returns list of {fy, fp, val} sorted most-recent-first.
    """
    facts = get_company_facts(ticker)
    if not facts:
        return []
    
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if concept not in us_gaap:
        logger.debug(f"Concept {concept} not found for {ticker}")
        return []
    
    units = us_gaap[concept].get("units", {})
    
    # Collect all quarterly entries
    raw_entries = []
    for currency, entries in units.items():
        for e in entries:
            fp = e.get("fp", "")
            fy = e.get("fy")
            val = e.get("val")
            form = e.get("form", "")
            frame = e.get("frame", "")
            
            # Only Q1-Q4 quarterly data
            if fp not in ("Q1", "Q2", "Q3", "Q4"):
                continue
            if not fy or val is None:
                continue
            
            raw_entries.append({
                "fy": fy,
                "fp": fp,
                "val": val,
                "form": form,
                "frame": frame,
                "priority": 0 if frame in ("", "none", "None") else 1,
            })
        break  # First currency only
    
    # De-duplicate: for each (fy, fp), keep the entry with lowest priority (= primary filing)
    seen: Dict[str, dict] = {}
    for entry in raw_entries:
        key = f"{entry['fy']}_{entry['fp']}"
        if key not in seen or entry["priority"] < seen[key]["priority"]:
            seen[key] = entry
    
    results = list(seen.values())
    results.sort(key=lambda r: (r["fy"], r["fp"]), reverse=True)
    return results


def get_sec_annual(ticker: str, concept: str) -> List[Dict]:
    """
    Get annual data for a US-GAAP concept from SEC.
    Works for both US (10-K) and foreign (20-F) companies.
    """
    facts = get_company_facts(ticker)
    if not facts:
        return []
    
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if concept not in us_gaap:
        return []
    
    units = us_gaap[concept].get("units", {})
    results = []
    for currency, entries in units.items():
        for e in entries:
            fp = e.get("fp")
            fy = e.get("fy")
            val = e.get("val")
            form = e.get("form", "")
            frame = e.get("frame", "")
            
            # Annual data: fp=FY, form=10-K or 20-F
            if fp == "FY" and form in ("10-K", "20-F", ""):
                seen = False
                for r in results:
                    if r["fy"] == fy:
                        seen = True
                        break
                if not seen:
                    results.append({"fy": fy, "val": val, "form": form, "frame": frame})
        break
    
    results.sort(key=lambda r: r["fy"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# Batch fetch: get all key fundamentals at once
# ═══════════════════════════════════════════════════════════════════

KEY_CONCEPTS = {
    "revenue": "Revenues",
    "revenue_alt": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "net_income": "NetIncomeLoss",
    "eps_diluted": "EarningsPerShareDiluted",
    "eps_basic": "EarningsPerShareBasic",
    "operating_income": "OperatingIncomeLoss",
    "gross_profit": "GrossProfit",
    "cost_revenue": "CostOfRevenue",
    "total_assets": "Assets",
    "current_assets": "AssetsCurrent",
    "current_liabilities": "LiabilitiesCurrent",
    "total_equity": "StockholdersEquity",
    "cash_and_equivalents": "CashAndCashEquivalentsAtCarryingValue",
    "depreciation": "DepreciationDepletionAndAmortization",
    "tax_expense": "IncomeTaxExpenseBenefit",
    "operating_expenses": "OperatingExpenses",
    "selling_general_admin": "SellingGeneralAndAdministrativeExpense",
    "research_development": "ResearchAndDevelopmentExpense",
    "interest_expense": "InterestExpense",
    "debt_current": "DebtCurrent",
    "debt_longterm": "LongTermDebt",
    "inventory": "InventoryNet",
    "receivables": "AccountsReceivableNetCurrent",
}


def get_fundamentals_sec(ticker: str) -> dict:
    """
    Fetch all key fundamentals from SEC EDGAR.
    Returns dict with quarterly + annual data per concept.
    """
    result = {
        "ticker": ticker,
        "source": "sec_edgar",
        "is_foreign": is_foreign_issuer(ticker),
        "quarterly": {},
        "annual": {},
        "freshness": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    
    for name, concept in KEY_CONCEPTS.items():
        try:
            q = get_sec_quarterly(ticker, concept)
            if q:
                result["quarterly"][name] = q
            a = get_sec_annual(ticker, concept)
            if a:
                result["annual"][name] = a
        except Exception as e:
            logger.debug(f"  {concept} for {ticker}: {e}")
    
    return result


# ═══════════════════════════════════════════════════════════════════
# Quick check if SEC has data for a ticker
# ═══════════════════════════════════════════════════════════════════

def has_sec_data(ticker: str) -> bool:
    """Quick check if SEC has fundamentals data for this ticker."""
    facts = get_company_facts(ticker)
    if not facts:
        return False
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    # At minimum should have Revenue or NetIncome
    return "Revenues" in us_gaap or "NetIncomeLoss" in us_gaap


# ═══════════════════════════════════════════════════════════════════
# Clear cache
# ═══════════════════════════════════════════════════════════════════

def clear_cache():
    """Clear all cached SEC data."""
    for f in _CACHE_DIR.glob("sec_*.json"):
        f.unlink()
    logger.info("SEC cache cleared")


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    for ticker in ["AAPL", "TMDX", "INMD", "MSFT", "MNDY"]:
        print(f"\n=== {ticker} ===")
        
        # Is it foreign?
        foreign = is_foreign_issuer(ticker)
        print(f"  Foreign issuer: {foreign}")
        
        # Annual data
        rev = get_sec_annual(ticker, "Revenues")
        if rev:
            for r in rev[:3]:
                print(f"  Annual Revenue FY{r['fy']}: ${r['val']/1e6:.2f}M ({r.get('form','?')})")
        
        # Quarterly (if US stock)
        if foreign == False:
            eps = get_sec_quarterly(ticker, "EarningsPerShareDiluted")
            if eps:
                print(f"  Quarterly EPS:")
                for e in eps[:4]:
                    print(f"    FY{e['fy']} {e['fp']}: ${e['val']:.2f}")
            rev_q = get_sec_quarterly(ticker, "Revenues")
            if rev_q:
                print(f"  Quarterly Revenue:")
                for r in rev_q[:4]:
                    print(f"    FY{r['fy']} {r['fp']}: ${r['val']/1e6:.2f}M")
