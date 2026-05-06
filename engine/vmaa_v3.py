#!/usr/bin/env python3
"""
VMAA v3 — Unified Pipeline Orchestrator
=========================================
Complete multi-engine pipeline: data → quality → momentum → technical →
chip → earnings → composite → risk → decision → monitor.

Pipeline Flow:
```
                ┌──────────────────┐
                │   Data Layer     │
                │ (Tiger+SEC+yf)   │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  Part 1 Quality  │
                │  (Value + FCF)   │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │ Part 2 MAGNA     │
                │ (Momentum+ Gap)  │
                └────────┬─────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│ Technical  │  │   Chip     │  │  Earnings  │
│ Indicators │  │  Analysis  │  │  Consensus │
└──────┬─────┘  └──────┬─────┘  └──────┬─────┘
       └───────────────┼───────────────┘
                       ▼
                ┌────────────┐
                │  Composite │
                │   Score    │
                └──────┬─────┘
                       ▼
                ┌────────────┐
                │   Risk     │
                │  Assess    │
                └──────┬─────┘
                       ▼
                ┌────────────┐
                │  Decision  │
                │  + Execute │
                └────────────┘
```

Design Rules:
  1. All sub-engine imports use try/except ImportError
  2. Output is always JSON-serializable
  3. Backward compatible with existing pipeline.py
  4. Reuses existing config thresholds (P1C, P2C, RC)
  5. Reuses existing data/hybrid.py for data
  6. Quick mode targets < 10 min
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure VMAA root is importable
_vmaa_root = Path(__file__).resolve().parent.parent
if str(_vmaa_root) not in sys.path:
    sys.path.insert(0, str(_vmaa_root))

logger = logging.getLogger("vmaa.engine.vmaa_v3")


# ═══════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(
    engine: Any,  # VMAAEngine instance (avoid circular import)
    tickers: Optional[List[str]] = None,
    universe: str = "sp500",
    max_tickers: int = 50,
    mode: str = "full",
    progress: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete VMAA pipeline.

    Args:
        engine: VMAAEngine instance
        tickers: Specific tickers (overrides universe)
        universe: Universe source name
        max_tickers: Max tickers to process
        mode: "quick" | "full" | "backtest"
        progress: Show progress output

    Returns:
        JSON-serializable dict with complete results
    """
    cfg = engine.config

    result: Dict[str, Any] = {
        "pipeline_meta": {
            "version": "3.0.0",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": mode,
            "engines_loaded": engine._engine_status,
            "stages": {},
        },
        "market": {},
        "pipeline_summary": {},
        "engines": {},
        "candidates": [],
        "risk": {},
        "decisions": [],
    }

    stage_times: Dict[str, float] = {}

    # ── Stage 0: Universe Resolution ──
    t0 = time.time()
    if tickers and len(tickers) > 0:
        universe_tickers = [t.strip().upper() for t in tickers if t.strip()]
    else:
        universe_tickers = _resolve_universe(universe, max_tickers, engine)
    universe_tickers = universe_tickers[:max_tickers]
    result["pipeline_summary"]["scanned"] = len(universe_tickers)
    result["pipeline_meta"]["universe_source"] = "custom" if tickers else universe
    stage_times["universe"] = round(time.time() - t0, 1)

    if progress:
        print(f"📊 Universe: {len(universe_tickers)} tickers ({universe})")

    # ── Stage 1: Market Regime ──
    t0 = time.time()
    market_data = _assess_market_regime()
    result["market"] = market_data
    stage_times["market"] = round(time.time() - t0, 1)

    if progress:
        print(f"📈 Market: SPY={market_data.get('spy', 'N/A')} | Regime={market_data.get('regime', 'UNKNOWN')}")

    # ── Stage 2: Data Fetching ──
    t0 = time.time()
    price_data, fundamentals_data = _fetch_data_layer(universe_tickers, mode, engine)
    result["pipeline_meta"]["stages"]["data"] = {
        "tickers_with_price": len(price_data),
        "tickers_with_fundamentals": len(fundamentals_data),
    }
    stage_times["data"] = round(time.time() - t0, 1)

    if progress:
        print(f"💾 Data: {len(price_data)} price | {len(fundamentals_data)} fundamentals | {stage_times['data']}s")

    # ── Stage 3: Part 1 — Quality Screening ──
    t0 = time.time()
    quality_results = _run_quality_screen(universe_tickers, fundamentals_data, price_data, engine)
    result["pipeline_summary"]["quality"] = quality_results.get("passed", 0)
    result["pipeline_meta"]["stages"]["quality"] = quality_results
    stage_times["quality"] = round(time.time() - t0, 1)

    if progress:
        print(f"🔍 Part 1 Quality: {quality_results.get('passed', 0)}/{quality_results.get('screened', 0)} passed | {stage_times['quality']}s")

    # ── Stage 4: Part 2 — MAGNA Screening ──
    t0 = time.time()
    # Pass raw Part1Result objects (NOT ticker strings) — batch_screen_magna expects List[Part1Result]
    magna_pool = quality_results.get("raw_pool", []) or \
                 [c["ticker"] for c in quality_results.get("all_results", []) if c.get("quality_passed")]
    magna_results = _run_magna_screen(magna_pool, price_data, engine)
    result["pipeline_summary"]["signals"] = magna_results.get("passed", 0)
    result["pipeline_meta"]["stages"]["magna"] = magna_results
    stage_times["magna"] = round(time.time() - t0, 1)

    if progress:
        print(f"⚡ Part 2 MAGNA: {magna_results.get('passed', 0)}/{magna_results.get('screened', 0)} signals | {stage_times['magna']}s")

    # ── Stage 5: Technical Analysis ──
    tech_tickers = [c["ticker"] for c in magna_results.get("all_results", []) if c.get("magna_passed")]
    t0 = time.time()
    tech_results = _run_technical_analysis(tech_tickers, engine)
    result["engines"]["technical"] = tech_results
    stage_times["technical"] = round(time.time() - t0, 1)

    if progress:
        print(f"📐 Technical: {tech_results.get('stocks_analyzed', 0)} analyzed | {stage_times['technical']}s")

    # ── Stage 6: Chip Analysis ──
    t0 = time.time()
    chip_results = _run_chip_analysis(tech_tickers, engine)
    result["engines"]["chip"] = chip_results
    stage_times["chip"] = round(time.time() - t0, 1)

    if progress:
        print(f"🎯 Chip: {chip_results.get('stocks_analyzed', 0)} analyzed | {stage_times['chip']}s")

    # ── Stage 7: Earnings Consensus ──
    t0 = time.time()
    earnings_results = _run_earnings_consensus(tech_tickers)
    result["engines"]["earnings"] = earnings_results
    stage_times["earnings"] = round(time.time() - t0, 1)

    if progress:
        print(f"💰 Earnings: {earnings_results.get('stocks_analyzed', 0)} analyzed | {stage_times['earnings']}s")

    # ── Stage 8: Composite Scoring ──
    t0 = time.time()
    candidates, composite_scores = _compute_composite_score(
        quality_results,
        magna_results,
        tech_results,
        chip_results,
        earnings_results,
        cfg,
    )
    result["pipeline_summary"]["entry_ready"] = len(candidates)
    result["candidates"] = candidates
    result["pipeline_meta"]["stages"]["composite"] = {
        "candidates": len(candidates),
        "score_range": {
            "min": round(min(composite_scores.values()) if composite_scores else 0, 3),
            "max": round(max(composite_scores.values()) if composite_scores else 0, 3),
            "avg": round(np.mean(list(composite_scores.values())) if composite_scores else 0, 3),
        },
    }
    stage_times["composite"] = round(time.time() - t0, 1)

    if progress:
        print(f"🏆 Composite: {len(candidates)} candidates | {stage_times['composite']}s")

    # ── Stage 9: Risk Assessment ──
    t0 = time.time()
    risk_data = _run_risk_assessment(candidates, engine)
    result["risk"] = risk_data
    stage_times["risk"] = round(time.time() - t0, 1)

    if progress:
        print(f"🛡️ Risk: score={risk_data.get('risk_score', 'N/A')} | {stage_times['risk']}s")

    # ── Stage 10: Trade Decisions ──
    t0 = time.time()
    decisions = _generate_decisions(candidates, risk_data, market_data, engine)
    result["decisions"] = decisions
    stage_times["decisions"] = round(time.time() - t0, 1)

    if progress:
        print(f"💡 Decisions: {len(decisions)} generated | {stage_times['decisions']}s")

    # ── Stage 11: Monitor Alerts (if enabled) ──
    t0 = time.time()
    if engine.monitor and cfg.enable_monitor:
        try:
            monitor_tickers = [d["ticker"] for d in decisions if d.get("action") in ("BUY", "ENTRY_READY")]
            if monitor_tickers:
                engine.monitor.set_watchlist(monitor_tickers)
            result["pipeline_meta"]["stages"]["monitor"] = {
                "watchlist_size": len(engine.monitor.watchlist),
            }
        except Exception as e:
            logger.warning(f"Monitor setup failed: {e}")
            result["pipeline_meta"]["stages"]["monitor"] = {"error": str(e)}
    stage_times["monitor"] = round(time.time() - t0, 1)

    # ── Summary ──
    result["pipeline_meta"]["stage_times"] = stage_times
    result["pipeline_meta"]["total_elapsed_seconds"] = round(sum(stage_times.values()), 1)

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage: Universe Resolution
# ═══════════════════════════════════════════════════════════════════

def _resolve_universe(
    source: str,
    max_tickers: int,
    engine: Any,
) -> List[str]:
    """Resolve the stock universe."""
    try:
        # Try using pipeline.py's existing get_ticker_universe
        from pipeline import get_ticker_universe
        return get_ticker_universe(source=source)[:max_tickers]
    except ImportError:
        pass

    # Fallback: try yfinance for S&P 500
    try:
        import yfinance as yf
        import requests
        import io

        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            return df['Symbol'].str.replace('.', '-', regex=False).tolist()[:max_tickers]
    except Exception:
        pass

    # Hardcoded fallback
    return [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
        "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH",
        "HD", "DIS", "BAC", "XOM", "NFLX", "ADBE", "CRM", "AMD",
        "INTC", "QCOM", "TXN", "CSCO", "PEP", "KO", "COST",
    ][:max_tickers]


# ═══════════════════════════════════════════════════════════════════
# Stage: Market Regime
# ═══════════════════════════════════════════════════════════════════

def _assess_market_regime() -> Dict[str, Any]:
    """Assess current market regime using SPY as proxy."""
    regime: Dict[str, Any] = {
        "spy": 0.0,
        "spy_change_5d": 0.0,
        "spy_change_20d": 0.0,
        "vix_proxy": 0.0,
        "regime": "UNKNOWN",
        "above_50ma": False,
        "above_200ma": False,
    }

    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y")

        if not hist.empty and len(hist) >= 200:
            close = hist["Close"]
            regime["spy"] = round(float(close.iloc[-1]), 2)
            regime["spy_change_5d"] = round(float(close.pct_change(5).iloc[-1]), 4)
            regime["spy_change_20d"] = round(float(close.pct_change(20).iloc[-1]), 4)

            # Moving averages
            ma50 = float(close.rolling(50).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1])
            regime["spy_ma50"] = round(ma50, 2)
            regime["spy_ma200"] = round(ma200, 2)
            regime["above_50ma"] = regime["spy"] > ma50
            regime["above_200ma"] = regime["spy"] > ma200

            # VIX proxy (20d annualized vol)
            daily_ret = close.pct_change().dropna()
            vol_20d = float(daily_ret.rolling(20).std().iloc[-1])
            regime["vix_proxy"] = round(vol_20d * np.sqrt(252), 4)

            # Regime classification
            if regime["vix_proxy"] > 0.35:
                regime["regime"] = "CRISIS"
            elif regime["vix_proxy"] > 0.25:
                regime["regime"] = "VOLATILE"
            elif regime["spy"] < ma200:
                regime["regime"] = "BEARISH"
            elif regime["spy"] < ma50:
                regime["regime"] = "CORRECTION"
            elif regime["spy_change_20d"] > 0.05:
                regime["regime"] = "BULLISH"
            else:
                regime["regime"] = "NORMAL"

    except Exception as e:
        logger.warning(f"Market regime assessment failed: {e}")
        regime["warning"] = str(e)

    return regime


# ═══════════════════════════════════════════════════════════════════
# Stage: Data Fetching
# ═══════════════════════════════════════════════════════════════════

def _fetch_data_layer(
    tickers: List[str],
    mode: str,
    engine: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Fetch price and fundamentals data using the hybrid data layer.

    Returns:
        (price_data, fundamentals_data) dicts keyed by ticker
    """
    price_data: Dict[str, Dict[str, Any]] = {}
    fundamentals_data: Dict[str, Dict[str, Any]] = {}

    try:
        from data.hybrid import get_price as hybrid_get_price

        for ticker in tickers:
            try:
                price, volume, source, date = hybrid_get_price(ticker)
                if price > 0:
                    price_data[ticker] = {
                        "price": price,
                        "volume": volume,
                        "source": source,
                        "date": date,
                    }
            except Exception:
                pass
    except ImportError:
        logger.warning("data/hybrid.py unavailable, falling back to yfinance")
        try:
            import yfinance as yf
            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="5d")
                    if not hist.empty:
                        close = float(hist["Close"].iloc[-1])
                        vol = int(hist["Volume"].iloc[-1])
                        if close > 0:
                            price_data[ticker] = {
                                "price": close,
                                "volume": vol,
                                "source": "yfinance",
                                "date": hist.index[-1].strftime("%Y-%m-%d"),
                            }
                except Exception:
                    pass
        except ImportError:
            logger.error("yfinance not available for data fetching")

    # Fundamentals via SEC EDGAR + yfinance
    try:
        from data.hybrid import HybridData
        hd = HybridData()
        for ticker in tickers:
            try:
                data = hd.get_fundamentals(ticker)
                if data:
                    fundamentals_data[ticker] = data
            except Exception:
                pass
    except ImportError:
        logger.warning("HybridData unavailable for fundamentals — using yfinance")
        try:
            import yfinance as yf
            for ticker in tickers:
                try:
                    t = yf.Ticker(ticker)
                    info = t.info
                    if info:
                        fundamentals_data[ticker] = {
                            "marketCap": info.get("marketCap"),
                            "bookValue": info.get("bookValue"),
                            "returnOnAssets": info.get("returnOnAssets"),
                            "ebitdaMargins": info.get("ebitdaMargins"),
                            "freeCashflow": info.get("freeCashflow"),
                            "netIncomeToCommon": info.get("netIncomeToCommon"),
                            "totalAssets": info.get("totalAssets"),
                            "debtToEquity": info.get("debtToEquity"),
                            "beta": info.get("beta"),
                            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
                            "averageVolume": info.get("averageVolume"),
                            "sector": info.get("sector", ""),
                            "source": "yfinance",
                        }
                except Exception:
                    pass
        except ImportError:
            pass

    return price_data, fundamentals_data


# ═══════════════════════════════════════════════════════════════════
# Stage: Part 1 — Quality Screening
# ═══════════════════════════════════════════════════════════════════

def _run_quality_screen(
    tickers: List[str],
    fundamentals_data: Dict[str, Dict[str, Any]],
    price_data: Dict[str, Dict[str, Any]],
    engine: Any,
) -> Dict[str, Any]:
    """Run Part 1 quality (value + FCF) screening."""
    result: Dict[str, Any] = {
        "screened": 0,
        "passed": 0,
        "failed": 0,
        "all_results": [],
        "raw_pool": [],        # Raw Part1Result objects for Part 2
        "engine_used": "none",
    }

    # Try using pipeline.py's existing Part 1 batch_screen
    try:
        from part1_fundamentals import batch_screen as part1_batch
        from config import P1C

        screened = part1_batch(tickers)
        if screened is not None:
            result["engine_used"] = "part1_fundamentals.batch_screen"
            result["screened"] = len(screened)

            passed = [s for s in screened if getattr(s, "quality_score", 0) >= P1C.min_quality_score]
            result["passed"] = len(passed)
            result["raw_pool"] = passed  # Store raw objects for Part 2

            for stock in screened:
                qs = getattr(stock, "quality_score", 0)
                ticker = getattr(stock, "ticker", "unknown")
                price = price_data.get(ticker, {}).get("price", 0)
                entry = {
                    "ticker": ticker,
                    "quality_score": round(float(qs), 4),
                    "quality_passed": qs >= P1C.min_quality_score,
                    "fcf_yield": round(float(getattr(stock, "fcf_yield", 0) or 0), 4),
                    "bm_ratio": round(float(getattr(stock, "bm_ratio", 0) or 0), 4),
                    "roa": round(float(getattr(stock, "roa", 0) or 0), 4),
                    "price": price,
                }
                result["all_results"].append(entry)
                if not entry["quality_passed"]:
                    result["failed"] += 1

            return result
    except (ImportError, Exception) as e:
        logger.warning(f"Part 1 batch_screen unavailable: {e}")

    # Fallback: manual quality screening from fundamentals
    try:
        from config import P1C
        result["engine_used"] = "manual_fallback"

        for ticker in tickers:
            fd = fundamentals_data.get(ticker, {})
            price = price_data.get(ticker, {}).get("price", 0)

            # Extract key metrics
            market_cap = float(fd.get("marketCap", 0) or 0)
            bv = float(fd.get("bookValue", 0) or 0)
            roa = float(fd.get("returnOnAssets", 0) or 0)
            ebitda_margin = float(fd.get("ebitdaMargins", 0) or 0)
            fcf = float(fd.get("freeCashflow", 0) or 0)
            fcf_yield = fcf / market_cap if market_cap > 0 else 0
            ni = float(fd.get("netIncomeToCommon", 0) or 0)
            fcf_conv = fcf / ni if ni > 0 and fcf > 0 else 0
            wk_low = float(fd.get("fiftyTwoWeekLow", 0) or 0)
            ptl = price / wk_low if wk_low > 0 else 99

            # Score
            scores = {
                "bm": min(bv / max(price, 1), 2.0) / 2.0 if price > 0 and bv > 0 else 0,
                "roa": min(max(roa, 0) / P1C.target_roa, 1.0) if P1C.target_roa else 0,
                "ebitda": min(max(ebitda_margin, 0) / P1C.target_ebitda_margin, 1.0),
                "fcf_yield": min(fcf_yield / P1C.target_fcf_yield, 1.0),
                "fcf_conv": min(fcf_conv / P1C.target_fcf_conversion, 1.0),
                "ptl": max(0, 1 - (ptl - 1) / (P1C.max_ptl_ratio - 1)) if P1C.max_ptl_ratio > 1 else 0,
            }

            quality = sum(scores[k] * getattr(P1C, f"weight_{k}", 0.2) for k in scores) / max(
                sum(getattr(P1C, f"weight_{k}", 0.2) for k in scores), 0.01
            )

            entry = {
                "ticker": ticker,
                "quality_score": round(quality, 4),
                "quality_passed": quality >= P1C.min_quality_score,
                "fcf_yield": round(fcf_yield, 4),
                "bm_ratio": round(bv / max(price, 1), 4) if price > 0 else 0,
                "roa": round(roa, 4),
                "price": price,
            }
            result["all_results"].append(entry)
            result["screened"] += 1
            if entry["quality_passed"]:
                result["passed"] += 1
            else:
                result["failed"] += 1

    except ImportError:
        logger.warning("Cannot import P1C for quality screen fallback")
        result["engine_used"] = "disabled"

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage: Part 2 — MAGNA Screening
# ═══════════════════════════════════════════════════════════════════

def _run_magna_screen(
    pool: List[Any],
    price_data: Dict[str, Dict[str, Any]],
    engine: Any,
) -> Dict[str, Any]:
    """Run Part 2 MAGNA 53/10 momentum screening.
    
    Args:
        pool: List of Part1Result objects (preferred) OR ticker strings
    """
    result: Dict[str, Any] = {
        "screened": 0,
        "passed": 0,
        "failed": 0,
        "all_results": [],
        "engine_used": "none",
    }

    if not pool:
        return result

    try:
        from part2_magna import batch_screen_magna as part2_batch
        from config import P2C

        # batch_screen_magna expects List[Part1Result]
        # If we got strings, convert to Part1Result stubs (signal quality will be lower)
        if pool and isinstance(pool[0], str):
            screened = part2_batch(pool)
        else:
            screened = part2_batch(pool)
        if screened is not None:
            result["engine_used"] = "part2_magna.batch_screen_magna"
            result["screened"] = len(screened)

            for stock in screened:
                ticker = getattr(stock, "ticker", "unknown")
                ms = getattr(stock, "magna_score", 0)
                price = price_data.get(ticker, {}).get("price", 0)

                entry = {
                    "ticker": ticker,
                    "magna_score": int(ms),
                    "magna_passed": ms >= P2C.magna_pass_threshold,
                    "eps_growth": round(float(getattr(stock, "eps_growth", 0) or 0), 4),
                    "revenue_growth": round(float(getattr(stock, "revenue_growth", 0) or 0), 4),
                    "has_gap": bool(getattr(stock, "has_gap", False)),
                    "in_base": bool(getattr(stock, "in_base", False)),
                    "price": price,
                }
                result["all_results"].append(entry)
                if entry["magna_passed"]:
                    result["passed"] += 1
                else:
                    result["failed"] += 1

            return result
    except (ImportError, Exception) as e:
        logger.warning(f"Part 2 batch_screen_magna unavailable: {e}")

    # Fallback: minimal MAGNA check from price data
    result["engine_used"] = "manual_fallback"
    try:
        import yfinance as yf
        import numpy as np

        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="6mo")
                info = t.info

                magna_score = 0
                details = {}

                # Gap check
                if len(hist) >= 3:
                    prev_close = float(hist["Close"].iloc[-2])
                    today_open = float(hist["Open"].iloc[-1])
                    today_close = float(hist["Close"].iloc[-1])
                    gap_pct = (today_open - prev_close) / prev_close if prev_close > 0 else 0
                    if gap_pct >= 0.04:
                        magna_score += 2
                        details["gap_pct"] = round(gap_pct, 4)

                    # Momentum (price above 50MA)
                    if len(hist) >= 50:
                        ma50 = float(hist["Close"].rolling(50).mean().iloc[-1])
                        if today_close > ma50:
                            magna_score += 1

                # EPS growth
                eps_growth = float(info.get("earningsQuarterlyGrowth", 0) or 0)
                if eps_growth >= 0.20:
                    magna_score += 2
                revenue_growth = float(info.get("revenueGrowth", 0) or 0)
                if revenue_growth >= 0.10:
                    magna_score += 2

                price = price_data.get(ticker, {}).get("price", 0)
                entry = {
                    "ticker": ticker,
                    "magna_score": magna_score,
                    "magna_passed": magna_score >= 3,
                    "eps_growth": round(eps_growth, 4),
                    "revenue_growth": round(revenue_growth, 4),
                    "price": price,
                }
                result["all_results"].append(entry)
                result["screened"] += 1
                if entry["magna_passed"]:
                    result["passed"] += 1
                else:
                    result["failed"] += 1

            except Exception as e:
                logger.debug(f"MAGNA fallback failed for {ticker}: {e}")
                entry = {
                    "ticker": ticker,
                    "magna_score": 0,
                    "magna_passed": False,
                    "error": str(e)[:100],
                }
                result["all_results"].append(entry)
                result["screened"] += 1
                result["failed"] += 1

    except ImportError:
        result["engine_used"] = "disabled"

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage: Technical Analysis
# ═══════════════════════════════════════════════════════════════════

def _run_technical_analysis(
    tickers: List[str],
    engine: Any,
) -> Dict[str, Any]:
    """Run technical analysis on MAGNA-passed tickers."""
    result: Dict[str, Any] = {
        "enabled": False,
        "stocks_analyzed": 0,
        "signals": {},
        "error": None,
    }

    if not tickers or not engine.technical:
        return result

    result["enabled"] = True
    try:
        batch = engine.technical.batch_analysis(tickers)

        for item in batch:
            ticker = item.get("ticker", "unknown")
            if "error" in item:
                continue

            signals = item.get("signals", {})
            result["signals"][ticker] = {
                "signal": signals.get("signal", "HOLD"),
                "strength": signals.get("strength", 0),
                "latest_price": signals.get("latest_price", 0),
                "indicators_count": len(item.get("indicators", {})),
                "key_indicators": {
                    "rsi_14": item.get("indicators", {}).get("rsi_14"),
                    "macd_histogram": item.get("indicators", {}).get("macd_histogram"),
                    "above_sma_50": item.get("indicators", {}).get("above_sma_50"),
                    "bb_pct_b": item.get("indicators", {}).get("bb_pct_b"),
                },
            }

        result["stocks_analyzed"] = len(batch)
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Technical analysis failed: {e}")

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage: Chip Analysis
# ═══════════════════════════════════════════════════════════════════

def _run_chip_analysis(
    tickers: List[str],
    engine: Any,
) -> Dict[str, Any]:
    """Run chip (volume profile) analysis on MAGNA-passed tickers."""
    result: Dict[str, Any] = {
        "enabled": False,
        "stocks_analyzed": 0,
        "signals": {},
        "error": None,
    }

    if not tickers or not engine.chip:
        return result

    result["enabled"] = True
    for ticker in tickers:
        try:
            chip_report = engine.chip.analyze(ticker, period="6mo")
            if chip_report and not chip_report.errors:
                result["signals"][ticker] = {
                    "current_price": chip_report.current_price,
                    "below_vwap": (
                        chip_report.current_price < chip_report.cost_basis.get("vwap", 0)
                        if chip_report.cost_basis else False
                    ),
                    "support_level": (
                        chip_report.support_resistance.get("nearest_support", {}).get("price")
                        if chip_report.support_resistance else None
                    ),
                    "resistance_level": (
                        chip_report.support_resistance.get("nearest_resistance", {}).get("price")
                        if chip_report.support_resistance else None
                    ),
                    "concentration_score": (
                        chip_report.concentration.get("concentration_score")
                        if chip_report.concentration else None
                    ),
                    "profitability_rating": (
                        chip_report.profitability.get("rating")
                        if chip_report.profitability else None
                    ),
                }
                result["stocks_analyzed"] += 1
            elif chip_report and chip_report.errors:
                result["signals"][ticker] = {"error": chip_report.errors[0]}
        except Exception as e:
            logger.warning(f"Chip analysis failed for {ticker}: {e}")
            result["signals"][ticker] = {"error": str(e)[:100]}

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage: Earnings Consensus
# ═══════════════════════════════════════════════════════════════════

def _run_earnings_consensus(tickers: List[str]) -> Dict[str, Any]:
    """Fetch earnings consensus data from yfinance."""
    result: Dict[str, Any] = {
        "enabled": True,
        "stocks_analyzed": 0,
        "consensus": {},
    }

    if not tickers:
        return result

    try:
        import yfinance as yf

        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info

                consensus = {
                    "recommendation": info.get("recommendationKey", "none"),
                    "target_mean": float(info.get("targetMeanPrice", 0) or 0),
                    "target_high": float(info.get("targetHighPrice", 0) or 0),
                    "target_low": float(info.get("targetLowPrice", 0) or 0),
                    "num_analysts": int(info.get("numberOfAnalystOpinions", 0) or 0),
                    "earnings_growth": float(info.get("earningsGrowth", 0) or 0),
                    "revenue_growth": float(info.get("revenueGrowth", 0) or 0),
                    "forward_pe": float(info.get("forwardPE", 0) or 0),
                    "peg_ratio": float(info.get("pegRatio", 0) or 0),
                }

                # Target premium
                price = float(info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0)
                if consensus["target_mean"] > 0 and price > 0:
                    consensus["target_premium_pct"] = round(
                        (consensus["target_mean"] - price) / price, 4
                    )
                else:
                    consensus["target_premium_pct"] = None

                result["consensus"][ticker] = consensus
                result["stocks_analyzed"] += 1
            except Exception as e:
                logger.debug(f"Earnings consensus failed for {ticker}: {e}")

    except ImportError:
        result["enabled"] = False
        result["error"] = "yfinance not available"

    return result


# ═══════════════════════════════════════════════════════════════════
# Stage: Composite Scoring
# ═══════════════════════════════════════════════════════════════════

def _compute_composite_score(
    quality_results: Dict[str, Any],
    magna_results: Dict[str, Any],
    tech_results: Dict[str, Any],
    chip_results: Dict[str, Any],
    earnings_results: Dict[str, Any],
    cfg: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Compute composite score by weighting contributions from all engines.

    Returns:
        (candidates list, composite_scores dict)
    """
    candidates: List[Dict[str, Any]] = []
    composite_scores: Dict[str, float] = {}

    # Collect all tickers that passed Part 1 + Part 2
    magna_passed = {
        r["ticker"]: r
        for r in magna_results.get("all_results", [])
        if r.get("magna_passed")
    }

    quality_passed = {
        r["ticker"]: r
        for r in quality_results.get("all_results", [])
        if r.get("quality_passed")
    }

    # Only tickers that pass BOTH quality and MAGNA
    dual_pass = set(magna_passed.keys()) & set(quality_passed.keys())

    for ticker in dual_pass:
        q = quality_passed[ticker]
        m = magna_passed[ticker]

        # Quality score (0-1)
        quality_norm = min(q.get("quality_score", 0), 1.0)

        # Momentum score (0-1), normalize MAGNA score (0-10 → 0-1)
        magna_norm = min(m.get("magna_score", 0) / 10.0, 1.0)

        # Technical score (0-1)
        tech_signal = tech_results.get("signals", {}).get(ticker, {})
        tech_score = _normalize_tech_signal(tech_signal)

        # Chip score (0-1)
        chip_signal = chip_results.get("signals", {}).get(ticker, {})
        chip_score = _normalize_chip_signal(chip_signal)

        # Earnings score (0-1)
        earnings_consensus = earnings_results.get("consensus", {}).get(ticker, {})
        earnings_score = _normalize_earnings_signal(earnings_consensus)

        # Weighted composite
        composite = (
            quality_norm * cfg.weight_quality
            + magna_norm * cfg.weight_momentum
            + tech_score * cfg.weight_technical
            + earnings_score * cfg.weight_earnings
            + chip_score * cfg.weight_chip
        )
        composite_scores[ticker] = round(composite, 4)

        candidate = {
            "ticker": ticker,
            "composite_score": round(composite, 4),
            "quality_score": round(quality_norm, 4),
            "magna_score": int(m.get("magna_score", 0)),
            "technical_score": round(tech_score, 4),
            "chip_score": round(chip_score, 4),
            "earnings_score": round(earnings_score, 4),
            "price": q.get("price", m.get("price", 0)),
            "technical_signal": tech_signal.get("signal", "UNKNOWN"),
            "chip_rating": chip_signal.get("profitability_rating", "UNKNOWN"),
            "analyst_consensus": earnings_consensus.get("recommendation", "none"),
            "target_premium": earnings_consensus.get("target_premium_pct"),
        }

        if composite >= cfg.min_composite_score:
            candidates.append(candidate)

    # Sort by composite score descending
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)

    return candidates, composite_scores


def _normalize_tech_signal(signal: Dict[str, Any]) -> float:
    """Normalize technical signal to 0-1 score."""
    if not signal or signal.get("error"):
        return 0.5  # neutral

    sig_type = signal.get("signal", "HOLD")
    strength = float(signal.get("strength", 0))

    # Map signals to scores
    signal_map = {
        "STRONG_BUY": 0.9 + strength * 0.1,
        "BUY": 0.65 + strength * 0.15,
        "HOLD": 0.40 + strength * 0.20,
        "SELL": 0.20 - strength * 0.10,
        "STRONG_SELL": 0.05 - strength * 0.05,
    }

    base = signal_map.get(sig_type, 0.40)

    # Adjust based on key indicators
    indicators = signal.get("key_indicators", {})
    rsi = indicators.get("rsi_14")
    if rsi is not None:
        if 30 <= rsi <= 70:
            base += 0.05  # healthy RSI range
        elif rsi < 30:
            base += 0.10  # oversold — potential bounce

    above_sma = indicators.get("above_sma_50")
    if above_sma:
        base += 0.05

    return round(max(0.0, min(1.0, base)), 4)


def _normalize_chip_signal(signal: Dict[str, Any]) -> float:
    """Normalize chip analysis signal to 0-1 score."""
    if not signal or signal.get("error"):
        return 0.5

    score = 0.5

    # Below VWAP = accumulation potential
    if signal.get("below_vwap"):
        score += 0.15

    # Concentration score
    conc = signal.get("concentration_score")
    if conc is not None:
        score += float(conc) * 0.1

    # Profitability rating
    rating = (signal.get("profitability_rating") or "").upper()
    rating_map = {"EXCELLENT": 0.15, "GOOD": 0.10, "FAIR": 0.05, "POOR": -0.05}
    score += rating_map.get(rating, 0)

    return round(max(0.0, min(1.0, score)), 4)


def _normalize_earnings_signal(consensus: Dict[str, Any]) -> float:
    """Normalize earnings consensus to 0-1 score."""
    if not consensus:
        return 0.5

    score = 0.5

    # Analyst recommendation
    rec = (consensus.get("recommendation") or "").lower()
    rec_map = {"strong_buy": 0.20, "buy": 0.15, "hold": 0.0, "underperform": -0.10, "sell": -0.20}
    score += rec_map.get(rec, 0)

    # Target premium
    premium = consensus.get("target_premium_pct")
    if premium is not None:
        if premium > 0.20:
            score += 0.15
        elif premium > 0.10:
            score += 0.10
        elif premium > 0:
            score += 0.05
        elif premium < -0.10:
            score -= 0.10

    # Growth metrics
    eg = consensus.get("earnings_growth", 0)
    if eg > 0.20:
        score += 0.10
    elif eg > 0.10:
        score += 0.05

    # PEG ratio
    peg = consensus.get("peg_ratio", 0)
    if 0 < peg < 2:
        score += 0.05
    elif 0 < peg < 1:
        score += 0.10

    return round(max(0.0, min(1.0, score)), 4)


# ═══════════════════════════════════════════════════════════════════
# Stage: Risk Assessment
# ═══════════════════════════════════════════════════════════════════

def _run_risk_assessment(
    candidates: List[Dict[str, Any]],
    engine: Any,
) -> Dict[str, Any]:
    """Run risk assessment on composite candidates."""
    risk_data: Dict[str, Any] = {
        "enabled": False,
        "risk_score": None,
        "risk_level": "UNKNOWN",
        "var_95": None,
        "max_dd_pct": None,
        "circuit_breakers": "DISABLED",
        "max_position_size": None,
        "position_sizing": [],
    }

    if not engine.risk or not candidates:
        return risk_data

    risk_data["enabled"] = True

    try:
        from engine.risk.engine import Portfolio, Position, quick_assess

        tickers = [c["ticker"] for c in candidates]
        prices = [c.get("price", 0) for c in candidates]

        report = quick_assess(tickers, prices)
        risk_data["risk_score"] = report.risk_score
        risk_data["risk_level"] = report.risk_level
        risk_data["var_95"] = (
            report.var.get("historical_var_pct", {}).get("var95")
            if report.var else None
        )
        risk_data["max_dd_pct"] = next(
            iter(report.stress_tests.values()), {}
        ).get("drawdown", None) if report.stress_tests else None

        # Circuit breakers
        cb = engine.risk.check_circuit_breakers()
        risk_data["circuit_breakers"] = "OK" if cb.trading_allowed else cb.recommendation

        # Position sizing for candidates
        total_value = sum(prices) * 100  # approximate portfolio value
        sizing = engine.risk.suggest_sizing(
            candidates=candidates,
            portfolio_value=total_value,
            market_ok=engine._engine_status.get("risk") == "loaded",
        )
        risk_data["position_sizing"] = sizing
        risk_data["max_position_size"] = engine.risk.config.sizing.max_risk_per_trade if hasattr(engine.risk, 'config') else None

    except Exception as e:
        risk_data["error"] = str(e)
        logger.warning(f"Risk assessment failed: {e}")

    return risk_data


# ═══════════════════════════════════════════════════════════════════
# Stage: Trade Decisions
# ═══════════════════════════════════════════════════════════════════

def _generate_decisions(
    candidates: List[Dict[str, Any]],
    risk_data: Dict[str, Any],
    market_data: Dict[str, Any],
    engine: Any,
) -> List[Dict[str, Any]]:
    """Generate trade decisions from composite scores and risk assessment."""
    decisions: List[Dict[str, Any]] = []

    regime = market_data.get("regime", "NORMAL")
    trading_allowed = regime not in ("CRISIS",)
    cb_ok = risk_data.get("circuit_breakers") == "OK"

    for i, cand in enumerate(candidates):
        comp = cand["composite_score"]
        price = cand.get("price", 0)

        # Decision logic
        sizing = risk_data.get("position_sizing", [])
        position_size = sizing[i].get("suggested_shares", 0) if i < len(sizing) else 0

        if not trading_allowed:
            action = "HOLD_MARKET"
            reason = f"Market regime {regime} — no new entries"
        elif not cb_ok:
            action = "HOLD_CIRCUIT"
            reason = f"Circuit breaker: {risk_data.get('circuit_breakers', 'ACTIVE')}"
        elif comp >= 0.75:
            action = "STRONG_BUY"
            reason = f"Composite {comp:.3f} — top tier"
        elif comp >= 0.55:
            action = "BUY"
            reason = f"Composite {comp:.3f} — good entry"
        elif comp >= 0.45:
            action = "ENTRY_READY"
            reason = f"Composite {comp:.3f} — monitor for trigger"
        else:
            action = "WATCH"
            reason = f"Composite {comp:.3f} — below entry threshold"

        decisions.append({
            "ticker": cand["ticker"],
            "action": action,
            "composite_score": comp,
            "quality": cand["quality_score"],
            "momentum": cand["magna_score"],
            "technical": cand["technical_score"],
            "chip": cand["chip_score"],
            "earnings": cand["earnings_score"],
            "price": price,
            "position_size": position_size,
            "reason": reason,
            "technical_signal": cand.get("technical_signal", "UNKNOWN"),
            "analyst_consensus": cand.get("analyst_consensus", "none"),
            "target_premium": cand.get("target_premium"),
        })

    return decisions


# ═══════════════════════════════════════════════════════════════════
# Convenience: Run standalone
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("VMAA v3 Pipeline Orchestrator — Standalone Test")
    print("=" * 60)

    from engine import VMAAEngine
    engine = VMAAEngine()
    engine.print_status()

    print("\n🔍 Running quick scan on 10 tickers...")
    result = engine.quick_scan(tickers=[
        "AAPL", "MSFT", "GOOGL", "NVDA", "META",
        "AMZN", "TSLA", "JPM", "V", "JNJ",
    ])

    print(f"\nPipeline Summary:")
    print(f"  Scanned: {result['pipeline_summary']['scanned']}")
    print(f"  Quality passed: {result['pipeline_summary']['quality']}")
    print(f"  MAGNA signals: {result['pipeline_summary']['signals']}")
    print(f"  Entry ready: {result['pipeline_summary']['entry_ready']}")
    print(f"  Total time: {result['pipeline_meta']['total_elapsed_seconds']}s")

    if result["candidates"]:
        print(f"\nTop Candidates:")
        for c in result["candidates"][:5]:
            print(f"  {c['ticker']:6s}  composite={c['composite_score']:.3f}  "
                  f"q={c['quality_score']:.2f} m={c['magna_score']} "
                  f"t={c['technical_score']:.2f} c={c['chip_score']:.2f}")

    if result["decisions"]:
        print(f"\nDecisions:")
        for d in result["decisions"][:5]:
            print(f"  {d['ticker']:6s}  {d['action']:12s}  {d['reason']}")
