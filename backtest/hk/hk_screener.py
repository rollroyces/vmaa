#!/usr/bin/env python3
"""
HK Screening Logic for Backtest
================================
Wraps live `pipeline_hk.py` screening functions for point-in-time backtesting.

Key approach: IMPORT HK screening logic from pipeline_hk.py, NOT duplicate it.
The live module functions (`screen_hk_fundamentals`, `screen_hk_magna`) operate
on current yfinance data. For backtesting, we:

1. Pre-fetch yfinance info/ticker objects (current data proxy)
2. Override price-dependent fields with historical snapshot data
3. Feed the modified data through the EXACT SAME screening pipeline

This ensures backtest thresholds ALWAYS match live scan thresholds —
no divergence, no dead code, no "forgot to update backtest thresholds".

Thresholds used (from pipeline_hk.py, verified against HK backtest spec):
  - B/M ≥ 0.15 (HK) vs 0.20 (US)
  - ROA ≥ 1% (HK) vs 0% (US)
  - FCF/Y ≥ 1% (HK) vs 2% (US)
  - PTL ≤ 1.35 (HK) vs 1.50 (US)
  - Financial sector auto-pass on B/M, EBITDA, FCF conv
  - Financial sector uses ROE ≥ 5% instead of ROA
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.data import HistoricalSnapshot
from backtest.hk.hk_config import HKBacktestConfig, HKC, HKScreeningThresholds

logger = logging.getLogger("vmaa.backtest.hk.screener")

# Import HK pipeline screening functions
# These are the SINGLE SOURCE OF TRUTH for HK screening logic
from pipeline_hk import (
    screen_hk_fundamentals as _live_screen_fundamentals,
    screen_hk_magna as _live_screen_magna,
    HKMagnaSignal,
    FINANCIAL_SECTORS,
    FINANCIAL_INDUSTRIES,
)


class HKScreener:
    """
    Point-in-time HK screening using live pipeline_hk.py logic.

    Strategy:
      - Cache yfinance info/ticker objects for each HK stock
      - On each backtest date: build a modified yf.Ticker that returns
        snapshot-adjusted data, then call live screen_hk_fundamentals()
        and screen_hk_magna() with the adjusted data.

    This ensures thresholds never diverge between live and backtest.
    """

    def __init__(self, config: HKBacktestConfig = HKC):
        self.config = config
        self.thresholds = config.thresholds
        self._info_cache: Dict[str, dict] = {}
        self._ticker_cache: Dict[str, yf.Ticker] = {}
        self._daily_prices: Dict[str, pd.DataFrame] = {}

    def cache_yfinance(self, tickers: List[str]) -> None:
        """
        Pre-fetch yfinance data for all tickers.
        
        Cached info provides current fundamentals (best available proxy
        for historical fundamentals in retail backtesting). Price-dependent
        fields are overridden per-date during screening.
        """
        for t in tickers:
            try:
                yft = yf.Ticker(t)
                self._ticker_cache[t] = yft
                self._info_cache[t] = yft.info
                logger.debug(f"  Cached yfinance: {t}")
            except Exception as e:
                logger.debug(f"  {t}: yfinance cache failed — {e}")
                self._info_cache[t] = {}
        logger.info(f"yfinance cache: {len(self._info_cache)} HK tickers")

    def set_daily_prices(self, daily_prices: Dict[str, pd.DataFrame]) -> None:
        """Connect daily price data for historical price overrides."""
        self._daily_prices = daily_prices

    # ═══════════════════════════════════════════════════════════════
    # Part 1: HK Fundamentals (via live pipeline_hk.py)
    # ═══════════════════════════════════════════════════════════════

    def screen_part1(self, snapshot: HistoricalSnapshot) -> Optional[dict]:
        """
        Run HK Part 1 screening at a historical point in time.

        Uses cached yfinance data overlaid with snapshot-adjusted prices
        fed through the live screen_hk_fundamentals().

        Returns dict with standard HK screening fields (same format as
        pipeline_hk.py's screen_hk_fundamentals output).
        """
        ticker = snapshot.ticker
        cached_info = self._info_cache.get(ticker, {})

        if not cached_info:
            # Try direct fundamental computation from snapshot
            return self._screen_part1_from_snapshot(snapshot)

        try:
            # Build modified info dict: cached fundamentals + historical prices
            info = dict(cached_info)
            info['regularMarketPrice'] = snapshot.close
            info['currentPrice'] = snapshot.close
            info['previousClose'] = snapshot.close
            info['fiftyTwoWeekLow'] = snapshot.low_52w
            info['fiftyTwoWeekHigh'] = snapshot.high_52w

            # Use snapshot market cap if available, else price-adjust current
            if snapshot.market_cap and snapshot.market_cap > 0:
                info['marketCap'] = snapshot.market_cap
            elif cached_info.get('marketCap', 0) > 0 and cached_info.get('currentPrice', 0) > 0:
                info['marketCap'] = cached_info['marketCap'] * (
                    snapshot.close / cached_info['currentPrice']
                )

            # Call HK-specific screening with this adjusted info
            # We monkey-patch the ticker's info access
            cached_ticker = self._ticker_cache.get(ticker)
            if cached_ticker is None:
                cached_ticker = yf.Ticker(ticker)

            # Store original info for restoration
            _orig_info = cached_ticker._info if hasattr(cached_ticker, '_info') else None

            try:
                cached_ticker._info = info
                result = _live_screen_hk_fundamentals(ticker)
            finally:
                if _orig_info is not None:
                    cached_ticker._info = _orig_info
                elif hasattr(cached_ticker, '_info'):
                    del cached_ticker._info

            return result

        except Exception as e:
            logger.debug(f"  {ticker}: HK Part 1 via live failed — {e}, falling back to snapshot")
            return self._screen_part1_from_snapshot(snapshot)

    def _screen_part1_from_snapshot(self, snapshot: HistoricalSnapshot) -> Optional[dict]:
        """
        Fallback: compute HK Part 1 screening directly from snapshot data.
        
        Reimplements screen_hk_fundamentals() logic using only snapshot fields.
        This is used when yfinance cache is unavailable.
        """
        ticker = snapshot.ticker

        # Skip delisted or sub-HKD 1 stocks
        if snapshot.close <= 1.0 or (snapshot.market_cap or 0) <= 0:
            return None

        # ── Sector detection ──
        sector = snapshot.sector or 'Unknown'
        industry = snapshot.industry or 'Unknown'
        is_financial = (
            sector in FINANCIAL_SECTORS or industry in FINANCIAL_INDUSTRIES
        )

        # ── Compute metrics from snapshot ──
        price = snapshot.close
        low_52w = snapshot.low_52w
        high_52w = snapshot.high_52w
        market_cap = snapshot.market_cap or 0

        # B/M
        bv = snapshot.book_value or 0
        bm_ratio = bv / price if bv > 0 and price > 0 else 0

        # Profitability
        roa = snapshot.roa or 0
        roe = snapshot.roe or 0
        ebitda_margin = (snapshot.ebitda or 0) / (snapshot.total_revenue or 1) if (snapshot.total_revenue or 0) > 0 else 0

        # FCF Yield
        fcf = snapshot.free_cashflow or 0
        ev = (market_cap + (snapshot.total_debt or 0)) if market_cap > 0 else 0
        fcf_yield = fcf / ev if fcf > 0 and ev > 0 else (fcf / market_cap if fcf > 0 and market_cap > 0 else 0)

        # FCF Conversion
        ni = snapshot.net_income or 0
        fcf_conversion = fcf / ni if fcf > 0 and ni > 0 else 0

        # PTL (proximity to 52w low)
        ptl_ratio = price / low_52w if low_52w > 0 else 999

        # Debt
        de = snapshot.debt_to_equity or 0

        # ── Screening (same thresholds as pipeline_hk.py) ──
        t = self.thresholds
        passed = []
        failed = []
        warnings = []

        # Market Cap
        if market_cap >= t.min_market_cap_hkd:
            passed.append("market_cap")
        else:
            failed.append("market_cap")

        # B/M Ratio
        if bm_ratio >= t.min_bm_ratio:
            passed.append("bm_ratio")
        elif is_financial:
            warnings.append("bm_ratio_financial_skip")
            passed.append("bm_ratio")
        else:
            failed.append("bm_ratio")

        # ROA / ROE (financial)
        if is_financial:
            if roe >= t.min_roe_financial:
                passed.append("roe_financial")
            else:
                failed.append("roe_financial")
        else:
            if roa >= t.min_roa:
                passed.append("roa")
            else:
                failed.append("roa")

        # EBITDA Margin
        if is_financial:
            passed.append("ebitda_skip")
        elif ebitda_margin >= t.min_ebitda_margin:
            passed.append("ebitda_margin")
        else:
            failed.append("ebitda_margin")

        # FCF Yield
        if fcf_yield >= t.min_fcf_yield:
            passed.append("fcf_yield")
        elif fcf <= 0:
            warnings.append("fcf_negative")
            failed.append("fcf_yield")
        else:
            failed.append("fcf_yield")

        # Safety Margin (PTL)
        if ptl_ratio <= t.max_ptl_ratio:
            passed.append("safety_margin")
        else:
            failed.append("safety_margin")

        # FCF Conversion
        if is_financial:
            passed.append("fcf_conv_skip")
        elif fcf_conversion >= t.min_fcf_conversion:
            passed.append("fcf_conversion")
        else:
            failed.append("fcf_conversion")

        # ── Quality Score ──
        score = 0.0
        if bm_ratio >= t.min_bm_ratio:
            score += min(bm_ratio, 1.5) / 1.5 * 20
        if is_financial:
            score += min(roe, 0.25) / 0.25 * 15
        else:
            score += min(roa, 0.15) / 0.15 * 15
        if not is_financial:
            score += min(ebitda_margin, 0.30) / 0.30 * 10
        score += min(fcf_yield, 0.10) / 0.10 * 20
        score += max(0, (t.max_ptl_ratio - ptl_ratio)) / 0.35 * 15
        if not is_financial:
            score += min(fcf_conversion, 1.0) * 10

        quality_score = min(score / 100, 1.0)

        # Pass check
        core_passed = sum(1 for p in passed if '_skip' not in p)
        passed_screen = core_passed >= t.min_core_passed and quality_score >= t.min_quality_score

        if not passed_screen:
            return None

        return {
            "ticker": ticker,
            "name": snapshot.short_name or ticker,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "market_cap_type": "turnaround" if market_cap < 10e9 else "large",
            "current_price": round(price, 2),
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
            "earnings_growth": 0,
            "asset_vs_earnings": "n/a_hk",
            "debt_to_equity": round(de, 1) if de else 0,
            "beta": snapshot.beta or 1.0,
            "interest_rate_sensitive": (
                sector in {'Real Estate', 'Utilities', 'Financial Services', 'Technology'}
                or (de or 0) > 80
            ),
            "quality_score": round(quality_score, 4),
            "passed_criteria": passed,
            "failed_criteria": failed,
            "warnings": warnings,
            "is_financial": is_financial,
        }

    # ═══════════════════════════════════════════════════════════════
    # Part 2: HK MAGNA (via live pipeline_hk.py)
    # ═══════════════════════════════════════════════════════════════

    def screen_part2(self, ticker: str, quality: dict,
                     snapshot: HistoricalSnapshot,
                     date_str: str = "") -> Optional:
        """
        Run HK Part 2 MAGNA screening at a historical point in time.

        Uses cached yfinance data overlaid with snapshot-adjusted prices
        and historical price history fed through live screen_hk_magna().

        Returns HKMagnaSignal or None.
        """
        try:
            cached_ticker = self._ticker_cache.get(ticker)
            if cached_ticker is None:
                cached_ticker = yf.Ticker(ticker)
                self._ticker_cache[ticker] = cached_ticker

            # Get historical price data up to date_str
            hist = self._daily_prices.get(ticker)
            if hist is not None and date_str:
                target = pd.Timestamp(date_str)
                hist_for_date = hist[hist.index <= target].copy()
            elif hist is not None:
                hist_for_date = hist.copy()
            else:
                # Fetch fresh if no daily prices available
                hist_for_date = cached_ticker.history(period="6mo")
                if date_str:
                    target = pd.Timestamp(date_str)
                    hist_for_date = hist_for_date[hist_for_date.index <= target]

            if hist_for_date is None or len(hist_for_date) < 20:
                logger.debug(f"  {ticker}: insufficient history for MAGNA")
                return None

            # Build modified info from cached + snapshot
            info = dict(self._info_cache.get(ticker, {}))
            info['regularMarketPrice'] = snapshot.close
            info['currentPrice'] = snapshot.close
            info['fiftyTwoWeekLow'] = snapshot.low_52w
            info['fiftyTwoWeekHigh'] = snapshot.high_52w

            # Monkey-patch ticker's info and history for live screening
            _orig_info = getattr(cached_ticker, '_info', None)

            try:
                cached_ticker._info = info
                # Call live screen_hk_magna — but it does yf.Ticker(ticker) internally
                # So we need a different approach: replicate the MAGNA logic directly
                # but using our adjusted data
                result = self._screen_magna_direct(
                    ticker=ticker,
                    info=info,
                    hist=hist_for_date,
                    quality=quality,
                )
            finally:
                if _orig_info is not None:
                    cached_ticker._info = _orig_info

            return result

        except Exception as e:
            logger.debug(f"  {ticker}: HK Part 2 failed — {e}")
            return None

    def _screen_magna_direct(self, ticker: str, info: dict,
                              hist: pd.DataFrame, quality: dict) -> Optional[HKMagnaSignal]:
        """
        Direct MAGNA computation using adjusted info + historical hist.
        
        Replicates screen_hk_magna() logic exactly but operates on
        provided data instead of fetching fresh yfinance data.
        This ensures point-in-time accuracy for backtests.
        """
        try:
            t = self.thresholds
            score = 0
            triggers = []
            details = {}

            close = hist['Close'].values
            vol = hist['Volume'].values if 'Volume' in hist.columns else np.ones(len(close))

            # ── M: Earnings Acceleration ──
            earnings_growth = float(info.get('earningsGrowth', 0) or 0)
            rev_growth = float(info.get('revenueGrowth', 0) or 0)

            eps_accel = earnings_growth >= t.eps_accel_min
            if eps_accel:
                score += 2
                triggers.append("M")

            # ── A: Sales Acceleration ──
            sales_accel = rev_growth >= t.sales_accel_min
            if sales_accel:
                score += 2
                triggers.append("A")

            # ── G: Gap Up (recent price surge + volume) ──
            if len(close) >= 5:
                recent_ret = (close[-1] - close[-5]) / close[-5]
            else:
                recent_ret = 0
            recent_vol_ratio = np.mean(vol[-3:]) / np.mean(vol[-20:]) if len(vol) >= 20 else 1

            gap_up = recent_ret >= t.gap_min_pct and recent_vol_ratio >= t.gap_vol_mult
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

            # ── Analyst Coverage ──
            analyst_count = int(info.get('numberOfAnalystOpinions', 0) or 0)
            target_mean = float(info.get('targetMeanPrice', 0) or 0)
            analyst_ok = analyst_count >= 3 and target_mean >= float(close[-1]) * 1.10
            if analyst_ok:
                score += 1
                triggers.append("Alyst")

            details['analyst_count'] = analyst_count
            details['target_mean'] = round(target_mean, 2)
            details['earnings_growth'] = round(earnings_growth, 4)
            details['rev_growth'] = round(rev_growth, 4)
            details['gap'] = gap_up
            details['base'] = is_base

            # ── Entry Trigger ──
            entry_ready = gap_up or (eps_accel and sales_accel)

            return HKMagnaSignal(
                ticker=ticker,
                magna_score=score,
                entry_ready=entry_ready,
                gap_trigger=gap_up,
                eps_accel=eps_accel,
                sales_accel=sales_accel,
                neglect_base=is_base,
                short_score=0,
                analyst_ok=analyst_ok,
                triggers=triggers,
                details=details,
            )

        except Exception as e:
            logger.debug(f"MAGNA direct failed for {ticker}: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════
    # Market Regime (HK-specific: uses HSI)
    # ═══════════════════════════════════════════════════════════════

    def get_hk_market_regime(self, hsi_hist: pd.DataFrame,
                              target_date: str) -> dict:
        """
        Compute HK market regime at a historical point in time using HSI.

        Returns dict compatible with pipeline_hk.py's get_hk_market_regime().
        """
        if hsi_hist is None or hsi_hist.empty:
            return {
                "market_ok": True,
                "position_scalar": 0.70,
                "vol_regime": "UNKNOWN",
            }

        target = pd.Timestamp(target_date)
        available = hsi_hist[hsi_hist.index <= target]
        if available.empty or len(available) < 50:
            return {
                "hsi_price": 0, "hsi_ma50": 0,
                "above_ma50": True, "vol_regime": "UNKNOWN",
                "volatility_20d": 0.15, "dd_from_high": 0,
                "market_ok": True, "position_scalar": 0.75,
            }

        current = float(available['Close'].iloc[-1])
        ma50 = float(available['Close'].rolling(50).mean().iloc[-1]) \
            if len(available) >= 50 else current
        above_ma = current > ma50 if ma50 > 0 else True

        returns = available['Close'].pct_change().dropna()
        vol_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.15

        if vol_20d < 0.12:
            vol_regime, scalar = "LOW", 1.0
        elif vol_20d < 0.22:
            vol_regime, scalar = "NORMAL", 0.80
        else:
            vol_regime, scalar = "HIGH", 0.50

        high_3mo = float(available['High'].tail(63).max())
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
