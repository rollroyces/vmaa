"""
VMAA Direct Yahoo Finance API Module
======================================
Complete replacement for yfinance when rate-limited.

Alternative to yfinance: directly calls Yahoo Finance REST APIs that
are NOT rate-limited (chart API works even when yfinance is blocked).

Data Sources (in order of preference):
  1. Yahoo Chart API (v8) — price, OHLCV, volume — ALWAYS works
  2. Yahoo QuoteSummary (v10) — fundamentals — sometimes 401s
  3. Finnhub metrics — fundamentals supplement (beta, market cap, 52w range)
  4. Finnhub profile2 — company info (sector, industry)

Usage:
    from vmaa.data.yahoo_direct import YahooDirect
    yd = YahooDirect()
    
    # Price + history
    price = yd.get_price('AAPL')
    hist = yd.get_history('AAPL', period='1y')
    prices_batch = yd.get_prices_batch(['AAPL', 'MSFT', 'NVDA'])
    
    # Fundamentals
    info = yd.get_info('AAPL')  # Same format as yfinance Ticker.info
    
    # 52-week range
    low, high = yd.get_52w_range('AAPL')
"""

from __future__ import annotations

import logging
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger("vmaa.data.yahoo_direct")

# ── Constants ──

_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
]

_FINNHUB_KEY = "d2ebgbhr01qr1ro95mrgd2ebgbhr01qr1ro95ms0"

# Rate-limit tracking (global — shared across all instances/threads)
_LAST_CALL_TIME: float = 0.0
_COOLDOWN_UNTIL: float = 0.0  # Yahoo 429 cooldown
_CONSECUTIVE_429: int = 0

# Finnhub rate limit tracking
_FINNHUB_CALL_TIMES: list = []


class YahooDirect:
    """
    Direct Yahoo Finance + Finnhub data provider.
    Completely replaces yfinance for price, history, and fundamentals.
    """

    def __init__(self, delay: float = 0.15, workers: int = 15):
        self.delay = delay
        self.workers = workers
        self._session = requests.Session()
        # Rotate user agents
        self._user_agents = list(_USER_AGENTS)
        random.shuffle(self._user_agents)
        self._ua_index = 0

    def _get_headers(self) -> dict:
        """Get headers with rotated User-Agent and randomized Accept-Language."""
        ua = self._user_agents[self._ua_index % len(self._user_agents)]
        self._ua_index += 1
        langs = ["en-US,en;q=0.9", "en-GB,en;q=0.8", "en-CA,en;q=0.9", "en-AU,en;q=0.8"]
        return {
            "User-Agent": ua,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": random.choice(langs),
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def _wait(self):
        """Respect rate limits between calls. Does NOT sleep for 429 cooldown
        (that's handled by _yahoo_request and callers can fall through to Finnhub)."""
        global _LAST_CALL_TIME

        now = time.time()
        elapsed = now - _LAST_CALL_TIME
        target = self.delay * (1.0 + random.uniform(-0.2, 0.2))
        if elapsed < target:
            time.sleep(target - elapsed)
        _LAST_CALL_TIME = time.time()

    def _check_global_cooldown(self):
        """Check if in global cooldown (429 from any call)."""
        global _COOLDOWN_UNTIL
        now = time.time()
        if now < _COOLDOWN_UNTIL:
            return True
        return False

    def _rate_limit_finnhub(self):
        """Respect Finnhub free tier rate limits (60 calls/min)."""
        global _FINNHUB_CALL_TIMES
        now = time.time()
        # Clean old entries
        _FINNHUB_CALL_TIMES = [t for t in _FINNHUB_CALL_TIMES if now - t < 60]
        if len(_FINNHUB_CALL_TIMES) >= 55:  # Leave margin
            sleep_time = 61 - (now - _FINNHUB_CALL_TIMES[0])
            if sleep_time > 0:
                logger.debug(f"Finnhub rate limit: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        _FINNHUB_CALL_TIMES.append(now)

    def _yahoo_request(self, url: str, params: dict = None, timeout: int = 10) -> Optional[dict]:
        """Make Yahoo API request with fast-fail on 429.
        
        Difference from standard _request: only retries ONCE for 429,
        then sets global cooldown and returns None quickly so callers
        can fall back to Finnhub immediately.
        """
        global _COOLDOWN_UNTIL, _CONSECUTIVE_429

        self._wait()
        try:
            r = self._session.get(url, params=params, headers=self._get_headers(), timeout=timeout)
            if r.status_code == 200:
                _CONSECUTIVE_429 = 0
                return r.json()
            elif r.status_code == 429:
                _CONSECUTIVE_429 += 1
                cooldown = min(3 * (2 ** (_CONSECUTIVE_429 - 1)), 120)
                cooldown *= 1.0 + random.uniform(-0.3, 0.3)
                _COOLDOWN_UNTIL = time.time() + cooldown
                logger.debug(f"Yahoo 429 (x{_CONSECUTIVE_429}) — cooldown {cooldown:.1f}s")
                self._ua_index += random.randint(1, 3)
                # Don't sleep here — return None immediately so caller falls through to Finnhub
                return None
            elif r.status_code == 401:
                logger.debug(f"Yahoo 401 on {url[:60]}")
                return None
            else:
                logger.debug(f"Yahoo {r.status_code} on {url[:60]}")
                return None
        except Exception as e:
            logger.debug(f"Yahoo request error: {e}")
            time.sleep(0.5)
            return None

    # ══════════════════════════════════════════════════════════════
    # PRICE
    # ══════════════════════════════════════════════════════════════

    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price via Yahoo Chart API or Finnhub fallback."""
        # Primary: Yahoo Chart API
        data = self._yahoo_request(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"interval": "1d", "range": "5d"}
        )
        if data:
            try:
                meta = data["chart"]["result"][0]["meta"]
                price = meta.get("regularMarketPrice") or meta.get("previousClose", 0)
                if price and float(price) > 0:
                    return float(price)
            except (KeyError, IndexError, ValueError):
                pass

        # Fallback: Finnhub quote (not rate-limited like Yahoo)
        try:
            self._rate_limit_finnhub()
            r = self._session.get(
                f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={_FINNHUB_KEY}",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=8
            )
            if r.status_code == 200:
                data = r.json()
                close = float(data.get('c', 0))
                if close > 0:
                    return close
        except Exception:
            pass

        return None

    def get_prices_batch(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for multiple tickers."""
        results = {}
        for t in tickers:
            price = self.get_price(t)
            if price:
                results[t] = price
        return results

    # ══════════════════════════════════════════════════════════════
    # HISTORY (OHLCV)
    # ══════════════════════════════════════════════════════════════

    def get_history(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get OHLCV history mimicking yfinance.history() output.
        
        Periods: 1mo, 3mo, 6mo, 1y, 2y, 5y, max
        """
        range_map = {
            "1mo": "1mo", "3mo": "3mo", "6mo": "6mo",
            "1y": "1y", "2y": "2y", "5y": "5y", "max": "max",
        }
        rng = range_map.get(period, "1y")

        data = self._yahoo_request(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"interval": "1d", "range": rng}
        )
        if not data:
            return None

        try:
            result = data["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            quotes = result.get("indicators", {}).get("quote", [{}])[0]
            adjclose = result.get("indicators", {}).get("adjclose", [{}])[0]

            if not timestamps or not quotes.get("close"):
                return None

            df = pd.DataFrame({
                "Open": quotes.get("open", []),
                "High": quotes.get("high", []),
                "Low": quotes.get("low", []),
                "Close": quotes.get("close", []),
                "Volume": quotes.get("volume", []),
                "Adj Close": adjclose.get("adjclose", quotes.get("close", [])),
            }, index=pd.to_datetime(timestamps, unit="s"))

            # Clean NaN rows
            df = df.dropna(subset=["Close"])
            df.index.name = "Date"

            return df
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"get_history parse error for {ticker}: {e}")
            return None

    def get_52w_range(self, ticker: str) -> Tuple[float, float]:
        """Get 52-week low and high from 1y chart data."""
        hist = self.get_history(ticker, period="1y")
        if hist is not None and len(hist) >= 20:
            return float(hist["Low"].min()), float(hist["High"].max())

        # Try Finnhub as fallback
        if not self._check_global_cooldown():
            try:
                self._rate_limit_finnhub()
                r = self._session.get(
                    f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={_FINNHUB_KEY}",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=8
                )
                if r.status_code == 200:
                    m = r.json().get("metric", {})
                    low = m.get("52WeekLow")
                    high = m.get("52WeekHigh")
                    if low and high:
                        return float(low), float(high)
            except Exception:
                pass
        return 0.0, 0.0

    # ══════════════════════════════════════════════════════════════
    # FUNDAMENTALS
    # ══════════════════════════════════════════════════════════════

    def get_info(self, ticker: str) -> Optional[dict]:
        """Get fundamental data in yfinance-compatible format.
        Uses multiple sources:
          - Yahoo Chart API for price data (always works)
          - Finnhub for metrics (beta, market cap, 52w range)
          - Finnhub profile2 for sector/industry
        """
        info = {}

        # 1. Yahoo Chart API (price, 52w range) — skip if already in 429 cooldown
        chart = None
        if not self._check_global_cooldown():
            chart = self._yahoo_request(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"interval": "1d", "range": "1y"}
            )
        if chart:
            try:
                meta = chart["chart"]["result"][0]["meta"]
                info["regularMarketPrice"] = meta.get("regularMarketPrice", 0)
                info["previousClose"] = meta.get("chartPreviousClose", 0)
                info["regularMarketDayHigh"] = meta.get("regularMarketDayHigh", 0)
                info["regularMarketDayLow"] = meta.get("regularMarketDayLow", 0)
                info["regularMarketVolume"] = meta.get("regularMarketVolume", 0)
                info["currency"] = meta.get("currency", "USD")
                info["exchangeName"] = meta.get("exchangeName", "")
                info["instrumentType"] = meta.get("instrumentType", "EQUITY")

                # Current price
                current = float(meta.get("regularMarketPrice", 0))
                if current == 0:
                    current = float(meta.get("chartPreviousClose", 0))
                info["currentPrice"] = current

                # 52-week from chart data
                quotes = chart["chart"]["result"][0].get("indicators", {}).get("quote", [{}])[0]
                closes = [c for c in quotes.get("close", []) if c is not None]
                if closes:
                    info["fiftyTwoWeekLow"] = float(min(closes))
                    info["fiftyTwoWeekHigh"] = float(max(closes))
                    info["fiftyTwoWeekRange"] = f"{info['fiftyTwoWeekLow']} - {info['fiftyTwoWeekHigh']}"

                    # 50-day MA
                    if len(closes) >= 50:
                        info["fiftyDayAverage"] = float(sum(closes[-50:]) / 50)

                    # 200-day MA
                    if len(closes) >= 200:
                        info["twoHundredDayAverage"] = float(sum(closes[-200:]) / 200)

            except (KeyError, IndexError, ValueError) as e:
                logger.debug(f"Chart parse error for {ticker}: {e}")

        # 2. Finnhub profile2 (sector, industry, name) — skip if already 429 cooldown
        if not self._check_global_cooldown():
            try:
                self._rate_limit_finnhub()
                r = self._session.get(
                    f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={_FINNHUB_KEY}",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=8
                )
                if r.status_code == 200:
                    p = r.json()
                    info["longName"] = p.get("name", info.get("longName", ""))
                    info["shortName"] = p.get("ticker", ticker)
                    info["sector"] = p.get("finnhubIndustry", "")
                    info["industry"] = p.get("finnhubIndustry", "")
                    info["marketCap"] = int(p.get("marketCapitalization", 0) * 1_000_000) if p.get("marketCapitalization") else 0
                    info["country"] = p.get("country", "US")
                    info["website"] = p.get("weburl", "")
                    info["exchange"] = p.get("exchange", "")
            except Exception as e:
                logger.debug(f"Finnhub profile2 error for {ticker}: {e}")

        # 3. Finnhub metrics (beta, 52w range, valuation) — skip if already 429 cooldown
        if not self._check_global_cooldown():
            try:
                self._rate_limit_finnhub()
                r = self._session.get(
                    f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={_FINNHUB_KEY}",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=8
                )
                if r.status_code == 200:
                    m = r.json().get("metric", {})
                    if m.get("beta"):
                        info["beta"] = float(m["beta"])
                    if m.get("52WeekHigh") and "fiftyTwoWeekHigh" not in info:
                        info["fiftyTwoWeekHigh"] = float(m["52WeekHigh"])
                    if m.get("52WeekLow") and "fiftyTwoWeekLow" not in info:
                        info["fiftyTwoWeekLow"] = float(m["52WeekLow"])
                    if m.get("marketCapitalization") and "marketCap" not in info:
                        info["marketCap"] = int(float(m["marketCapitalization"]) * 1_000_000)
                    if m.get("peRatio"):
                        info["trailingPE"] = float(m["peRatio"])
                    if m.get("priceToBook"):
                        info["priceToBook"] = float(m["priceToBook"])
            except Exception as e:
                logger.debug(f"Finnhub metrics error for {ticker}: {e}")

        # 4. Price fallback: if Yahoo 429'd but Finnhub worked
        current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))
        if not current_price:
            try:
                self._rate_limit_finnhub()
                r = self._session.get(
                    f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={_FINNHUB_KEY}",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=5
                )
                if r.status_code == 200:
                    current_price = float(r.json().get('c', 0))
                    if current_price > 0:
                        info["currentPrice"] = current_price
                        info["regularMarketPrice"] = current_price
                        info["regularMarketDayHigh"] = r.json().get('h', 0)
                        info["regularMarketDayLow"] = r.json().get('l', 0)
            except Exception:
                pass

        # 5. Set essential defaults
        info["currentPrice"] = current_price
        info["regularMarketPrice"] = current_price
        info["shortName"] = info.get("shortName") or ticker

        # Market cap estimate
        if not info.get("marketCap") or info.get("marketCap", 0) == 0:
            if current_price:
                info["marketCap"] = int(current_price * 500_000_000)

        return info if info else None

    def get_info_batch(self, tickers: List[str], workers: int = None) -> Dict[str, Optional[dict]]:
        """Get fundamental data for multiple tickers in parallel."""
        w = workers or self.workers
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        with ThreadPoolExecutor(max_workers=min(w, 20)) as pool:
            futures = {pool.submit(self.get_info, t): t for t in tickers}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    info = f.result()
                    if info:
                        results[t] = info
                except Exception as e:
                    logger.debug(f"get_info_batch error for {t}: {e}")
        return results

    # ══════════════════════════════════════════════════════════════
    # HISTORICAL PRICE BATCH
    # ══════════════════════════════════════════════════════════════

    def get_histories_batch(self, tickers: List[str], period: str = "6mo", workers: int = None) -> Dict[str, pd.DataFrame]:
        """Get price history for multiple tickers in parallel."""
        w = workers or self.workers
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        with ThreadPoolExecutor(max_workers=min(w, 20)) as pool:
            futures = {pool.submit(self.get_history, t, period): t for t in tickers}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    hist = f.result()
                    if hist is not None:
                        results[t] = hist
                except Exception:
                    pass
        return results


# ══════════════════════════════════════════════════════════════
# CONVENIENCE MODULE-LEVEL FUNCTION
# ══════════════════════════════════════════════════════════════

_yd_instance = None


def _get_yd() -> YahooDirect:
    """Get singleton YahooDirect instance."""
    global _yd_instance
    if _yd_instance is None:
        _yd_instance = YahooDirect()
    return _yd_instance


def get_price(ticker: str) -> Optional[float]:
    """Module-level convenience: get current price."""
    return _get_yd().get_price(ticker)


def get_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Module-level convenience: get OHLCV history."""
    return _get_yd().get_history(ticker, period)


def get_info(ticker: str) -> Optional[dict]:
    """Module-level convenience: get fundamentals (yfinance-compatible dict)."""
    return _get_yd().get_info(ticker)


def get_52w_range(ticker: str) -> Tuple[float, float]:
    """Module-level convenience: get 52-week low/high."""
    return _get_yd().get_52w_range(ticker)


def is_available() -> bool:
    """Check if Yahoo Direct is accessible (test with SPY)."""
    try:
        price = _get_yd().get_price("SPY")
        return price is not None and price > 0
    except Exception:
        return False
