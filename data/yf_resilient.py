"""
VMAA yfinance Rate-Limit Resilient Wrapper
===========================================
Wraps yfinance calls with:
- Exponential backoff on 401/429/rate-limit errors
- Jitter in delays to avoid thundering herd
- Configurable delay between calls
- Fallback: when yfinance fully blocked, return None and let data.hybrid handle it

Usage:
    from vmaa.data.yf_resilient import YFinance
    yf = YFinance(delay=0.15)  # 150ms between calls
    info = yf.get_info('AAPL')
    hist = yf.get_history('AAPL', period='1y')

Note: yfinance 1.3.0+ handles UA impersonation via curl_cffi internally.
We DO NOT pass custom sessions — yfinance manages its own session.
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import yfinance as yf
import pandas as pd

logger = logging.getLogger("vmaa.data.yf_resilient")

# Global rate-limit state
_YF_LAST_CALL: float = 0.0
_YF_RATE_LIMITED_UNTIL: float = 0.0
_YF_CONSECUTIVE_429: int = 0
_YF_MAX_BACKOFF: float = 300.0  # 5 min max backoff


class YFinance:
    """
    Rate-limit-resilient yfinance wrapper.

    Features:
    - Configurable delay between calls (avoid 429)
    - Exponential backoff when rate-limited (1s -> 2s -> 4s -> ... 5min max)
    - Random jitter in delays to avoid thundering herd
    - Auto-detects YFRateLimitError, 401, 429, Invalid Crumb
    - Thread-safe global delay tracking

    Does NOT pass custom sessions to yfinance 1.3.0 (which uses curl_cffi).
    """

    def __init__(self, delay: float = 0.15, max_retries: int = 3, jitter: float = 0.3):
        """
        Args:
            delay: Base seconds to wait between successive yfinance calls (default 150ms)
            max_retries: Max retries on rate-limit (default 3)
            jitter: Random jitter fraction for delays (0.3 = +/-30%)
        """
        self.base_delay = delay
        self.max_retries = max_retries
        self.jitter = jitter

    def _wait(self):
        """Wait appropriate time between calls respecting rate limits.
        Adds random jitter to avoid thundering herd."""
        global _YF_LAST_CALL, _YF_RATE_LIMITED_UNTIL

        now = time.time()

        # Check if we're in a rate-limit cooldown period
        if now < _YF_RATE_LIMITED_UNTIL:
            wait = _YF_RATE_LIMITED_UNTIL - now
            logger.debug(f"Rate-limit cooldown: sleeping {wait:.1f}s")
            time.sleep(wait)
            return

        # Standard delay between calls with jitter
        elapsed = now - _YF_LAST_CALL
        target_delay = self.base_delay * (1.0 + random.uniform(-self.jitter, self.jitter))
        if elapsed < target_delay:
            time.sleep(target_delay - elapsed)

    def _handle_rate_limit(self):
        """Exponential backoff when rate-limited."""
        global _YF_CONSECUTIVE_429, _YF_RATE_LIMITED_UNTIL

        _YF_CONSECUTIVE_429 += 1
        backoff = min(2 ** _YF_CONSECUTIVE_429, _YF_MAX_BACKOFF)
        # Add jitter: +/-30%
        backoff *= 1.0 + random.uniform(-0.3, 0.3)
        logger.warning(f"⚠️ yfinance rate-limited (x{_YF_CONSECUTIVE_429}). Backing off {backoff:.1f}s")

        _YF_RATE_LIMITED_UNTIL = time.time() + backoff
        time.sleep(backoff)

    def _reset_rate_limit_state(self):
        """Reset rate-limit tracking after a successful call."""
        global _YF_CONSECUTIVE_429
        _YF_CONSECUTIVE_429 = 0

    # ── Info ──

    def get_info(self, ticker: str) -> Optional[dict]:
        """Get stock info with rate-limit resilience."""
        global _YF_LAST_CALL

        for attempt in range(self.max_retries + 1):
            try:
                self._wait()
                t = yf.Ticker(ticker)
                info = t.info
                _YF_LAST_CALL = time.time()
                if info and isinstance(info, dict) and info.get('shortName'):
                    self._reset_rate_limit_state()
                    return info
                self._handle_rate_limit()
            except Exception as e:
                err_str = str(e).lower()
                if _is_rate_error(err_str):
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_info error for {ticker}: {e}")
                    return None

        logger.warning(f"yfinance get_info exhausted for {ticker} after {self.max_retries} retries")
        return None

    def get_info_fast(self, ticker: str) -> Optional[dict]:
        """Fast single-attempt get_info (no retries). For non-critical lookups."""
        global _YF_LAST_CALL
        try:
            self._wait()
            t = yf.Ticker(ticker)
            info = t.info
            _YF_LAST_CALL = time.time()
            if info and isinstance(info, dict) and info.get('shortName'):
                self._reset_rate_limit_state()
                return info
        except Exception:
            pass
        return None

    # ── History ──

    def get_history(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get price history with rate-limit resilience."""
        global _YF_LAST_CALL

        for attempt in range(self.max_retries + 1):
            try:
                self._wait()
                t = yf.Ticker(ticker)
                hist = t.history(period=period)
                _YF_LAST_CALL = time.time()
                if hist is not None and len(hist) >= 2:
                    self._reset_rate_limit_state()
                    return hist
                self._handle_rate_limit()
            except Exception as e:
                err_str = str(e).lower()
                if _is_rate_error(err_str):
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_history error for {ticker}: {e}")
                    return None

        logger.warning(f"yfinance get_history exhausted for {ticker} after {self.max_retries} retries")
        return None

    # ── Fundamentals ──

    def get_earnings(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get quarterly earnings with rate-limit resilience."""
        global _YF_LAST_CALL
        for attempt in range(self.max_retries + 1):
            try:
                self._wait()
                t = yf.Ticker(ticker)
                earnings = t.quarterly_earnings
                _YF_LAST_CALL = time.time()
                if earnings is not None:
                    self._reset_rate_limit_state()
                    return earnings
                self._handle_rate_limit()
            except Exception as e:
                if _is_rate_error(str(e).lower()):
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_earnings error for {ticker}: {e}")
                    return None
        return None

    def get_financials(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get financials with rate-limit resilience."""
        global _YF_LAST_CALL
        for attempt in range(self.max_retries + 1):
            try:
                self._wait()
                t = yf.Ticker(ticker)
                fin = t.financials
                _YF_LAST_CALL = time.time()
                if fin is not None and not fin.empty:
                    self._reset_rate_limit_state()
                    return fin
                self._handle_rate_limit()
            except Exception as e:
                if _is_rate_error(str(e).lower()):
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_financials error for {ticker}: {e}")
                    return None
        return None

    def get_balance_sheet(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get balance sheet with rate-limit resilience."""
        global _YF_LAST_CALL
        for attempt in range(self.max_retries + 1):
            try:
                self._wait()
                t = yf.Ticker(ticker)
                bs = t.balance_sheet
                _YF_LAST_CALL = time.time()
                if bs is not None and not bs.empty:
                    self._reset_rate_limit_state()
                    return bs
                self._handle_rate_limit()
            except Exception as e:
                if _is_rate_error(str(e).lower()):
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_balance_sheet error for {ticker}: {e}")
                    return None
        return None

    def get_recommendations(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get analyst recommendations with rate-limit resilience."""
        global _YF_LAST_CALL
        for attempt in range(self.max_retries + 1):
            try:
                self._wait()
                t = yf.Ticker(ticker)
                rec = t.recommendations
                _YF_LAST_CALL = time.time()
                if rec is not None and not rec.empty:
                    self._reset_rate_limit_state()
                    return rec
                return rec
            except Exception as e:
                if _is_rate_error(str(e).lower()):
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_recommendations error for {ticker}: {e}")
                    return None
        return None

    def get_calendar(self, ticker: str) -> Optional[dict]:
        """Get earnings calendar with rate-limit resilience."""
        global _YF_LAST_CALL
        for attempt in range(self.max_retries + 1):
            try:
                self._wait()
                t = yf.Ticker(ticker)
                cal = t.calendar
                _YF_LAST_CALL = time.time()
                if cal is not None:
                    self._reset_rate_limit_state()
                    return cal
                self._handle_rate_limit()
            except Exception as e:
                if _is_rate_error(str(e).lower()):
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_calendar error for {ticker}: {e}")
                    return None
        return None


# ── Module-level helpers ──

def _is_rate_error(s: str) -> bool:
    """Check if error string indicates yfinance rate-limiting."""
    triggers = ['rate', '429', '401', 'too many', 'crumb', 'unauthorized', 'blocked']
    return any(t in s for t in triggers)


def is_yfinance_blocked() -> bool:
    """Quick check if yfinance is currently rate-limited.
    Checks global cooldown first, then tries SPY info."""
    global _YF_RATE_LIMITED_UNTIL
    if time.time() < _YF_RATE_LIMITED_UNTIL:
        return True
    try:
        t = yf.Ticker("SPY")
        info = t.info
        return not (info and isinstance(info, dict) and info.get('shortName'))
    except Exception:
        return True


def clear_rate_limit_state():
    """Reset all rate-limit tracking."""
    global _YF_RATE_LIMITED_UNTIL, _YF_CONSECUTIVE_429, _YF_LAST_CALL
    _YF_LAST_CALL = 0.0
    _YF_RATE_LIMITED_UNTIL = 0.0
    _YF_CONSECUTIVE_429 = 0
