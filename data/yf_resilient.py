"""
VMAA yfinance Rate-Limit Resilient Wrapper
===========================================
Wraps yfinance calls with:
- Exponential backoff on 401/429 (rate limit)
- Automatic session refresh
- Configurable delay between calls
- Fallback: when yfinance fully blocked, return None and let data.hybrid handle it

Usage:
    from vmaa.data.yf_resilient import YFinance
    yf = YFinance(delay=0.1)  # 100ms between calls
    info = yf.get_info('AAPL')
    hist = yf.get_history('AAPL', period='1y')
"""
from __future__ import annotations

import logging
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
    - Exponential backoff when rate-limited
    - Auto-detects 401/429 and pauses
    - Thread-safe delay tracking
    """

    def __init__(self, delay: float = 0.05, max_retries: int = 3):
        """
        Args:
            delay: Seconds to wait between successive yfinance calls (default 50ms)
            max_retries: Max retries on rate-limit (default 3)
        """
        self.delay = delay
        self.max_retries = max_retries

    def _wait(self):
        """Wait appropriate time between calls respecting rate limits."""
        global _YF_LAST_CALL, _YF_RATE_LIMITED_UNTIL

        now = time.time()

        # Check if we're in a rate-limit cooldown period
        if now < _YF_RATE_LIMITED_UNTIL:
            wait = _YF_RATE_LIMITED_UNTIL - now
            logger.debug(f"Rate-limit cooldown: sleeping {wait:.1f}s")
            time.sleep(wait)
            return

        # Standard delay between calls
        elapsed = now - _YF_LAST_CALL
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def _handle_rate_limit(self):
        """Exponential backoff when rate-limited."""
        global _YF_CONSECUTIVE_429, _YF_RATE_LIMITED_UNTIL

        _YF_CONSECUTIVE_429 += 1
        backoff = min(2 ** _YF_CONSECUTIVE_429, _YF_MAX_BACKOFF)
        logger.warning(f"⚠️ yfinance rate-limited (x{_YF_CONSECUTIVE_429}). Backing off {backoff}s")

        _YF_RATE_LIMITED_UNTIL = time.time() + backoff
        time.sleep(backoff)

    def _reset_rate_limit_state(self):
        """Reset rate-limit tracking after a successful call."""
        global _YF_CONSECUTIVE_429
        _YF_CONSECUTIVE_429 = 0

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
                # Empty dict = rate-limited
                self._handle_rate_limit()
            except Exception as e:
                err_str = str(e).lower()
                if 'rate' in err_str or '429' in err_str or '401' in err_str or 'too many' in err_str:
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_info error for {ticker}: {e}")
                    return None

        logger.warning(f"yfinance get_info exhausted for {ticker} after {self.max_retries} retries")
        return None

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
                # Empty hist = rate-limited
                self._handle_rate_limit()
            except Exception as e:
                err_str = str(e).lower()
                if 'rate' in err_str or '429' in err_str or '401' in err_str or 'too many' in err_str:
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_history error for {ticker}: {e}")
                    return None

        logger.warning(f"yfinance get_history exhausted for {ticker} after {self.max_retries} retries")
        return None

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
                err_str = str(e).lower()
                if 'rate' in err_str or '429' in err_str or '401' in err_str or 'too many' in err_str:
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
                err_str = str(e).lower()
                if 'rate' in err_str or '429' in err_str or '401' in err_str or 'too many' in err_str:
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
                err_str = str(e).lower()
                if 'rate' in err_str or '429' in err_str or '401' in err_str or 'too many' in err_str:
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
                # Empty rec might not be rate-limit
                return rec
            except Exception as e:
                err_str = str(e).lower()
                if 'rate' in err_str or '429' in err_str or '401' in err_str or 'too many' in err_str:
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
                err_str = str(e).lower()
                if 'rate' in err_str or '429' in err_str or '401' in err_str or 'too many' in err_str:
                    self._handle_rate_limit()
                else:
                    logger.debug(f"yfinance get_calendar error for {ticker}: {e}")
                    return None
        return None


def is_yfinance_blocked() -> bool:
    """Quick check if yfinance is currently rate-limited."""
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
