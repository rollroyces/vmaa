#!/usr/bin/env python3
"""
VMAA Signal Generation
======================
Generate trading signals from technical indicators.

Signal Types:
  - BUY, SELL: Standard signals
  - STRONG_BUY, STRONG_SELL: Stronger confirmation
  - NEUTRAL: No clear signal

Multi-indicator confirmation with signal strength tracking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.technical.config import TC
from engine.technical.indicators import (
    sma, ema, macd, rsi, stochastic, kdj, williams_r, adx, cci, mfi,
    bollinger_bands, ichimoku,
    _ensure_array, _extract_columns,
)

logger = logging.getLogger("vmaa.engine.technical.signals")


# ═══════════════════════════════════════════════════════════════════
# Signal Types
# ═══════════════════════════════════════════════════════════════════

class SignalType(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class IndicatorSignal:
    """A single indicator's signal."""
    indicator: str                     # Indicator name
    signal: SignalType
    value: float                       # Indicator value at signal point
    threshold: float                   # Threshold that was crossed
    direction: str                     # "above" or "below"
    strength: int = 1                  # Individual signal strength (1-3)


@dataclass
class SignalResult:
    """Aggregated signal result for a ticker at a point in time."""
    ticker: str = ""
    date: str = ""
    latest_price: float = 0.0
    signal: SignalType = SignalType.NEUTRAL
    strength: int = 1                  # 1-5: number of confirming indicators
    indicators: List[IndicatorSignal] = field(default_factory=list)
    buy_count: int = 0
    sell_count: int = 0
    neutral_count: int = 0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "latest_price": self.latest_price,
            "signal": self.signal.value,
            "strength": self.strength,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "neutral_count": self.neutral_count,
            "indicators": [
                {
                    "indicator": s.indicator,
                    "signal": s.signal.value,
                    "value": round(s.value, 4) if not np.isnan(s.value) else None,
                    "threshold": round(s.threshold, 4) if not np.isnan(s.threshold) else None,
                    "direction": s.direction,
                }
                for s in self.indicators
            ],
            "summary": self.summary,
        }


# ═══════════════════════════════════════════════════════════════════
# Individual Signal Generators
# ═══════════════════════════════════════════════════════════════════


def _latest(arr: np.ndarray) -> float:
    """Get the latest non-NaN value from an array."""
    valid = arr[~np.isnan(arr)]
    return float(valid[-1]) if len(valid) > 0 else np.nan


def _prev(arr: np.ndarray) -> float:
    """Get the second-to-last non-NaN value from an array."""
    valid = arr[~np.isnan(arr)]
    return float(valid[-2]) if len(valid) >= 2 else np.nan


def signal_macd(close: np.ndarray) -> Optional[IndicatorSignal]:
    """Generate signal from MACD crossover.

    BUY: MACD line crosses ABOVE signal line
    SELL: MACD line crosses BELOW signal line
    """
    macd_line, signal_line, hist = macd(close, TC.macd_fast, TC.macd_slow, TC.macd_signal)

    curr_macd = _latest(macd_line)
    curr_signal = _latest(signal_line)
    curr_hist = _latest(hist)
    prev_macd = _prev(macd_line)
    prev_signal = _prev(signal_line)

    if np.isnan(curr_macd) or np.isnan(curr_signal) or np.isnan(prev_macd) or np.isnan(prev_signal):
        return None

    if prev_macd <= prev_signal and curr_macd > curr_signal:
        return IndicatorSignal(
            indicator="MACD",
            signal=SignalType.BUY,
            value=float(curr_hist),
            threshold=0.0,
            direction="above",
            strength=2,
        )

    if prev_macd >= prev_signal and curr_macd < curr_signal:
        return IndicatorSignal(
            indicator="MACD",
            signal=SignalType.SELL,
            value=float(curr_hist),
            threshold=0.0,
            direction="below",
            strength=2,
        )

    return None


def signal_rsi(close: np.ndarray) -> Optional[IndicatorSignal]:
    """Generate signal from RSI oversold/overbought.

    STRONG_BUY: RSI < strong_oversold (default 20)
    BUY: RSI < oversold (default 30)
    STRONG_SELL: RSI > strong_overbought (default 80)
    SELL: RSI > overbought (default 70)
    """
    rsi_val = rsi(close, TC.rsi_period)
    curr = _latest(rsi_val)

    if np.isnan(curr):
        return None

    if curr < TC.rsi_strong_oversold:
        return IndicatorSignal(
            indicator="RSI",
            signal=SignalType.STRONG_BUY,
            value=curr,
            threshold=TC.rsi_strong_oversold,
            direction="below",
            strength=3,
        )
    elif curr < TC.rsi_oversold:
        return IndicatorSignal(
            indicator="RSI",
            signal=SignalType.BUY,
            value=curr,
            threshold=TC.rsi_oversold,
            direction="below",
            strength=2,
        )
    elif curr > TC.rsi_strong_overbought:
        return IndicatorSignal(
            indicator="RSI",
            signal=SignalType.STRONG_SELL,
            value=curr,
            threshold=TC.rsi_strong_overbought,
            direction="above",
            strength=3,
        )
    elif curr > TC.rsi_overbought:
        return IndicatorSignal(
            indicator="RSI",
            signal=SignalType.SELL,
            value=curr,
            threshold=TC.rsi_overbought,
            direction="above",
            strength=2,
        )

    return None


def signal_bollinger(close: np.ndarray) -> Optional[IndicatorSignal]:
    """Generate signal from Bollinger Bands.

    BUY: Price at/below lower band (touch or break)
    SELL: Price at/above upper band (touch or break)
    Uses %B for detection.
    """
    _, bb_upper, bb_lower, _, bb_pct_b = bollinger_bands(close, TC.bb_period, TC.bb_num_std)

    curr_b = _latest(bb_pct_b)
    curr_close = _latest(close)
    curr_lower = _latest(bb_lower)
    curr_upper = _latest(bb_upper)

    if np.isnan(curr_b) or np.isnan(curr_lower) or np.isnan(curr_upper):
        return None

    if curr_b <= TC.bb_pct_b_oversold + 0.05:  # slight tolerance
        return IndicatorSignal(
            indicator="Bollinger",
            signal=SignalType.BUY,
            value=curr_close,
            threshold=curr_lower,
            direction="below" if curr_b <= 0 else "at",
            strength=2,
        )

    if curr_b >= TC.bb_pct_b_overbought - 0.05:
        return IndicatorSignal(
            indicator="Bollinger",
            signal=SignalType.SELL,
            value=curr_close,
            threshold=curr_upper,
            direction="above" if curr_b >= 1 else "at",
            strength=2,
        )

    return None


def signal_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[IndicatorSignal]:
    """Generate signal from KDJ crossover.

    BUY: K crosses ABOVE D (golden cross)
    SELL: K crosses BELOW D (death cross)
    Also checks J line extremes.
    """
    k, d, j = kdj(high, low, close, TC.kdj_k, TC.kdj_d, TC.kdj_j_weight)

    curr_k = _latest(k)
    curr_d = _latest(d)
    curr_j = _latest(j)
    prev_k = _prev(k)
    prev_d = _prev(d)

    if np.isnan(curr_k) or np.isnan(curr_d) or np.isnan(prev_k) or np.isnan(prev_d):
        return None

    # K crosses D
    if prev_k <= prev_d and curr_k > curr_d and curr_k < 50:  # oversold region cross = stronger
        return IndicatorSignal(
            indicator="KDJ",
            signal=SignalType.BUY,
            value=curr_k,
            threshold=curr_d,
            direction="above",
            strength=2,
        )

    if prev_k >= prev_d and curr_k < curr_d and curr_k > 50:  # overbought region cross
        return IndicatorSignal(
            indicator="KDJ",
            signal=SignalType.SELL,
            value=curr_k,
            threshold=curr_d,
            direction="below",
            strength=2,
        )

    # J line extremes (early warning)
    if not np.isnan(curr_j):
        if curr_j < 0:
            return IndicatorSignal(
                indicator="KDJ (J-line)",
                signal=SignalType.BUY,
                value=curr_j,
                threshold=0.0,
                direction="below",
                strength=1,
            )
        if curr_j > 100:
            return IndicatorSignal(
                indicator="KDJ (J-line)",
                signal=SignalType.SELL,
                value=curr_j,
                threshold=100.0,
                direction="above",
                strength=1,
            )

    return None


def signal_ma_crossover(close: np.ndarray, fast: int = 5, slow: int = 20) -> Optional[IndicatorSignal]:
    """Generate signal from Moving Average crossover.

    BUY: Fast MA crosses ABOVE Slow MA (golden cross)
    SELL: Fast MA crosses BELOW Slow MA (death cross)
    """
    fast_ma = sma(close, fast)
    slow_ma = sma(close, slow)

    curr_fast = _latest(fast_ma)
    curr_slow = _latest(slow_ma)
    prev_fast = _prev(fast_ma)
    prev_slow = _prev(slow_ma)

    if np.isnan(curr_fast) or np.isnan(curr_slow) or np.isnan(prev_fast) or np.isnan(prev_slow):
        return None

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        indicator_name = "Golden Cross" if fast == 50 and slow == 200 else f"MA({fast})/MA({slow})"
        return IndicatorSignal(
            indicator=indicator_name,
            signal=SignalType.BUY,
            value=curr_fast,
            threshold=curr_slow,
            direction="above",
            strength=3 if fast == 50 and slow == 200 else 1,
        )

    if prev_fast >= prev_slow and curr_fast < curr_slow:
        indicator_name = "Death Cross" if fast == 50 and slow == 200 else f"MA({fast})/MA({slow})"
        return IndicatorSignal(
            indicator=indicator_name,
            signal=SignalType.SELL,
            value=curr_fast,
            threshold=curr_slow,
            direction="below",
            strength=3 if fast == 50 and slow == 200 else 1,
        )

    return None


def signal_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[IndicatorSignal]:
    """Generate signal from Stochastic Oscillator.

    BUY: %K < oversold AND %K crosses above %D
    SELL: %K > overbought AND %K crosses below %D
    """
    k_arr, d_arr = stochastic(high, low, close, TC.stoch_k, TC.stoch_d)

    curr_k = _latest(k_arr)
    curr_d = _latest(d_arr)
    prev_k = _prev(k_arr)
    prev_d = _prev(d_arr)

    if np.isnan(curr_k) or np.isnan(curr_d) or np.isnan(prev_k) or np.isnan(prev_d):
        return None

    # Oversold crossover
    if curr_k < TC.stoch_oversold and prev_k <= prev_d and curr_k > curr_d:
        return IndicatorSignal(
            indicator="Stochastic",
            signal=SignalType.BUY,
            value=curr_k,
            threshold=TC.stoch_oversold,
            direction="above",
            strength=2,
        )

    # Overbought crossover
    if curr_k > TC.stoch_overbought and prev_k >= prev_d and curr_k < curr_d:
        return IndicatorSignal(
            indicator="Stochastic",
            signal=SignalType.SELL,
            value=curr_k,
            threshold=TC.stoch_overbought,
            direction="below",
            strength=2,
        )

    return None


def signal_ichimoku(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Optional[IndicatorSignal]:
    """Generate signal from Ichimoku Cloud.

    BUY: Price moves ABOVE the cloud (Senkou Span A > B)
    SELL: Price moves BELOW the cloud
    """
    ichi = ichimoku(high, low)
    tenkan = ichi["tenkan"]
    kijun = ichi["kijun"]
    senkou_a = ichi["senkou_a"]
    senkou_b = ichi["senkou_b"]

    n = len(close)
    displacement = TC.ichimoku_displacement
    if n < displacement + 1:
        return None

    curr_close = _latest(close)

    # Current cloud values (at index n-1, from senkou computed displacement ago)
    idx = n - 1
    if idx < len(senkou_a) and idx < len(senkou_b):
        cloud_a = senkou_a[idx]
        cloud_b = senkou_b[idx]
    else:
        return None

    if np.isnan(cloud_a) or np.isnan(cloud_b) or np.isnan(curr_close):
        return None

    cloud_top = max(cloud_a, cloud_b)
    cloud_bot = min(cloud_a, cloud_b)

    # Price above cloud
    if curr_close > cloud_top:
        prev_close = _prev(close)
        prev_top = cloud_top  # approximate
        if not np.isnan(prev_close) and prev_close <= cloud_top:
            return IndicatorSignal(
                indicator="Ichimoku",
                signal=SignalType.BUY,
                value=curr_close,
                threshold=cloud_top,
                direction="above",
                strength=2,
            )

    # Price below cloud
    if curr_close < cloud_bot:
        prev_close = _prev(close)
        if not np.isnan(prev_close) and prev_close >= cloud_bot:
            return IndicatorSignal(
                indicator="Ichimoku",
                signal=SignalType.SELL,
                value=curr_close,
                threshold=cloud_bot,
                direction="below",
                strength=2,
            )

    # Tenkan/Kijun cross (TK Cross)
    curr_t = _latest(tenkan)
    curr_k = _latest(kijun)
    prev_t = _prev(tenkan)
    prev_k = _prev(kijun)
    if not any(np.isnan(x) for x in [curr_t, curr_k, prev_t, prev_k]):
        if prev_t <= prev_k and curr_t > curr_k:
            return IndicatorSignal(
                indicator="Ichimoku (TK Cross)",
                signal=SignalType.BUY,
                value=curr_t,
                threshold=curr_k,
                direction="above",
                strength=1,
            )
        if prev_t >= prev_k and curr_t < curr_k:
            return IndicatorSignal(
                indicator="Ichimoku (TK Cross)",
                signal=SignalType.SELL,
                value=curr_t,
                threshold=curr_k,
                direction="below",
                strength=1,
            )

    return None


def signal_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[IndicatorSignal]:
    """Generate signal from Williams %R.

    BUY: %R < -80 (oversold)
    SELL: %R > -20 (overbought)
    """
    wr = williams_r(high, low, close, TC.williams_r_period)
    curr = _latest(wr)

    if np.isnan(curr):
        return None

    if curr < TC.williams_r_oversold:
        return IndicatorSignal(
            indicator="Williams %R",
            signal=SignalType.BUY,
            value=curr,
            threshold=TC.williams_r_oversold,
            direction="below",
            strength=1,
        )
    elif curr > TC.williams_r_overbought:
        return IndicatorSignal(
            indicator="Williams %R",
            signal=SignalType.SELL,
            value=curr,
            threshold=TC.williams_r_overbought,
            direction="above",
            strength=1,
        )

    return None


def signal_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Optional[IndicatorSignal]:
    """Generate signal from CCI.

    BUY: CCI < -100 (oversold)
    SELL: CCI > 100 (overbought)
    """
    cci_val = cci(high, low, close, TC.cci_period)
    curr = _latest(cci_val)

    if np.isnan(curr):
        return None

    if curr < TC.cci_oversold:
        return IndicatorSignal(
            indicator="CCI",
            signal=SignalType.BUY,
            value=curr,
            threshold=TC.cci_oversold,
            direction="below",
            strength=1,
        )
    elif curr > TC.cci_overbought:
        return IndicatorSignal(
            indicator="CCI",
            signal=SignalType.SELL,
            value=curr,
            threshold=TC.cci_overbought,
            direction="above",
            strength=1,
        )

    return None


def signal_mfi(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> Optional[IndicatorSignal]:
    """Generate signal from Money Flow Index.

    BUY: MFI < 20 (oversold)
    SELL: MFI > 80 (overbought)
    """
    mfi_val = mfi(high, low, close, volume, TC.mfi_period)
    curr = _latest(mfi_val)

    if np.isnan(curr):
        return None

    if curr < TC.mfi_oversold:
        return IndicatorSignal(
            indicator="MFI",
            signal=SignalType.BUY,
            value=curr,
            threshold=TC.mfi_oversold,
            direction="below",
            strength=2,
        )
    elif curr > TC.mfi_overbought:
        return IndicatorSignal(
            indicator="MFI",
            signal=SignalType.SELL,
            value=curr,
            threshold=TC.mfi_overbought,
            direction="above",
            strength=2,
        )

    return None


def signal_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
) -> Optional[IndicatorSignal]:
    """Generate signal from ADX (trend strength) and DI direction.

    BUY: +DI > -DI AND ADX > trending threshold
    SELL: -DI > +DI AND ADX > trending threshold
    STRONG: ADX > strong trend threshold
    """
    adx_arr, plus_di, minus_di, _ = adx(high, low, close, TC.adx_period)

    curr_adx = _latest(adx_arr)
    curr_pdi = _latest(plus_di)
    curr_mdi = _latest(minus_di)
    prev_pdi = _prev(plus_di)
    prev_mdi = _prev(minus_di)

    if np.isnan(curr_adx) or np.isnan(curr_pdi) or np.isnan(curr_mdi):
        return None

    if curr_adx < TC.adx_trending:
        return None  # No trend

    # Strong trend
    is_strong = curr_adx >= TC.adx_strong_trend
    base_strength = 3 if is_strong else 1

    # DI crossover
    if not np.isnan(prev_pdi) and not np.isnan(prev_mdi):
        if prev_pdi <= prev_mdi and curr_pdi > curr_mdi:
            sig_type = SignalType.STRONG_BUY if is_strong else SignalType.BUY
            return IndicatorSignal(
                indicator="ADX (DI Cross)",
                signal=sig_type,
                value=curr_adx,
                threshold=TC.adx_trending,
                direction="above",
                strength=base_strength,
            )
        if prev_pdi >= prev_mdi and curr_pdi < curr_mdi:
            sig_type = SignalType.STRONG_SELL if is_strong else SignalType.SELL
            return IndicatorSignal(
                indicator="ADX (DI Cross)",
                signal=sig_type,
                value=curr_adx,
                threshold=TC.adx_trending,
                direction="above",
                strength=base_strength,
            )

    return None


# ═══════════════════════════════════════════════════════════════════
# Signal Aggregation
# ═══════════════════════════════════════════════════════════════════

# List of all signal generators (indicator_name, function, args_spec)
_SIGNAL_GENERATORS: List[Tuple[str, callable, List[str]]] = [
    ("MACD", signal_macd, ["close"]),
    ("RSI", signal_rsi, ["close"]),
    ("Bollinger", signal_bollinger, ["close"]),
    ("KDJ", signal_kdj, ["high", "low", "close"]),
    ("Stochastic", signal_stochastic, ["high", "low", "close"]),
    ("Ichimoku", signal_ichimoku, ["high", "low", "close"]),
    ("Williams %R", signal_williams_r, ["high", "low", "close"]),
    ("CCI", signal_cci, ["high", "low", "close"]),
    ("MFI", signal_mfi, ["high", "low", "close", "volume"]),
    ("ADX", signal_adx, ["high", "low", "close"]),
    # MA Crossovers (various periods)
    ("MA(5)/MA(20)", lambda c: signal_ma_crossover(c, 5, 20), ["close"]),
    ("MA(20)/MA(50)", lambda c: signal_ma_crossover(c, 20, 50), ["close"]),
    ("MA(50)/MA(200)", lambda c: signal_ma_crossover(c, 50, 200), ["close"]),
]


def _build_data_dict(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build a data dict for signal generators."""
    return {"close": close, "high": high, "low": low, "volume": volume}


def generate_all_signals(df: pd.DataFrame, ticker: str = "") -> SignalResult:
    """Generate all available trading signals for a ticker.

    Args:
        df: DataFrame with Open/High/Low/Close/Volume columns.
        ticker: Ticker symbol for identification.

    Returns:
        SignalResult with all generated signals and aggregation.
    """
    o, h, l, c, v = _extract_columns(df)
    data = _build_data_dict(c, h, l, v)

    signals: List[IndicatorSignal] = []
    buy_count = sell_count = neutral_count = 0

    for name, gen_func, args in _SIGNAL_GENERATORS:
        try:
            func_args = [data[arg] for arg in args]
            sig = gen_func(*func_args)
            if sig is not None:
                signals.append(sig)
                if sig.signal in (SignalType.STRONG_BUY, SignalType.BUY):
                    buy_count += 1
                elif sig.signal in (SignalType.STRONG_SELL, SignalType.SELL):
                    sell_count += 1
                else:
                    neutral_count += 1
        except Exception as e:
            logger.debug(f"Signal generator '{name}' failed: {e}")
            continue

    # Aggregate signal
    net = buy_count - sell_count
    if net >= 3:
        if any(s.strength >= 3 for s in signals if s.signal in (SignalType.STRONG_BUY, SignalType.BUY)):
            overall = SignalType.STRONG_BUY
        else:
            overall = SignalType.BUY
    elif net >= 1:
        overall = SignalType.BUY
    elif net <= -3:
        if any(s.strength >= 3 for s in signals if s.signal in (SignalType.STRONG_SELL, SignalType.SELL)):
            overall = SignalType.STRONG_SELL
        else:
            overall = SignalType.SELL
    elif net <= -1:
        overall = SignalType.SELL
    else:
        overall = SignalType.NEUTRAL

    # Signal strength = number of confirming indicators (capped at 5)
    if overall == SignalType.BUY or overall == SignalType.STRONG_BUY:
        strength = min(buy_count, 5)
    elif overall == SignalType.SELL or overall == SignalType.STRONG_SELL:
        strength = min(sell_count, 5)
    else:
        strength = 1

    latest_price = float(c[~np.isnan(c)][-1]) if len(c[~np.isnan(c)]) > 0 else 0.0
    date_str = str(df.index[-1]) if hasattr(df.index[-1], 'strftime') else str(df.index[-1])

    # Summary
    buy_names = [s.indicator for s in signals if s.signal in (SignalType.BUY, SignalType.STRONG_BUY)]
    sell_names = [s.indicator for s in signals if s.signal in (SignalType.SELL, SignalType.STRONG_SELL)]
    summary_parts = []
    if buy_names:
        summary_parts.append(f"BUY: {', '.join(buy_names)}")
    if sell_names:
        summary_parts.append(f"SELL: {', '.join(sell_names)}")
    summary = f"[{overall.value} | Strength: {strength}/5] {' | '.join(summary_parts)}" if summary_parts else f"[{overall.value}] No clear signals"

    return SignalResult(
        ticker=ticker,
        date=date_str,
        latest_price=latest_price,
        signal=overall,
        strength=strength,
        indicators=signals,
        buy_count=buy_count,
        sell_count=sell_count,
        neutral_count=neutral_count,
        summary=summary,
    )


# ═══════════════════════════════════════════════════════════════════
# Signal History Tracking
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SignalHistoryEntry:
    """A historical signal record with outcome."""
    ticker: str
    date: str
    signal: SignalType
    entry_price: float
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    profitable: Optional[bool] = None
    pnl_pct: Optional[float] = None
    days_held: Optional[int] = None


class SignalTracker:
    """Track signal history and compute accuracy metrics.

    Usage:
        tracker = SignalTracker()
        tracker.record("AAPL", "2024-01-15", SignalType.BUY, 185.0)
        tracker.close("AAPL", "2024-01-30", 192.0)
        stats = tracker.stats()
    """

    def __init__(self, max_history: int = 500):
        self.history: List[SignalHistoryEntry] = []
        self.max_history = max_history
        self._open_positions: Dict[str, SignalHistoryEntry] = {}

    def record(self, ticker: str, date: str, signal: SignalType, price: float) -> None:
        """Record a new signal."""
        entry = SignalHistoryEntry(
            ticker=ticker,
            date=date,
            signal=signal,
            entry_price=price,
        )
        self.history.append(entry)
        if signal in (SignalType.BUY, SignalType.STRONG_BUY):
            self._open_positions[ticker] = entry
        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def close(self, ticker: str, date: str, price: float) -> Optional[SignalHistoryEntry]:
        """Close a position and record outcome."""
        if ticker not in self._open_positions:
            return None
        entry = self._open_positions.pop(ticker)
        entry.exit_price = price
        entry.exit_date = date
        entry.profitable = price > entry.entry_price
        entry.pnl_pct = (price - entry.entry_price) / entry.entry_price * 100.0
        return entry

    def stats(self) -> Dict[str, Any]:
        """Compute signal accuracy statistics."""
        closed = [e for e in self.history if e.profitable is not None]
        total_closed = len(closed)
        if total_closed == 0:
            return {"total_signals": len(self.history), "closed_trades": 0}

        profitable_trades = sum(1 for e in closed if e.profitable)
        win_rate = profitable_trades / total_closed if total_closed > 0 else 0.0
        avg_pnl = np.mean([e.pnl_pct for e in closed if e.pnl_pct is not None])
        avg_win = np.mean([e.pnl_pct for e in closed if e.profitable and e.pnl_pct is not None])
        avg_loss = np.mean([e.pnl_pct for e in closed if not e.profitable and e.pnl_pct is not None])

        # By signal type
        by_type = {}
        for sig_type in SignalType:
            typed = [e for e in closed if e.signal == sig_type]
            if typed:
                wins = sum(1 for e in typed if e.profitable)
                by_type[sig_type.value] = {
                    "count": len(typed),
                    "profitable": wins,
                    "win_rate": wins / len(typed) if len(typed) > 0 else 0.0,
                }

        return {
            "total_signals": len(self.history),
            "closed_trades": total_closed,
            "open_trades": len(self._open_positions),
            "profitable_trades": profitable_trades,
            "win_rate": round(win_rate * 100, 2),
            "avg_pnl_pct": round(avg_pnl, 2) if not np.isnan(avg_pnl) else 0.0,
            "avg_win_pct": round(avg_win, 2) if not np.isnan(avg_win) else 0.0,
            "avg_loss_pct": round(avg_loss, 2) if not np.isnan(avg_loss) else 0.0,
            "by_signal_type": by_type,
        }
