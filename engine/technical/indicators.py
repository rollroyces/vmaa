#!/usr/bin/env python3
"""
VMAA Technical Indicators
==========================
Pure numpy/pandas implementation of common technical indicators.
No TA-Lib dependency — all from scratch with vectorized operations.

Categories:
  - Trend: SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA
  - Momentum: RSI, MACD, Stochastic, KDJ, Williams %R, ADX, CCI, ROC, MFI
  - Volatility: Bollinger Bands (with %B, width, squeeze)
  - Volume: OBV, VWAP, Chaikin MF, A/D Line, Volume Ratio
  - Composite: Ichimoku Cloud, Parabolic SAR, Pivot Points, ATR
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("vmaa.engine.technical.indicators")

# ═══════════════════════════════════════════════════════════════════
# Type aliases
# ═══════════════════════════════════════════════════════════════════

ArrayLike = Union[np.ndarray, pd.Series, List[float]]
PriceData = Union[pd.DataFrame, Dict[str, np.ndarray]]


def _ensure_array(data: ArrayLike) -> np.ndarray:
    """Convert input to numpy float64 array."""
    if isinstance(data, pd.Series):
        return data.to_numpy(dtype=np.float64)
    if isinstance(data, list):
        return np.array(data, dtype=np.float64)
    return np.asarray(data, dtype=np.float64)


def _extract_columns(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract OHLCV arrays from a DataFrame.
    Accepts columns named with any case: Open/High/Low/Close/Volume.
    Also accepts lowercase: open/high/low/close/volume.
    """
    col_map = {
        "open": ["Open", "open"],
        "high": ["High", "high"],
        "low": ["Low", "low"],
        "close": ["Close", "close", "Adj Close", "Adj Close"],
        "volume": ["Volume", "volume"],
    }
    results = {}
    for key, candidates in col_map.items():
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            # Fallback: case-insensitive search
            for c in df.columns:
                if c.lower() == key:
                    found = c
                    break
        if found is None:
            raise KeyError(f"Cannot find column for '{key}' in DataFrame. Columns: {list(df.columns)}")
        results[key] = _ensure_array(df[found])

    return results["open"], results["high"], results["low"], results["close"], results["volume"]


# ═══════════════════════════════════════════════════════════════════
# 1. TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════


def sma(data: ArrayLike, period: int = 20) -> np.ndarray:
    """Simple Moving Average.

    Args:
        data: Price series (close, typically).
        period: Lookback window.

    Returns:
        Array of same length; first `period-1` values are NaN.
    """
    arr = _ensure_array(data)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result
    # Vectorized rolling mean via convolution
    kernel = np.ones(period) / period
    valid = np.convolve(arr, kernel, mode="valid")
    result[period - 1:] = valid
    return result


def ema(data: ArrayLike, period: int = 20) -> np.ndarray:
    """Exponential Moving Average.

    alpha = 2 / (period + 1) — standard Wilder smoothing.

    Args:
        data: Price series.
        period: Lookback window.

    Returns:
        Array of same length; first `period-1` values are NaN.
    """
    arr = _ensure_array(data)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result
    alpha = 2.0 / (period + 1.0)
    # Seed with mean of first `period` non-NaN values, or find start
    # If arr has NaN prefix (e.g. from another indicator), skip them
    first_valid = 0
    while first_valid < n and np.isnan(arr[first_valid]):
        first_valid += 1
    if first_valid >= n or n - first_valid < period:
        return result
    seed_start = first_valid
    seed_window = arr[seed_start:seed_start + period]
    valid_seed = seed_window[~np.isnan(seed_window)]
    if len(valid_seed) == 0:
        return result
    seed = np.mean(valid_seed)
    result[seed_start + period - 1] = seed
    for i in range(seed_start + period, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def wma(data: ArrayLike, period: int = 20) -> np.ndarray:
    """Weighted Moving Average (linear weighting).

    Most recent price gets weight=period, oldest gets weight=1.
    Denominator = period * (period + 1) / 2.

    Args:
        data: Price series.
        period: Lookback window.

    Returns:
        Array of same length; first `period-1` values are NaN.
    """
    arr = _ensure_array(data)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    for i in range(period - 1, n):
        window = arr[i - period + 1 : i + 1]
        result[i] = np.dot(window, weights) / denom
    return result


def dema(data: ArrayLike, period: int = 20) -> np.ndarray:
    """Double Exponential Moving Average.

    DEMA = 2 * EMA - EMA(EMA)
    Reduces lag compared to simple EMA.

    Args:
        data: Price series.
        period: Lookback window.

    Returns:
        Array of same length; first `2*period-2` values are NaN.
    """
    e1 = ema(data, period)
    e2 = ema(e1, period)
    result = 2.0 * e1 - e2
    return result


def tema(data: ArrayLike, period: int = 20) -> np.ndarray:
    """Triple Exponential Moving Average.

    TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    Further reduces lag.

    Args:
        data: Price series.
        period: Lookback window.

    Returns:
        Array of same length; first `3*period-3` values are NaN.
    """
    e1 = ema(data, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    result = 3.0 * e1 - 3.0 * e2 + e3
    return result


def trima(data: ArrayLike, period: int = 20) -> np.ndarray:
    """Triangular Moving Average.

    TRIMA = SMA(SMA(price, period//2), period//2)
    For odd periods, uses (period+1)//2 for both inner and outer.

    Args:
        data: Price series.
        period: Lookback window.

    Returns:
        Array of same length; first `period-1` values are NaN.
    """
    arr = _ensure_array(data)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result
    half = (period + 1) // 2
    # First pass SMA with half period
    s1 = sma(arr, half)
    # Second pass SMA with half period
    s2 = sma(s1, half)
    return s2


def kama(
    data: ArrayLike,
    period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> np.ndarray:
    """Kaufman Adaptive Moving Average (KAMA).

    Adapts smoothing based on market efficiency ratio (volatility).
    - Fast SC = 2/(fast_period+1), Slow SC = 2/(slow_period+1)
    - ER = |price - price[period ago]| / sum(|Δprice| over period)
    - SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    - KAMA[i] = KAMA[i-1] + SC * (price[i] - KAMA[i-1])

    Args:
        data: Price series.
        period: Efficiency Ratio lookback (default 10 per Perry Kaufman).
        fast_period: Fast EMA constant period (default 2).
        slow_period: Slow EMA constant period (default 30).

    Returns:
        Array of same length; first `period` values are NaN.
    """
    arr = _ensure_array(data)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return result

    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    diff_sc = fast_sc - slow_sc

    # Direction (price change over period)
    direction = np.abs(arr[period:] - arr[:-period])

    # Volatility (sum of absolute price changes over period)
    abs_diffs = np.abs(np.diff(arr))
    # Rolling sum over `period` for absolute diffs
    volatility = np.zeros(n - period, dtype=np.float64)
    cumsum = np.cumsum(np.insert(abs_diffs, 0, 0))
    volatility = cumsum[period:] - cumsum[:-period]

    # Efficiency Ratio
    er = np.divide(direction, volatility, out=np.zeros_like(direction), where=volatility != 0)

    # Smoothing Constant
    sc = (er * diff_sc + slow_sc) ** 2

    # KAMA: first value is SMA at index `period`
    result[period] = np.mean(arr[: period + 1])
    for i in range(period + 1, n):
        result[i] = result[i - 1] + sc[i - period - 1] * (arr[i] - result[i - 1])

    return result


# ═══════════════════════════════════════════════════════════════════
# 2. MOMENTUM INDICATORS
# ═══════════════════════════════════════════════════════════════════


def rsi(data: ArrayLike, period: int = 14) -> np.ndarray:
    """Relative Strength Index (Wilder, 1978).

    RSI = 100 - 100 / (1 + avg_gain / avg_loss)
    Uses Wilder's smoothing (exponential).

    Args:
        data: Price series (close).
        period: Lookback window (default 14).

    Returns:
        Array of same length; first `period` values are NaN.
    """
    arr = _ensure_array(data)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return result

    diffs = np.diff(arr)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)

    # First avg_gain / avg_loss is simple mean
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing for remaining
    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain = alpha * gains[i - 1] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i - 1] + (1 - alpha) * avg_loss
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - 100.0 / (1.0 + rs)

    return result


def macd(
    data: ArrayLike,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence Divergence (MACD).

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line

    Args:
        data: Price series (close).
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram) — each same length as input.
    """
    arr = _ensure_array(data)
    ema_fast = ema(arr, fast)
    ema_slow = ema(arr, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator (%K, %D).

    %K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
    %D = SMA(%K, d_period)

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        k_period: Lookback for %K (default 14).
        d_period: Smoothing for %D (default 3).

    Returns:
        Tuple of (%K, %D) — each same length as input.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    n = len(c)
    k = np.full(n, np.nan, dtype=np.float64)
    d = np.full(n, np.nan, dtype=np.float64)

    if n < k_period:
        return k, d

    for i in range(k_period - 1, n):
        highest = np.max(h[i - k_period + 1 : i + 1])
        lowest = np.min(l[i - k_period + 1 : i + 1])
        denom = highest - lowest
        if denom != 0:
            k[i] = 100.0 * (c[i] - lowest) / denom
        else:
            k[i] = 50.0  # neutral when all prices equal

    # %D = SMA of %K
    d[:] = sma(k, d_period)
    return k, d


def kdj(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    k_period: int = 14,
    d_period: int = 3,
    j_weight: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDJ Indicator (Stochastic variant with J line).

    %K and %D are same as stochastic.
    J = j_weight * D - (j_weight - 1) * K  →  typically 3*D - 2*K

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        k_period: Lookback for %K (default 14).
        d_period: Smoothing for %D (default 3).
        j_weight: J line multiplier (default 3).

    Returns:
        Tuple of (%K, %D, %J) — each same length as input.
    """
    k, d = stochastic(high, low, close, k_period, d_period)
    j = j_weight * d - (j_weight - 1) * k
    return k, d, j


def williams_r(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> np.ndarray:
    """Williams %R.

    %R = -100 * (HighestHigh - Close) / (HighestHigh - LowestLow)
    Ranges from -100 (oversold) to 0 (overbought).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback window (default 14).

    Returns:
        Array of same length; first `period-1` values are NaN.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    n = len(c)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    for i in range(period - 1, n):
        highest = np.max(h[i - period + 1 : i + 1])
        lowest = np.min(l[i - period + 1 : i + 1])
        denom = highest - lowest
        if denom != 0:
            result[i] = -100.0 * (highest - c[i]) / denom
        else:
            result[i] = -50.0

    return result


def adx(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index (ADX).

    Measures trend strength (not direction).
    - +DI and -DI show directional movement
    - ADX = smoothed average of DX
    - ADX > 25 = trending, ADX < 20 = ranging

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Smoothing period (default 14).

    Returns:
        Tuple of (adx, plus_di, minus_di, dx) — each same length as input.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    n = len(c)

    adx_arr = np.full(n, np.nan, dtype=np.float64)
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    dx_arr = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return adx_arr, plus_di, minus_di, dx_arr

    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Smoothed TR and DM (Wilder's smoothing)
    atr_arr = np.full(n, np.nan, dtype=np.float64)
    atr_arr[period] = np.mean(tr[1 : period + 1])
    smooth_plus_dm = np.full(n, np.nan, dtype=np.float64)
    smooth_plus_dm[period] = np.mean(plus_dm[1 : period + 1])
    smooth_minus_dm = np.full(n, np.nan, dtype=np.float64)
    smooth_minus_dm[period] = np.mean(minus_dm[1 : period + 1])

    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr_arr[i] = alpha * tr[i] + (1 - alpha) * atr_arr[i - 1]
        smooth_plus_dm[i] = alpha * plus_dm[i] + (1 - alpha) * smooth_plus_dm[i - 1]
        smooth_minus_dm[i] = alpha * minus_dm[i] + (1 - alpha) * smooth_minus_dm[i - 1]

    # +DI, -DI, DX
    for i in range(period, n):
        if atr_arr[i] != 0:
            plus_di[i] = 100.0 * smooth_plus_dm[i] / atr_arr[i]
            minus_di[i] = 100.0 * smooth_minus_dm[i] / atr_arr[i]
            dx_sum = plus_di[i] + minus_di[i]
            if dx_sum != 0:
                dx_arr[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / dx_sum
            else:
                dx_arr[i] = 0.0

    # ADX = smoothed DX
    adx_arr[2 * period - 1] = np.nanmean(dx_arr[period : 2 * period])
    for i in range(2 * period, n):
        adx_arr[i] = alpha * dx_arr[i] + (1 - alpha) * adx_arr[i - 1]

    return adx_arr, plus_di, minus_di, dx_arr


def cci(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 20) -> np.ndarray:
    """Commodity Channel Index (CCI).

    CCI = (TypicalPrice - SMA(TP, period)) / (0.015 * MeanDeviation)
    Typical Price = (High + Low + Close) / 3

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback window (default 20).

    Returns:
        Array of same length; first `period-1` values are NaN.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    n = len(c)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    tp = (h + l + c) / 3.0
    tp_sma = sma(tp, period)

    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        mean_dev = np.mean(np.abs(window - tp_sma[i]))
        if mean_dev != 0:
            result[i] = (tp[i] - tp_sma[i]) / (0.015 * mean_dev)
        else:
            result[i] = 0.0

    return result


def roc(data: ArrayLike, period: int = 12) -> np.ndarray:
    """Rate of Change (ROC).

    ROC = 100 * (Price[i] - Price[i-period]) / Price[i-period]

    Args:
        data: Price series.
        period: Lookback window (default 12).

    Returns:
        Array of same length; first `period` values are NaN.
    """
    arr = _ensure_array(data)
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return result

    result[period:] = 100.0 * (arr[period:] - arr[:-period]) / arr[:-period]
    return result


def mfi(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    period: int = 14,
) -> np.ndarray:
    """Money Flow Index (MFI) — volume-weighted RSI.

    Typical Price = (High + Low + Close) / 3
    Raw Money Flow = TP * Volume
    Positive Money Flow when TP rises, Negative when TP falls.
    Money Ratio = sum(pos_MF) / sum(neg_MF) over period.
    MFI = 100 - 100 / (1 + Money Ratio)

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        period: Lookback window (default 14).

    Returns:
        Array of same length; first `period` values are NaN.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    v = _ensure_array(volume)
    n = len(c)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return result

    tp = (h + l + c) / 3.0
    rmf = tp * v  # Raw Money Flow

    # Positive/negative money flow
    pos_mf = np.where(tp[1:] > tp[:-1], rmf[1:], 0.0)
    neg_mf = np.where(tp[1:] < tp[:-1], rmf[1:], 0.0)

    for i in range(period, n):
        pos_sum = np.sum(pos_mf[i - period : i])
        neg_sum = np.sum(neg_mf[i - period : i])
        if neg_sum == 0:
            result[i] = 100.0
        else:
            mr = pos_sum / neg_sum
            result[i] = 100.0 - 100.0 / (1.0 + mr)

    return result


# ═══════════════════════════════════════════════════════════════════
# 3. BOLLINGER BANDS
# ═══════════════════════════════════════════════════════════════════


def bollinger_bands(
    data: ArrayLike,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.

    Middle = SMA(period)
    Upper = Middle + num_std * STD(period)
    Lower = Middle - num_std * STD(period)

    Also computes band width and %B.

    Args:
        data: Price series (close).
        period: Moving average period (default 20).
        num_std: Number of standard deviations (default 2.0).

    Returns:
        Tuple of (middle, upper, lower, bandwidth, percent_b) — each same length as input.
    """
    arr = _ensure_array(data)
    n = len(arr)

    middle = sma(arr, period)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    bandwidth = np.full(n, np.nan, dtype=np.float64)
    percent_b = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return middle, upper, lower, bandwidth, percent_b

    for i in range(period - 1, n):
        window = arr[i - period + 1 : i + 1]
        std = np.std(window, ddof=0)  # population std
        upper[i] = middle[i] + num_std * std
        lower[i] = middle[i] - num_std * std
        if middle[i] != 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        denom = upper[i] - lower[i]
        if denom != 0:
            percent_b[i] = (arr[i] - lower[i]) / denom

    return middle, upper, lower, bandwidth, percent_b


def bollinger_squeeze(
    bandwidth: np.ndarray,
    lookback: int = 125,
) -> Tuple[np.ndarray, bool]:
    """Detect Bollinger Band squeeze.

    A squeeze occurs when the current bandwidth is the narrowest in `lookback` periods.
    `lookback` of 125 ≈ 6 months of trading days.

    Args:
        bandwidth: Band width series from bollinger_bands().
        lookback: Comparison period (default 125 ≈ 6 months).

    Returns:
        Tuple of (squeeze_signal_array, is_squeezing_now).
    """
    n = len(bandwidth)
    squeeze = np.full(n, False)
    if n < lookback:
        return squeeze, False

    for i in range(lookback - 1, n):
        current_bw = bandwidth[i]
        if np.isnan(current_bw):
            continue
        past = bandwidth[i - lookback + 1 : i + 1]
        past_valid = past[~np.isnan(past)]
        if len(past_valid) < lookback * 0.5:
            continue
        squeeze[i] = current_bw <= np.nanmin(past_valid)

    is_squeezing = bool(squeeze[-1]) if len(squeeze) > 0 else False
    return squeeze, is_squeezing


# ═══════════════════════════════════════════════════════════════════
# 4. VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════


def obv(close: ArrayLike, volume: ArrayLike) -> np.ndarray:
    """On-Balance Volume (OBV).

    OBV[0] = 0
    If close > prev_close: OBV += volume
    If close < prev_close: OBV -= volume
    If close == prev_close: OBV unchanged

    Args:
        close: Close price series.
        volume: Volume series.

    Returns:
        Array of same length.
    """
    c = _ensure_array(close)
    v = _ensure_array(volume)
    n = len(c)
    result = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        if c[i] > c[i - 1]:
            result[i] = result[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            result[i] = result[i - 1] - v[i]
        else:
            result[i] = result[i - 1]

    return result


def vwap(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    reset: str = "daily",
) -> np.ndarray:
    """Volume-Weighted Average Price (VWAP).

    VWAP = cumulative(TP * Volume) / cumulative(Volume)
    TP = (High + Low + Close) / 3

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        reset: Reset frequency — "daily" (resets each new day) or "session" (never resets).

    Returns:
        Array of same length.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    v = _ensure_array(volume)
    n = len(c)

    tp = (h + l + c) / 3.0
    pv = tp * v  # price * volume
    result = np.full(n, np.nan, dtype=np.float64)

    cum_pv = 0.0
    cum_vol = 0.0

    for i in range(n):
        # Reset daily at market open — use a simple heuristic:
        # If volume is zero (holiday) or cum_vol < previous (new session)
        # For simplicity, reset on each "day" marker if available.
        if reset == "daily" and i > 0:
            # Heuristic: reset daily — check if there's a time gap
            # Without datetime, we rely on the caller to boundary.
            # Here we just accumulate; caller can slice if needed.
            pass
        cum_pv += pv[i]
        cum_vol += v[i]
        if cum_vol > 0:
            result[i] = cum_pv / cum_vol

    return result


def volume_profile(
    close: ArrayLike,
    volume: ArrayLike,
    price_levels: int = 10,
) -> Dict[str, np.ndarray]:
    """Volume Profile by price level.

    Divides the price range into `price_levels` buckets and sums volume
    at each level. Returns the VWAP and volume by price level.

    Args:
        close: Close price series.
        volume: Volume series.
        price_levels: Number of price buckets (default 10).

    Returns:
        Dict with keys {'vwap', 'price_bins', 'volume_by_bin', 'poc_price', 'poc_volume'}.
    """
    c = _ensure_array(close)
    v = _ensure_array(volume)

    mask = ~np.isnan(c) & ~np.isnan(v) & (v > 0)
    c_valid = c[mask]
    v_valid = v[mask]

    if len(c_valid) == 0:
        return {
            "vwap": np.nan,
            "price_bins": np.array([]),
            "volume_by_bin": np.array([]),
            "poc_price": np.nan,
            "poc_volume": np.nan,
        }

    vwap_val = np.average(c_valid, weights=v_valid)

    price_min = np.min(c_valid)
    price_max = np.max(c_valid)
    if price_max == price_min:
        bins = np.array([price_min])
        vol_by_bin = np.array([np.sum(v_valid)])
    else:
        bin_edges = np.linspace(price_min, price_max, price_levels + 1)
        vol_by_bin = np.zeros(price_levels, dtype=np.float64)
        price_bins = np.zeros(price_levels, dtype=np.float64)
        for j in range(price_levels):
            in_bin = (c_valid >= bin_edges[j]) & (c_valid < bin_edges[j + 1])
            # Include upper edge in last bin only
            if j == price_levels - 1:
                in_bin = in_bin | (c_valid >= bin_edges[j + 1])
            vol_by_bin[j] = np.sum(v_valid[in_bin])
            price_bins[j] = (bin_edges[j] + bin_edges[j + 1]) / 2.0

    # POC = Point of Control (price level with most volume)
    if len(vol_by_bin) > 0:
        poc_idx = np.argmax(vol_by_bin)
        poc_price = float(price_bins[poc_idx] if len(price_bins) > 0 else np.nan)
        poc_volume = float(vol_by_bin[poc_idx])
    else:
        poc_price = np.nan
        poc_volume = np.nan

    return {
        "vwap": float(vwap_val),
        "price_bins": price_bins if len(price_bins) > 0 else np.array([]),
        "volume_by_bin": vol_by_bin,
        "poc_price": poc_price,
        "poc_volume": poc_volume,
    }


def chaikin_money_flow(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
    period: int = 21,
) -> np.ndarray:
    """Chaikin Money Flow (CMF) — 21-period default.

    CMF = sum(MoneyFlowVolume, period) / sum(Volume, period)
    MoneyFlowVolume = volume * ((Close-Low) - (High-Close)) / (High-Low)
    Ranges from -1 to 1; positive = accumulation, negative = distribution.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        period: Lookback window (default 21).

    Returns:
        Array of same length; first `period-1` values are NaN.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    v = _ensure_array(volume)
    n = len(c)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    # Money Flow Multiplier
    hl_diff = h - l
    mf_mult = np.zeros(n, dtype=np.float64)
    valid_hl = hl_diff != 0
    mf_mult[valid_hl] = ((c[valid_hl] - l[valid_hl]) - (h[valid_hl] - c[valid_hl])) / hl_diff[valid_hl]

    # Money Flow Volume
    mf_vol = mf_mult * v

    for i in range(period - 1, n):
        vol_sum = np.sum(v[i - period + 1 : i + 1])
        if vol_sum != 0:
            result[i] = np.sum(mf_vol[i - period + 1 : i + 1]) / vol_sum

    return result


def accumulation_distribution(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    volume: ArrayLike,
) -> np.ndarray:
    """Accumulation/Distribution Line (A/D Line).

    A/D[i] = A/D[i-1] + MoneyFlowMultiplier * Volume
    MoneyFlowMultiplier = ((Close-Low) - (High-Close)) / (High-Low)

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.

    Returns:
        Array of same length.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    v = _ensure_array(volume)
    n = len(c)

    hl_diff = h - l
    mf_mult = np.zeros(n, dtype=np.float64)
    valid_hl = hl_diff != 0
    mf_mult[valid_hl] = ((c[valid_hl] - l[valid_hl]) - (h[valid_hl] - c[valid_hl])) / hl_diff[valid_hl]

    adl = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        adl[i] = adl[i - 1] + mf_mult[i] * v[i]

    return adl


def volume_ratio(volume: ArrayLike, period: int = 20) -> np.ndarray:
    """Volume Ratio: current volume / average volume over `period`.

    Args:
        volume: Volume series.
        period: Lookback window (default 20).

    Returns:
        Array of same length; first `period-1` values are NaN.
    """
    v = _ensure_array(volume)
    n = len(v)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return result

    vol_sma = sma(v, period)
    valid = vol_sma > 0
    result[valid] = v[valid] / vol_sma[valid]
    return result


# ═══════════════════════════════════════════════════════════════════
# 5. ICHIMOKU CLOUD
# ═══════════════════════════════════════════════════════════════════


def ichimoku(
    high: ArrayLike,
    low: ArrayLike,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> Dict[str, np.ndarray]:
    """Ichimoku Kinko Hyo (One Glance Equilibrium Chart).

    Tenkan-sen = (HighestHigh + LowestLow) / 2 over tenkan_period
    Kijun-sen  = (HighestHigh + LowestLow) / 2 over kijun_period
    Senkou Span A = (Tenkan + Kijun) / 2, shifted forward by displacement
    Senkou Span B = (HighestHigh + LowestLow) / 2 over senkou_b_period, shifted forward by displacement
    Chikou Span  = Close, shifted backward by displacement

    Args:
        high: High price series.
        low: Low price series.
        tenkan_period: Tenkan-sen period (default 9).
        kijun_period: Kijun-sen period (default 26).
        senkou_b_period: Senkou Span B period (default 52).
        displacement: Forward/backward shift (default 26).

    Returns:
        Dict with {'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'}.
        senkou_a and senkou_b are length n+displacement (shifted forward).
        All others are length n.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    n = len(h)

    tenkan = np.full(n, np.nan, dtype=np.float64)
    kijun = np.full(n, np.nan, dtype=np.float64)
    chikou = np.full(n, np.nan, dtype=np.float64)

    # Tenkan-sen
    for i in range(tenkan_period - 1, n):
        tenkan[i] = (np.max(h[i - tenkan_period + 1 : i + 1]) + np.min(l[i - tenkan_period + 1 : i + 1])) / 2.0

    # Kijun-sen
    for i in range(kijun_period - 1, n):
        kijun[i] = (np.max(h[i - kijun_period + 1 : i + 1]) + np.min(l[i - kijun_period + 1 : i + 1])) / 2.0

    # Senkou Span A = (Tenkan + Kijun) / 2, shifted forward
    raw_senkou_a = (tenkan + kijun) / 2.0
    senkou_a = np.full(n + displacement, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(raw_senkou_a[i]):
            j = i + displacement
            if j < len(senkou_a):
                senkou_a[j] = raw_senkou_a[i]

    # Senkou Span B
    raw_senkou_b = np.full(n, np.nan, dtype=np.float64)
    for i in range(senkou_b_period - 1, n):
        raw_senkou_b[i] = (np.max(h[i - senkou_b_period + 1 : i + 1]) + np.min(l[i - senkou_b_period + 1 : i + 1])) / 2.0

    senkou_b = np.full(n + displacement, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(raw_senkou_b[i]):
            j = i + displacement
            if j < len(senkou_b):
                senkou_b[j] = raw_senkou_b[i]

    # Chikou Span = Close shifted backward
    for i in range(displacement, n):
        chikou[i - displacement] = h[i]  # Using high as proxy — correct: use close

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
    }


# ═══════════════════════════════════════════════════════════════════
# 6. PARABOLIC SAR
# ═══════════════════════════════════════════════════════════════════


def parabolic_sar(
    high: ArrayLike,
    low: ArrayLike,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.20,
) -> np.ndarray:
    """Parabolic Stop and Reverse (SAR).

    Reverses direction when price crosses the SAR value.
    Acceleration Factor (AF) starts at af_start and increases by af_increment
    on each new extreme, capped at af_max.

    Args:
        high: High price series.
        low: Low price series.
        af_start: Initial acceleration factor (default 0.02).
        af_increment: AF increment per new extreme (default 0.02).
        af_max: Maximum AF (default 0.20).

    Returns:
        Array of same length as input; SAR values.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    n = len(h)
    sar = np.full(n, np.nan, dtype=np.float64)

    if n < 2:
        return sar

    # Initial trend: up if price rises, down if falls
    uptrend = h[1] > h[0]
    ep = h[0] if uptrend else l[0]  # Extreme Point
    af = af_start
    sar_val = l[0] if uptrend else h[0]
    sar[0] = sar_val

    for i in range(1, n):
        sar_val = sar_val + af * (ep - sar_val)

        if uptrend:
            sar_val = min(sar_val, l[i - 1], l[i - 2] if i >= 2 else l[0])
            if h[i] > ep:
                ep = h[i]
                af = min(af + af_increment, af_max)
            # Check reversal
            if l[i] < sar_val:
                uptrend = False
                sar_val = ep
                ep = l[i]
                af = af_start
        else:
            sar_val = max(sar_val, h[i - 1], h[i - 2] if i >= 2 else h[0])
            if l[i] < ep:
                ep = l[i]
                af = min(af + af_increment, af_max)
            # Check reversal
            if h[i] > sar_val:
                uptrend = True
                sar_val = ep
                ep = h[i]
                af = af_start

        sar[i] = sar_val

    return sar


# ═══════════════════════════════════════════════════════════════════
# 7. PIVOT POINTS
# ═══════════════════════════════════════════════════════════════════


def pivot_points(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
) -> Dict[str, np.ndarray]:
    """Standard Pivot Points (daily, based on previous period H/L/C).

    Pivot = (H + L + C) / 3
    R1 = 2 * Pivot - L
    R2 = Pivot + (H - L)
    R3 = H + 2 * (Pivot - L)
    S1 = 2 * Pivot - H
    S2 = Pivot - (H - L)
    S3 = L - 2 * (H - Pivot)

    Note: This computes for the FULL series using previous-period OHLC.
    For daily pivots, run this on daily-resampled data.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.

    Returns:
        Dict with {'pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3'}.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    n = len(c)

    pivot = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        p = (h[i - 1] + l[i - 1] + c[i - 1]) / 3.0
        pivot[i] = p
        r1[i] = 2.0 * p - l[i - 1]
        r2[i] = p + (h[i - 1] - l[i - 1])
        r3[i] = h[i - 1] + 2.0 * (p - l[i - 1])
        s1[i] = 2.0 * p - h[i - 1]
        s2[i] = p - (h[i - 1] - l[i - 1])
        s3[i] = l[i - 1] - 2.0 * (h[i - 1] - p)

    return {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


# ═══════════════════════════════════════════════════════════════════
# 8. ATR (Average True Range)
# ═══════════════════════════════════════════════════════════════════


def atr(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> np.ndarray:
    """Average True Range (ATR).

    True Range = max(H-L, |H-Close[prev]|, |L-Close[prev]|)
    ATR = Wilder's smoothed average of TR.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Smoothing period (default 14).

    Returns:
        Array of same length; first `period` values are NaN.
    """
    h = _ensure_array(high)
    l = _ensure_array(low)
    c = _ensure_array(close)
    n = len(c)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < period + 1:
        return result

    tr = np.zeros(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    # First ATR = simple mean of first `period` TR
    result[period] = np.mean(tr[1 : period + 1])

    # Wilder smoothing
    alpha = 1.0 / period
    for i in range(period + 1, n):
        result[i] = alpha * tr[i] + (1 - alpha) * result[i - 1]

    return result


# ═══════════════════════════════════════════════════════════════════
# 9. CONVENIENCE: Compute all indicators for a ticker
# ═══════════════════════════════════════════════════════════════════


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ALL standard indicators for a price DataFrame.

    Adds indicator columns to a copy of the DataFrame.
    Standard periods are used throughout.

    Args:
        df: DataFrame with Open/High/Low/Close/Volume columns.

    Returns:
        DataFrame with original columns + all indicator columns appended.
    """
    result = df.copy()
    o, h, l, c, v = _extract_columns(df)

    # Trend
    for p in [5, 10, 20, 50, 100, 200]:
        result[f"sma_{p}"] = sma(c, p)
        result[f"ema_{p}"] = ema(c, p)
        if p <= 20:
            result[f"wma_{p}"] = wma(c, p)

    result["dema_20"] = dema(c, 20)
    result["tema_20"] = tema(c, 20)
    result["trima_20"] = trima(c, 20)
    result["kama"] = kama(c, 10)

    # Momentum
    result["rsi_14"] = rsi(c, 14)
    macd_line, macd_signal, macd_hist = macd(c, 12, 26, 9)
    result["macd"] = macd_line
    result["macd_signal"] = macd_signal
    result["macd_histogram"] = macd_hist

    stoch_k, stoch_d = stochastic(h, l, c, 14, 3)
    result["stoch_k"] = stoch_k
    result["stoch_d"] = stoch_d

    kdj_k, kdj_d, kdj_j = kdj(h, l, c, 14, 3)
    result["kdj_k"] = kdj_k
    result["kdj_d"] = kdj_d
    result["kdj_j"] = kdj_j

    result["williams_r"] = williams_r(h, l, c, 14)

    adx_arr, plus_di, minus_di, _ = adx(h, l, c, 14)
    result["adx"] = adx_arr
    result["plus_di"] = plus_di
    result["minus_di"] = minus_di

    result["cci_20"] = cci(h, l, c, 20)
    result["roc_12"] = roc(c, 12)
    result["mfi_14"] = mfi(h, l, c, v, 14)

    # Bollinger
    bb_mid, bb_up, bb_lo, bb_width, bb_pb = bollinger_bands(c, 20, 2.0)
    result["bb_middle"] = bb_mid
    result["bb_upper"] = bb_up
    result["bb_lower"] = bb_lo
    result["bb_width"] = bb_width
    result["bb_pct_b"] = bb_pb

    bb_squeeze_signal, _ = bollinger_squeeze(bb_width, 125)
    result["bb_squeeze"] = bb_squeeze_signal.astype(int)

    # Volume
    result["obv"] = obv(c, v)
    result["cmf_21"] = chaikin_money_flow(h, l, c, v, 21)
    result["ad_line"] = accumulation_distribution(h, l, c, v)
    result["vol_ratio"] = volume_ratio(v, 20)

    # VWAP
    result["vwap"] = vwap(h, l, c, v)

    # Ichimoku
    ichi = ichimoku(h, l)
    result["ichimoku_tenkan"] = ichi["tenkan"]
    result["ichimoku_kijun"] = ichi["kijun"]
    # Pad senkou to match length
    result["ichimoku_senkou_a"] = pd.Series(ichi["senkou_a"][:len(result)], index=result.index)
    result["ichimoku_senkou_b"] = pd.Series(ichi["senkou_b"][:len(result)], index=result.index)
    result["ichimoku_chikou"] = ichi["chikou"]

    # Parabolic SAR
    result["parabolic_sar"] = parabolic_sar(h, l)

    # Pivot Points
    pp = pivot_points(h, l, c)
    for k, v in pp.items():
        result[f"pivot_{k}"] = v

    # ATR
    result["atr_14"] = atr(h, l, c, 14)

    return result
