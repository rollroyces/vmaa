#!/usr/bin/env python3
"""
VMAA Technical Analysis Configuration
======================================
Default periods, thresholds, and signal parameters for the technical engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TechnicalConfig:
    """Central configuration for the Technical Analysis Engine."""

    # ── Trend Indicator Defaults ──
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    wma_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    dema_period: int = 20
    tema_period: int = 20
    trima_period: int = 20
    kama_period: int = 10
    kama_fast: int = 2
    kama_slow: int = 30

    # ── Momentum Indicator Defaults ──
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    kdj_k: int = 14
    kdj_d: int = 3
    kdj_j_weight: float = 3.0
    williams_r_period: int = 14
    adx_period: int = 14
    cci_period: int = 20
    roc_period: int = 12
    mfi_period: int = 14

    # ── Bollinger Band Defaults ──
    bb_period: int = 20
    bb_num_std: float = 2.0
    bb_squeeze_lookback: int = 125  # ~6 months

    # ── Volume Indicator Defaults ──
    cmf_period: int = 21
    vol_ratio_period: int = 20
    volume_profile_levels: int = 10

    # ── Ichimoku Defaults ──
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    ichimoku_displacement: int = 26

    # ── Parabolic SAR Defaults ──
    psar_af_start: float = 0.02
    psar_af_increment: float = 0.02
    psar_af_max: float = 0.20

    # ── ATR Defaults ──
    atr_period: int = 14

    # ── Signal Thresholds ──
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_strong_oversold: float = 20.0
    rsi_strong_overbought: float = 80.0

    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    williams_r_oversold: float = -80.0
    williams_r_overbought: float = -20.0

    mfi_oversold: float = 20.0
    mfi_overbought: float = 80.0

    adx_trending: float = 25.0
    adx_strong_trend: float = 40.0

    cci_oversold: float = -100.0
    cci_overbought: float = 100.0

    bb_pct_b_oversold: float = 0.0   # price at lower band
    bb_pct_b_overbought: float = 1.0  # price at upper band

    # ── MA Crossover Triggers ──
    ma_crossover_fast_periods: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "golden_cross": (50, 200),       # 50 crosses above 200
        "death_cross": (200, 50),        # 50 crosses below 200 → (slow, fast) for signal detection
        "short_term": (5, 20),
        "medium_term": (20, 50),
    })

    # ── Signal Strength Settings ──
    # Number of confirming indicators needed for each signal strength
    signal_strength_thresholds: Dict[int, int] = field(default_factory=lambda: {
        1: 1,   # 1 indicator → STRONG_BUY/SELL
        2: 2,   # 2 indicators → confirmed
        3: 3,   # 3 indicators → strong
        4: 4,   # 4 indicators → very strong
        5: 5,   # 5 indicators → extreme
    })

    # ── Custom Indicator Settings ──
    max_custom_indicators: int = 50
    custom_indicator_storage: str = "vmaa/engine/technical/custom_indicators.json"


# Singleton
TC = TechnicalConfig()
