#!/usr/bin/env python3
"""
VMAA 2.0 — Configuration & Thresholds
======================================
All screening parameters for:
  Part 1: Core Financial Fundamentals (Quality + Value)
  Part 2: MAGNA 53/10 (Momentum + Breakout)

Also includes risk management and pipeline operational params.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ═══════════════════════════════════════════════════════════════════
# Part 1: Core Financial Fundamentals Thresholds
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Part1Config:
    """Thresholds for Stage 1 — Quality & Value screening."""

    # ── 1. Market Cap Positioning ──
    deep_value_max_cap: float = 250e6         # $250M — deep value threshold
    turnaround_max_cap: float = 10e9          # $10B — turnaround growth ceiling
    large_cap_enabled: bool = True            # Accept stocks > $10B as "large_cap"
    # Classification: < deep_value_max_cap → "deep_value"
    #                 < turnaround_max_cap  → "turnaround"
    #                 >= turnaround_max_cap → "large_cap" (if enabled) or REJECTED

    # ── 2. Quality: B/M Ratio ──
    min_bm_ratio: float = 0.20                # Minimum Book-to-Market (was 0.3 — relaxed for mid-cap tech/growth)
    target_bm_ratio: float = 0.60             # Target B/M (was 0.8)

    # ── 3. Quality: ROA ──
    min_roa: float = 0.0                      # Minimum ROA (must be profitable)
    target_roa: float = 0.05                  # Target ROA (5%+)

    # ── 4. Quality: EBITDA Margin ──
    min_ebitda_margin: float = 0.05           # Minimum 5% EBITDA margin
    target_ebitda_margin: float = 0.15        # Target 15%+

    # ── 5. Cash Flow: FCF Yield ──
    min_fcf_yield: float = 0.02               # Minimum 2% FCF yield (was 3% — more inclusive in bull markets)
    target_fcf_yield: float = 0.06            # Target 6%+ (was 8%)

    # ── 6. Cash Flow: FCF Conversion (Earnings Authenticity) ──
    min_fcf_conversion: float = 0.50          # FCF/NI >= 50% (earnings have substance)
    target_fcf_conversion: float = 0.80       # Target 80%+ (high quality earnings)

    # ── 7. Safety Margin: Price / 52-week Low ──
    max_ptl_ratio: float = 1.50               # Max 50% above 52w-low (was 30% — bull market adjustment)
    target_ptl_ratio: float = 1.20            # Target within 20% of 52w-low (was 10%)

    # ── 8. Asset Expansion Constraint ──
    # ΔAssets < ΔEarnings → capital-efficient growth
    require_asset_lt_earnings: bool = True    # Hard requirement or soft preference

    # ── 9. Interest Rate Sensitivity ──
    interest_sensitive_sectors: List[str] = field(default_factory=lambda: [
        'Technology',
        'Financial Services',
        'Real Estate',
        'Utilities',
        'Communication Services',
    ])
    ir_debt_to_equity_threshold: float = 80   # Debt/Equity > 80 → sensitive
    ir_beta_threshold: float = 1.5            # Beta > 1.5 → sensitive

    # ── 10. Minimum Data Requirements ──
    min_price: float = 1.0                    # Skip sub-$1 stocks
    min_avg_volume: int = 50000               # Minimum average daily volume
    max_avg_volume: int = 50_000_000          # Cap for liquidity check

    # ── Scoring Weights (for quality_score) ──
    weight_bm: float = 0.15              # Reduced from 0.20 to make room for FCF conversion (core metric)
    weight_roa: float = 0.15
    weight_ebitda: float = 0.10
    weight_fcf_yield: float = 0.15       # Reduced from 0.20 to make room for FCF conversion
    weight_fcf_conversion: float = 0.20  # Raised from 0.10 — FCF/NI is 核心 (core) per spec review
    weight_ptl: float = 0.15
    weight_asset_efficiency: float = 0.10

    # ── Pass threshold ──
    min_quality_score: float = 0.40           # Minimum composite quality to pass


# ═══════════════════════════════════════════════════════════════════
# Part 2: MAGNA 53/10 Thresholds
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Part2Config:
    """Thresholds for Stage 2 — MAGNA 53/10 momentum screening."""

    # ── M: Massive Earnings Acceleration ──
    eps_accel_min: float = 0.20               # EPS growth acceleration ≥ 20%
    eps_growth_min: float = 0.20              # Current EPS growth ≥ 20% (raised from 15%)

    # ── A: Acceleration of Sales ──
    revenue_accel_min: float = 0.10           # Revenue growth acceleration ≥ 10%
    revenue_growth_min: float = 0.10          # Current revenue growth ≥ 10%

    # ── G: Gap Up ──
    gap_min_pct: float = 0.04                 # Gap ≥ 4%
    gap_premarket_vol_min: int = 100_000      # Pre-market volume ≥ 100K (DEPRECATED: yfinance never returns this)
    gap_volume_multiplier: float = 1.5        # Gap day volume must be ≥ 1.5x 20-day avg (replaces broken preMarketVolume)
    gap_lookback_days: int = 20               # Look back 20 trading days

    # ── N: Neglect / Base Pattern ──
    base_min_months: float = 6.0              # Minimum 6 months in base (was 3.0 — spec says 數月至數年, not weeks)
    base_max_range_pct: float = 0.30          # Price range ≤ 30% within base
    base_vol_decline_pct: float = 0.30        # Volume decline from early base

    # ── 5: Short Interest ──
    short_ratio_high: float = 5.0             # SI ratio > 5 → 2 points
    short_ratio_moderate: float = 3.0          # SI ratio > 3 → 1 point
    short_pct_float_min: float = 0.05         # 5%+ of float short

    # ── 3: Analyst Upgrades ──
    analyst_count_min: int = 3                # At least 3 analysts
    analyst_target_premium_pct: float = 0.15  # Target ≥ 15% above current price

    # ── Cap 10: Market Cap ──
    max_market_cap: float = 10e9              # $10B ceiling
    large_cap_enabled: bool = True            # Accept stocks > $10B as "large_cap"

    # ── 10: IPO Tenure ──
    max_ipo_years: float = 10.0               # Listed ≤ 10 years
    ipo_hard_requirement: bool = False         # Soft flag, not hard reject (relaxed)

    # ── Scoring ──
    magna_points: dict = field(default_factory=lambda: {
        'm_earnings_accel': 2,
        'a_sales_accel': 2,
        'g_gap_up': 2,
        'n_neglect_base': 1,
        'short_interest': 2,                  # 0/1/2 based on threshold
        'analyst_upgrades': 1,
        'cap_under_10b': 0,                   # Prerequisite, not scored
        'ipo_within_10yr': 0,                 # Prerequisite, not scored
    })
    magna_pass_threshold: int = 3             # Minimum MAGNA score to consider

    # ── Entry Triggers ──
    # Entry is triggered when G (Gap) OR both M+A fire simultaneously
    require_gap_or_ma: bool = True


# ═══════════════════════════════════════════════════════════════════
# Risk Management
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RiskConfig:
    """Comprehensive risk management parameters."""

    # Portfolio-level — WIDE_STOP strategy
    max_positions: int = 5
    max_positions_per_sector: int = 2
    cash_reserve_pct: float = 0.15
    max_portfolio_heat: float = 0.70
    max_daily_loss_pct: float = 0.03
    max_correlation: float = 0.70

    # Position-level
    max_position_pct: float = 0.18
    kelly_fraction: float = 0.15
    min_position_size: float = 500.0
    max_position_size: float = 80000.0

    # HK market position limits
    max_position_size_hkd: float = 80000.0
    max_shares_hk: int = 5000

    # Entry
    max_slippage_pct: float = 0.02
    require_volume: bool = True
    min_avg_volume: int = 50000

    # Stop management — WIDE_STOP strategy (backtest-optimized 2026-05-07)
    atr_stop_multiplier: float = 3.0            # Wider ATR stop for breathing room
    hard_stop_pct: float = 0.25                # 25% — wide to allow mean-reversion
    trailing_stop_pct: float = 0.12            # Slightly wider trail
    trailing_activate_after: float = 0.18       # Activate trail after 18% gain
    time_stop_days: int = 9999                 # No time limit — let trades play out

    # Take Profit — FULL EXIT at first target (no partial fills)
    # Partial fills were the #1 cause of losses in backtesting
    # Selling 30% at +12% locked tiny wins while remaining 70% bled out
    tp1_level_pct: float = 0.15                # TP1: +15%
    tp2_level_pct: float = 0.25                # TP2: +25% (secondary)
    tp3_level_pct: float = 0.40                # TP3: +40% (tertiary)

    # Market conditions
    vix_proxy_threshold: float = 0.25
    market_trend_ma: int = 50
    volume_drop_threshold: float = 0.50


# ═══════════════════════════════════════════════════════════════════
# Pipeline Operational Config
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PipelineConfig:
    """Operational parameters for the pipeline runner."""

    # Rate limiting
    yfinance_delay: float = 0.15              # Seconds between yfinance calls
    batch_size: int = 50                      # Report progress every N stocks

    # Universe
    default_universe: str = "sp500"           # "sp500" | "russell2000" | "custom"
    custom_tickers: List[str] = field(default_factory=list)

    # Output
    output_dir: str = "output"
    save_quality_pool: bool = True            # Save Part 1 results for later Part 2 scans

    # Monitoring (for continuous mode)
    monitor_interval_minutes: int = 15        # Check for Part 2 triggers every N min


# ═══════════════════════════════════════════════════════════════════
# Singleton instances
# ═══════════════════════════════════════════════════════════════════

P1C = Part1Config()
P2C = Part2Config()
RC = RiskConfig()
PC = PipelineConfig()

# Sentiment config is in part3_sentiment.py (SENT_CONFIG)
