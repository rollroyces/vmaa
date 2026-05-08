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
from types import MappingProxyType
from typing import List


# ═══════════════════════════════════════════════════════════════════
# Part 1: Core Financial Fundamentals Thresholds
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Part1Config:
    """Thresholds for Stage 1 — Quality & Value screening."""

    # ── 1. Market Cap Positioning ──
    deep_value_max_cap: float = 250e6         # $250M — deep value threshold (微型股)
    turnaround_max_cap: float = 10e9          # $10B — turnaround growth ceiling
    large_cap_enabled: bool = True            # Accept stocks > $10B as "large_cap"
    # Classification: < deep_value_max_cap → "deep_value"
    #                 < turnaround_max_cap  → "turnaround"
    #                 >= turnaround_max_cap → "large_cap" (if enabled) or REJECTED

    # ── 2. Quality: B/M Ratio ──
    min_bm_ratio: float = 0.25                # Minimum Book-to-Market — tightened: high B/M required
    target_bm_ratio: float = 0.60             # Target B/M

    # ── 3. Quality: ROA ──
    min_roa: float = 0.02                     # Minimum ROA 2% — tightened: genuine profitability
    target_roa: float = 0.05                  # Target ROA (5%+)

    # ── 4. Quality: EBITDA Margin ──
    min_ebitda_margin: float = 0.05           # Minimum 5% EBITDA margin
    target_ebitda_margin: float = 0.15        # Target 15%+

    # ── 5. Cash Flow: FCF Yield ──
    min_fcf_yield: float = 0.03               # Minimum 3% FCF yield — tightened: strong cash flow only
    target_fcf_yield: float = 0.06            # Target 6%+

    # ── 6. Cash Flow: FCF Conversion (Earnings Authenticity) ──
    min_fcf_conversion: float = 0.60          # FCF/NI >= 60% — tightened: real cash behind earnings
    target_fcf_conversion: float = 0.80       # Target 80%+ (high quality earnings)

    # ── 7. Safety Margin: Price / 52-week Low ──
    max_ptl_ratio: float = 1.35               # Max 35% above 52w-low — tightened: closer to bottom
    target_ptl_ratio: float = 1.20            # Target within 20% of 52w-low

    # ── 8. Asset Expansion Constraint ──
    # ΔAssets < ΔEarnings → capital-efficient growth
    require_asset_lt_earnings: bool = True    # Hard requirement: asset expansion must not outpace earnings

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
    # 權重偏重現金流品質：獲利品質與現金轉換為核心，成長迷思不計分
    weight_bm: float = 0.10              # Book-to-Market (value anchor)
    weight_roa: float = 0.12             # ROA (profitability)
    weight_ebitda: float = 0.08          # EBITDA margin
    weight_fcf_yield: float = 0.20       # FCF Yield — 核心：強勁現金流
    weight_fcf_conversion: float = 0.25  # FCF/NI — 核心：獲利品質與現金轉換
    weight_ptl: float = 0.15             # Price-to-52w-Low (safety margin)
    weight_asset_efficiency: float = 0.10 # Asset expansion vs earnings

    # ── Pass threshold ──
    min_quality_score: float = 0.50           # Minimum composite quality — tightened from 0.40

    def __post_init__(self):
        w_sum = (self.weight_bm + self.weight_roa + self.weight_ebitda +
                 self.weight_fcf_yield + self.weight_fcf_conversion +
                 self.weight_ptl + self.weight_asset_efficiency)
        if abs(w_sum - 1.0) > 0.001:
            raise ValueError(f"Part1 weights sum to {w_sum}, expected 1.0")


# ═══════════════════════════════════════════════════════════════════
# Part 2: MAGNA 53/10 Thresholds
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Part2Config:
    """Thresholds for Stage 2 — MAGNA 53/10 momentum screening."""

    # ── M: Massive Earnings Acceleration ──
    eps_accel_min: float = 0.25               # EPS growth acceleration ≥ 25% — tightened: massive only
    eps_growth_min: float = 0.25              # Current EPS growth ≥ 25% — tightened

    # ── A: Acceleration of Sales ──
    revenue_accel_min: float = 0.15           # Revenue growth acceleration ≥ 15% — tightened
    revenue_growth_min: float = 0.15          # Current revenue growth ≥ 15% — tightened

    # ── G: Gap Up ──
    gap_min_pct: float = 0.04                 # Gap ≥ 4%
    gap_volume_multiplier: float = 1.5        # Gap day volume must be ≥ 1.5x 20-day avg
    gap_min_volume: int = 100000              # Absolute: gap day volume ≥ 100K shares
    gap_lookback_days: int = 20               # Look back 20 trading days

    # ── N: Neglect / Base Pattern ──
    base_min_months: float = 9.0              # Minimum 9 months in base — tightened: 數月至數年
    base_max_range_pct: float = 0.30          # Price range ≤ 30% within base
    base_vol_decline_pct: float = 0.30        # Volume decline from early base

    # ── 5: Short Interest ──
    short_ratio_high: float = 5.0             # SI ratio > 5 → 2 points
    short_ratio_moderate: float = 3.0          # SI ratio > 3 → 1 point
    short_pct_float_min: float = 0.05         # 5%+ of float short

    # ── 3: Analyst Upgrades ──
    analyst_count_min: int = 3                # At least 3 analysts
    analyst_target_premium_pct: float = 0.15  # Target ≥ 15% above current price

    # ── Cap 10: Market Cap — 硬上限 $10B ──
    max_market_cap: float = 10e9              # $10B ceiling
    large_cap_enabled: bool = False           # Tightened: Cap 10 hard limit, no large_cap exceptions

    # ── 10: IPO Tenure — 硬上限 10 年 ──
    max_ipo_years: float = 10.0               # Listed ≤ 10 years
    ipo_hard_requirement: bool = True          # Tightened: hard reject if IPO > 10yr

    # ── Scoring ──
    magna_points: dict = field(default_factory=lambda: {
        'm_earnings_accel': 2,
        'a_sales_accel': 2,
        'g_gap_up': 2,
        'n_neglect_base': 1,
        'short_interest': 2,                  # 0/1/2 based on threshold
        'analyst_upgrades': 1,
        'cap_under_10b': 0,                   # Hard prerequisite, not scored
        'ipo_within_10yr': 0,                 # Hard prerequisite, not scored
    })
    magna_pass_threshold: int = 4             # Minimum MAGNA score — tightened from 3

    # ── Entry Triggers ──
    # Entry is triggered when G (Gap) OR both M+A fire simultaneously
    require_gap_or_ma: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'magna_points', MappingProxyType(self.magna_points))


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

    # Stop management — FIXED R:R v3.2
    # Core fix: inverted R:R (risk 25%→make 15%) is mathematically broken
    # New: risk 15%→make 20% = R:R 1:1.33, breakeven WR = 42.9%
    # Small-cap (<$2B): risk 12%→make 18% = R:R 1:1.5, breakeven WR = 40%
    # v3.2: Trailing stop DISABLED — was killing 6/10 wins before TP1
    #       Set trailing_stop_pct high (0.99) to effectively disable
    atr_stop_multiplier: float = 2.0            # Tighter ATR stop (was 3.0)
    hard_stop_pct: float = 0.15                 # 15% hard stop — FIXED from 25%
    trailing_stop_pct: float = 0.12              # Base 12% trail — overridden per-stock by compute_trailing_stop()
    trailing_activate_after: float = 0.15       # 15% activate  — per-stock adaptive (v3.2.1)
    time_stop_days: int = 9999                  # No time limit — let trades play out
    # Small-cap specific (<$2B market cap)
    small_cap_max_market_cap: float = 2e9       # $2B threshold
    small_cap_hard_stop_pct: float = 0.12       # 12% stop for small caps
    small_cap_tp1_pct: float = 0.18             # 18% TP1 for small caps

    # Take Profit — FULL EXIT at first target (no partial fills)
    # Partial fills were the #1 cause of losses in backtesting
    tp1_level_pct: float = 0.20                # TP1: +20% (was 15% — FIXED R:R)
    tp2_level_pct: float = 0.30                # TP2: +30%
    tp3_level_pct: float = 0.50                # TP3: +50%

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
