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
    gap_volume_multiplier: float = 2.0        # Gap day volume must be ≥ 2.0x 20-day avg (V3)
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

    # ── Cap 10: Market Cap — V3 expanded to $50B ──
    max_market_cap: float = 50e9              # $50B ceiling (V3: 500M-50B range)
    large_cap_enabled: bool = False           # Tightened: Cap 50B hard limit, no large_cap exceptions

    # ── 10: IPO Tenure — V3 expanded to 20 年 ──
    max_ipo_years: float = 20.0               # Listed ≤ 20 years (V3: was 10yr)
    ipo_hard_requirement: bool = True          # Tightened: hard reject if IPO > 20yr

    # ── Scoring (V3 re-weighted) ──
    magna_points: dict = field(default_factory=lambda: {
        'm_earnings_accel': 2,
        'a_sales_accel': 2,
        'g_gap_up': 3,                        # V3: 2→3 — gap is strongest signal
        'n_neglect_base': 2,                   # V3: 1→2 — base pattern gets more weight
        'short_interest': 1,                   # V3: 2→1 — less weight on squeeze
        'analyst_upgrades': 1,
        'cap_under_10b': 0,                   # Hard prerequisite, not scored
        'ipo_within_10yr': 0,                 # Hard prerequisite, not scored
    })
    magna_pass_threshold: int = 5             # Minimum MAGNA score — V3: 4→5

    # ── Entry Filters (V3) ──
    magna_entry_min_1m_return: float = -0.05  # V3: don't catch falling knives, 1m return > -5%

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

    # Portfolio-level — OPTION_C v3 (MAGNA-optimized)
    max_positions: int = 4                  # V3: 3→4, more concurrent positions
    max_positions_per_sector: int = 2
    cash_reserve_pct: float = 0.15
    max_portfolio_heat: float = 0.70
    max_daily_loss_pct: float = 0.03
    max_correlation: float = 0.70

    # Blacklist — stocks that fail risk regardless of signal
    blacklisted_tickers: List[str] = field(default_factory=lambda: ['TSLA'])

    # Universe volatility filter — exclude stocks with >6% avg daily range (V3: was 5%)
    max_avg_daily_range_pct: float = 0.06

    # Position-level
    max_position_pct: float = 0.18
    kelly_fraction: float = 0.12            # V3: 0.08→0.12, higher bet sizing
    min_position_size: float = 500.0
    max_position_size: float = 80000.0

    # HK market position limits
    max_position_size_hkd: float = 80000.0
    max_shares_hk: int = 5000

    # Entry
    max_slippage_pct: float = 0.02
    require_volume: bool = True
    min_avg_volume: int = 50000

    # Stop management — ATR-BASED v3
    # V3: Hard stop = max(12%, 2*ATR). Trailing from high watermark.
    # No more static hard stop — stops dynamically adjust to volatility.
    atr_stop_multiplier: float = 2.0            # ATR multiplier for hard stop
    hard_stop_pct: float = 0.12                 # Floor: hard stop never tighter than 12%
    trailing_stop_pct: float = 0.10              # Trailing stop = 0.5 * ATR from high watermark
    trailing_activate_after: float = 0.10        # Activate trailing stop after 10% gain (was 15%)
    time_stop_days: int = 9999                  # No time limit — let trades play out
    # Small-cap specific (<$2B market cap)
    small_cap_max_market_cap: float = 2e9       # $2B threshold
    small_cap_hard_stop_pct: float = 0.12       # 12% stop for small caps
    small_cap_tp1_pct: float = 0.18             # 18% TP1 for small caps

    # Take Profit — REBALANCED R:R v3
    # TP: 25%, SL: max(12%, 2*ATR). R:R ~ 1:2 (favorable).
    # Split TP: 50% at +15%, 50% at +25%
    tp1_a_level_pct: float = 0.15              # TP1-A: +15% (50% sell) — V3: 10%→15%
    tp1_a_sell_pct: int = 50                   # Sell 50% at TP1-A
    tp1_b_level_pct: float = 0.25              # TP1-B: +25% (50% sell) — V3: 20%→25%
    tp1_b_sell_pct: int = 50                   # Sell 50% at TP1-B
    # Legacy — used by non-MAGNA strategies
    tp1_level_pct: float = 0.20                # TP1: +20%
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
