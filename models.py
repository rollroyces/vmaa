#!/usr/bin/env python3
"""
VMAA 2.0 — Shared Data Models
==============================
Two-stage value + momentum framework:
  Stage 1: Core Financial Fundamentals (Quality Pool)
  Stage 2: MAGNA 53/10 Momentum Triggers (Entry Signal)

All dataclasses shared across Part1, Part2, Pipeline, and Risk modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# Part 1: Core Financial Fundamentals Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Part1Result:
    """Output of Stage 1 screening — a quality-verified candidate."""
    ticker: str
    name: str
    sector: str
    industry: str

    # Market positioning
    market_cap: float                    # in USD
    market_cap_type: str                 # "deep_value" (<$250M) or "turnaround" (<$10B)
    current_price: float
    low_52w: float
    high_52w: float
    ptl_ratio: float                     # Price / 52w-low (safety margin)

    # Quality metrics
    bm_ratio: float                      # Book-to-Market
    roa: float                           # Return on Assets
    roe: float                           # Return on Equity
    ebitda_margin: float                 # EBITDA / Revenue
    fcf_yield: float                     # Free Cash Flow / Market Cap
    fcf_conversion: float                # FCF / Net Income (earnings authenticity)

    # Efficiency
    asset_growth: float                  # YoY ΔTotal Assets
    earnings_growth: float               # YoY ΔNet Income
    asset_vs_earnings: str               # "asset<earnings", "asset>=earnings", "n/a"

    # Risk
    debt_to_equity: float
    beta: float
    interest_rate_sensitive: bool

    # Screening metadata
    quality_score: float                 # 0.0–1.0 composite quality
    passed_criteria: List[str]           # Which criteria passed
    failed_criteria: List[str]           # Which criteria failed
    warnings: List[str]                  # Soft warnings (not hard fails)
    rationale: str                       # Human-readable summary

    # Data freshness
    data_date: str = ""                  # When fundamental data was fetched


# ═══════════════════════════════════════════════════════════════════
# Part 2: MAGNA 53/10 Signal Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Part2Signal:
    """Output of Stage 2 screening — a MAGNA momentum signal."""
    ticker: str

    # MAGNA 53/10 components (each scored 0 or 1, except where noted)
    m_earnings_accel: bool = False        # Massive Earnings Acceleration
    a_sales_accel: bool = False           # Acceleration of Sales
    g_gap_up: bool = False               # Gap Up >4%
    g_premarket_vol_ok: bool = False     # Pre-market volume >100K
    n_neglect_base: bool = False         # Neglect / Base pattern detected
    short_interest_high: bool = False    # Short Interest elevated (score 0-2)
    short_interest_score: int = 0        # 0, 1, or 2 points
    analyst_upgrades: bool = False       # ≥3 analysts + target premium + recent upgrade
    analyst_recently_upgraded: bool = False  # Proxy: target ≥5% higher than last cached run
    cap_under_10b: bool = False          # Market Cap < $10B
    ipo_within_10yr: bool = False        # IPO within 10 years

    # Detailed metrics
    eps_growth_qoq: float = 0.0          # Latest quarter EPS growth
    eps_growth_prev_qoq: float = 0.0     # Previous quarter EPS growth
    eps_acceleration: float = 0.0        # Acceleration (current - prev)
    revenue_growth_qoq: float = 0.0      # Latest quarter revenue growth
    revenue_growth_prev_qoq: float = 0.0
    revenue_acceleration: float = 0.0
    gap_pct: float = 0.0                 # Gap-up percentage
    premarket_volume: int = 0            # Pre-market volume
    short_ratio: float = 0.0             # Short interest ratio
    short_pct_float: float = 0.0         # % of float short
    base_duration_months: float = 0.0    # How long in base (months)
    base_range_pct: float = 0.0          # Price range within base (%)
    analyst_count: int = 0               # Number of analysts
    analyst_target_mean: float = 0.0     # Mean analyst target
    ipo_years: Optional[float] = None    # Years since IPO

    # Composite
    magna_score: float = 0.0             # 0–10 composite score (graduated, half-point increments)
    trigger_signals: List[str] = field(default_factory=list)  # Which signals fired
    entry_ready: bool = False            # True when G or MA triggers fire

    # Data freshness
    signal_date: str = ""                # When signal was detected


# ═══════════════════════════════════════════════════════════════════
# Combined: A stock that passed Part 1 AND has Part 2 signal
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VMAACandidate:
    """A stock that passed both Stage 1 (Quality) and Stage 2 (Momentum)."""
    ticker: str
    part1: Part1Result
    part2: Part2Signal
    sentiment: Optional[Any] = None     # SentimentResult from part3_sentiment

    # Combined assessment
    composite_rank: float = 0.0          # Overall ranking score
    entry_triggered: bool = False        # Entry signal active

    @property
    def quality_score(self) -> float:
        return self.part1.quality_score

    @property
    def magna_score(self) -> float:
        return self.part2.magna_score


# ═══════════════════════════════════════════════════════════════════
# Trade Decision (carried forward from v1, refined)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TradeDecision:
    """Complete trade decision with entry, stops, sizing, and risk."""
    ticker: str
    action: str                        # BUY / SELL / HOLD / MONITOR
    quantity: int
    entry_price: float
    entry_method: str                  # "gap_entry" | "52w_low_bounce" | "base_breakout"
    stop_loss: float
    stop_type: str                     # "ATR" | "Hard" | "Structural" | "Fallback"
    take_profits: List[Dict[str, Any]]
    trailing_stop_pct: float
    time_stop_days: int
    position_pct: float
    risk_amount: float
    reward_ratio: float
    confidence_score: float            # 0.0–1.0
    risk_flags: List[str]
    rationale: str


# ═══════════════════════════════════════════════════════════════════
# Market Regime
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    """Overall market conditions affecting sizing and risk appetite."""
    spy_price: float
    spy_ma50: float
    above_ma50: bool
    volatility_20d: float
    vol_regime: str                    # "LOW" | "NORMAL" | "HIGH"
    dd_from_3mo_high: float
    market_ok: bool                    # Favorable for long positions
    position_scalar: float             # Multiplier for position sizing (0.5–1.0)
    vix_proxy: float = 0.0
    interest_rate_regime: str = ""     # "rising" | "falling" | "stable"


# ═══════════════════════════════════════════════════════════════════
# Pipeline Run Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Complete pipeline run output."""
    timestamp: str
    mode: str                          # "DRY_RUN" | "LIVE"

    # Market
    market: Dict[str, Any]

    # Stage 1
    part1_total_scanned: int
    part1_passed: int
    part1_results: List[Part1Result]

    # Stage 2
    part2_signals: List[Part2Signal]

    # Combined
    candidates: List[VMAACandidate]

    # Decisions
    decisions: List[TradeDecision]

    # Execution
    executed_count: int
    skipped_count: int

    # Performance
    elapsed_seconds: float
    status: str                        # "complete" | "no_candidates" | "error"
