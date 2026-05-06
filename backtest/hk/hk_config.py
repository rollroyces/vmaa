#!/usr/bin/env python3
"""
HK Backtest Configuration
==========================
HK-specific parameters for historical simulation of VMAA-HK pipeline.

Key differences from US config:
  - B/M threshold: 0.15 (vs US 0.20) — HK stocks trade at lower B/M
  - ROA threshold: 1% (vs US 0%) — non-zero floor for HK
  - FCF/Y threshold: 1% (vs US 2%) — HK dividend culture affects FCF
  - Financial sector: auto-pass on B/M, ROA→ROE, EBITDA skip
  - Currency: HKD
  - Benchmark: 2800.HK (TraHK ETF)
  - Universe: HSI constituents

Matches thresholds from pipeline_hk.py live scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from pipeline_hk import HSI_TICKERS, FINANCIAL_SECTORS, FINANCIAL_INDUSTRIES


# ═══════════════════════════════════════════════════════════════════
# HK Thresholds (matching pipeline_hk.py screening logic)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HKScreeningThresholds:
    """HK-specific screening thresholds — mirrors pipeline_hk.py logic."""

    # ── Part 1: Fundamentals ──
    min_bm_ratio: float = 0.15           # HK: 0.15 (US: 0.20)
    min_roa: float = 0.01                # HK: 1% (US: 0%)
    min_roe_financial: float = 0.05      # Financials use ROE ≥ 5%
    min_ebitda_margin: float = 0.03      # HK: 3% (US: 5%)
    min_fcf_yield: float = 0.01          # HK: 1% (US: 2%)
    min_fcf_conversion: float = 0.30     # HK: 30% (US: 50%)
    max_ptl_ratio: float = 1.35          # HK: 1.35x 52w-low (US: 1.50)
    min_rev_growth: float = -0.15        # Revenue not declining >15% YoY
    min_market_cap_hkd: float = 10e9     # HK: HKD 10B (≈ USD 1.3B)

    # ── Pass thresholds ──
    min_core_passed: int = 4             # Minimum core criteria to pass Part 1
    min_quality_score: float = 0.25      # Minimum composite quality score

    # ── Part 2: MAGNA ──
    eps_accel_min: float = 0.15          # HK: 15% EPS accel (US: 20%)
    sales_accel_min: float = 0.10        # HK: 10% revenue accel (US: 10%)
    gap_min_pct: float = 0.04            # Gap ≥ 4%
    gap_vol_mult: float = 1.5            # Volume ≥ 1.5x 20-day avg
    magna_min_score: int = 1             # Minimum MAGNA score to consider

    # ── Financial Sector Bypasses ──
    financial_sectors: Set[str] = field(default_factory=lambda: FINANCIAL_SECTORS)
    financial_industries: Set[str] = field(default_factory=lambda: FINANCIAL_INDUSTRIES)


# ═══════════════════════════════════════════════════════════════════
# HK Backtest Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HKSlippageConfig:
    """HK transaction cost model — HKD denominated."""

    # Fixed commission per trade (HKD)
    fixed_commission: float = 50.0           # ~HKD 50 broker fee

    # Percentage-based commission (0.25% typical HK brokerage)
    pct_commission: float = 0.0025

    # Stamp duty (0.1% per side — HK government)
    stamp_duty: float = 0.001

    # Slippage as percentage of trade
    slippage_pct: float = 0.002              # 0.2% — wider spreads on HK stocks

    # Minimum slippage per share
    min_slippage_per_share: float = 0.05     # HKD 0.05 — wider spreads

    def total_cost(self, price: float, quantity: int,
                   daily_volume: int = 0) -> float:
        """Calculate total HKD transaction cost for a trade."""
        notional = price * quantity
        fixed = self.fixed_commission
        pct = notional * self.pct_commission
        stamp = notional * self.stamp_duty
        slippage = notional * self.slippage_pct

        min_slippage = quantity * self.min_slippage_per_share
        slippage = max(slippage, min_slippage)

        return round(fixed + pct + stamp + slippage, 2)


@dataclass
class HKBacktestConfig:
    """Master configuration for an HK backtest run."""

    # ── Thresholds ──
    thresholds: HKScreeningThresholds = field(default_factory=HKScreeningThresholds)

    # ── Date Range ──
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"

    # ── Capital (HKD) ──
    initial_capital: float = 500_000.0       # HKD 500K

    # ── Frequency ──
    rebalance_frequency: str = "monthly"

    # ── Benchmark ──
    benchmark_ticker: str = "2800.HK"         # TraHK ETF (use ^HSI for index)
    hsi_ticker: str = "^HSI"

    # ── Cost Model ──
    slippage: HKSlippageConfig = field(default_factory=HKSlippageConfig)

    # ── Universe ──
    tickers: List[str] = field(default_factory=lambda: list(HSI_TICKERS))
    max_tickers: int = 0

    # ── Position Sizing ──
    max_positions: int = 8
    max_positions_per_sector: int = 2
    max_position_pct: float = 0.20
    kelly_fraction: float = 0.25
    min_position_hkd: float = 2000.0          # Min HKD 2K position

    # ── Stop Loss / Take Profit ──
    atr_stop_multiplier: float = 2.0
    hard_stop_pct: float = 0.15              # 15% for HK (wider: higher vol)
    trailing_stop_pct: float = 0.10
    trailing_activate_after: float = 0.12
    time_stop_days: int = 90                 # HK: longer runway
    tp_levels: List[float] = field(default_factory=lambda: [0.12, 0.22, 0.35])
    tp_sell_fractions: List[float] = field(default_factory=lambda: [0.30, 0.30, 0.40])

    # ── Data ──
    fundamental_report_lag_days: int = 45
    cache_dir: str = "backtest/cache"
    min_history_days: int = 252

    # ── Output ──
    output_dir: str = "backtest/hk/output"
    save_trade_log: bool = True
    save_equity_curve: bool = True

    # ── Currency ──
    currency: str = "HKD"

    def to_dict(self) -> dict:
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'currency': self.currency,
            'rebalance_frequency': self.rebalance_frequency,
            'benchmark_ticker': self.benchmark_ticker,
            'max_positions': self.max_positions,
            'tickers': self.tickers[:10] + ['...'],
        }


# Singleton
HKC = HKBacktestConfig()
