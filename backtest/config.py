#!/usr/bin/env python3
"""
Backtest Configuration
======================
All parameters for historical simulation of the VMAA 2.0 pipeline.

Includes:
  - Date ranges and frequency
  - Capital and cost modeling
  - Benchmark selection
  - Walk-forward parameters
  - Slippage and commission models
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SlippageConfig:
    """Transaction cost / slippage model."""
    # Fixed commission per trade (USD)
    fixed_commission: float = 0.99

    # Percentage-based commission (e.g., 0.005 = 0.5 bps)
    pct_commission: float = 0.00005

    # Slippage as percentage of trade (0.001 = 0.1% per side)
    slippage_pct: float = 0.001

    # Minimum slippage cost per share (to model spread for low-price stocks)
    min_slippage_per_share: float = 0.01

    # Volume-based slippage: additional pct per 1% of daily volume traded
    volume_impact_factor: float = 0.0001

    def total_cost(self, price: float, quantity: int,
                   daily_volume: int = 0) -> float:
        """Calculate total transaction cost for a trade."""
        notional = price * quantity
        fixed = self.fixed_commission
        pct = notional * self.pct_commission
        slippage = notional * self.slippage_pct

        # Volume impact: if trading > 1% of daily volume, add slippage
        vol_impact = 0.0
        if daily_volume > 0:
            vol_fraction = quantity / daily_volume
            if vol_fraction > 0.01:
                vol_impact = notional * (vol_fraction - 0.01) * self.volume_impact_factor

        # Floor slippage at min per share
        min_slippage = quantity * self.min_slippage_per_share
        slippage = max(slippage, min_slippage)

        return round(fixed + pct + slippage + vol_impact, 2)


@dataclass
class BacktestConfig:
    """Master configuration for a backtest run."""

    # ── Date Range ──
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"

    # ── Capital ──
    initial_capital: float = 100_000.0

    # ── Frequency ──
    # "daily", "weekly", "monthly"
    rebalance_frequency: str = "monthly"

    # ── Benchmark ──
    benchmark_ticker: str = "SPY"

    # ── Cost Model ──
    slippage: SlippageConfig = field(default_factory=SlippageConfig)

    # ── Universe ──
    # If empty, will use the pipeline's default universe
    tickers: List[str] = field(default_factory=list)
    max_tickers: int = 0  # 0 = no limit

    # ── Screening ──
    # Whether to re-run Part 1 on each rebalance date (expensive but accurate)
    re_screen_fundamentals: bool = True
    # Whether to re-run Part 2 on each rebalance date
    re_screen_magna: bool = True
    # If fundamentals rescreening is off, use a fixed quality pool
    quality_pool_path: str = ""

    # ── Position Sizing ──
    # Override RiskConfig values for backtest
    max_positions: int = 5
    max_positions_per_sector: int = 2
    max_position_pct: float = 0.25
    kelly_fraction: float = 0.25              # Base Kelly fraction
    kelly_fraction_bull: float = 0.40         # Aggressive in bull markets
    kelly_fraction_bear: float = 0.15         # Conservative in bear markets
    min_position_size: float = 500.0

    # ── Stop Loss / Take Profit ──
    atr_stop_multiplier: float = 2.0
    hard_stop_pct: float = 0.15
    trailing_stop_pct: float = 0.10
    trailing_activate_after: float = 0.12  # Activate trailing after 10% gain
    time_stop_days: int = 180
    # Take profit levels (pct above entry)
    tp_levels: List[float] = field(default_factory=lambda: [0.12, 0.22, 0.35])
    tp_sell_fractions: List[float] = field(default_factory=lambda: [0.30, 0.30, 0.40])

    # ── Sector Rotation ──
    sector_momentum_enabled: bool = True
    sector_momentum_lookback: int = 60
    sector_momentum_top_n: int = 3
    sector_momentum_boost: int = 2

    # ── Walk-Forward Parameters ──
    # If enabled, split into in-sample / out-of-sample periods
    walk_forward: bool = False
    wf_train_months: int = 24   # In-sample training period
    wf_test_months: int = 6     # Out-of-sample test period
    wf_step_months: int = 6     # How much to advance each walk-forward step

    # ── Data ──
    # Reporting lag for fundamental data (days after quarter end)
    fundamental_report_lag_days: int = 45
    # Cache directory for downloaded data
    cache_dir: str = "backtest/cache"

    # ── Output ──
    output_dir: str = "backtest/output"
    save_trade_log: bool = True
    save_equity_curve: bool = True

    # ── Performance ──
    # Number of parallel workers for data fetching
    workers: int = 1
    # Skip tickers with insufficient data
    min_history_days: int = 252  # 1 year minimum

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'rebalance_frequency': self.rebalance_frequency,
            'benchmark_ticker': self.benchmark_ticker,
            'max_positions': self.max_positions,
            'tickers': self.tickers,
        }


# Singleton
BTC = BacktestConfig()
