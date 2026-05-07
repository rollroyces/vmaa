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
    # Entry slippage penalty when next-day open is unavailable (0.005 = 0.5%)
    entry_slippage_pct: float = 0.005

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
    # WIDE_STOP optimized values
    max_positions: int = 5
    max_positions_per_sector: int = 2
    max_position_pct: float = 0.18
    kelly_fraction: float = 0.15              # Matches winning WIDE_STOP tuning
    kelly_fraction_bull: float = 0.40         # Aggressive in bull markets
    kelly_fraction_bear: float = 0.15         # Conservative in bear markets
    min_position_size: float = 500.0

    # ── Stop Loss / Take Profit ──
    # WIDE_STOP strategy (backtest winner 2026-05-07)
    atr_stop_multiplier: float = 3.0
    hard_stop_pct: float = 0.25              # Wide stop — allows mean-reversion
    trailing_stop_pct: float = 0.12
    trailing_activate_after: float = 0.18    # Activate trailing after 18% gain
    time_stop_days: int = 9999               # No time limit — let trades fully play out
    # Take profit: full exit at TP1 (the #1 improvement — partial fills were destroying returns)
    tp_levels: List[float] = field(default_factory=lambda: [0.15, 0.25, 0.40])
    tp_sell_fractions: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # TP1: sell 100% at +15% | TP2: 100% at +25% | TP3: 100% at +40%

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

    # ── Signal Staleness ──
    # Days since last Part 2 signal before a ticker is considered stale.
    # Stale tickers (passing Part 1 but failing Part 2 for >90 days) are
    # skipped to avoid wasted compute on tickers that never generate entries.
    signal_stale_days: int = 90

    # ── Bear Market Defense ──
    bear_market_ma_period: int = 200          # MA period for bear detection
    bear_market_enabled: bool = True          # Enable bear-market risk reduction
    bear_max_positions: int = 2               # Max positions when SPY < MA200
    bear_hard_stop_pct: float = 0.10          # Tighter stop in bear markets
    bear_kelly_fraction: float = 0.10         # Conservative sizing in bear
    bear_min_quality_score: float = 0.50      # Higher quality floor in bear

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
            'signal_stale_days': self.signal_stale_days,
            'tickers': self.tickers,
        }


# Singleton
BTC = BacktestConfig()
