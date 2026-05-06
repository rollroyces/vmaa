#!/usr/bin/env python3
"""
Core Backtesting Engine
========================
Walk-forward simulation of the VMAA 2.0 pipeline over historical data.

Architecture:
  1. Walk-through time: loop over rebalance dates
  2. On each date: reconstruct available data → run Part 1 → Part 2 → produce signals
  3. Execute simulated trades: enter positions, track P&L, manage stops
  4. Record all events: signals, trades, stops, daily equity

Key Design Decisions:
  - Monthly rebalance is the default (matching pipeline's quarterly data cadence)
  - Signal regeneration uses historical snapshots (no forward-looking bias)
  - Portfolio simulation tracks cash, positions, and open orders
  - Stop loss / take profit checked daily on bar data
  - Transaction costs modeled per SlippageConfig
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Parent VMAA imports
import sys
from pathlib import Path
_vmaa_root = Path(__file__).resolve().parent.parent
if str(_vmaa_root) not in sys.path:
    sys.path.insert(0, str(_vmaa_root))

from models import (
    Part1Result, Part2Signal, VMAACandidate, TradeDecision, MarketRegime
)
from config import P1C, P2C, RC
from backtest.config import BacktestConfig, BTC, SlippageConfig
from backtest.data import HistoricalDataLoader, HistoricalSnapshot

logger = logging.getLogger("vmaa.backtest.engine")


# ═══════════════════════════════════════════════════════════════════
# Portfolio State
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Tracks a live position during backtest."""
    ticker: str
    entry_date: str
    entry_price: float
    quantity: int
    cost_basis: float
    stop_loss: float
    take_profits: List[Dict[str, Any]]
    trailing_stop_pct: float
    trailing_activated: bool = False
    trailing_high: float = 0.0
    time_stop_days: int = 60
    entry_method: str = ""
    confidence: float = 0.0
    sector: str = ""

    # P&L tracking
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update(self, price: float) -> None:
        """Update position state with current price."""
        self.current_price = price
        self.market_value = price * self.quantity
        self.unrealized_pnl = self.market_value - self.cost_basis
        self.unrealized_pnl_pct = (price / self.entry_price - 1) * 100

        # Trailing stop logic
        if not self.trailing_activated:
            if self.unrealized_pnl_pct >= 10.0:  # Activate after 10% gain
                self.trailing_activated = True
                self.trailing_high = price
        else:
            self.trailing_high = max(self.trailing_high, price)

    @property
    def trailing_stop_price(self) -> float:
        """Current trailing stop level."""
        if self.trailing_activated:
            return self.trailing_high * (1 - self.trailing_stop_pct)
        return 0.0

    def check_stop(self, bar_low: float, bar_high: float,
                   bar_open: float) -> Tuple[bool, str, float]:
        """
        Check if a stop was triggered during this bar.
        Returns: (triggered, reason, exit_price)
        """
        # 1. Hard stop
        stop = self.stop_loss
        if bar_low <= stop:
            return True, "hard_stop", stop

        # 2. Trailing stop
        trail = self.trailing_stop_price
        if self.trailing_activated and trail > 0 and bar_low <= trail:
            return True, "trailing_stop", trail

        # 3. Take profit
        for tp in self.take_profits:
            if bar_high >= tp['level']:
                # Partially fill: sell fraction at TP level
                return True, f"take_profit_{tp['label']}", tp['level']

        return False, "", 0.0


@dataclass
class PortfolioState:
    """Full portfolio state at a point in time."""
    date: str
    cash: float
    equity: float  # cash + market value of positions
    positions: Dict[str, Position] = field(default_factory=dict)
    peak_equity: float = 0.0
    drawdown: float = 0.0

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def invested(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def invested_pct(self) -> float:
        return (self.invested / self.equity * 100) if self.equity > 0 else 0.0

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update all positions with current prices and compute equity."""
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos.update(prices[ticker])
        self._recalc_equity()

    def _recalc_equity(self) -> None:
        self.equity = self.cash + self.invested
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        if self.peak_equity > 0:
            self.drawdown = (self.equity - self.peak_equity) / self.peak_equity


# ═══════════════════════════════════════════════════════════════════
# Trade Record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Record of a completed trade (entry → exit)."""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    cost: float
    return_pct: float
    exit_reason: str  # "take_profit", "hard_stop", "trailing_stop", "time_stop", "signal_exit"
    holding_days: int
    entry_method: str = ""
    sector: str = ""
    confidence: float = 0.0
    is_win: bool = False


@dataclass
class DailyRecord:
    """Daily portfolio snapshot for equity curve."""
    date: str
    equity: float
    cash: float
    invested: float
    daily_return: float
    drawdown: float
    num_positions: int


# ═══════════════════════════════════════════════════════════════════
# Backtest Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    config: Dict[str, Any]
    start_date: str
    end_date: str

    # Equity
    initial_capital: float
    final_equity: float
    total_return: float
    total_return_pct: float

    # Daily series
    equity_curve: List[DailyRecord]
    benchmark_curve: List[Dict[str, Any]]

    # Trades
    trades: List[TradeRecord]
    num_trades: int

    # Monthly returns
    monthly_returns: Dict[str, float]

    # Statistics (populated by metrics module)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def trade_log_df(self) -> pd.DataFrame:
        """Trade log as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            'ticker': t.ticker,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'gross_pnl': t.gross_pnl,
            'net_pnl': t.net_pnl,
            'cost': t.cost,
            'return_pct': t.return_pct,
            'exit_reason': t.exit_reason,
            'holding_days': t.holding_days,
            'is_win': t.is_win,
        } for t in self.trades])

    @property
    def equity_df(self) -> pd.DataFrame:
        """Equity curve as DataFrame."""
        return pd.DataFrame([{
            'date': r.date,
            'equity': r.equity,
            'daily_return': r.daily_return,
            'drawdown': r.drawdown,
            'num_positions': r.num_positions,
        } for r in self.equity_curve])


# ═══════════════════════════════════════════════════════════════════
# Signal Generators (Historical Versions)
# ═══════════════════════════════════════════════════════════════════

class HistoricalSignalGenerator:
    """
    Regenerate Part 1 and Part 2 signals at historical points-in-time.
    
    FIXED (2026-05): Previously duplicated ~200+ lines of screening
    logic from part1_fundamentals.py and part2_magna.py. Changes to
    thresholds in config.py or logic in the live modules did NOT
    propagate to backtests.
    
    Now delegates to live modules via prefetched data dicts:
      - screen_fundamentals(ticker, prefetched={...}) from part1_fundamentals
      - screen_magna(ticker, prefetched={...}) from part2_magna
    
    Live modules use prefetched info dict + price history when provided,
    avoiding redundant yfinance calls. Quarterly financials (balance_sheet,
    quarterly_financials) still accessed via yf.Ticker (current data) —
    this is an acknowledged limitation for point-in-time accuracy.
    """

    def __init__(self, config: BacktestConfig = BTC):
        self.config = config
        self._daily_prices: Dict[str, pd.DataFrame] = {}
        # Cache of current yfinance data (used as historical proxy — 
        # yfinance does not provide historical fundamentals)
        self._yf_cache: Dict[str, dict] = {}
        self._yf_ticker_cache: Dict[str, Any] = {}
        # Set by BacktestEngine after data load

    def cache_yfinance(self, tickers: list[str]):
        """
        Pre-fetch current yfinance data for all tickers.
        Cached data is used as a proxy for historical fundamentals.
        This is the standard approach for retail backtesting — the
        limitation is documented and acknowledged.
        """
        import yfinance as yf
        for t in tickers:
            try:
                yft = yf.Ticker(t)
                self._yf_ticker_cache[t] = yft
                self._yf_cache[t] = yft.info
            except Exception:
                self._yf_cache[t] = {}

    def _get_cached_info(self, ticker: str, snapshot: HistoricalSnapshot) -> dict:
        """
        Build an info dict from cached yfinance data + historical snapshot.
        
        Strategy:
          - Price-dependent fields use snapshot (historical) data
          - Fundamental fields use cached (current) data as proxy
          - Also includes balance_sheet / quarterly_financials via cached ticker
        """
        cached = self._yf_cache.get(ticker, {})
        
        info = dict(cached)  # Start with current yfinance data
        
        # Override price-dependent fields with historical snapshot data
        info['regularMarketPrice'] = snapshot.close
        info['currentPrice'] = snapshot.close
        info['previousClose'] = snapshot.close
        info['fiftyTwoWeekLow'] = snapshot.low_52w
        info['fiftyTwoWeekHigh'] = snapshot.high_52w
        info['marketCap'] = snapshot.market_cap
        info['fiftyTwoWeekLow'] = snapshot.low_52w
        info['fiftyTwoWeekHigh'] = snapshot.high_52w
        
        return info

    @staticmethod
    def _snapshot_to_info(snapshot: HistoricalSnapshot) -> dict:
        """
        Legacy method — kept for backward compatibility.
        Use _get_cached_info() instead for richer data.
        """
        return {
            'marketCap': snapshot.market_cap,
            'bookValue': snapshot.book_value,
            'returnOnAssets': snapshot.roa,
            'returnOnEquity': snapshot.roe,
            'ebitda': snapshot.ebitda,
            'totalRevenue': snapshot.total_revenue,
            'freeCashflow': snapshot.free_cashflow,
            'netIncomeToCommon': snapshot.net_income,
            'netIncome': snapshot.net_income,
            'totalAssets': snapshot.total_assets,
            'totalDebt': snapshot.total_debt,
            'debtToEquity': snapshot.debt_to_equity,
            'beta': snapshot.beta,
            'sector': snapshot.sector,
            'industry': snapshot.industry,
            'shortName': snapshot.short_name,
            'shortRatio': snapshot.short_ratio,
            'shortPercentOfFloat': snapshot.short_pct_float,
            'numberOfAnalystOpinions': snapshot.analyst_count,
            'targetMeanPrice': snapshot.analyst_target,
            'averageVolume': snapshot.volume,
            'regularMarketPrice': snapshot.close,
            'currentPrice': snapshot.close,
            'previousClose': snapshot.close,
            'fiftyTwoWeekLow': snapshot.low_52w,
            'fiftyTwoWeekHigh': snapshot.high_52w,
        }

    def _build_prefetched(self, snapshot: HistoricalSnapshot) -> dict:
        """
        Build a prefetched dict from cached yfinance data + historical snapshot.
        
        Uses cached CURRENT fundamentals (only source available) overlaid with
        historical price-dependent fields from the snapshot for point-in-time
        accuracy on price-based checks (PTL, gaps, volume).
        """
        ticker = snapshot.ticker
        hist = self._daily_prices.get(ticker) if self._daily_prices else None
        if hist is not None:
            # Clip to snapshot date (no forward-looking)
            target = pd.Timestamp(snapshot.date)
            hist = hist[hist.index <= target].copy()

        # Use cached yfinance data with historical price overrides
        if self._yf_cache:
            info = self._get_cached_info(ticker, snapshot)
        else:
            info = self._snapshot_to_info(snapshot)
        
        # Use cached ticker object (avoids re-fetching current data for each call)
        ticker_obj = self._yf_ticker_cache.get(ticker) if self._yf_ticker_cache else None
        if ticker_obj is None:
            import yfinance as yf
            ticker_obj = yf.Ticker(ticker)

        return {
            'info': info,
            'hist': hist,
            'ticker': ticker_obj,
        }

    def screen_part1(self, snapshot: HistoricalSnapshot) -> Optional[Part1Result]:
        """
        Run Part 1 quality screening via live module.
        Uses prefetched snapshot data — no duplicated logic.
        """
        try:
            from part1_fundamentals import screen_fundamentals
            prefetched = self._build_prefetched(snapshot)
            return screen_fundamentals(
                snapshot.ticker, sector_medians=None, prefetched=prefetched
            )
        except Exception as e:
            logger.debug(f"  {snapshot.ticker}: Part 1 via live module failed — {e}")
            return None

    def screen_part2(self, snapshot: HistoricalSnapshot,
                     part1: Optional[Part1Result] = None) -> Optional[Part2Signal]:
        """
        Run Part 2 MAGNA screening via live module.
        Uses prefetched snapshot data — no duplicated logic.
        """
        try:
            from part2_magna import screen_magna
            prefetched = self._build_prefetched(snapshot)
            return screen_magna(
                snapshot.ticker, part1=part1, prefetched=prefetched
            )
        except Exception as e:
            logger.debug(f"  {snapshot.ticker}: Part 2 via live module failed — {e}")
            return None

    def get_market_regime(self, benchmark_hist: pd.DataFrame,
                          target_date: str) -> MarketRegime:
        """Compute market regime at a historical point in time."""
        target = pd.Timestamp(target_date)
        available = benchmark_hist[benchmark_hist.index <= target]
        if available.empty or len(available) < 50:
            return MarketRegime(
                spy_price=0, spy_ma50=0, above_ma50=True,
                volatility_20d=0.15, vol_regime="UNKNOWN",
                dd_from_3mo_high=0, market_ok=True, position_scalar=0.75,
            )

        current = float(available['Close'].iloc[-1])
        ma50 = float(available['Close'].rolling(50).mean().iloc[-1]) if len(available) >= 50 else current
        above_ma = current > ma50 if ma50 > 0 else True

        returns = available['Close'].pct_change().dropna()
        vol_20d = float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else 0.15

        if vol_20d < 0.12:
            vol_regime = "LOW"
            scalar = 1.0
        elif vol_20d < 0.22:
            vol_regime = "NORMAL"
            scalar = 0.80
        else:
            vol_regime = "HIGH"
            scalar = 0.50

        high_3mo = float(available['High'].tail(63).max())
        dd_from_high = (current - high_3mo) / high_3mo if high_3mo > 0 else 0
        market_ok = above_ma and (dd_from_high > -0.12)

        return MarketRegime(
            spy_price=round(current, 2),
            spy_ma50=round(ma50, 2),
            above_ma50=above_ma,
            volatility_20d=round(vol_20d, 4),
            vol_regime=vol_regime,
            dd_from_3mo_high=round(dd_from_high, 4),
            market_ok=market_ok,
            position_scalar=scalar,
        )


# ═══════════════════════════════════════════════════════════════════
# Core Backtesting Engine
# ═══════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Walk-forward backtesting engine for VMAA 2.0.

    Usage:
        engine = BacktestEngine(config)
        result = engine.run()
        print(f"Total Return: {result.total_return_pct:.1f}%")
    """

    def __init__(self, config: BacktestConfig = BTC):
        self.config = config
        self.loader = HistoricalDataLoader(config)
        self.signal_gen = HistoricalSignalGenerator(config)
        self.cost_model = config.slippage

        # State
        self.portfolio: PortfolioState = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[DailyRecord] = []
        self.benchmark_curve: List[Dict[str, Any]] = []

        # Daily tracking
        self._prev_equity: float = 0.0
        self._daily_prices: Dict[str, pd.DataFrame] = {}
        self._market_regime = None  # Set per-date in _run_rebalance

    def run(self, tickers: Optional[List[str]] = None) -> BacktestResult:
        """
        Execute the full backtest.

        Args:
            tickers: Universe to trade. If None, uses config.tickers.

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        start_ts = time.time()
        tickers = tickers or self.config.tickers
        if not tickers:
            tickers = self._get_default_universe()
        if self.config.max_tickers > 0:
            tickers = tickers[:self.config.max_tickers]

        logger.info("=" * 70)
        logger.info(f"VMAA Backtest: {self.config.start_date} → {self.config.end_date}")
        logger.info(f"Universe: {len(tickers)} tickers | "
                    f"Capital: ${self.config.initial_capital:,.0f} | "
                    f"Frequency: {self.config.rebalance_frequency}")
        logger.info("=" * 70)

        # ── Step 1: Load Data ──
        logger.info("\n[1/5] Loading historical data...")
        bench_ticker = self.config.benchmark_ticker
        all_tickers = list(dict.fromkeys([bench_ticker] + tickers))
        self.loader.fetch_all(all_tickers,
                             self.config.start_date,
                             self.config.end_date)

        # Verify tickers have data
        valid_tickers = [t for t in tickers if self.loader.get_price_history(t) is not None
                         and len(self.loader.get_price_history(t)) >= self.config.min_history_days]
        skipped = len(tickers) - len(valid_tickers)
        if skipped:
            logger.info(f"  Skipped {skipped} tickers with insufficient data")
        tickers = valid_tickers
        logger.info(f"  Loaded data for {len(tickers)} valid tickers")

        # Benchmark
        bench_hist = self.loader.get_price_history(bench_ticker)
        if bench_hist is None or bench_hist.empty:
            logger.warning(f"  Benchmark {bench_ticker} data not available")
            bench_hist = pd.DataFrame()

        # ── Step 2: Build Rebalance Dates ──
        logger.info("\n[2/5] Building rebalance schedule...")
        rebalance_dates = self._get_rebalance_dates(
            self.config.start_date, self.config.end_date, tickers
        )
        logger.info(f"  {len(rebalance_dates)} rebalance dates "
                    f"({self.config.rebalance_frequency})")

        # ── Step 3: Initialize Portfolio ──
        logger.info("\n[3/5] Initializing portfolio...")
        self.portfolio = PortfolioState(
            date=self.config.start_date,
            cash=self.config.initial_capital,
            equity=self.config.initial_capital,
            peak_equity=self.config.initial_capital,
        )
        self.trades = []
        self.equity_curve = []
        self._prev_equity = self.config.initial_capital

        # Pre-load daily price data for all tickers
        self._daily_prices = {}
        for t in tickers:
            hist = self.loader.get_price_history(t)
            if hist is not None:
                self._daily_prices[t] = hist

        # Connect daily prices to signal generator (enables live-module prefetch)
        self.signal_gen._daily_prices = self._daily_prices
        
        # Pre-fetch current yfinance data for all tickers
        # Used as proxy for historical fundamentals in screening
        all_universe = tickers or self.config.tickers or []
        if all_universe:
            logger.info(f"  Pre-fetching yfinance data for {len(all_universe)} tickers...")
            self.signal_gen.cache_yfinance(all_universe)
            logger.info(f"  yfinance cache: {len(self.signal_gen._yf_cache)} tickers")
        else:
            logger.warning("  No ticker universe — yfinance cache skipped")

        # ── Step 4: Walk-Forward Simulation ──
        logger.info("\n[4/5] Running walk-forward simulation...")
        all_dates = self._get_trading_days(tickers, self.config.start_date,
                                           self.config.end_date)
        logger.info(f"  {len(all_dates)} trading days to simulate")

        rebalance_idx = 0
        for di, date_str in enumerate(all_dates):
            if (di + 1) % 100 == 0:
                logger.info(f"  Day {di+1}/{len(all_dates)} | "
                           f"Equity: ${self.portfolio.equity:,.0f} | "
                           f"Positions: {self.portfolio.num_positions} | "
                           f"Trades: {len(self.trades)}")

            # Get prices for today
            prices_today = self._get_prices_on_date(tickers, date_str)

            # Update existing positions
            self.portfolio.update_prices(prices_today)
            self._check_exits(date_str, prices_today)

            # Rebalance — use >= to handle holidays where date ≠ trading day
            while (rebalance_idx < len(rebalance_dates) and
                date_str >= rebalance_dates[rebalance_idx]):
                self._do_rebalance(date_str, tickers, bench_hist)
                rebalance_idx += 1

            # Check time stops
            self._check_time_stops(date_str, prices_today)

            # Record daily equity
            daily_ret = 0.0
            if self._prev_equity > 0:
                daily_ret = (self.portfolio.equity - self._prev_equity) / self._prev_equity
            self._prev_equity = self.portfolio.equity

            self.equity_curve.append(DailyRecord(
                date=date_str,
                equity=self.portfolio.equity,
                cash=self.portfolio.cash,
                invested=self.portfolio.invested,
                daily_return=round(daily_ret, 6),
                drawdown=round(self.portfolio.drawdown, 6),
                num_positions=self.portfolio.num_positions,
            ))

            # Benchmark tracking
            bm_val = 0.0
            if bench_hist is not None and not bench_hist.empty:
                bm_sub = bench_hist[bench_hist.index <= pd.Timestamp(date_str)]
                if not bm_sub.empty:
                    bm_val = float(bm_sub['Close'].iloc[-1])
            self.benchmark_curve.append({
                'date': date_str,
                'price': bm_val,
                'value': bm_val / (self._bench_start_price or 1.0) * self.config.initial_capital
                if self._bench_start_price else 0.0,
            })

        # ── Step 5: Close Remaining Positions ──
        final_date = all_dates[-1] if all_dates else self.config.end_date
        final_prices = self._get_prices_on_date(tickers, final_date)
        self._close_all_positions(final_date, final_prices)

        # ── Build Result ──
        total_return = self.portfolio.equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital * 100)

        elapsed = time.time() - start_ts
        logger.info(f"\n[5/5] Backtest complete in {elapsed:.0f}s")
        logger.info(f"  Initial: ${self.config.initial_capital:,.0f}")
        logger.info(f"  Final:   ${self.portfolio.equity:,.0f}")
        logger.info(f"  Return:  {total_return_pct:.1f}%")
        logger.info(f"  Trades:  {len(self.trades)}")
        logger.info(f"  Win Rate:{sum(1 for t in self.trades if t.is_win)/max(len(self.trades),1)*100:.0f}%")

        # Monthly returns
        monthly = self._compute_monthly_returns()

        return BacktestResult(
            config=self.config.to_dict(),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_equity=self.portfolio.equity,
            total_return=round(total_return, 2),
            total_return_pct=round(total_return_pct, 2),
            equity_curve=self.equity_curve,
            benchmark_curve=self.benchmark_curve,
            trades=self.trades,
            num_trades=len(self.trades),
            monthly_returns=monthly,
        )

    # ── Rebalance Logic ──

    def _do_rebalance(self, date_str: str, tickers: List[str],
                      bench_hist: pd.DataFrame) -> None:
        """Run full VMAA pipeline on a rebalance date."""
        logger.debug(f"  Rebalance: {date_str}")

        # Get market regime (store for adaptive stop in _execute_trade)
        market = self.signal_gen.get_market_regime(bench_hist, date_str)
        self._market_regime = market
        scalar = market.position_scalar if market.market_ok else 0.25

        # Run Part 1 on all tickers
        candidates: List[VMAACandidate] = []
        for ticker in tickers:
            # Skip if already holding
            if ticker in self.portfolio.positions:
                continue

            snapshot = self.loader.get_snapshot(ticker, date_str)
            if snapshot is None:
                continue

            # Part 1
            if self.config.re_screen_fundamentals:
                p1 = self.signal_gen.screen_part1(snapshot)
            else:
                p1 = None  # Would load from quality pool
            if p1 is None:
                continue

            # Part 2
            if self.config.re_screen_magna:
                p2 = self.signal_gen.screen_part2(snapshot, p1)
            else:
                p2 = None
            if p2 is None:
                continue

            # Composite rank
            composite = p1.quality_score * 0.50 + (p2.magna_score / 10) * 0.35
            if p2.entry_ready:
                composite += 0.15

            candidates.append(VMAACandidate(
                ticker=p1.ticker,
                part1=p1,
                part2=p2,
                composite_rank=round(composite, 4),
                entry_triggered=p2.entry_ready,
            ))

        # Sort: entry-triggered first, then by composite rank
        candidates.sort(key=lambda c: (c.entry_triggered, c.composite_rank), reverse=True)

        # Execute entries
        for candidate in candidates:
            if self.portfolio.num_positions >= self.config.max_positions:
                break

            # Sector limit
            sector = candidate.part1.sector
            sector_count = sum(1 for p in self.portfolio.positions.values()
                              if p.sector == sector)
            if sector_count >= self.config.max_positions_per_sector:
                continue

            # Entry decision
            self._execute_entry(candidate, date_str, scalar)

    def _execute_entry(self, candidate: VMAACandidate, date_str: str,
                       scalar: float) -> None:
        """Execute a simulated buy order."""
        ticker = candidate.ticker
        p1 = candidate.part1
        p2 = candidate.part2
        entry_price = p1.current_price
        low_52w = p1.low_52w

        # Position sizing (Quarter-Kelly)
        confidence = self._compute_confidence(candidate)
        risk_per_share = entry_price * self.config.hard_stop_pct
        # Base win probability 50% modulated by confidence (0.45 → 0.55 range)
        win_prob = 0.45 + confidence * 0.10
        payout = 2.0
        kelly = (payout * win_prob - (1 - win_prob)) / payout
        kelly = max(0, min(kelly, 0.25))

        portfolio_value = self.portfolio.equity
        risk_capital = portfolio_value * kelly * self.config.kelly_fraction * scalar
        raw_qty = int(risk_capital / risk_per_share) if risk_per_share > 0 else 0

        # Allocation-based limit
        max_alloc = portfolio_value * self.config.max_position_pct
        alloc_qty = int(max_alloc / entry_price) if entry_price > 0 else 0

        quantity = min(raw_qty, alloc_qty)
        if quantity <= 0:
            return

        # Min position size check
        position_value = quantity * entry_price
        if position_value < self.config.min_position_size:
            return

        # Transaction cost
        hist = self._daily_prices.get(ticker)
        daily_vol = int(hist.loc[hist.index <= pd.Timestamp(date_str)]['Volume'].iloc[-1]) \
            if hist is not None else 0
        cost = self.cost_model.total_cost(entry_price, quantity, daily_vol)

        if self.portfolio.cash < position_value + cost:
            # Can't afford
            return

        # Execute
        self.portfolio.cash -= (position_value + cost)

        # Compute stops using ADAPTIVE selection (matching live risk_adaptive.py)
        # Phase 1 (2026-05-06): Dynamically adjusts for price level, vol, regime
        from risk_adaptive import compute_stops_adaptive
        
        # Build a mini hist-like DataFrame for the date
        mini_hist = self._daily_prices.get(ticker, pd.DataFrame())
        if not mini_hist.empty:
            mini_hist = mini_hist[mini_hist.index <= pd.Timestamp(date_str)].copy()
        
        stop_loss, stop_type = compute_stops_adaptive(
            entry_price, low_52w, mini_hist, market=self._market_regime
        )

        # Take profits
        tps = []
        for i, (level_pct, sell_pct) in enumerate(
            zip(self.config.tp_levels, self.config.tp_sell_fractions)):
            tps.append({
                'level': round(entry_price * (1 + level_pct), 2),
                'sell_pct': sell_pct,
                'label': f'TP{i+1}',
            })

        pos = Position(
            ticker=ticker,
            entry_date=date_str,
            entry_price=entry_price,
            quantity=quantity,
            cost_basis=position_value,
            stop_loss=round(stop_loss, 2),
            take_profits=tps,
            trailing_stop_pct=self.config.trailing_stop_pct,
            trailing_high=entry_price,
            time_stop_days=self.config.time_stop_days,
            entry_method="backtest",
            confidence=confidence,
            sector=p1.sector,
            current_price=entry_price,
            market_value=position_value,
        )
        self.portfolio.positions[ticker] = pos

        logger.debug(f"    BUY  {ticker:6s} {quantity:4d}sh @ ${entry_price:.2f} "
                    f"stop=${stop_loss:.2f} conf={confidence:.0%}")

    def _check_exits(self, date_str: str,
                     prices_today: Dict[str, float]) -> None:
        """Check all positions for stop/take profit triggers."""
        exited = []
        for ticker, pos in list(self.portfolio.positions.items()):
            price = prices_today.get(ticker, pos.current_price)
            bar_high = price
            bar_low = price
            bar_open = price

            # Try to get actual OHLC data
            hist = self._daily_prices.get(ticker)
            if hist is not None:
                bar = hist[hist.index == pd.Timestamp(date_str)]
                if not bar.empty:
                    bar_high = float(bar['High'].iloc[0])
                    bar_low = float(bar['Low'].iloc[0])
                    bar_open = float(bar['Open'].iloc[0])

            triggered, reason, exit_price = pos.check_stop(bar_low, bar_high, bar_open)
            if triggered:
                exit_price = max(exit_price, 0.01)
                self._close_position(ticker, date_str, exit_price, reason)
                exited.append(ticker)

        for t in exited:
            self.portfolio.positions.pop(t, None)

    def _check_time_stops(self, date_str: str,
                          prices_today: Dict[str, float]) -> None:
        """Check for time-based exit (max holding period)."""
        target_date = pd.Timestamp(date_str)
        exited = []
        for ticker, pos in list(self.portfolio.positions.items()):
            entry_date = pd.Timestamp(pos.entry_date)
            days_held = (target_date - entry_date).days
            if days_held >= pos.time_stop_days:
                price = prices_today.get(ticker, pos.current_price)
                self._close_position(ticker, date_str, price, "time_stop")
                exited.append(ticker)
        for t in exited:
            self.portfolio.positions.pop(t, None)

    def _close_position(self, ticker: str, date_str: str,
                        exit_price: float, reason: str) -> None:
        """Close a position and record the trade."""
        pos = self.portfolio.positions.get(ticker)
        if pos is None:
            return

        hist = self._daily_prices.get(ticker)
        daily_vol = 0
        if hist is not None:
            try:
                bar = hist[hist.index <= pd.Timestamp(date_str)]
                if not bar.empty:
                    daily_vol = int(bar['Volume'].iloc[-1])
            except Exception:
                pass

        # For TP partial exits: sell only the configured fraction
        if 'take_profit' in reason:
            for tp in pos.take_profits:
                if tp['label'] in reason:
                    sell_qty = max(1, int(pos.quantity * tp['sell_pct']))
                    sell_qty = min(sell_qty, pos.quantity)

                    gross_pnl = (exit_price - pos.entry_price) * sell_qty
                    cost = self.cost_model.total_cost(exit_price, sell_qty, daily_vol)
                    net_pnl = gross_pnl - cost
                    proceeds = exit_price * sell_qty - cost
                    self.portfolio.cash += proceeds

                    # Reduce position + remove consumed TP level
                    pos.quantity -= sell_qty
                    pos.cost_basis = pos.entry_price * pos.quantity
                    pos.take_profits = [t for t in pos.take_profits if t['label'] != tp['label']]

                    # Record partial trade
                    trade = TradeRecord(
                        ticker=ticker,
                        entry_date=pos.entry_date,
                        exit_date=date_str,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        quantity=sell_qty,
                        gross_pnl=round(gross_pnl, 2),
                        net_pnl=round(net_pnl, 2),
                        cost=round(cost, 2),
                        return_pct=round((exit_price / pos.entry_price - 1) * 100, 2),
                        exit_reason=reason + "_partial",
                        holding_days=(pd.Timestamp(date_str) - pd.Timestamp(pos.entry_date)).days,
                        entry_method=pos.entry_method,
                        sector=pos.sector,
                        confidence=pos.confidence,
                        is_win=exit_price > pos.entry_price,
                    )
                    self.trades.append(trade)

                    if pos.quantity <= 0:
                        self.portfolio.positions.pop(ticker, None)
                    return

        # Full close (non-TP reason)
        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        cost = self.cost_model.total_cost(exit_price, pos.quantity, daily_vol)
        net_pnl = gross_pnl - cost
        proceeds = exit_price * pos.quantity - cost
        self.portfolio.cash += proceeds

        entry_dt = pd.Timestamp(pos.entry_date)
        exit_dt = pd.Timestamp(date_str)
        holding_days = (exit_dt - entry_dt).days

        trade = TradeRecord(
            ticker=ticker,
            entry_date=pos.entry_date,
            exit_date=date_str,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            cost=round(cost, 2),
            return_pct=round((exit_price / pos.entry_price - 1) * 100, 2),
            exit_reason=reason,
            holding_days=holding_days,
            entry_method=pos.entry_method,
            sector=pos.sector,
            confidence=pos.confidence,
            is_win=exit_price > pos.entry_price,
        )
        self.trades.append(trade)

        logger.debug(f"    SELL {ticker:6s} ${exit_price:.2f} "
                    f"PnL=${trade.net_pnl:,.0f} ({trade.return_pct:+.1f}%) [{reason}]")

    def _close_all_positions(self, date_str: str,
                             prices: Dict[str, float]) -> None:
        """Close all open positions at final date."""
        for ticker in list(self.portfolio.positions.keys()):
            price = prices.get(ticker, self.portfolio.positions[ticker].entry_price)
            self._close_position(ticker, date_str, price, "signal_exit")
        self.portfolio.positions.clear()
        self.portfolio._recalc_equity()

    # ── Date Management ──

    def _get_rebalance_dates(self, start: str, end: str,
                             tickers: List[str]) -> List[str]:
        """Get rebalance dates based on frequency."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        freq_map = {'daily': 'B', 'weekly': 'W-FRI', 'monthly': 'BME'}
        freq = freq_map.get(self.config.rebalance_frequency, 'BME')

        all_dates = pd.date_range(start_ts, end_ts, freq=freq)
        return [d.strftime('%Y-%m-%d') for d in all_dates]

    @staticmethod
    def _safe_index_mask(hist: pd.DataFrame, start: str, end: str):
        """Create a boolean mask for date range, handling tz-aware indexes."""
        idx = hist.index
        # Strip timezone if present
        if hasattr(idx, 'tz') and idx.tz is not None:
            idx = idx.tz_localize(None)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return (idx >= start_ts) & (idx <= end_ts)

    def _get_trading_days(self, tickers: List[str],
                          start: str, end: str) -> List[str]:
        """Get all trading days from the benchmark's price history."""
        bench_hist = self.loader.get_price_history(self.config.benchmark_ticker)
        if bench_hist is not None and not bench_hist.empty:
            mask = self._safe_index_mask(bench_hist, start, end)
            return [d.strftime('%Y-%m-%d') for d in bench_hist.index[mask]]
        # Fallback: use first ticker's history
        for t in tickers:
            hist = self.loader.get_price_history(t)
            if hist is not None and not hist.empty:
                mask = self._safe_index_mask(hist, start, end)
                return [d.strftime('%Y-%m-%d') for d in hist.index[mask]]
        # Last resort: business days
        biz_days = pd.bdate_range(start, end)
        return [d.strftime('%Y-%m-%d') for d in biz_days]

    def _get_prices_on_date(self, tickers: List[str],
                            date_str: str) -> Dict[str, float]:
        """Get closing prices for all tickers on a given date."""
        prices = {}
        target = pd.Timestamp(date_str)
        for t in tickers:
            hist = self._daily_prices.get(t)
            if hist is None:
                continue
            available = hist[hist.index <= target]
            if not available.empty:
                prices[t] = float(available['Close'].iloc[-1])
        return prices

    # ── Helpers ──

    @property
    def _bench_start_price(self) -> Optional[float]:
        """Benchmark price at start of backtest."""
        bench_hist = self.loader.get_price_history(self.config.benchmark_ticker)
        if bench_hist is None or bench_hist.empty:
            return None
        start_ts = pd.Timestamp(self.config.start_date)
        available = bench_hist[bench_hist.index >= start_ts]
        if available.empty:
            return None
        return float(available['Close'].iloc[0])

    def _compute_atr(self, ticker: str, date_str: str, period: int = 14) -> float:
        """Compute ATR at a historical point in time."""
        hist = self._daily_prices.get(ticker)
        if hist is None or len(hist) < period + 1:
            return 0.0
        available = hist[hist.index <= pd.Timestamp(date_str)]
        if len(available) < period + 1:
            return 0.0
        recent = available.tail(period + 1)
        high = recent['High']
        low = recent['Low']
        close = recent['Close']
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return float(tr.tail(period).mean())

    def _compute_confidence(self, candidate: VMAACandidate) -> float:
        """Compute trade confidence score."""
        p1 = candidate.part1
        p2 = candidate.part2
        confidence = p1.quality_score * 0.40
        confidence += (p2.magna_score / 10) * 0.35
        confidence += 0.10  # Base market contribution
        if p1.ptl_ratio < 1.05:
            confidence += 0.05
        if p2.entry_ready:
            confidence += 0.05
        if p2.short_interest_score >= 2:
            confidence += 0.03
        return round(min(confidence, 1.0), 3)

    def _compute_monthly_returns(self) -> Dict[str, float]:
        """Compute monthly return series from equity curve."""
        monthly = {}
        if not self.equity_curve:
            return monthly

        df = pd.DataFrame([{
            'date': pd.Timestamp(r.date),
            'equity': r.equity,
        } for r in self.equity_curve])
        df = df.set_index('date')
        df['return'] = df['equity'].pct_change()

        monthly_returns = df['return'].resample('ME').apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
        )
        for idx, val in monthly_returns.items():
            monthly[idx.strftime('%Y-%m')] = round(float(val) * 100, 2)

        return monthly

    def _get_default_universe(self) -> List[str]:
        """Get a default universe of liquid tickers."""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM",
            "DIS", "NFLX", "ADBE", "CRM", "CSCO", "INTC", "VZ", "T", "PFE",
            "MRK", "ABBV", "PEP", "KO", "TMO", "NKE", "ABT", "DHR", "MDT",
            "BMY", "AMGN", "LOW", "UPS", "QCOM", "TXN", "HON", "GE", "CAT",
            "DE", "SBUX", "MCD", "ORCL", "IBM", "CVX", "COP", "GILD", "UBER",
            "PLTR", "SOFI", "RKLB", "COIN", "DKNG", "RBLX", "HOOD", "AFRM",
        ]


# ═══════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════

def _fmt_cap(cap: float) -> str:
    """Format market cap for display."""
    if cap >= 1e9:
        return f"${cap/1e9:.1f}B"
    elif cap >= 1e6:
        return f"${cap/1e6:.0f}M"
    return f"${cap:,.0f}"
