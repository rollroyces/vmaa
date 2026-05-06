#!/usr/bin/env python3
"""
HK Backtest Engine
==================
Walk-forward simulation of VMAA-HK pipeline over historical data.

Architecture mirrors US BacktestEngine but adapted for HK:
  - Monthly rebalance on HK trading calendar
  - HK screening via HKScreener (imports pipeline_hk.py logic)
  - HKD-denominated portfolio and trade tracking
  - HSI-based market regime
  - Adaptive stop via risk_adaptive.py
  - HKD transaction costs via HKSlippageConfig
"""
from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# VMAA imports
_vmaa_root = Path(__file__).resolve().parent.parent.parent
if str(_vmaa_root) not in sys.path:
    sys.path.insert(0, str(_vmaa_root))

from backtest.hk.hk_config import HKBacktestConfig, HKC, HKSlippageConfig
from backtest.hk.hk_data import HKDataLoader
from backtest.hk.hk_screener import HKScreener
from risk_adaptive import compute_stops_adaptive

logger = logging.getLogger("vmaa.backtest.hk.engine")


# ═══════════════════════════════════════════════════════════════════
# HK Portfolio State
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HKPosition:
    """Tracks a live position during HK backtest (HKD)."""
    ticker: str
    entry_date: str
    entry_price: float
    quantity: int
    cost_basis: float           # HKD
    stop_loss: float
    take_profits: List[Dict[str, Any]]
    trailing_stop_pct: float
    trailing_activated: bool = False
    trailing_high: float = 0.0
    time_stop_days: int = 90
    entry_method: str = ""
    confidence: float = 0.0
    sector: str = ""

    # P&L tracking
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    def update(self, price: float) -> None:
        self.current_price = price
        self.market_value = price * self.quantity
        self.unrealized_pnl = self.market_value - self.cost_basis
        self.unrealized_pnl_pct = (price / self.entry_price - 1) * 100

        if not self.trailing_activated:
            if self.unrealized_pnl_pct >= 12.0:
                self.trailing_activated = True
                self.trailing_high = price
        else:
            self.trailing_high = max(self.trailing_high, price)

    @property
    def trailing_stop_price(self) -> float:
        if self.trailing_activated:
            return self.trailing_high * (1 - self.trailing_stop_pct)
        return 0.0

    def check_stop(self, bar_low: float, bar_high: float,
                   bar_open: float) -> Tuple[bool, str, float]:
        """Check stop triggers. Returns (triggered, reason, exit_price)."""
        # 1. Hard stop
        if bar_low <= self.stop_loss:
            return True, "hard_stop", self.stop_loss

        # 2. Trailing stop
        trail = self.trailing_stop_price
        if self.trailing_activated and trail > 0 and bar_low <= trail:
            return True, "trailing_stop", trail

        # 3. Take profit
        for tp in self.take_profits:
            if bar_high >= tp['level']:
                return True, f"take_profit_{tp['label']}", tp['level']

        return False, "", 0.0


@dataclass
class HKPortfolioState:
    """Full HK portfolio state (HKD)."""
    date: str
    cash: float              # HKD
    equity: float            # HKD
    positions: Dict[str, HKPosition] = field(default_factory=dict)
    peak_equity: float = 0.0
    drawdown: float = 0.0

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def invested(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    def update_prices(self, prices: Dict[str, float]) -> None:
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
# HK Trade Record
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HKTradeRecord:
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float        # HKD
    net_pnl: float          # HKD
    cost: float             # HKD
    return_pct: float
    exit_reason: str
    holding_days: int
    sector: str = ""
    entry_method: str = ""
    confidence: float = 0.0
    is_win: bool = False


@dataclass
class HKDailyRecord:
    date: str
    equity: float           # HKD
    cash: float             # HKD
    invested: float         # HKD
    daily_return: float
    drawdown: float
    num_positions: int


# ═══════════════════════════════════════════════════════════════════
# HK Backtest Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HKBacktestResult:
    config: Dict[str, Any]
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return: float
    total_return_pct: float
    currency: str = "HKD"
    equity_curve: List[HKDailyRecord] = field(default_factory=list)
    benchmark_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[HKTradeRecord] = field(default_factory=list)
    num_trades: int = 0
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'date': r.date, 'equity': r.equity,
            'daily_return': r.daily_return, 'drawdown': r.drawdown,
            'num_positions': r.num_positions,
        } for r in self.equity_curve])

    @property
    def trade_log_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            'ticker': t.ticker, 'entry_date': t.entry_date,
            'exit_date': t.exit_date, 'entry_price': t.entry_price,
            'exit_price': t.exit_price, 'quantity': t.quantity,
            'gross_pnl': t.gross_pnl, 'net_pnl': t.net_pnl,
            'cost': t.cost, 'return_pct': t.return_pct,
            'exit_reason': t.exit_reason, 'holding_days': t.holding_days,
            'is_win': t.is_win, 'sector': t.sector,
        } for t in self.trades])


# ═══════════════════════════════════════════════════════════════════
# HK Candidate (for rebalance ranking)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HKCandidate:
    ticker: str
    quality: dict               # Part 1 result dict
    magna_signal: Any           # HKMagnaSignal
    composite_rank: float
    entry_triggered: bool


# ═══════════════════════════════════════════════════════════════════
# HK Backtest Engine
# ═══════════════════════════════════════════════════════════════════

class HKBacktestEngine:
    """
    Walk-forward backtesting engine for VMAA-HK.

    Usage:
        engine = HKBacktestEngine(config)
        result = engine.run()
        print(f"HKD Return: {result.total_return_pct:.1f}%")
    """

    def __init__(self, config: HKBacktestConfig = HKC):
        self.config = config
        self.loader = HKDataLoader(config)
        self.screener = HKScreener(config)
        self.cost_model = config.slippage

        # State
        self.portfolio: HKPortfolioState = None
        self.trades: List[HKTradeRecord] = []
        self.equity_curve: List[HKDailyRecord] = []
        self.benchmark_curve: List[Dict[str, Any]] = []

        self._prev_equity: float = 0.0
        self._daily_prices: Dict[str, pd.DataFrame] = {}
        self._market_regime: dict = {}
        self._bench_start_price: Optional[float] = None
        
        # Improvement: track hard-stopped tickers for cooldown (6 months)
        self._hard_stopped_tickers: Dict[str, str] = {}  # ticker → date
        self._hs_cooldown_days: int = 180  # 6-month ban after hard stop
        
        # Improvement: track per-sector hard stop streaks
        self._sector_hs_streak: Dict[str, int] = {}

    def run(self, tickers: Optional[List[str]] = None) -> HKBacktestResult:
        """Execute the full HK backtest."""
        start_ts = time.time()
        tickers = tickers or self.config.tickers
        if not tickers:
            tickers = []  # Will use HSI default from config
        if self.config.max_tickers > 0:
            tickers = tickers[:self.config.max_tickers]

        logger.info("=" * 70)
        logger.info(f"🇭🇰 VMAA-HK Backtest: {self.config.start_date} → {self.config.end_date}")
        logger.info(f"Universe: {len(tickers)} HK tickers | "
                    f"Capital: HKD {self.config.initial_capital:,.0f} | "
                    f"Frequency: {self.config.rebalance_frequency}")
        logger.info("=" * 70)

        # ── Step 1: Load Data ──
        logger.info("\n[1/5] Loading HK historical data...")
        all_tickers = list(dict.fromkeys(
            [self.config.benchmark_ticker, self.config.hsi_ticker] +
            [t for t in tickers if t not in (self.config.benchmark_ticker, self.config.hsi_ticker)]
        ))
        self.loader.fetch_all(all_tickers, self.config.start_date, self.config.end_date)

        # Verify tickers have data
        valid_tickers = []
        for t in tickers:
            hist = self.loader.get_price_history(t)
            if hist is not None and len(hist) >= self.config.min_history_days:
                valid_tickers.append(t)
        skipped = len(tickers) - len(valid_tickers)
        if skipped:
            logger.info(f"  Skipped {skipped} tickers with insufficient data")
        tickers = valid_tickers
        logger.info(f"  Loaded data for {len(tickers)} valid HK tickers")

        # Benchmark data
        bench_hist = self.loader.get_benchmark_history()
        hsi_hist = self.loader.get_hsi_history()

        # Pre-load daily prices
        self._daily_prices = {}
        for t in tickers:
            hist = self.loader.get_price_history(t)
            if hist is not None:
                self._daily_prices[t] = hist
        self.screener.set_daily_prices(self._daily_prices)

        # Cache yfinance data for screening
        logger.info(f"  Pre-fetching yfinance data for {len(tickers)} HK tickers...")
        self.screener.cache_yfinance(tickers)

        # Benchmark start price
        if bench_hist is not None and not bench_hist.empty:
            bm_start_ts = pd.Timestamp(self.config.start_date)
            avail = bench_hist[bench_hist.index >= bm_start_ts]
            if not avail.empty:
                self._bench_start_price = float(avail['Close'].iloc[0])

        # ── Step 2: Build Dates ──
        logger.info("\n[2/5] Building HK rebalance schedule...")
        trading_days = self.loader.get_trading_days(
            self.config.start_date, self.config.end_date
        )
        rebalance_dates = self.loader.get_rebalance_dates(
            self.config.start_date, self.config.end_date
        )
        logger.info(f"  {len(trading_days)} HK trading days, "
                    f"{len(rebalance_dates)} rebalance dates")

        # ── Step 3: Initialize Portfolio ──
        logger.info("\n[3/5] Initializing HK portfolio...")
        self.portfolio = HKPortfolioState(
            date=self.config.start_date,
            cash=self.config.initial_capital,
            equity=self.config.initial_capital,
            peak_equity=self.config.initial_capital,
        )
        self.trades = []
        self.equity_curve = []
        self._prev_equity = self.config.initial_capital

        # ── Step 4: Walk-Forward Simulation ──
        logger.info("\n[4/5] Running HK walk-forward simulation...")
        logger.info(f"  {len(trading_days)} trading days to simulate")

        rebalance_idx = 0
        for di, date_str in enumerate(trading_days):
            if (di + 1) % 50 == 0:
                logger.info(
                    f"  Day {di+1}/{len(trading_days)} | "
                    f"Equity: HKD {self.portfolio.equity:,.0f} | "
                    f"Pos: {self.portfolio.num_positions} | "
                    f"Trades: {len(self.trades)}"
                )

            prices_today = self._get_prices_on_date(tickers, date_str)
            self.portfolio.update_prices(prices_today)
            self._check_exits(date_str, prices_today)

            # Rebalance — use >= to handle holidays where BME date ≠ trading day
            while (rebalance_idx < len(rebalance_dates) and
                    date_str >= rebalance_dates[rebalance_idx]):
                self._do_rebalance(date_str, tickers, hsi_hist)
                rebalance_idx += 1

            # Time stops
            self._check_time_stops(date_str, prices_today)

            # Record daily equity
            daily_ret = 0.0
            if self._prev_equity > 0:
                daily_ret = (self.portfolio.equity - self._prev_equity) / self._prev_equity
            self._prev_equity = self.portfolio.equity

            self.equity_curve.append(HKDailyRecord(
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
                'value': (bm_val / (self._bench_start_price or 1.0) * self.config.initial_capital)
                if self._bench_start_price else 0.0,
            })

        # ── Step 5: Close Remaining ──
        final_date = trading_days[-1] if trading_days else self.config.end_date
        final_prices = self._get_prices_on_date(tickers, final_date)
        self._close_all_positions(final_date, final_prices)

        # Build result
        total_return = self.portfolio.equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital * 100)
        monthly = self._compute_monthly_returns()

        elapsed = time.time() - start_ts
        logger.info(f"\n[5/5] HK Backtest complete in {elapsed:.0f}s")
        logger.info(f"  Initial: HKD {self.config.initial_capital:,.0f}")
        logger.info(f"  Final:   HKD {self.portfolio.equity:,.0f}")
        logger.info(f"  Return:  {total_return_pct:.1f}%")
        logger.info(f"  Trades:  {len(self.trades)}")

        return HKBacktestResult(
            config=self.config.to_dict(),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_equity=self.portfolio.equity,
            total_return=round(total_return, 2),
            total_return_pct=round(total_return_pct, 2),
            currency="HKD",
            equity_curve=self.equity_curve,
            benchmark_curve=self.benchmark_curve,
            trades=self.trades,
            num_trades=len(self.trades),
            monthly_returns=monthly,
        )

    # ═══════════════════════════════════════════════════════════════
    # Rebalance
    # ═══════════════════════════════════════════════════════════════

    def _do_rebalance(self, date_str: str, tickers: List[str],
                      hsi_hist: pd.DataFrame) -> None:
        """Run full VMAA-HK pipeline on a rebalance date."""
        # Market regime
        self._market_regime = self.screener.get_hk_market_regime(hsi_hist, date_str)
        scalar = self._market_regime.get("position_scalar", 0.8)
        if not self._market_regime.get("market_ok", True):
            scalar = 0.50  # Floor raised from 0.25 — avoid over-penalising

        # Run Part 1 + Part 2 on all tickers
        candidates: List[HKCandidate] = []
        parti_pass = 0
        part2_pass = 0
        for ticker in tickers:
            if ticker in self.portfolio.positions:
                continue

            snapshot = self.loader.get_snapshot(ticker, date_str)
            if snapshot is None:
                continue

            # Part 1: HK fundamentals
            quality = self.screener.screen_part1(snapshot)
            if quality is None:
                continue
            parti_pass += 1

            # Part 2: HK MAGNA
            magna = self.screener.screen_part2(ticker, quality, snapshot, date_str)
            if magna is None or magna.magna_score < self.config.thresholds.magna_min_score:
                continue
            part2_pass += 1

            # Composite rank
            composite = quality.get("quality_score", 0) * 0.50 + (magna.magna_score / 10) * 0.35
            if magna.entry_ready:
                composite += 0.15

            candidates.append(HKCandidate(
                ticker=ticker,
                quality=quality,
                magna_signal=magna,
                composite_rank=round(composite, 4),
                entry_triggered=magna.entry_ready,
            ))

        # Sort: entry-ready first, then composite
        candidates.sort(key=lambda c: (c.entry_triggered, c.composite_rank), reverse=True)

        logger.info(
            f"  🔄 {date_str} | Cand={parti_pass} part1/{part2_pass} part2/"
            f"{sum(1 for c in candidates if c.entry_triggered)} entry | "
            f"Pos={self.portfolio.num_positions} | Cash=HKD {self.portfolio.cash:,.0f} | "
            f"Scalar={scalar:.2f}"
        )

        # Execute entries
        entered = 0
        skipped_max_pos = 0
        skipped_sector = 0
        skipped_sizing = 0
        skipped_momentum = 0
        skipped_cooldown = 0
        for candidate in candidates:
            if self.portfolio.num_positions >= self.config.max_positions:
                skipped_max_pos += 1
                continue

            sector = candidate.quality.get("sector", "")
            sector_count = sum(
                1 for p in self.portfolio.positions.values() if p.sector == sector
            )
            
            # Fix #7: Sector loss tracker — halve per-sector limit
            hs_streak = self._sector_hs_streak.get(sector, 0)
            sector_limit = self.config.max_positions_per_sector
            if hs_streak >= 2:
                sector_limit = max(1, sector_limit // 2)
            if sector_count >= sector_limit:
                skipped_sector += 1
                continue

            # Fix #2: Repeat-offender filter — skip tickers that hit hard stop recently
            ticker = candidate.ticker
            if ticker in self._hard_stopped_tickers:
                last_hs = pd.Timestamp(self._hard_stopped_tickers[ticker])
                cooldown_end = last_hs + pd.Timedelta(days=self._hs_cooldown_days)
                if pd.Timestamp(date_str) < cooldown_end:
                    skipped_cooldown += 1
                    continue
                else:
                    del self._hard_stopped_tickers[ticker]  # cooldown expired

            # Fix #3: Entry momentum filter — require price > 20-day MA
            hist = self._daily_prices.get(ticker)
            if hist is not None and len(hist) >= 20:
                target = pd.Timestamp(date_str)
                available = hist[hist.index <= target]
                if len(available) >= 20:
                    ma20 = float(available['Close'].tail(20).mean())
                    if available['Close'].iloc[-1] < ma20 * 0.98:  # 2% tolerance
                        skipped_momentum += 1
                        continue

            pos_before = self.portfolio.num_positions
            self._execute_entry(candidate, date_str, scalar)
            if self.portfolio.num_positions > pos_before:
                entered += 1
            else:
                skipped_sizing += 1

        if skipped_max_pos + skipped_sector + skipped_sizing + skipped_momentum + skipped_cooldown > 0 or entered > 0:
            logger.info(
                f"    Entered: {entered} | Skipped: {skipped_max_pos} max_pos, "
                f"{skipped_sector} sector, {skipped_sizing} sizing, "
                f"{skipped_momentum} mom, {skipped_cooldown} cooldown"
            )

    def _execute_entry(self, candidate: HKCandidate, date_str: str,
                       scalar: float) -> None:
        """Execute a simulated buy order in HKD."""
        ticker = candidate.ticker
        quality = candidate.quality
        magna = candidate.magna_signal
        entry_price = quality.get("current_price", 0)
        low_52w = quality.get("low_52w", entry_price * 0.5)

        if entry_price <= 0:
            return

        # Confidence
        confidence = quality.get("quality_score", 0) * 0.40
        confidence += (magna.magna_score / 10) * 0.35 + 0.10
        if quality.get("ptl_ratio", 999) < 1.05:
            confidence += 0.05
        if magna.entry_ready:
            confidence += 0.05
        confidence = round(min(confidence, 1.0), 3)

        # Position sizing (Kelly)
        risk_per_share = entry_price * self.config.hard_stop_pct
        # Base win probability 50% modulated by confidence (0.45 → 0.55 range)
        win_prob = 0.45 + confidence * 0.10
        payout = 2.0
        kelly = (payout * win_prob - (1 - win_prob)) / payout
        kelly = max(0, min(kelly, 0.25))

        portfolio_value = self.portfolio.equity
        risk_capital = portfolio_value * kelly * self.config.kelly_fraction * scalar
        raw_qty = int(risk_capital / risk_per_share) if risk_per_share > 0 else 0

        max_alloc = portfolio_value * self.config.max_position_pct
        alloc_qty = int(max_alloc / entry_price) if entry_price > 0 else 0

        quantity = min(raw_qty, alloc_qty)
        if quantity <= 0:
            return

        position_value = quantity * entry_price
        if position_value < self.config.min_position_hkd:
            return

        # Transaction cost
        cost = self.cost_model.total_cost(entry_price, quantity)

        if self.portfolio.cash < position_value + cost:
            return

        # Execute
        self.portfolio.cash -= (position_value + cost)

        # Adaptive stop via risk_adaptive.py
        mini_hist = self._daily_prices.get(ticker, pd.DataFrame())
        if not mini_hist.empty:
            mini_hist = mini_hist[mini_hist.index <= pd.Timestamp(date_str)].copy()

        # Build market regime object compatible with risk_adaptive
        class HKMarketRegimeProxy:
            def __init__(self, regime_dict):
                self.vol_regime = regime_dict.get('vol_regime', 'NORMAL')
                self.market_ok = regime_dict.get('market_ok', True)
                self.position_scalar = regime_dict.get('position_scalar', 0.8)

        market_proxy = HKMarketRegimeProxy(self._market_regime) if self._market_regime else None

        stop_loss, stop_type = compute_stops_adaptive(
            entry_price, low_52w, mini_hist, market=market_proxy
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

        pos = HKPosition(
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
            entry_method="hk_backtest",
            confidence=confidence,
            sector=quality.get("sector", ""),
            current_price=entry_price,
            market_value=position_value,
        )
        self.portfolio.positions[ticker] = pos

        logger.debug(
            f"    🇭🇰 BUY  {ticker:10s} {quantity:5d}sh @ HKD {entry_price:.2f} "
            f"stop=HKD {stop_loss:.2f} ({stop_type}) conf={confidence:.0%}"
        )

    # ═══════════════════════════════════════════════════════════════
    # Exit Management
    # ═══════════════════════════════════════════════════════════════

    def _check_exits(self, date_str: str, prices_today: Dict[str, float]) -> None:
        """Check all positions for stop/take profit triggers."""
        exited = []
        for ticker, pos in list(self.portfolio.positions.items()):
            price = prices_today.get(ticker, pos.current_price)
            bar_high = price
            bar_low = price
            bar_open = price

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
                # Partial take-profit: sell only the configured fraction
                if reason.startswith("take_profit_"):
                    tp_label = reason.replace("take_profit_", "")
                    sell_fraction = 1.0
                    for tp in pos.take_profits:
                        if tp['label'] == tp_label:
                            sell_fraction = tp.get('sell_pct', 1.0)
                            break
                    if sell_fraction < 1.0 and pos.quantity > 1:
                        self._close_partial_position(ticker, date_str, exit_price, reason, sell_fraction)
                        # Remove consumed TP level so it doesn't re-trigger
                        pos.take_profits = [tp for tp in pos.take_profits if tp['label'] != tp_label]
                    else:
                        self._close_position(ticker, date_str, exit_price, reason)
                        exited.append(ticker)
                else:
                    self._close_position(ticker, date_str, exit_price, reason)
                    exited.append(ticker)

        for t in exited:
            self.portfolio.positions.pop(t, None)

    def _check_time_stops(self, date_str: str, prices_today: Dict[str, float]) -> None:
        """Check time-based exits."""
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

    def _close_partial_position(self, ticker: str, date_str: str,
                                 exit_price: float, reason: str,
                                 sell_fraction: float) -> None:
        """Partially close a position (take-profit scaling)."""
        pos = self.portfolio.positions.get(ticker)
        if pos is None or pos.quantity <= 1:
            return

        sell_qty = max(1, int(pos.quantity * sell_fraction))
        sell_qty = min(sell_qty, pos.quantity)
        if sell_qty <= 0:
            return

        gross_pnl = (exit_price - pos.entry_price) * sell_qty
        cost = self.cost_model.total_cost(exit_price, sell_qty)
        net_pnl = gross_pnl - cost
        proceeds = exit_price * sell_qty - cost
        self.portfolio.cash += proceeds

        # Update position: reduce quantity, proportional cost basis
        original_qty = pos.quantity + sell_qty  # before sale
        pos.cost_basis = pos.cost_basis * (pos.quantity / original_qty) if original_qty > 0 else 0
        pos.quantity -= sell_qty
        pos.market_value = pos.current_price * pos.quantity

        entry_dt = pd.Timestamp(pos.entry_date)
        exit_dt = pd.Timestamp(date_str)
        holding_days = (exit_dt - entry_dt).days

        trade = HKTradeRecord(
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
            holding_days=holding_days,
            sector=pos.sector,
            entry_method=pos.entry_method,
            confidence=pos.confidence,
            is_win=exit_price > pos.entry_price,
        )
        self.trades.append(trade)

        logger.debug(
            f"    🇭🇰 SELL {ticker:10s} {sell_qty}sh ({sell_fraction:.0%}) @ HKD {exit_price:.2f} "
            f"PnL=HKD {trade.net_pnl:,.0f} ({trade.return_pct:+.1f}%) [{reason}] "
            f"Remain: {pos.quantity}sh"
        )

    def _close_position(self, ticker: str, date_str: str,
                        exit_price: float, reason: str) -> None:
        """Close a position and record trade."""
        pos = self.portfolio.positions.get(ticker)
        if pos is None:
            return

        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        cost = self.cost_model.total_cost(exit_price, pos.quantity)
        net_pnl = gross_pnl - cost

        proceeds = exit_price * pos.quantity - cost
        self.portfolio.cash += proceeds

        entry_dt = pd.Timestamp(pos.entry_date)
        exit_dt = pd.Timestamp(date_str)
        holding_days = (exit_dt - entry_dt).days

        trade = HKTradeRecord(
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
            sector=pos.sector,
            entry_method=pos.entry_method,
            confidence=pos.confidence,
            is_win=exit_price > pos.entry_price,
        )
        self.trades.append(trade)

        # Track hard stops for repeat-offender + sector streak filters
        if "hard_stop" in reason:
            self._hard_stopped_tickers[ticker] = date_str
            sector = pos.sector
            self._sector_hs_streak[sector] = self._sector_hs_streak.get(sector, 0) + 1
        else:
            # Reset sector streak on any non-HS exit (wins + other stops)
            sector = pos.sector
            if sector in self._sector_hs_streak and self._sector_hs_streak[sector] > 0:
                self._sector_hs_streak[sector] = 0

        logger.debug(
            f"    🇭🇰 SELL {ticker:10s} HKD {exit_price:.2f} "
            f"PnL=HKD {trade.net_pnl:,.0f} ({trade.return_pct:+.1f}%) [{reason}]"
        )

    def _close_all_positions(self, date_str: str, prices: Dict[str, float]) -> None:
        """Close all open positions at final date."""
        for ticker in list(self.portfolio.positions.keys()):
            price = prices.get(ticker, self.portfolio.positions[ticker].entry_price)
            self._close_position(ticker, date_str, price, "signal_exit")
        self.portfolio.positions.clear()
        self.portfolio._recalc_equity()

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

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

    def _compute_monthly_returns(self) -> Dict[str, float]:
        """Compute monthly return series."""
        monthly = {}
        if not self.equity_curve:
            return monthly
        df = pd.DataFrame([{
            'date': pd.Timestamp(r.date),
            'equity': r.equity,
        } for r in self.equity_curve]).set_index('date')
        df['return'] = df['equity'].pct_change()
        mret = df['return'].resample('ME').apply(
            lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0
        )
        for idx, val in mret.items():
            monthly[idx.strftime('%Y-%m')] = round(float(val) * 100, 2)
        return monthly
