#!/usr/bin/env python3
"""
HK Performance Metrics
======================
Comprehensive performance analytics for HK backtest results.

In addition to standard metrics:
  - HKD-denominated returns
  - vs HSI benchmark comparison (2800.HK TraHK ETF)
  - HK sector breakdown
  - HK-specific win/loss patterns
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.hk.hk_backtest import HKBacktestResult

logger = logging.getLogger("vmaa.backtest.hk.metrics")


class HKMetricsCalculator:
    """
    Compute all performance metrics from an HKBacktestResult.

    Usage:
        calc = HKMetricsCalculator()
        metrics = calc.compute(result)
        print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """

    def __init__(self, risk_free_rate: float = 0.03):
        """
        Args:
            risk_free_rate: Annual risk-free rate (3% for HKD, lower than USD).
        """
        self.risk_free_rate = risk_free_rate

    def compute(self, result: HKBacktestResult) -> Dict[str, Any]:
        """Compute all performance metrics."""
        metrics: Dict[str, Any] = {}
        equity_df = result.equity_df
        trades = result.trades

        if equity_df.empty:
            metrics['error'] = "No equity curve data"
            return metrics

        # ── Return Metrics ──
        metrics['returns'] = self._compute_return_metrics(result, equity_df)

        # ── Risk-Adjusted ──
        metrics['risk_adjusted'] = self._compute_risk_adjusted(equity_df)

        # ── Drawdown ──
        metrics['drawdown'] = self._compute_drawdown_metrics(equity_df)

        # ── Trade Statistics ──
        if trades:
            metrics['trades'] = self._compute_trade_stats(trades)
        else:
            metrics['trades'] = {'num_trades': 0, 'note': 'No trades executed'}

        # ── Benchmark Comparison ──
        if result.benchmark_curve:
            metrics['benchmark'] = self._compute_benchmark_metrics(result, equity_df)

        # ── Sector Breakdown ──
        if trades:
            metrics['sectors'] = self._compute_sector_metrics(trades)

        # ── Summary ──
        metrics['summary'] = self._compute_summary(metrics, result)

        result.metrics = metrics
        return metrics

    # ── Return Metrics ──

    def _compute_return_metrics(self, result: HKBacktestResult,
                                 equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute total return, CAGR, and volatility metrics in HKD."""
        initial = result.initial_capital
        final = result.final_equity
        total_return_pct = result.total_return_pct

        # Trading days
        eq = equity_df.set_index('date')
        if eq.index.empty:
            return {'total_return_pct': total_return_pct}

        n_days = len(eq)
        n_years = n_days / 252

        # CAGR
        cagr = (final / initial) ** (1 / n_years) - 1 if n_years > 0 and initial > 0 else 0

        # Annualized volatility
        daily_rets = eq['daily_return'].dropna()
        ann_vol = float(daily_rets.std() * np.sqrt(252)) if len(daily_rets) > 0 else 0

        # Best/worst month
        monthly = result.monthly_returns
        best_month = max(monthly.values()) if monthly else 0
        worst_month = min(monthly.values()) if monthly else 0

        # Positive months %
        pos_months = sum(1 for v in monthly.values() if v > 0)
        pct_positive = pos_months / len(monthly) * 100 if monthly else 0

        return {
            'total_return_pct': round(total_return_pct, 2),
            'total_return_hkd': round(final - initial, 2),
            'cagr_pct': round(cagr * 100, 2),
            'annual_volatility_pct': round(ann_vol * 100, 2),
            'best_month_pct': round(best_month, 2),
            'worst_month_pct': round(worst_month, 2),
            'positive_months_pct': round(pct_positive, 1),
            'n_years': round(n_years, 2),
            'currency': 'HKD',
        }

    # ── Risk-Adjusted Metrics ──

    def _compute_risk_adjusted(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute Sharpe, Sortino, Calmar ratios."""
        daily_rets = equity_df['daily_return'].dropna()
        if len(daily_rets) < 2:
            return {}

        mean_ret = float(daily_rets.mean())
        std_ret = float(daily_rets.std())

        # Sharpe
        excess = mean_ret - self.risk_free_rate / 252
        sharpe = (excess / std_ret * np.sqrt(252)) if std_ret > 0 else 0

        # Sortino
        downside = daily_rets[daily_rets < 0]
        down_std = float(downside.std()) if len(downside) > 0 else 0
        sortino = (excess / down_std * np.sqrt(252)) if down_std > 0 else 0

        # Calmar
        dd = self._compute_drawdown_metrics(equity_df)
        max_dd = abs(dd.get('max_drawdown_pct', 0)) / 100
        cagr = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) ** (
            252 / len(equity_df)) - 1 if len(equity_df) > 0 else 0
        calmar = cagr / max_dd if max_dd > 0 else 0

        return {
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'calmar_ratio': round(calmar, 2),
            'risk_free_rate_pct': round(self.risk_free_rate * 100, 1),
        }

    # ── Drawdown ──

    def _compute_drawdown_metrics(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute max drawdown and related metrics."""
        equity = equity_df['equity'].values
        if len(equity) < 2:
            return {'max_drawdown_pct': 0.0, 'max_drawdown_hkd': 0.0}

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(np.min(dd))

        # Duration
        in_dd = dd < 0
        if not any(in_dd):
            return {'max_drawdown_pct': 0.0, 'max_drawdown_hkd': 0.0, 'max_dd_duration_days': 0}

        # Find longest drawdown stretch
        dd_start = None
        max_duration = 0
        current_duration = 0
        for i, is_dd in enumerate(in_dd):
            if is_dd:
                if current_duration == 0:
                    dd_start = i
                current_duration += 1
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                current_duration = 0

        # Average drawdown
        avg_dd = float(np.mean(dd[in_dd]))

        return {
            'max_drawdown_pct': round(max_dd * 100, 2),
            'max_drawdown_hkd': round(float(np.min(equity - peak)), 2),
            'avg_drawdown_pct': round(avg_dd * 100, 2),
            'max_dd_duration_days': max_duration,
        }

    # ── Trade Statistics ──

    def _compute_trade_stats(self, trades: List) -> Dict[str, Any]:
        """Compute trade-level statistics."""
        if not trades:
            return {}

        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = len(trades)

        win_rate = n_wins / n_total * 100 if n_total > 0 else 0

        # Profit factor
        gross_profit = sum(t.gross_pnl for t in wins)
        gross_loss = abs(sum(t.gross_pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average win/loss
        avg_win = gross_profit / n_wins if n_wins > 0 else 0
        avg_loss = gross_loss / n_losses if n_losses > 0 else 0

        # Expectancy
        expectancy = sum(t.net_pnl for t in trades) / n_total if n_total > 0 else 0

        # Holding period
        avg_hold = sum(t.holding_days for t in trades) / n_total if n_total > 0 else 0

        # Best/worst trade
        best_trade = max(trades, key=lambda t: t.net_pnl) if trades else None
        worst_trade = min(trades, key=lambda t: t.net_pnl) if trades else None

        # Consecutive wins/losses
        cons_wins = cons_losses = cur_wins = cur_losses = 0
        for t in trades:
            if t.is_win:
                cur_wins += 1
                cons_wins = max(cons_wins, cur_wins)
                cur_losses = 0
            else:
                cur_losses += 1
                cons_losses = max(cons_losses, cur_losses)
                cur_wins = 0

        # Exit reason breakdown
        exits = {}
        for t in trades:
            reason = t.exit_reason
            exits[reason] = exits.get(reason, 0) + 1

        return {
            'num_trades': n_total,
            'num_wins': n_wins,
            'num_losses': n_losses,
            'win_rate_pct': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2),
            'avg_win_hkd': round(avg_win, 2),
            'avg_loss_hkd': round(avg_loss, 2),
            'avg_win_pct': round(
                sum(t.return_pct for t in wins) / n_wins, 2
            ) if n_wins > 0 else 0,
            'avg_loss_pct': round(
                sum(t.return_pct for t in losses) / n_losses, 2
            ) if n_losses > 0 else 0,
            'expectancy_hkd': round(expectancy, 2),
            'avg_hold_days': round(avg_hold, 1),
            'best_trade_hkd': round(best_trade.net_pnl, 2) if best_trade else 0,
            'worst_trade_hkd': round(worst_trade.net_pnl, 2) if worst_trade else 0,
            'best_trade_pct': round(best_trade.return_pct, 2) if best_trade else 0,
            'worst_trade_pct': round(worst_trade.return_pct, 2) if worst_trade else 0,
            'consecutive_wins': cons_wins,
            'consecutive_losses': cons_losses,
            'exit_reasons': exits,
        }

    # ── Benchmark Comparison ──

    def _compute_benchmark_metrics(self, result: HKBacktestResult,
                                    equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare strategy returns vs 2800.HK benchmark."""
        bm_df = pd.DataFrame(result.benchmark_curve)
        if bm_df.empty or 'value' not in bm_df.columns:
            return {}

        bm_df = bm_df.set_index('date')
        bm_rets = bm_df['value'].pct_change().dropna()

        eq = equity_df.set_index('date')
        common_dates = eq.index.intersection(bm_df.index)
        if len(common_dates) < 10:
            return {}

        strat_rets = eq.loc[common_dates, 'daily_return']

        # Align benchmark returns
        bm_common = bm_rets.loc[bm_rets.index.isin(common_dates)]

        # Benchmark total return
        bm_start = float(bm_df['value'].iloc[0])
        bm_end = float(bm_df['value'].iloc[-1])
        bm_return_pct = (bm_end / bm_start - 1) * 100 if bm_start > 0 else 0

        # Excess return
        excess_return = result.total_return_pct - bm_return_pct

        # Alpha / Beta
        min_len = min(len(strat_rets), len(bm_common))
        s = strat_rets.iloc[:min_len]
        b = bm_common.iloc[:min_len]
        cov = np.cov(s, b)[0][1] if min_len > 1 else 0
        bm_var = np.var(b) if min_len > 1 else 0
        beta = cov / bm_var if bm_var > 0 else 1.0

        alpha = (s.mean() - beta * b.mean()) * 252 if min_len > 0 else 0

        # Correlation
        corr = np.corrcoef(s, b)[0][1] if min_len > 1 else 0

        # Tracking error
        tracking_err = (s - b).std() * np.sqrt(252) if min_len > 1 else 0

        # Information ratio
        info_ratio = (s.mean() - b.mean()) / (s - b).std() * np.sqrt(252) \
            if min_len > 1 and (s - b).std() > 0 else 0

        return {
            'benchmark_ticker': result.config.get('benchmark_ticker', '2800.HK'),
            'benchmark_return_pct': round(bm_return_pct, 2),
            'excess_return_pct': round(excess_return, 2),
            'alpha': round(float(alpha) * 100, 2),
            'beta': round(float(beta), 2),
            'correlation': round(float(corr), 2),
            'tracking_error_pct': round(float(tracking_err) * 100, 2),
            'information_ratio': round(float(info_ratio), 2),
        }

    # ── Sector Breakdown ──

    def _compute_sector_metrics(self, trades: List) -> Dict[str, Any]:
        """Compute performance by HK sector."""
        sectors: Dict[str, Dict] = {}
        for t in trades:
            sec = t.sector or "Unknown"
            if sec not in sectors:
                sectors[sec] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'return_pcts': []}
            sectors[sec]['trades'] += 1
            if t.is_win:
                sectors[sec]['wins'] += 1
            sectors[sec]['pnl'] += t.net_pnl
            sectors[sec]['return_pcts'].append(t.return_pct)

        result = {}
        for sec, s in sorted(sectors.items(), key=lambda x: x[1]['pnl'], reverse=True):
            win_rate = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
            avg_ret = np.mean(s['return_pcts']) if s['return_pcts'] else 0
            result[sec] = {
                'trades': s['trades'],
                'win_rate_pct': round(win_rate, 1),
                'total_pnl_hkd': round(s['pnl'], 2),
                'avg_return_pct': round(float(avg_ret), 2),
            }

        return result

    # ── Summary ──

    def _compute_summary(self, metrics: Dict[str, Any],
                          result: HKBacktestResult) -> Dict[str, Any]:
        """Compile a summary dict for easy display."""
        ret = metrics.get('returns', {})
        ra = metrics.get('risk_adjusted', {})
        dd = metrics.get('drawdown', {})
        tr = metrics.get('trades', {})
        bm = metrics.get('benchmark', {})

        return {
            'total_return_pct': ret.get('total_return_pct', 0),
            'cagr_pct': ret.get('cagr_pct', 0),
            'sharpe_ratio': ra.get('sharpe_ratio', 0),
            'max_drawdown_pct': dd.get('max_drawdown_pct', 0),
            'win_rate_pct': tr.get('win_rate_pct', 0),
            'profit_factor': tr.get('profit_factor', 0),
            'num_trades': tr.get('num_trades', 0),
            'benchmark_return_pct': bm.get('benchmark_return_pct', 0),
            'excess_return_pct': bm.get('excess_return_pct', 0),
            'alpha': bm.get('alpha', 0),
            'beta': bm.get('beta', 0),
            'information_ratio': bm.get('information_ratio', 0),
            'currency': 'HKD',
        }
