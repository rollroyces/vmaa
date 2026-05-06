#!/usr/bin/env python3
"""
Performance Metrics
===================
Comprehensive performance analytics for backtest results.

Metrics computed:
  - Return metrics: Total Return, CAGR, Annualized Volatility
  - Risk-adjusted: Sharpe, Sortino, Calmar Ratios
  - Drawdown analysis: Max Drawdown, Duration, Underwater curve
  - Trade statistics: Win Rate, Profit Factor, Expectancy
  - Benchmark comparison: Alpha, Beta, Correlation, Info Ratio
  - Market regime sensitivity
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult

logger = logging.getLogger("vmaa.backtest.metrics")


# ═══════════════════════════════════════════════════════════════════
# Main Metric Computer
# ═══════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """
    Compute all performance metrics from a BacktestResult.

    Usage:
        calc = MetricsCalculator()
        metrics = calc.compute(result)
        print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 4%).
        """
        self.risk_free_rate = risk_free_rate

    def compute(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Compute all performance metrics.

        Returns:
            Dict with all metrics organized by category.
        """
        metrics: Dict[str, Any] = {}

        # Extract series
        equity_df = result.equity_df
        trades = result.trades

        if equity_df.empty:
            metrics['error'] = "No equity curve data"
            return metrics

        # ── Return Metrics ──
        metrics['returns'] = self._compute_return_metrics(result, equity_df)

        # ── Risk-Adjusted Metrics ──
        metrics['risk_adjusted'] = self._compute_risk_adjusted(equity_df)

        # ── Drawdown Analysis ──
        metrics['drawdown'] = self._compute_drawdown_metrics(equity_df)

        # ── Trade Statistics ──
        if trades:
            metrics['trades'] = self._compute_trade_stats(trades)
        else:
            metrics['trades'] = {'num_trades': 0, 'note': 'No trades executed'}

        # ── Benchmark Comparison ──
        if result.benchmark_curve:
            metrics['benchmark'] = self._compute_benchmark_metrics(result, equity_df)

        # ── Market Regime Sensitivity ──
        if trades and len(equity_df) > 0:
            metrics['regime'] = self._compute_regime_sensitivity(equity_df, result)

        # ── Summary ──
        metrics['summary'] = self._compute_summary(metrics, result)

        # Store in result
        result.metrics = metrics
        return metrics

    # ── Return Metrics ──

    def _compute_return_metrics(self, result: BacktestResult,
                                 equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute total return, CAGR, and volatility metrics."""
        m = {}

        m['total_return'] = result.total_return
        m['total_return_pct'] = result.total_return_pct

        # CAGR
        days = (pd.Timestamp(result.end_date) - pd.Timestamp(result.start_date)).days
        years = days / 365.25
        if years > 0 and result.initial_capital > 0:
            m['cagr_pct'] = round(
                ((result.final_equity / result.initial_capital) ** (1 / years) - 1) * 100, 2
            )
        else:
            m['cagr_pct'] = 0.0
        m['years'] = round(years, 2)

        # Annualized volatility
        daily_returns = equity_df['daily_return'].dropna()
        if len(daily_returns) > 0:
            m['annualized_volatility_pct'] = round(
                float(daily_returns.std() * np.sqrt(252) * 100), 2
            )
            m['daily_volatility_pct'] = round(float(daily_returns.std() * 100), 4)
            m['best_day_pct'] = round(float(daily_returns.max() * 100), 2)
            m['worst_day_pct'] = round(float(daily_returns.min() * 100), 2)
        else:
            m['annualized_volatility_pct'] = 0.0

        # Monthly returns
        monthly = result.monthly_returns
        if monthly:
            monthly_values = list(monthly.values())
            m['best_month_pct'] = round(max(monthly_values), 2)
            m['worst_month_pct'] = round(min(monthly_values), 2)
            m['positive_months_pct'] = round(
                sum(1 for v in monthly_values if v > 0) / len(monthly_values) * 100, 1
            )
            m['avg_monthly_return_pct'] = round(
                sum(monthly_values) / len(monthly_values), 2
            )

        return m

    # ── Risk-Adjusted Metrics ──

    def _compute_risk_adjusted(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute Sharpe, Sortino, Calmar ratios."""
        m = {}
        daily_returns = equity_df['daily_return'].dropna()
        if len(daily_returns) < 2:
            return m

        mean_daily = float(daily_returns.mean())
        std_daily = float(daily_returns.std())
        rf_daily = self.risk_free_rate / 252

        # Sharpe Ratio
        excess_daily = mean_daily - rf_daily
        if std_daily > 0:
            m['sharpe_ratio'] = round(float(excess_daily / std_daily * np.sqrt(252)), 3)
        else:
            m['sharpe_ratio'] = 0.0

        # Sortino Ratio (downside deviation)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0:
            downside_std = float(downside.std() * np.sqrt(252))
            if downside_std > 0:
                m['sortino_ratio'] = round(float(excess_daily / downside_std * np.sqrt(252)), 3)
            else:
                m['sortino_ratio'] = 0.0
        else:
            m['sortino_ratio'] = 0.0

        # Calmar Ratio = CAGR / Max Drawdown
        days = len(equity_df)
        years = days / 252
        if years > 0:
            cagr = (float(equity_df['equity'].iloc[-1]) / float(equity_df['equity'].iloc[0])) ** (1 / years) - 1
        else:
            cagr = 0.0
        max_dd = abs(float(equity_df['drawdown'].min()))
        if max_dd > 0:
            m['calmar_ratio'] = round(cagr / max_dd, 3)
        else:
            m['calmar_ratio'] = 0.0

        return m

    # ── Drawdown Analysis ──

    def _compute_drawdown_metrics(self, equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute drawdown metrics and underwater curve."""
        m = {}
        drawdowns = equity_df['drawdown']

        m['max_drawdown_pct'] = round(float(drawdowns.min() * 100), 2)
        m['avg_drawdown_pct'] = round(float(drawdowns[drawdowns < 0].mean() * 100), 2)

        # Drawdown duration analysis
        dd_durations = self._compute_drawdown_durations(drawdowns.values)
        if dd_durations:
            m['max_drawdown_days'] = max(dd_durations)
            m['avg_drawdown_days'] = round(sum(dd_durations) / len(dd_durations), 0)
            m['drawdown_periods'] = len(dd_durations)
        else:
            m['max_drawdown_days'] = 0
            m['avg_drawdown_days'] = 0
            m['drawdown_periods'] = 0

        # Underwater curve data (for plotting)
        m['underwater'] = [
            {'date': equity_df.iloc[i]['date'],
             'drawdown_pct': round(float(equity_df.iloc[i]['drawdown']) * 100, 2)}
            for i in range(len(equity_df))
        ]

        # Recovery analysis
        m['num_new_highs'] = int(sum(1 for dd in drawdowns if dd == 0.0))

        return m

    @staticmethod
    def _compute_drawdown_durations(drawdowns: np.ndarray) -> List[int]:
        """Compute durations of each drawdown period."""
        durations = []
        in_dd = False
        current_duration = 0
        for dd in drawdowns:
            if dd < 0:
                if not in_dd:
                    in_dd = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_dd:
                    in_dd = False
                    durations.append(current_duration)
                    current_duration = 0
        if in_dd:
            durations.append(current_duration)
        return durations

    # ── Trade Statistics ──

    def _compute_trade_stats(self, trades) -> Dict[str, Any]:
        """Compute comprehensive trade statistics."""
        m = {}
        n = len(trades)

        m['num_trades'] = n
        if n == 0:
            return m

        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]
        n_wins = len(wins)
        n_losses = len(losses)

        # Win Rate
        m['win_rate_pct'] = round(n_wins / n * 100, 1)
        m['num_wins'] = n_wins
        m['num_losses'] = n_losses

        # Profit Factor
        gross_profit = sum(t.gross_pnl for t in wins)
        gross_loss = abs(sum(t.gross_pnl for t in losses))
        m['profit_factor'] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

        # Average Win / Average Loss
        if n_wins > 0:
            m['avg_win'] = round(sum(t.gross_pnl for t in wins) / n_wins, 2)
            m['avg_win_pct'] = round(sum(t.return_pct for t in wins) / n_wins, 2)
        else:
            m['avg_win'] = 0.0
            m['avg_win_pct'] = 0.0

        if n_losses > 0:
            m['avg_loss'] = round(sum(t.gross_pnl for t in losses) / n_losses, 2)
            m['avg_loss_pct'] = round(sum(t.return_pct for t in losses) / n_losses, 2)
        else:
            m['avg_loss'] = 0.0
            m['avg_loss_pct'] = 0.0

        # Win/Loss Ratio
        if abs(m['avg_loss']) > 0:
            m['win_loss_ratio'] = round(abs(m['avg_win'] / m['avg_loss']), 2)
        else:
            m['win_loss_ratio'] = float('inf')

        # Max Consecutive Wins / Losses
        m['max_consecutive_wins'] = self._max_consecutive(trades, True)
        m['max_consecutive_losses'] = self._max_consecutive(trades, False)

        # Average Holding Period
        m['avg_holding_days'] = round(sum(t.holding_days for t in trades) / n, 1)
        m['max_holding_days'] = max(t.holding_days for t in trades)
        m['min_holding_days'] = min(t.holding_days for t in trades)

        # Expectancy (average return per trade)
        total_gross = sum(t.gross_pnl for t in trades)
        m['total_gross_pnl'] = round(total_gross, 2)
        m['total_net_pnl'] = round(sum(t.net_pnl for t in trades), 2)
        m['total_cost'] = round(sum(t.cost for t in trades), 2)
        m['avg_trade_gross_pnl'] = round(total_gross / n, 2)
        m['avg_trade_net_pnl'] = round(sum(t.net_pnl for t in trades) / n, 2)
        m['avg_trade_return_pct'] = round(sum(t.return_pct for t in trades) / n, 2)

        # Expectancy formula: (Win% * AvgWin) - (Loss% * AvgLoss)
        m['expectancy'] = round(
            (m['win_rate_pct'] / 100 * m['avg_win']) -
            ((100 - m['win_rate_pct']) / 100 * abs(m['avg_loss'])), 2
        )
        m['expectancy_pct'] = round(
            (m['win_rate_pct'] / 100 * m['avg_win_pct']) -
            ((100 - m['win_rate_pct']) / 100 * abs(m['avg_loss_pct'])), 2
        )

        # Best/Worst trade
        best_trade = max(trades, key=lambda t: t.gross_pnl)
        worst_trade = min(trades, key=lambda t: t.gross_pnl)
        m['best_trade'] = {
            'ticker': best_trade.ticker,
            'pnl': round(best_trade.gross_pnl, 2),
            'return_pct': best_trade.return_pct,
            'holding_days': best_trade.holding_days,
        }
        m['worst_trade'] = {
            'ticker': worst_trade.ticker,
            'pnl': round(worst_trade.gross_pnl, 2),
            'return_pct': worst_trade.return_pct,
            'holding_days': worst_trade.holding_days,
        }

        # Exit reason breakdown
        reasons = {}
        for t in trades:
            reason = t.exit_reason
            if reason not in reasons:
                reasons[reason] = {'count': 0, 'total_pnl': 0.0, 'wins': 0}
            reasons[reason]['count'] += 1
            reasons[reason]['total_pnl'] += t.gross_pnl
            if t.is_win:
                reasons[reason]['wins'] += 1
        m['exit_reasons'] = {
            r: {
                'count': d['count'],
                'pct': round(d['count'] / n * 100, 1),
                'total_pnl': round(d['total_pnl'], 2),
                'win_rate': round(d['wins'] / d['count'] * 100, 1) if d['count'] > 0 else 0,
            }
            for r, d in reasons.items()
        }

        # Sector performance
        sectors = {}
        for t in trades:
            sec = t.sector or 'Unknown'
            if sec not in sectors:
                sectors[sec] = {'count': 0, 'total_pnl': 0.0, 'wins': 0}
            sectors[sec]['count'] += 1
            sectors[sec]['total_pnl'] += t.gross_pnl
            if t.is_win:
                sectors[sec]['wins'] += 1
        m['sectors'] = {
            s: {
                'count': d['count'],
                'total_pnl': round(d['total_pnl'], 2),
                'avg_pnl': round(d['total_pnl'] / d['count'], 2),
                'win_rate': round(d['wins'] / d['count'] * 100, 1) if d['count'] > 0 else 0,
            }
            for s, d in sectors.items()
        }

        return m

    @staticmethod
    def _max_consecutive(trades, is_win: bool) -> int:
        """Compute max consecutive wins or losses."""
        max_streak = 0
        current = 0
        for t in trades:
            if t.is_win == is_win:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    # ── Benchmark Comparison ──

    def _compute_benchmark_metrics(self, result: BacktestResult,
                                    equity_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute alpha, beta, correlation vs benchmark."""
        m = {}
        bm = pd.DataFrame(result.benchmark_curve)
        if bm.empty or 'value' not in bm.columns:
            return m

        # Align benchmark with equity curve
        bm = bm.set_index('date')
        eq = equity_df.set_index('date')

        # Benchmark returns
        bm['bm_return'] = bm['value'].pct_change()
        eq['strategy_return'] = eq['equity'].pct_change()

        # Merge
        merged = pd.merge(
            eq[['strategy_return']],
            bm[['bm_return']],
            left_index=True, right_index=True, how='inner'
        ).dropna()

        if len(merged) < 10:
            return m

        # Beta: covariance / variance
        cov = merged['strategy_return'].cov(merged['bm_return'])
        var = merged['bm_return'].var()
        m['beta'] = round(float(cov / var), 3) if var > 0 else 0.0

        # Correlation
        m['correlation'] = round(float(merged['strategy_return'].corr(merged['bm_return'])), 3)

        # Alpha (annualized)
        rf_daily = self.risk_free_rate / 252
        strategy_excess = merged['strategy_return'] - rf_daily
        benchmark_excess = merged['bm_return'] - rf_daily

        if len(strategy_excess) > 0:
            # Alpha = mean(strategy_excess) - beta * mean(benchmark_excess)
            alpha_daily = float(strategy_excess.mean() - m['beta'] * benchmark_excess.mean())
            m['alpha_annualized_pct'] = round(float(alpha_daily * 252 * 100), 2)
            m['alpha_daily_pct'] = round(float(alpha_daily * 100), 4)
        else:
            m['alpha_annualized_pct'] = 0.0

        # Tracking Error
        diff = merged['strategy_return'] - merged['bm_return']
        m['tracking_error_pct'] = round(float(diff.std() * np.sqrt(252) * 100), 2)

        # Information Ratio
        if m['tracking_error_pct'] > 0:
            m['information_ratio'] = round(float(
                (diff.mean() * 252) / (diff.std() * np.sqrt(252))
            ), 3)
        else:
            m['information_ratio'] = 0.0

        # Benchmark Total Return
        if result.initial_capital > 0:
            bm_first = bm['value'].iloc[0] if len(bm) > 0 else result.initial_capital
            bm_last = bm['value'].iloc[-1] if len(bm) > 0 else bm_first
            if bm_first > 0:
                m['benchmark_return_pct'] = round((bm_last / bm_first - 1) * 100, 2)
            else:
                m['benchmark_return_pct'] = 0.0

        # Excess return vs benchmark
        m['excess_return_pct'] = round(result.total_return_pct - m.get('benchmark_return_pct', 0), 2)

        return m

    # ── Market Regime Sensitivity ──

    def _compute_regime_sensitivity(self, equity_df: pd.DataFrame,
                                     result: BacktestResult) -> Dict[str, Any]:
        """Analyze performance under different market regimes."""
        m = {}

        if len(equity_df) < 50:
            return m

        # Compute 20-day rolling volatility and MA50 status
        eq = equity_df.copy()
        eq['equity_cumret'] = (1 + eq['daily_return']).cumprod()

        # Volatility regime: split into terciles of rolling 20d vol
        eq['rolling_vol'] = eq['daily_return'].rolling(20).std() * np.sqrt(252)
        vol_median = eq['rolling_vol'].median()

        high_vol = eq[eq['rolling_vol'] >= vol_median * 1.2]
        low_vol = eq[eq['rolling_vol'] <= vol_median * 0.8]
        normal_vol = eq[(eq['rolling_vol'] > vol_median * 0.8) &
                         (eq['rolling_vol'] < vol_median * 1.2)]

        m['vol_regime'] = {}
        for name, subset in [('HIGH', high_vol), ('NORMAL', normal_vol), ('LOW', low_vol)]:
            if len(subset) > 0:
                ret_sum = subset['daily_return'].sum()
                days = len(subset)
                m['vol_regime'][name] = {
                    'days': days,
                    'total_return_pct': round(float(ret_sum * 100), 2),
                    'daily_return_pct': round(float(subset['daily_return'].mean() * 100), 4),
                    'hit_rate_pct': round(float((subset['daily_return'] > 0).mean() * 100), 1),
                }

        # Trend regime: above/below MA50 (using 50-day rolling mean of equity)
        if len(eq) >= 50:
            eq['ma50'] = eq['equity'].rolling(50).mean()
            eq['above_ma50'] = eq['equity'] > eq['ma50']
            above = eq[eq['above_ma50']]
            below = eq[~eq['above_ma50']]

            m['trend_regime'] = {}
            for name, subset in [('ABOVE_MA50', above), ('BELOW_MA50', below)]:
                if len(subset) > 0:
                    m['trend_regime'][name] = {
                        'days': len(subset),
                        'total_return_pct': round(float(subset['daily_return'].sum() * 100), 2),
                        'daily_return_pct': round(float(subset['daily_return'].mean() * 100), 4),
                        'hit_rate_pct': round(float((subset['daily_return'] > 0).mean() * 100), 1),
                    }

        return m

    # ── Summary ──

    def _compute_summary(self, metrics: Dict[str, Any],
                         result: BacktestResult) -> Dict[str, Any]:
        """Compile a concise summary of key metrics."""
        returns = metrics.get('returns', {})
        risk = metrics.get('risk_adjusted', {})
        dd = metrics.get('drawdown', {})
        trades = metrics.get('trades', {})
        bm = metrics.get('benchmark', {})

        return {
            'start_date': result.start_date,
            'end_date': result.end_date,
            'initial_capital': result.initial_capital,
            'final_equity': result.final_equity,
            'total_return_pct': returns.get('total_return_pct', 0),
            'cagr_pct': returns.get('cagr_pct', 0),
            'annualized_volatility_pct': returns.get('annualized_volatility_pct', 0),
            'sharpe_ratio': risk.get('sharpe_ratio', 0),
            'sortino_ratio': risk.get('sortino_ratio', 0),
            'calmar_ratio': risk.get('calmar_ratio', 0),
            'max_drawdown_pct': dd.get('max_drawdown_pct', 0),
            'max_drawdown_days': dd.get('max_drawdown_days', 0),
            'num_trades': trades.get('num_trades', 0),
            'win_rate_pct': trades.get('win_rate_pct', 0),
            'profit_factor': trades.get('profit_factor', 0),
            'expectancy': trades.get('expectancy', 0),
            'avg_holding_days': trades.get('avg_holding_days', 0),
            'beta': bm.get('beta', None),
            'alpha_annualized_pct': bm.get('alpha_annualized_pct', None),
            'information_ratio': bm.get('information_ratio', None),
            'excess_return_pct': bm.get('excess_return_pct', None),
        }


# ═══════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(result: BacktestResult,
                    risk_free_rate: float = 0.04) -> Dict[str, Any]:
    """Convenience function to compute all metrics."""
    calc = MetricsCalculator(risk_free_rate=risk_free_rate)
    return calc.compute(result)
