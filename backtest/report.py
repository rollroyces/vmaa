#!/usr/bin/env python3
"""
Backtest Report Generator
==========================
Produces comprehensive reports from backtest results:
  - Text summary (human-readable)
  - JSON data (for programmatic use)
  - Equity curve data (for plotting)
  - Monthly returns heatmap data
  - Underwater chart data
  - Multi-run comparison
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from backtest.metrics import MetricsCalculator

logger = logging.getLogger("vmaa.backtest.report")


# ═══════════════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Generate reports from backtest results.

    Usage:
        gen = ReportGenerator(result)
        print(gen.text_report())
        gen.save_all("backtest/output/")
    """

    def __init__(self, result: BacktestResult):
        self.result = result
        if not result.metrics:
            calc = MetricsCalculator()
            calc.compute(result)

    def text_report(self) -> str:
        """Generate a human-readable text report."""
        r = self.result
        m = r.metrics
        lines = []
        add = lines.append

        # Header
        add("=" * 70)
        add("VMAA 2.0 — BACKTEST REPORT")
        add("=" * 70)
        add(f"Period:    {r.start_date} → {r.end_date}")
        add(f"Config:    {r.config.get('rebalance_frequency', 'monthly')} rebalance "
            f"| {r.config.get('initial_capital', 0):,.0f} initial capital")
        add(f"Universe:  {len(r.config.get('tickers', []))} tickers")
        add(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        add("")

        # Summary
        s = m.get('summary', {})
        add("─" * 70)
        add("PERFORMANCE SUMMARY")
        add("─" * 70)
        add(f"  Initial Capital:      ${s.get('initial_capital', 0):>12,.0f}")
        add(f"  Final Equity:         ${s.get('final_equity', 0):>12,.0f}")
        add(f"  Total Return:         {s.get('total_return_pct', 0):>11.1f}%")
        add(f"  CAGR:                 {s.get('cagr_pct', 0):>11.1f}%")
        add(f"  Annualized Vol:       {s.get('annualized_volatility_pct', 0):>11.1f}%")
        add("")

        add("─" * 70)
        add("RISK-ADJUSTED METRICS")
        add("─" * 70)
        add(f"  Sharpe Ratio:         {s.get('sharpe_ratio', 0):>11.2f}")
        add(f"  Sortino Ratio:        {s.get('sortino_ratio', 0):>11.2f}")
        add(f"  Calmar Ratio:         {s.get('calmar_ratio', 0):>11.2f}")
        add(f"  Max Drawdown:         {s.get('max_drawdown_pct', 0):>10.1f}%")
        add(f"  Max DD Duration:      {s.get('max_drawdown_days', 0):>10.0f} days")
        add("")

        add("─" * 70)
        add("TRADE STATISTICS")
        add("─" * 70)
        ts = m.get('trades', {})
        add(f"  Total Trades:         {ts.get('num_trades', 0):>10}")
        add(f"  Win Rate:             {ts.get('win_rate_pct', 0):>10.1f}%")
        add(f"  Profit Factor:        {ts.get('profit_factor', 0):>10.2f}")
        add(f"  Expectancy:           ${ts.get('expectancy', 0):>9,.2f}")
        add(f"  Avg Trade Return:     {ts.get('avg_trade_return_pct', 0):>10.2f}%")
        add(f"  Avg Win:              {ts.get('avg_win_pct', 0):>10.2f}%")
        add(f"  Avg Loss:             {ts.get('avg_loss_pct', 0):>10.2f}%")
        add(f"  Win/Loss Ratio:       {ts.get('win_loss_ratio', 0):>10.2f}")
        add(f"  Max Consec Wins:      {ts.get('max_consecutive_wins', 0):>10}")
        add(f"  Max Consec Losses:    {ts.get('max_consecutive_losses', 0):>10}")
        add(f"  Avg Holding Days:     {ts.get('avg_holding_days', 0):>10.0f}")
        add(f"  Total Gross P&L:      ${ts.get('total_gross_pnl', 0):>9,.0f}")
        add(f"  Total Net P&L:        ${ts.get('total_net_pnl', 0):>9,.0f}")
        add(f"  Total Costs:          ${ts.get('total_cost', 0):>9,.0f}")
        add("")

        # Benchmark
        bm = m.get('benchmark', {})
        if bm:
            add("─" * 70)
            add("BENCHMARK COMPARISON")
            add("─" * 70)
            add(f"  Benchmark Return:     {bm.get('benchmark_return_pct', 0):>10.1f}%")
            add(f"  Excess Return:        {bm.get('excess_return_pct', 0):>10.1f}%")
            add(f"  Alpha (annualized):   {bm.get('alpha_annualized_pct', 0):>10.2f}%")
            add(f"  Beta:                 {bm.get('beta', 0):>10.2f}")
            add(f"  Correlation:          {bm.get('correlation', 0):>10.2f}")
            add(f"  Tracking Error:       {bm.get('tracking_error_pct', 0):>10.2f}%")
            add(f"  Information Ratio:    {bm.get('information_ratio', 0):>10.2f}")
            add("")

        # Best/Worst
        if ts.get('best_trade'):
            bt = ts['best_trade']
            wt = ts['worst_trade']
            add("─" * 70)
            add("BEST / WORST TRADES")
            add("─" * 70)
            add(f"  🟢 Best:  {bt['ticker']:6s} ${bt['pnl']:>9,.0f} "
                f"({bt['return_pct']:+.1f}%) {bt['holding_days']}d")
            add(f"  🔴 Worst: {wt['ticker']:6s} ${wt['pnl']:>9,.0f} "
                f"({wt['return_pct']:+.1f}%) {wt['holding_days']}d")
            add("")

        # Exit reason breakdown
        er = ts.get('exit_reasons', {})
        if er:
            add("─" * 70)
            add("EXIT REASON BREAKDOWN")
            add("─" * 70)
            for reason, data in sorted(er.items()):
                add(f"  {reason:20s} {data['count']:4d} trades "
                    f"({data['pct']:5.1f}%)  P&L: ${data['total_pnl']:>10,.0f} "
                    f"Win: {data['win_rate']:.0f}%")
            add("")

        # Monthly returns heatmap data
        add("─" * 70)
        add("MONTHLY RETURNS")
        add("─" * 70)
        monthly = r.monthly_returns
        if monthly:
            # Group by year
            by_year: Dict[str, Dict[str, float]] = {}
            for key, val in sorted(monthly.items()):
                year, month = key.split('-')
                if year not in by_year:
                    by_year[year] = {}
                by_year[year][month] = val

            # Print horizontally
            months = ['01', '02', '03', '04', '05', '06',
                      '07', '08', '09', '10', '11', '12']
            header = "Year    " + " ".join(f"{m:>6s}" for m in months) + "    Annual"
            add(header)
            add("-" * len(header))
            for year in sorted(by_year.keys()):
                row_vals = []
                year_total = 0.0
                for m in months:
                    val = by_year[year].get(m)
                    if val is not None:
                        row_vals.append(f"{val:>+5.1f}%")
                        year_total += val
                    else:
                        row_vals.append(f"{'':>6s}")
                add(f"{year}   " + " ".join(row_vals) + f"   {year_total:>+5.1f}%")
            add("")

        # Footer
        add("═" * 70)
        add("END OF REPORT")
        add("═" * 70)

        return '\n'.join(lines)

    def json_report(self) -> Dict[str, Any]:
        """Generate a full JSON report."""
        r = self.result
        return {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'version': '2.0',
            },
            'config': r.config,
            'summary': r.metrics.get('summary', {}),
            'returns': r.metrics.get('returns', {}),
            'risk_adjusted': r.metrics.get('risk_adjusted', {}),
            'drawdown': r.metrics.get('drawdown', {}),
            'trades': r.metrics.get('trades', {}),
            'benchmark': r.metrics.get('benchmark', {}),
            'regime': r.metrics.get('regime', {}),
            'equity_curve': [
                {
                    'date': rec.date,
                    'equity': rec.equity,
                    'daily_return': rec.daily_return,
                    'drawdown': rec.drawdown,
                    'num_positions': rec.num_positions,
                }
                for rec in r.equity_curve
            ],
            'trade_log': [
                {
                    'ticker': t.ticker,
                    'entry_date': t.entry_date,
                    'exit_date': t.exit_date,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'gross_pnl': t.gross_pnl,
                    'net_pnl': t.net_pnl,
                    'return_pct': t.return_pct,
                    'exit_reason': t.exit_reason,
                    'holding_days': t.holding_days,
                    'is_win': t.is_win,
                }
                for t in r.trades
            ],
            'monthly_returns': r.monthly_returns,
            'benchmark_curve': r.benchmark_curve,
        }

    def equity_curve_data(self) -> List[Dict[str, Any]]:
        """Export equity curve as list of dicts for plotting."""
        return [
            {
                'date': rec.date,
                'equity': rec.equity,
                'daily_return': rec.daily_return,
                'drawdown': rec.drawdown,
                'num_positions': rec.num_positions,
            }
            for rec in self.result.equity_curve
        ]

    def underwater_data(self) -> List[Dict[str, Any]]:
        """Export underwater (drawdown) curve for plotting."""
        return self.result.metrics.get('drawdown', {}).get('underwater', [])

    def monthly_heatmap_data(self) -> Dict[str, Any]:
        """Export monthly returns data structured for heatmap visualization."""
        monthly = self.result.monthly_returns
        if not monthly:
            return {'years': [], 'data': []}

        years = sorted(set(k.split('-')[0] for k in monthly.keys()))
        months = ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12']

        data = []
        for year in years:
            for month in months:
                key = f"{year}-{month}"
                data.append({
                    'year': year,
                    'month': month,
                    'return': monthly.get(key),
                })

        return {'years': years, 'months': months, 'data': data}

    def save_all(self, output_dir: str = "backtest/output") -> Dict[str, str]:
        """
        Save all report formats to output directory.
        Returns dict mapping format to file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files = {}

        # Text report
        txt_path = out / f"backtest_report_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write(self.text_report())
        files['text'] = str(txt_path)

        # JSON report
        json_path = out / f"backtest_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.json_report(), f, indent=2, default=str)
        files['json'] = str(json_path)

        # Equity curve CSV
        csv_path = out / f"equity_curve_{timestamp}.csv"
        pd.DataFrame(self.equity_curve_data()).to_csv(csv_path, index=False)
        files['equity_csv'] = str(csv_path)

        # Trade log CSV
        trades_path = out / f"trade_log_{timestamp}.csv"
        self.result.trade_log_df.to_csv(trades_path, index=False)
        files['trades_csv'] = str(trades_path)

        logger.info(f"Reports saved to {out}/")
        return files


# ═══════════════════════════════════════════════════════════════════
# Multi-Run Comparison
# ═══════════════════════════════════════════════════════════════════

def compare_runs(run_files: List[str]) -> Dict[str, Any]:
    """
    Compare multiple backtest runs from saved JSON reports.

    Args:
        run_files: List of paths to backtest_report_*.json files.

    Returns:
        Comparison dict with side-by-side metrics.
    """
    runs = {}
    for path in run_files:
        try:
            with open(path) as f:
                data = json.load(f)
            name = Path(path).stem.replace('backtest_report_', '')
            runs[name] = data
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    if not runs:
        return {'error': 'No valid run files loaded'}

    # Key metrics to compare
    metrics_to_compare = [
        ('total_return_pct', 'Total Return %'),
        ('cagr_pct', 'CAGR %'),
        ('annualized_volatility_pct', 'Ann. Vol %'),
        ('sharpe_ratio', 'Sharpe'),
        ('sortino_ratio', 'Sortino'),
        ('calmar_ratio', 'Calmar'),
        ('max_drawdown_pct', 'Max DD %'),
        ('max_drawdown_days', 'Max DD Days'),
        ('num_trades', 'Trades'),
        ('win_rate_pct', 'Win Rate %'),
        ('profit_factor', 'Profit Factor'),
        ('expectancy', 'Expectancy $'),
        ('avg_holding_days', 'Avg Hold Days'),
        ('beta', 'Beta'),
        ('alpha_annualized_pct', 'Alpha %'),
        ('excess_return_pct', 'Excess Return %'),
    ]

    comparison = {'runs': list(runs.keys()), 'metrics': []}
    for key, label in metrics_to_compare:
        row = {'metric': label}
        for run_name, data in runs.items():
            summary = data.get('summary', {})
            row[run_name] = summary.get(key)
        comparison['metrics'].append(row)

    return comparison


def format_comparison_table(comparison: Dict[str, Any]) -> str:
    """Format a comparison dict as a readable table."""
    if 'error' in comparison:
        return f"Error: {comparison['error']}"

    runs = comparison.get('runs', [])
    metrics = comparison.get('metrics', [])

    if not runs or not metrics:
        return "No data to compare"

    lines = []
    add = lines.append

    # Header
    header = f"{'Metric':<24s}"
    for run in runs:
        header += f" {run[:15]:>16s}"
    add(header)
    add("-" * len(header))

    # Rows
    for row in metrics:
        line = f"{row['metric']:<24s}"
        for run in runs:
            val = row.get(run)
            if val is None:
                line += f" {'N/A':>16s}"
            elif isinstance(val, float):
                line += f" {val:>16.2f}"
            else:
                line += f" {str(val):>16s}"
        add(line)

    return '\n'.join(lines)
