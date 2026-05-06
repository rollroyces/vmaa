#!/usr/bin/env python3
"""
VMAA 2.0 — Backtest Runner CLI
===============================
Command-line interface for running backtests and analyzing results.

Usage:
    # Basic backtest
    python3 backtest/runner.py --tickers AAPL,MSFT,GOOGL --start 2020-01-01 --end 2024-12-31

    # With benchmark comparison
    python3 backtest/runner.py --tickers AAPL,MSFT,GOOGL --benchmark SPY --mode monthly_rebalance

    # Full S&P 500 backtest (monthly rebalance recommended)
    python3 backtest/runner.py --universe sp500 --mode monthly_rebalance --capital 100000

    # Walk-forward optimization
    python3 backtest/runner.py --walk-forward --wf-train 24 --wf-test 6

    # Compare multiple runs
    python3 backtest/runner.py --compare run1.json run2.json

    # Output options
    python3 backtest/runner.py --tickers AAPL --output-dir my_results/ --no-save
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure parent vmaa directory is importable
_vmaa_root = Path(__file__).resolve().parent.parent
if str(_vmaa_root) not in sys.path:
    sys.path.insert(0, str(_vmaa_root))

from backtest.config import BacktestConfig, SlippageConfig
from backtest.data import HistoricalDataLoader
from backtest.engine import BacktestEngine
from backtest.metrics import MetricsCalculator
from backtest.report import (
    ReportGenerator, compare_runs, format_comparison_table
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vmaa.backtest.runner")


# ═══════════════════════════════════════════════════════════════════
# S&P 500 Ticker List
# ═══════════════════════════════════════════════════════════════════

def get_sp500_tickers() -> List[str]:
    """Get current S&P 500 constituents."""
    try:
        import requests
        import io
        import pandas as pd
        url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            return df['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        logger.warning(f"Could not fetch S&P 500: {e}")

    # Fallback: top 100 liquid S&P 500 stocks
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM",
        "DIS", "NFLX", "ADBE", "CRM", "CSCO", "INTC", "VZ", "T", "PFE",
        "MRK", "ABBV", "PEP", "KO", "TMO", "NKE", "ABT", "DHR", "MDT",
        "BMY", "AMGN", "LOW", "UPS", "QCOM", "TXN", "HON", "GE", "CAT",
        "DE", "SBUX", "MCD", "ORCL", "IBM", "CVX", "COP", "GILD", "UBER",
        "PLTR", "SOFI", "COIN", "DKNG", "HOOD", "AFRM", "UBER", "LYFT",
        "CMG", "ISRG", "NOW", "INTU", "PANW", "AMD", "AVGO", "LLY",
        "COST", "AXP", "GS", "MS", "BLK", "SPGI", "C", "SCHW",
        "UNP", "BA", "LMT", "RTX", "NOC", "GD", "FDX", "EMR", "ETN",
        "DUK", "SO", "NEE", "D", "AEP", "EXC", "SRE", "PEG",
        "AMT", "PLD", "CCI", "EQIX", "WELL", "O", "SPG", "PSA",
        "PFE", "MRK", "JNJ", "ABBV", "BMY", "AMGN", "GILD", "REGN",
        "XLE", "CVX", "XOM", "COP", "EOG", "SLB", "PSX", "MPC",
        "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "DOW",
    ]


# ═══════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════

def run_backtest(args: argparse.Namespace) -> None:
    """Execute a backtest based on CLI arguments."""
    # Build config
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_frequency=args.mode.replace('_rebalance', ''),
        benchmark_ticker=args.benchmark,
        slippage=SlippageConfig(
            fixed_commission=args.commission_fixed,
            pct_commission=args.commission_pct,
            slippage_pct=args.slippage_pct,
        ),
        max_positions=args.max_positions,
        max_positions_per_sector=args.max_per_sector,
        max_position_pct=args.max_position_pct,
        kelly_fraction=args.kelly_fraction,
        hard_stop_pct=args.hard_stop,
        trailing_stop_pct=args.trailing_stop,
        time_stop_days=args.time_stop,
        atr_stop_multiplier=args.atr_mult,
        re_screen_fundamentals=not args.no_rescreen,
        re_screen_magna=not args.no_rescreen,
        walk_forward=args.walk_forward,
        wf_train_months=args.wf_train,
        wf_test_months=args.wf_test,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_tickers=args.max_tickers,
    )

    # Universe
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    elif args.universe == 'sp500':
        tickers = get_sp500_tickers()
        logger.info(f"Using S&P 500 universe: {len(tickers)} tickers")
    else:
        tickers = []  # Use engine default
    config.tickers = tickers

    # Create engine
    engine = BacktestEngine(config)

    # Run
    result = engine.run(tickers=tickers if tickers else None)

    # Compute metrics
    calc = MetricsCalculator(risk_free_rate=args.risk_free_rate)
    metrics = calc.compute(result)

    # Generate report
    gen = ReportGenerator(result)

    # Print text report
    print("\n")
    print(gen.text_report())

    # Save outputs
    if not args.no_save:
        files = gen.save_all(args.output_dir)
        print(f"\n📁 Reports saved:")
        for fmt, path in files.items():
            print(f"   {fmt}: {path}")

    # Save result as pickle for later comparison
    if args.save_run:
        run_path = Path(args.save_run)
        import pickle
        with open(run_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"\n💾 Run saved: {run_path}")


def run_comparison(args: argparse.Namespace) -> None:
    """Compare multiple backtest runs."""
    comparison = compare_runs(args.compare)
    table = format_comparison_table(comparison)
    print("\n" + table)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n📁 Comparison saved: {args.output}")


def run_tune(args: argparse.Namespace) -> None:
    """
    Run parameter sweep / tuning.
    Tests multiple values for a specified parameter.
    """
    param_path = args.tune.split('.')
    values = [float(v) if '.' in v else int(v) for v in args.values.split(',')]

    results = []
    for val in values:
        logger.info(f"\n{'='*50}")
        logger.info(f"Tuning {args.tune} = {val}")
        logger.info(f"{'='*50}")

        config = BacktestConfig(
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            rebalance_frequency=args.mode.replace('_rebalance', ''),
            benchmark_ticker=args.benchmark,
            max_tickers=args.max_tickers,
        )

        if args.tickers:
            config.tickers = [t.strip().upper() for t in args.tickers.split(',')]
        elif args.universe == 'sp500':
            config.tickers = get_sp500_tickers()

        # Set the tuned parameter
        obj = config
        for part in param_path[:-1]:
            obj = getattr(obj, part)
        setattr(obj, param_path[-1], val)

        engine = BacktestEngine(config)
        result = engine.run(tickers=config.tickers if config.tickers else None)
        calc = MetricsCalculator()
        metrics = calc.compute(result)

        summary = metrics.get('summary', {})
        results.append({
            'param': args.tune,
            'value': val,
            'total_return_pct': summary.get('total_return_pct', 0),
            'cagr_pct': summary.get('cagr_pct', 0),
            'sharpe_ratio': summary.get('sharpe_ratio', 0),
            'max_drawdown_pct': summary.get('max_drawdown_pct', 0),
            'win_rate_pct': summary.get('win_rate_pct', 0),
            'profit_factor': summary.get('profit_factor', 0),
            'num_trades': summary.get('num_trades', 0),
        })

    # Print tuning results
    print("\n" + "=" * 70)
    print(f"TUNING RESULTS: {args.tune}")
    print("=" * 70)
    header = f"{'Value':>10s} {'Return%':>9s} {'CAGR%':>7s} {'Sharpe':>7s} {'MaxDD%':>7s} {'Win%':>6s} {'PF':>6s} {'Trades':>7s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['value']:>10} {r['total_return_pct']:>8.1f}% {r['cagr_pct']:>6.1f}% "
              f"{r['sharpe_ratio']:>6.2f}  {r['max_drawdown_pct']:>6.1f}% "
              f"{r['win_rate_pct']:>5.0f}% {r['profit_factor']:>5.2f} "
              f"{r['num_trades']:>6}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Tuning results saved: {args.output}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VMAA 2.0 Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tickers AAPL,MSFT,GOOGL --start 2020-01-01 --end 2024-12-31
  %(prog)s --universe sp500 --mode monthly_rebalance --capital 100000
  %(prog)s --compare run1.json run2.json
  %(prog)s --tune part1.min_bm --values 0.2,0.3,0.4 --tickers AAPL
        """
    )

    # ── Core Options ──
    core = parser.add_argument_group('Core Parameters')
    core.add_argument('--tickers', type=str,
                      help='Comma-separated ticker list (e.g., AAPL,MSFT,GOOGL)')
    core.add_argument('--universe', type=str, default='default',
                      choices=['default', 'sp500'],
                      help='Stock universe (default: built-in 50 liquid tickers)')
    core.add_argument('--start', type=str, default='2020-01-01',
                      help='Start date YYYY-MM-DD')
    core.add_argument('--end', type=str, default='2024-12-31',
                      help='End date YYYY-MM-DD')
    core.add_argument('--mode', type=str, default='monthly_rebalance',
                      choices=['monthly_rebalance', 'weekly_rebalance', 'daily_rebalance'],
                      help='Rebalance frequency')
    core.add_argument('--capital', type=float, default=100000.0,
                      help='Initial capital')
    core.add_argument('--benchmark', type=str, default='SPY',
                      help='Benchmark ticker')
    core.add_argument('--max-tickers', type=int, default=0,
                      help='Limit ticker count (0=no limit, useful for testing)')

    # ── Cost Model ──
    costs = parser.add_argument_group('Transaction Costs')
    costs.add_argument('--commission-fixed', type=float, default=0.99,
                       help='Fixed commission per trade (USD)')
    costs.add_argument('--commission-pct', type=float, default=0.00005,
                       help='Percentage commission')
    costs.add_argument('--slippage-pct', type=float, default=0.001,
                       help='Slippage as decimal (0.001 = 0.1%%)')

    # ── Risk Parameters ──
    risk = parser.add_argument_group('Risk Management')
    risk.add_argument('--max-positions', type=int, default=8,
                      help='Max concurrent positions')
    risk.add_argument('--max-per-sector', type=int, default=2,
                      help='Max positions per sector')
    risk.add_argument('--max-position-pct', type=float, default=0.20,
                      help='Max position size as fraction of portfolio')
    risk.add_argument('--kelly-fraction', type=float, default=0.25,
                      help='Kelly fraction for position sizing')
    risk.add_argument('--hard-stop', type=float, default=0.10,
                      help='Hard stop loss (0.10 = 10%%)')
    risk.add_argument('--trailing-stop', type=float, default=0.08,
                      help='Trailing stop (0.08 = 8%%)')
    risk.add_argument('--time-stop', type=int, default=60,
                      help='Time stop in days')
    risk.add_argument('--atr-mult', type=float, default=2.0,
                      help='ATR stop multiplier')

    # ── Advanced ──
    adv = parser.add_argument_group('Advanced')
    adv.add_argument('--walk-forward', action='store_true',
                     help='Enable walk-forward optimization')
    adv.add_argument('--wf-train', type=int, default=24,
                     help='Walk-forward train months')
    adv.add_argument('--wf-test', type=int, default=6,
                     help='Walk-forward test months')
    adv.add_argument('--no-rescreen', action='store_true',
                     help='Skip fundamentals rescreening (faster but less accurate)')
    adv.add_argument('--risk-free-rate', type=float, default=0.04,
                     help='Annual risk-free rate (0.04 = 4%%)')
    adv.add_argument('--cache-dir', type=str, default='backtest/cache',
                     help='Data cache directory')

    # ── Output ──
    out = parser.add_argument_group('Output')
    out.add_argument('--output-dir', type=str, default='backtest/output',
                     help='Output directory for reports')
    out.add_argument('--no-save', action='store_true',
                     help='Do not save reports to disk')
    out.add_argument('--save-run', type=str, default='',
                     help='Save backtest result to pickle file')
    out.add_argument('--output', type=str, default='',
                     help='JSON output file for tuning/comparison results')
    out.add_argument('--quiet', action='store_true',
                     help='Suppress detailed logging')

    # ── Modes ──
    mode_group = parser.add_argument_group('Operation Mode')
    mode_group.add_argument('--compare', nargs='+',
                            help='Compare multiple backtest report JSON files')
    mode_group.add_argument('--tune', type=str,
                            help='Parameter to tune (e.g., part1.min_bm, hard_stop_pct)')
    mode_group.add_argument('--values', type=str,
                            help='Comma-separated values for tuning')

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Dispatch
    if args.compare:
        run_comparison(args)
    elif args.tune and args.values:
        run_tune(args)
    else:
        run_backtest(args)


if __name__ == '__main__':
    main()
