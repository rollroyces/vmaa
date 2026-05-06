#!/usr/bin/env python3
"""
VMAA-HK Backtest Runner CLI
============================
Command-line interface for HK backtesting.

Usage:
    # Test with specific HK tickers
    python3 backtest/hk/hk_runner.py --tickers 0700.HK,0388.HK,0016.HK --start 2023-01-01

    # Full HSI scan
    python3 backtest/hk/hk_runner.py --full-scan --start 2023-01-01 --end 2025-12-31

    # Compare with US backtest results
    python3 backtest/hk/hk_runner.py --compare

    # Custom capital
    python3 backtest/hk/hk_runner.py --tickers 0700.HK,0388.HK --capital 1000000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure vmaa root is importable
_vmaa_root = Path(__file__).resolve().parent.parent.parent
if str(_vmaa_root) not in sys.path:
    sys.path.insert(0, str(_vmaa_root))

from backtest.hk.hk_config import HKBacktestConfig, HKSlippageConfig, HKScreeningThresholds
from backtest.hk.hk_data import HKDataLoader
from backtest.hk.hk_backtest import HKBacktestEngine
from backtest.hk.hk_metrics import HKMetricsCalculator
from pipeline_hk import HSI_TICKERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vmaa.backtest.hk.runner")


# ═══════════════════════════════════════════════════════════════════
# Display Helpers
# ═══════════════════════════════════════════════════════════════════

def format_hk_report(result, metrics: dict) -> str:
    """Generate a Telegram-friendly text report for HK backtest."""
    s = metrics.get('summary', {})
    ret = metrics.get('returns', {})
    ra = metrics.get('risk_adjusted', {})
    dd = metrics.get('drawdown', {})
    tr = metrics.get('trades', {})
    bm = metrics.get('benchmark', {})
    sectors = metrics.get('sectors', {})

    lines = []
    lines.append("=" * 50)
    lines.append("🇭🇰 VMAA-HK Backtest Report")
    lines.append("=" * 50)
    lines.append(f"Period: {result.start_date} → {result.end_date}")
    lines.append(f"Capital: HKD {result.initial_capital:,.0f}")
    lines.append(f"Currency: {result.currency}")
    lines.append("")

    lines.append("── 📈 Returns ──")
    lines.append(f"  Total Return:  {s.get('total_return_pct', 0):+.1f}%")
    lines.append(f"  CAGR:          {s.get('cagr_pct', 0):+.1f}%")
    lines.append(f"  Final Equity:  HKD {result.final_equity:,.0f}")
    lines.append(f"  Total P&L:     HKD {result.total_return:+,.0f}")
    lines.append("")

    lines.append("── ⚠️ Risk ──")
    lines.append(f"  Volatility:    {ret.get('annual_volatility_pct', 0):.1f}%")
    lines.append(f"  Sharpe:        {s.get('sharpe_ratio', 0):.2f}")
    lines.append(f"  Max DD:        {s.get('max_drawdown_pct', 0):.1f}%")
    lines.append(f"  Max DD (HKD):  HKD {dd.get('max_drawdown_hkd', 0):,.0f}")
    lines.append(f"  Calmar:        {ra.get('calmar_ratio', 0):.2f}")
    lines.append("")

    lines.append("── 💼 Trades ──")
    lines.append(f"  Total Trades:  {s.get('num_trades', 0)}")
    lines.append(f"  Win Rate:      {s.get('win_rate_pct', 0):.0f}%")
    lines.append(f"  Profit Factor: {s.get('profit_factor', 0):.2f}")
    lines.append(f"  Avg Win:       HKD {tr.get('avg_win_hkd', 0):,.0f}")
    lines.append(f"  Avg Loss:      HKD {tr.get('avg_loss_hkd', 0):,.0f}")
    lines.append(f"  Expectancy:    HKD {tr.get('expectancy_hkd', 0):,.0f}")
    lines.append(f"  Avg Hold:      {tr.get('avg_hold_days', 0):.0f} days")
    lines.append(f"  Best Trade:    +{tr.get('best_trade_pct', 0):.1f}%")
    lines.append(f"  Worst Trade:   {tr.get('worst_trade_pct', 0):.1f}%")

    if tr.get('exit_reasons'):
        lines.append(f"  Exit Reasons:  {tr['exit_reasons']}")
    lines.append("")

    if bm:
        lines.append("── 📊 vs 2800.HK (TraHK) ──")
        lines.append(f"  Benchmark:     {bm.get('benchmark_return_pct', 0):+.1f}%")
        lines.append(f"  Excess Return: {bm.get('excess_return_pct', 0):+.1f}%")
        lines.append(f"  Alpha:         {bm.get('alpha', 0):+.1f}%")
        lines.append(f"  Beta:          {bm.get('beta', 0):.2f}")
        lines.append(f"  Info Ratio:    {bm.get('information_ratio', 0):.2f}")
        lines.append("")

    if sectors:
        lines.append("── 🏢 Sector Breakdown ──")
        for sec, data in list(sectors.items())[:8]:
            emoji = "🟢" if data['total_pnl_hkd'] > 0 else "🔴"
            lines.append(
                f"  {emoji} {sec[:25]:25s} "
                f"{data['trades']:3d} trades | "
                f"WR: {data['win_rate_pct']:.0f}% | "
                f"PnL: HKD {data['total_pnl_hkd']:,.0f}"
            )
        lines.append("")

    lines.append("=" * 50)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

def run_hk_backtest(args: argparse.Namespace) -> None:
    """Execute an HK backtest."""
    # Universe
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.full_scan:
        tickers = list(HSI_TICKERS)
        logger.info(f"Full HSI scan: {len(tickers)} tickers")
    else:
        tickers = list(HSI_TICKERS)[:10]  # Default: first 10 for quick test
        logger.info(f"Default universe (first 10 HSI): {tickers}")

    # Apply ticker limit
    if args.max_tickers > 0:
        tickers = tickers[:args.max_tickers]
        logger.info(f"Limited to {len(tickers)} tickers")

    # Build config
    config = HKBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        rebalance_frequency=args.mode.replace('_rebalance', ''),
        benchmark_ticker=args.benchmark,
        slippage=HKSlippageConfig(
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
        max_tickers=0,  # handled above
        tickers=tickers,
    )

    # Create engine and run
    engine = HKBacktestEngine(config)
    result = engine.run(tickers=tickers)

    # Compute metrics
    calc = HKMetricsCalculator(risk_free_rate=args.risk_free_rate)
    metrics = calc.compute(result)

    # Print report
    print("\n")
    print(format_hk_report(result, metrics))

    # Save outputs
    if not args.no_save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON summary
        summary_path = output_dir / f"hk_backtest_{args.start}_{args.end}.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'config': result.config,
                'summary': metrics.get('summary', {}),
                'returns': metrics.get('returns', {}),
                'risk_adjusted': metrics.get('risk_adjusted', {}),
                'drawdown': metrics.get('drawdown', {}),
                'trades': metrics.get('trades', {}),
                'benchmark': metrics.get('benchmark', {}),
                'sectors': metrics.get('sectors', {}),
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n📁 HK results: {summary_path}")

        # Save trade log CSV
        if result.trades and args.save_trades:
            import csv
            trades_path = output_dir / f"hk_trades_{args.start}_{args.end}.csv"
            with open(trades_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'ticker', 'entry_date', 'exit_date', 'entry_price',
                    'exit_price', 'quantity', 'gross_pnl', 'net_pnl',
                    'cost', 'return_pct', 'exit_reason', 'holding_days',
                    'sector', 'is_win'
                ])
                for t in result.trades:
                    writer.writerow([
                        t.ticker, t.entry_date, t.exit_date, t.entry_price,
                        t.exit_price, t.quantity, t.gross_pnl, t.net_pnl,
                        t.cost, t.return_pct, t.exit_reason, t.holding_days,
                        t.sector, t.is_win,
                    ])
            print(f"📁 Trade log: {trades_path}")


def run_hk_compare(args: argparse.Namespace) -> None:
    """Compare HK backtest with latest US backtest if available."""
    print("\n" + "=" * 50)
    print("🇭🇰 vs 🇺🇸 Backtest Comparison")
    print("=" * 50)

    # Try to load latest US backtest result
    us_result = None
    us_output_dir = Path("backtest/output")
    if us_output_dir.exists():
        json_files = sorted(us_output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                if 'summary' in data:
                    us_result = data
                    print(f"\nUS reference: {jf.name}")
                    break
            except Exception:
                continue

    # Run quick HK backtest for comparison
    test_tickers = args.tickers or "0700.HK,0388.HK,0016.HK,0005.HK,1299.HK"
    tickers = [t.strip() for t in test_tickers.split(',')]

    config = HKBacktestConfig(
        start_date=args.start or "2023-01-01",
        end_date=args.end or "2025-12-31",
        initial_capital=args.capital,
        tickers=tickers,
        max_tickers=0,
    )

    engine = HKBacktestEngine(config)
    result = engine.run(tickers=tickers)

    calc = HKMetricsCalculator()
    metrics = calc.compute(result)
    hk_summary = metrics.get('summary', {})

    print(f"\n🇭🇰 HK VMAA ({len(tickers)} stocks, {'-'.join(tickers[:3])}...):")
    print(f"   Return: {hk_summary.get('total_return_pct', 0):+.1f}%")
    print(f"   CAGR:   {hk_summary.get('cagr_pct', 0):+.1f}%")
    print(f"   Sharpe: {hk_summary.get('sharpe_ratio', 0):.2f}")
    print(f"   Max DD: {hk_summary.get('max_drawdown_pct', 0):.1f}%")
    print(f"   Win Rt: {hk_summary.get('win_rate_pct', 0):.0f}%")
    print(f"   Trades: {hk_summary.get('num_trades', 0)}")

    if us_result:
        us_summary = us_result.get('summary', {})
        print(f"\n🇺🇸 US VMAA (reference):")
        print(f"   Return: {us_summary.get('total_return_pct', 0):+.1f}%")
        print(f"   CAGR:   {us_summary.get('cagr_pct', 0):+.1f}%")
        print(f"   Sharpe: {us_summary.get('sharpe_ratio', 0):.2f}")
        print(f"   Max DD: {us_summary.get('max_drawdown_pct', 0):.1f}%")
        print(f"   Win Rt: {us_summary.get('win_rate_pct', 0):.0f}%")
        print(f"   Trades: {us_summary.get('num_trades', 0)}")

        # Comparison table
        print(f"\n{'Metric':<20s} {'🇭🇰 HK':>10s} {'🇺🇸 US':>10s} {'Diff':>10s}")
        print("-" * 52)
        for key, label in [
            ('total_return_pct', 'Total Return %'),
            ('cagr_pct', 'CAGR %'),
            ('sharpe_ratio', 'Sharpe'),
            ('max_drawdown_pct', 'Max DD %'),
            ('win_rate_pct', 'Win Rate %'),
        ]:
            hk_val = hk_summary.get(key, 0)
            us_val = us_summary.get(key, 0)
            diff = hk_val - us_val
            sign = "+" if diff > 0 else ""
            print(f"{label:<20s} {hk_val:>9.1f}  {us_val:>9.1f}  {sign}{diff:>9.1f}")

    print("\n" + "=" * 50)

    # Save comparison
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'hk': hk_summary,
                'us': us_result.get('summary', {}) if us_result else None,
            }, f, indent=2)
        print(f"📁 Comparison saved: {args.output}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🇭🇰 VMAA-HK Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tickers 0700.HK,0388.HK,0016.HK --start 2023-01-01
  %(prog)s --full-scan --start 2023-01-01 --end 2025-12-31
  %(prog)s --compare --tickers 0700.HK,0388.HK
        """
    )

    # ── Core Options ──
    core = parser.add_argument_group('Core Parameters')
    core.add_argument('--tickers', type=str,
                      help='Comma-separated HK tickers (e.g., 0700.HK,0388.HK,0016.HK)')
    core.add_argument('--full-scan', action='store_true',
                      help='Run on full HSI universe (~90 stocks)')
    core.add_argument('--start', type=str, default='2023-01-01',
                      help='Start date YYYY-MM-DD')
    core.add_argument('--end', type=str, default='2025-12-31',
                      help='End date YYYY-MM-DD')
    core.add_argument('--mode', type=str, default='monthly',
                      choices=['monthly', 'weekly', 'daily'],
                      help='Rebalance frequency')
    core.add_argument('--capital', type=float, default=500000.0,
                      help='Initial capital (HKD)')
    core.add_argument('--benchmark', type=str, default='2800.HK',
                      help='Benchmark ticker (default: 2800.HK TraHK)')
    core.add_argument('--max-tickers', type=int, default=0,
                      help='Limit ticker count (0=no limit)')

    # ── Cost Model ──
    costs = parser.add_argument_group('Transaction Costs (HKD)')
    costs.add_argument('--commission-fixed', type=float, default=50.0,
                       help='Fixed commission per trade (HKD)')
    costs.add_argument('--commission-pct', type=float, default=0.0025,
                       help='Percentage commission (0.0025 = 0.25%%)')
    costs.add_argument('--slippage-pct', type=float, default=0.002,
                       help='Slippage as decimal (0.002 = 0.2%%)')

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
    risk.add_argument('--hard-stop', type=float, default=0.15,
                      help='Hard stop loss (0.15 = 15%%)')
    risk.add_argument('--trailing-stop', type=float, default=0.10,
                      help='Trailing stop (0.10 = 10%%)')
    risk.add_argument('--time-stop', type=int, default=90,
                      help='Time stop in days')
    risk.add_argument('--atr-mult', type=float, default=2.0,
                      help='ATR stop multiplier')
    risk.add_argument('--risk-free-rate', type=float, default=0.03,
                      help='Annual risk-free rate (3%% for HKD)')

    # ── Output ──
    out = parser.add_argument_group('Output')
    out.add_argument('--output-dir', type=str, default='backtest/hk/output',
                     help='Output directory for reports')
    out.add_argument('--no-save', action='store_true',
                     help='Do not save reports to disk')
    out.add_argument('--save-trades', action='store_true', default=True,
                     help='Save trade log CSV')
    out.add_argument('--output', type=str, default='',
                     help='JSON output file for comparison')
    out.add_argument('--quiet', action='store_true',
                     help='Suppress detailed logging')

    # ── Modes ──
    mode_group = parser.add_argument_group('Operation Mode')
    mode_group.add_argument('--compare', action='store_true',
                            help='Compare HK vs US backtest results')

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.compare:
        run_hk_compare(args)
    else:
        run_hk_backtest(args)


if __name__ == '__main__':
    main()
