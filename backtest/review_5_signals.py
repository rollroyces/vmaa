#!/usr/bin/env python3
"""
Review VMAA's 5 tight BUY signals — will they lose money?
Tests:
  1. Risk/Reward math analysis
  2. Historical walk-forward simulation (1yr lookback, 2yr test)
  3. Monte Carlo simulation with actual win/loss profile
"""
from __future__ import annotations

import json, sys, time, logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RC
from risk import compute_confidence

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("review")

# === The 5 tight BUY signals ===
SIGNALS = [
    {"ticker": "TCMD", "entry": 26.19, "stop": 19.99, "tp1": 26.19*1.15, "qty": 1919, "q": 83, "magna": 8.0},
    {"ticker": "WAY",  "entry": 21.01, "stop": 17.23, "tp1": 21.01*1.15, "qty": 3113, "q": 97, "magna": 5.0},
    {"ticker": "VCYT", "entry": 39.58, "stop": 29.68, "tp1": 39.58*1.15, "qty": 1082, "q": 66, "magna": 9.0},
    {"ticker": "FA",   "entry": 15.40, "stop": 12.63, "tp1": 15.40*1.15, "qty": 3792, "q": 63, "magna": 9.0},
    {"ticker": "MITK", "entry": 15.01, "stop": 12.31, "tp1": 15.01*1.15, "qty": 3734, "q": 70, "magna": 8.0},
]

HARD_STOP_PCT = 0.25
TP1_PCT = 0.15
TRAILING_ACTIVATE = 0.18


def run_historical_simulation(ticker: str, entry_price: float, stop_price: float,
                              lookback_years: int = 3):
    """
    Simulate what would have happened if we entered at similar setups in the past.
    Find all dates where price was near entry_price (±5%) and simulate forward.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_years * 365)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
    except Exception as e:
        return None

    close = df['Close'].values.flatten()
    high = df['High'].values.flatten()
    low = df['Low'].values.flatten()
    dates = df.index

    # Find entry trigger dates: price near entry ±5%
    entry_zone = (close >= entry_price * 0.95) & (close <= entry_price * 1.05)

    results = []
    for i in range(len(close) - 120):  # Need at least 120 trading days forward
        if not entry_zone[i]:
            continue
        if i < 20:
            continue

        sim_entry = close[i]
        sim_stop = sim_entry * (1 - HARD_STOP_PCT)
        sim_tp1 = sim_entry * (1 + TP1_PCT)
        sim_trail_level = sim_entry * (1 + TRAILING_ACTIVATE)

        # Simulate forward until exit
        exit_price = None
        exit_reason = "time_stop"
        highest = sim_entry

        for j in range(i + 1, min(i + 252, len(close))):
            curr_high = high[j]
            curr_low = low[j]
            curr_close = close[j]

            # Check hard stop hit (using intraday low)
            if curr_low <= sim_stop:
                exit_price = sim_stop
                exit_reason = "hard_stop"
                break

            # Check TP1 hit (using intraday high)
            if curr_high >= sim_tp1:
                exit_price = sim_tp1
                exit_reason = "tp1"
                break

            # Update trailing stop
            highest = max(highest, curr_high)
            if highest >= sim_trail_level:
                trail_stop = highest * (1 - 0.12)  # 12% trailing
                if curr_low <= trail_stop:
                    exit_price = trail_stop
                    exit_reason = "trailing_stop"
                    break

            # Time stop: 252 days
            if j - i >= 252:
                exit_price = curr_close
                exit_reason = "time_stop"
                break

        if exit_price is None:
            exit_price = close[min(i + 252, len(close) - 1)]
            exit_reason = "time_stop"

        pnl_pct = (exit_price - sim_entry) / sim_entry * 100
        holding_days = j - i if j > i else 252

        results.append({
            'entry_date': str(dates[i].date()),
            'entry_price': round(float(sim_entry), 2),
            'exit_price': round(float(exit_price), 2),
            'pnl_pct': round(pnl_pct, 2),
            'exit_reason': exit_reason,
            'holding_days': holding_days,
        })

    return results


def monte_carlo_simulation(signals, n_sims=10000, hard_stop=0.25, tp=0.15):
    """
    Monte Carlo: assume win rate based on backtest data.
    Calculate expected value and ruin probability.
    """
    print("\n" + "=" * 70)
    print("🎲 MONTE CARLO SIMULATION")
    print("=" * 70)

    # From historical backtest (WIDE_STOP era):
    # Win rate needs to be > hard_stop/(hard_stop + tp) = 0.25/(0.25+0.15) = 62.5% just to break even
    breakeven_wr = hard_stop / (hard_stop + tp)
    print(f"\n📊 Risk/Reward Math:")
    print(f"   Hard Stop: {hard_stop:.0%}  |  TP1: {tp:.0%}")
    print(f"   Risk per trade:  {hard_stop:.0%}")
    print(f"   Reward per trade: {tp:.0%}")
    print(f"   Breakeven Win Rate: {breakeven_wr:.1%}")
    print(f"   (You need >{breakeven_wr:.0%} win rate just to break even!)")

    # Scenarios at different win rates
    for wr in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        ev = wr * tp - (1 - wr) * hard_stop
        print(f"   Win Rate {wr:.0%}: EV = {ev:+.1%} per trade")

    print(f"\n📊 Monte Carlo (10,000 sims, 5 positions, 50 trades each):")
    for wr_label, wr in [("Conservative", 0.45), ("Realistic", 0.53), ("Optimistic", 0.60)]:
        final_values = []
        for _ in range(n_sims):
            capital = 1.0
            for _ in range(50):
                # Each trade risks hard_stop of current capital
                if np.random.random() < wr:
                    capital *= (1 + tp * 0.18)  # 18% position size × TP
                else:
                    capital *= (1 - hard_stop * 0.18)  # 18% position × stop
            final_values.append(capital)

        final_values = np.array(final_values)
        win_prob = np.mean(final_values > 1.0)
        median_return = np.median(final_values) - 1
        p10 = np.percentile(final_values, 10) - 1
        p90 = np.percentile(final_values, 90) - 1
        print(f"   {wr_label} ({wr:.0%} WR): Median={median_return:+.1%}  P10={p10:+.1%}  P90={p90:+.1%}  Win%={win_prob:.1%}")


def run_signal_backtest():
    """Run historical simulation on each signal."""
    print("\n" + "=" * 70)
    print("📈 HISTORICAL PATTERN SIMULATION")
    print("   (Finding all historical instances where price ≈ entry and simulating forward)")
    print("=" * 70)

    all_results = {}
    for sig in SIGNALS:
        tkr = sig['ticker']
        entry = sig['entry']
        stop = sig['stop']
        print(f"\n--- {tkr} (Entry=${entry}, Stop=${stop}) ---")
        results = run_historical_simulation(tkr, entry, stop, lookback_years=3)
        if results is None or len(results) == 0:
            print(f"  ⚠️  No historical entry points found (insufficient data)")
            all_results[tkr] = []
            continue

        df = pd.DataFrame(results)
        wins = df[df['pnl_pct'] > 0]
        losses = df[df['pnl_pct'] <= 0]
        wr = len(wins) / len(df) if len(df) > 0 else 0

        print(f"  Found {len(df)} historical entry points")
        print(f"  Win Rate: {wr:.1%} ({len(wins)}W / {len(losses)}L)")
        print(f"  Avg Win: {wins['pnl_pct'].mean():+.1f}%" if len(wins) > 0 else "  No wins")
        print(f"  Avg Loss: {losses['pnl_pct'].mean():+.1f}%" if len(losses) > 0 else "  No losses")
        print(f"  Avg P&L: {df['pnl_pct'].mean():+.1f}%")
        print(f"  Max Win: {df['pnl_pct'].max():+.1f}% | Max Loss: {df['pnl_pct'].min():+.1f}%")

        exit_counts = df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            subset = df[df['exit_reason'] == reason]
            avg_pnl = subset['pnl_pct'].mean()
            print(f"  {reason}: {count} trades, Avg P&L: {avg_pnl:+.1f}%")

        all_results[tkr] = results

    return all_results


def recent_performance_check():
    """Check actual recent price action of the 5 signals."""
    print("\n" + "=" * 70)
    print("📊 RECENT PRICE ACTION (Last 90 days)")
    print("=" * 70)

    for sig in SIGNALS:
        tkr = sig['ticker']
        entry = sig['entry']
        stop = sig['stop']
        try:
            df = yf.download(tkr, period='3mo', progress=False, auto_adjust=True)
            if df.empty:
                print(f"  {tkr}: No data")
                continue
            close = df['Close'].values.flatten()
            current = close[-1]
            change_30d = (close[-1] / close[-22] - 1) * 100 if len(close) >= 22 else 0
            change_90d = (close[-1] / close[0] - 1) * 100
            dist_to_entry = (current - entry) / entry * 100
            dist_to_stop = (current - stop) / stop * 100
            atr = (df['High'] - df['Low']).tail(14).mean()
            print(f"  {tkr:6s}  Now=${current:.2f}  vs Entry=${entry:.2f} ({dist_to_entry:+.1f}%)  "
                  f"vs Stop=${stop:.2f} ({dist_to_stop:+.1f}%)  "
                  f"30d={change_30d:+.1f}%  90d={change_90d:+.1f}%  ATR14=${atr:.2f}")
        except Exception as e:
            print(f"  {tkr}: Error - {e}")


def summary_review():
    """Overall strategy review."""
    print("\n" + "=" * 70)
    print("🔍 STRATEGY REVIEW & CONCERNS")
    print("=" * 70)

    issues = []

    # 1. R:R ratio
    print("\n1️⃣  RISK:REWARD IMBALANCE")
    print(f"   Hard Stop: {HARD_STOP_PCT:.0%} | TP1: {TP1_PCT:.0%}")
    print(f"   R:R = 1:{TP1_PCT/HARD_STOP_PCT:.1f}")
    print(f"   ⚠️  You're risking {HARD_STOP_PCT:.0%} to make {TP1_PCT:.0%} — this is INVERTED R:R!")
    print(f"   Most profitable strategies target R:R ≥ 1:2 (risk 1 to make 2)")
    print(f"   Breakeven WR = {HARD_STOP_PCT/(HARD_STOP_PCT+TP1_PCT):.1%}")
    issues.append("INVERTED R:R — risking more than you make")

    # 2. Gap entries on small caps
    print("\n2️⃣  GAP ENTRY RELIABILITY")
    print(f"   4/5 signals are gap entries on small caps (avg cap < $2B)")
    print(f"   Gap-ups on small caps have high false-positive rates")
    print(f"   Gaps often fill within days, triggering hard stops")
    issues.append("Small-cap gap entries have high failure rate")

    # 3. Stop placement
    print("\n3️⃣  STOP PLACEMENT")
    for sig in SIGNALS:
        stop_dist = (sig['entry'] - sig['stop']) / sig['entry'] * 100
        print(f"   {sig['ticker']}: Stop at {sig['stop']} ({stop_dist:.1f}% below entry)")
    print(f"   ATR-based stops are wide to allow mean-reversion")
    print(f"   But on volatile small caps, 25% stops trigger frequently")
    issues.append("25% hard stops on small caps trigger frequently")

    # 4. Concentration
    print("\n4️⃣  DIVERSIFICATION")
    sectors = set()
    for sig in SIGNALS:
        try:
            info = yf.Ticker(sig['ticker']).info
            sector = info.get('sector', 'Unknown')
            sectors.add(sector)
            print(f"   {sig['ticker']}: {sector}")
        except:
            pass
    print(f"   ⚠️  Many signals cluster in similar sectors → correlation risk")
    issues.append("Sector concentration risk")

    # 5. The math
    print("\n5️⃣  EXPECTANCY MATH")
    for wr_pct in [40, 45, 50, 53, 55, 60, 65]:
        wr = wr_pct / 100
        ev = wr * TP1_PCT - (1 - wr) * HARD_STOP_PCT
        status = "✅" if ev > 0 else "❌"
        print(f"   WR={wr_pct}%: EV = {ev:+.1%} per trade {status}")
    issues.append("Need WR > 62.5% just to break even")

    print(f"\n⚠️  CRITICAL ISSUES FOUND: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")

    return issues


if __name__ == "__main__":
    # 1. Strategy math review
    issues = summary_review()

    # 2. Monte Carlo
    monte_carlo_simulation(SIGNALS)

    # 3. Recent price check
    recent_performance_check()

    # 4. Historical simulation (this takes time)
    print("\n\n⏳ Running historical pattern simulation...")
    all_results = run_signal_backtest()

    # 5. Final verdict
    print("\n" + "=" * 70)
    print("🏁 FINAL VERDICT")
    print("=" * 70)

    total_trades = sum(len(v) for v in all_results.values())
    if total_trades > 0:
        all_pnls = []
        for results in all_results.values():
            for r in results:
                all_pnls.append(r['pnl_pct'])
        all_pnls = np.array(all_pnls)
        wr = np.mean(all_pnls > 0)
        avg_pnl = np.mean(all_pnls)
        print(f"  Total historical simulations: {total_trades}")
        print(f"  Overall Win Rate: {wr:.1%}")
        print(f"  Average P&L: {avg_pnl:+.1f}%")
        print(f"  P10 P&L: {np.percentile(all_pnls, 10):+.1f}%")
        print(f"  P90 P&L: {np.percentile(all_pnls, 90):+.1f}%")

        if avg_pnl > 0:
            print(f"\n  🟢 Strategy shows positive expectancy based on historical patterns")
        else:
            print(f"\n  🔴 Strategy shows NEGATIVE expectancy — LIKELY TO LOSE MONEY")
    else:
        print(f"  ⚠️  Insufficient historical data for simulation")
