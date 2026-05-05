#!/usr/bin/env python3
"""
VMAA 2.0 — Two-Stage Pipeline Orchestrator
===========================================
Full Value Mean-reversion Algorithmic Advisor pipeline.

Architecture:
  Stage 1 (Quality) → Stage 2 (Momentum) → Risk → Execution

Flow:
  1. Part 1: Core Financial Fundamentals → Quality Pool
  2. Part 2: MAGNA 53/10 Signals → Entry Triggers
  3. Risk: Position sizing, stops, confidence scoring
  4. Execution: Order placement via Tiger Broker

Usage:
  python3 pipeline.py --scan-part1              # Build quality pool only
  python3 pipeline.py --scan-part2              # Check quality pool for triggers
  python3 pipeline.py --full-scan               # Both stages
  python3 pipeline.py --full-scan --dry-run     # Full scan, no execution
  python3 pipeline.py --full-scan --live        # Full scan + LIVE execution
  python3 pipeline.py --status                  # Portfolio + risk dashboard
  python3 pipeline.py --tickers AAPL,MSFT,...   # Scan specific tickers
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add current dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import VMAACandidate, TradeDecision
from config import PC, P1C, P2C, RC as RiskCfg

from part1_fundamentals import batch_screen as part1_batch
from part2_magna import batch_screen_magna as part2_batch
from risk import (
    get_market_regime, generate_trade_decision,
    check_correlation, compute_confidence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vmaa.pipeline")


# ═══════════════════════════════════════════════════════════════════
# Universe Management
# ═══════════════════════════════════════════════════════════════════

def get_ticker_universe(source: str = "sp500",
                        custom: Optional[List[str]] = None) -> List[str]:
    """
    Get stock universe for scanning.
    Sources: sp500, russell2000, custom
    """
    if custom and len(custom) > 0:
        return [t.strip().upper() for t in custom if t.strip()]

    import pandas as pd

    if source == "sp500":
        try:
            import requests, io
            url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                symbols = df['Symbol'].str.replace('.', '-', regex=False).tolist()
                logger.info(f"Loaded {len(symbols)} S&P 500 tickers from GitHub")
                return symbols
        except Exception as e:
            logger.warning(f"Could not fetch S&P 500 list: {e}, using fallback")

    if source == "russell2000":
        try:
            table = pd.read_html(
                'https://en.wikipedia.org/wiki/Russell_2000_Index'
            )[1]  # Usually the second table
            return table.iloc[:, 0].tolist()[:500]  # Take top components
        except Exception:
            logger.warning("Could not fetch Russell 2000, using fallback")

    # Fallback: curated list of 200 liquid tickers
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM",
        "DIS", "NFLX", "ADBE", "CRM", "CSCO", "INTC", "VZ", "T", "PFE",
        "MRK", "ABBV", "PEP", "KO", "TMO", "NKE", "ABT", "DHR", "MDT",
        "BMY", "AMGN", "LOW", "UPS", "QCOM", "TXN", "HON", "GE", "RTX",
        "CAT", "DE", "MMM", "ISRG", "SPGI", "BLK", "GS", "MS", "SCHW",
        "C", "PLD", "AMT", "CCI", "EQIX", "SBUX", "MCD", "CMG", "F",
        "GM", "TSM", "BABA", "ORCL", "IBM", "CVX", "COP", "SLB", "EOG",
        "OXY", "DHI", "LEN", "NVR", "ELV", "CI", "HUM", "CNC", "CVS",
        "ZTS", "REGN", "VRTX", "GILD", "BIIB", "ILMN", "A", "WAT",
        "EPAM", "MOS", "RVTY", "UHS",
        "SOFI", "RKLB", "PLTR", "AFRM", "HOOD", "IONQ", "NU", "SNAP",
        "UBER", "LYFT", "DKNG", "RBLX", "COIN", "MARA", "RIOT", "CLSK",
        "AI", "BBAI", "SOUN", "UPST", "OPEN", "CHWY", "BYND", "PTON",
        "LCID", "RIVN", "FUBO", "QS", "STEM", "CHPT", "BLNK", "PLUG",
        "FCEL", "SPWR", "RUN", "ENPH", "SEDG", "ARRY", "NOVA", "MAXN",
        "JMIA", "CIFR", "BITF", "HUT", "WULF", "IREN", "BTBT", "CAN",
        "GME", "AMC", "BBBYQ", "PRTYQ", "MULN", "NVOS", "CYN", "MCOM",
        "TOP", "HKD", "MGOL", "GNS", "TRKA", "TTOO", "FFIE", "HOLO",
        "AREB", "GFAI", "MEGL", "CJJD", "GDHG", "JXJT", "WETG", "CNXA",
        "VCIG", "LGMK", "AAMC", "AGRI", "ALBT", "BRSH", "CNVS", "DRCT",
        "EDBL", "FNGR", "GRTX", "HSCS", "IMNN", "JAGX", "KTRA", "LGMK",
        "MBRX", "NCPL", "OMIC", "PRFX", "QNRX", "RNXT", "SCPX", "TIVC",
        "UAVS", "VRM", "WINT", "XELA", "YGMZ", "ZCMD",
    ]


# ═══════════════════════════════════════════════════════════════════
# Quality Pool Persistence
# ═══════════════════════════════════════════════════════════════════

def save_quality_pool(results: List, path: str = "output/quality_pool.json"):
    """Save Part 1 results for later Part 2 scanning."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'timestamp': datetime.now().isoformat(),
        'count': len(results),
        'stocks': [
            {
                'ticker': r.ticker,
                'name': r.name,
                'sector': r.sector,
                'market_cap_type': r.market_cap_type,
                'quality_score': r.quality_score,
                'ptl_ratio': r.ptl_ratio,
                'bm_ratio': r.bm_ratio,
                'fcf_yield': r.fcf_yield,
                'rationale': r.rationale,
            }
            for r in results
        ]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Quality pool saved: {output_path} ({len(results)} stocks)")


def load_quality_pool(path: str = "output/quality_pool.json") -> List:
    """Load saved Part 1 results."""
    # This is a lightweight loader — full Part1Result objects are reconstructed
    # by re-running screen_fundamentals on the tickers during Part 2.
    with open(path) as f:
        data = json.load(f)
    return [s['ticker'] for s in data.get('stocks', [])]


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline Runner
# ═══════════════════════════════════════════════════════════════════

def run_stage1(tickers: List[str]) -> List:
    """
    Stage 1: Core Financial Fundamentals Screening.
    Returns quality pool (list of Part1Result).
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Core Financial Fundamentals")
    logger.info("=" * 60)
    logger.info(f"Universe: {len(tickers)} stocks")
    logger.info(f"Criteria: Cap($250M/$10B) | B/M≥{P1C.min_bm_ratio} | "
                f"FCF/Y≥{P1C.min_fcf_yield:.0%} | PTL≤{P1C.max_ptl_ratio}x | "
                f"ΔAssets<ΔEarnings | FCF/NI≥{P1C.min_fcf_conversion:.0%}")

    quality_pool = part1_batch(tickers)

    if quality_pool:
        # Print top 15
        logger.info(f"\n🏆 Top Quality Pool ({len(quality_pool)} total):")
        for i, r in enumerate(quality_pool[:15]):
            logger.info(f"  {i+1:2d}. {r.ticker:6s} {r.rationale}")
        if len(quality_pool) > 15:
            logger.info(f"  ... and {len(quality_pool)-15} more")

    return quality_pool


def run_stage2(quality_pool_tickers: List[str]) -> tuple:
    """
    Stage 2: MAGNA 53/10 Momentum Screening on quality pool.
    Re-runs Part 1 to get full Part1Result objects for each ticker,
    then runs Part 2 MAGNA.
    
    Returns: (part1_results, part2_signals, candidates)
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: MAGNA 53/10 Momentum Signals")
    logger.info("=" * 60)
    logger.info(f"Quality pool: {len(quality_pool_tickers)} stocks")
    logger.info(f"MAGNA: M(EPS↑) | A(Sales↑) | G(Gap>4%) | N(Base) | "
                f"5(SI) | 3(Analyst) | Cap<$10B | IPO≤10yr")

    # Re-run Part 1 to get full objects
    quality_pool = []
    logger.info("Refreshing Part 1 data for quality pool...")
    for i, ticker in enumerate(quality_pool_tickers):
        if (i + 1) % 30 == 0:
            logger.info(f"  Refresh: {i+1}/{len(quality_pool_tickers)}")
        try:
            from part1_fundamentals import screen_fundamentals
            p1 = screen_fundamentals(ticker)
            if p1:
                quality_pool.append(p1)
        except Exception:
            pass
        time.sleep(0.12)

    logger.info(f"  {len(quality_pool)}/{len(quality_pool_tickers)} still pass Part 1")

    # Run Part 2 MAGNA
    signals = part2_batch(quality_pool)

    # Combine into candidates
    p1_map = {p.ticker: p for p in quality_pool}
    candidates = []
    for sig in signals:
        p1 = p1_map.get(sig.ticker)
        if p1:
            # Compute composite rank
            composite = p1.quality_score * 0.50 + (sig.magna_score / 10) * 0.35
            if sig.entry_ready:
                composite += 0.15
            candidates.append(VMAACandidate(
                ticker=sig.ticker,
                part1=p1,
                part2=sig,
                composite_rank=round(composite, 4),
                entry_triggered=sig.entry_ready,
            ))

    candidates.sort(key=lambda c: (c.entry_triggered, c.composite_rank), reverse=True)

    # Print signals
    if signals:
        logger.info(f"\n🎯 MAGNA Signals ({len(signals)} total, "
                    f"{len([s for s in signals if s.entry_ready])} entry-ready):")
        for s in signals[:15]:
            entry_mark = "⚡ENTRY" if s.entry_ready else "📊     "
            sig_str = ",".join(s.trigger_signals) if s.trigger_signals else "-"
            logger.info(f"  {entry_mark} {s.ticker:6s} "
                        f"MAGNA={s.magna_score}/10 "
                        f"G={'✓' if s.g_gap_up else '✗'} "
                        f"M={'✓' if s.m_earnings_accel else '✗'} "
                        f"A={'✓' if s.a_sales_accel else '✗'} "
                        f"SI={s.short_interest_score} "
                        f"[{sig_str}]")

    return quality_pool, signals, candidates


def run_risk_and_execute(
    candidates: List[VMAACandidate],
    market,
    broker,
    existing_positions: List,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """
    Stage 3: Risk management → Trade decisions → Execution.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Risk Management & Execution")
    logger.info("=" * 60)

    from config import RC as RiskCfg

    account = broker.get_account()
    portfolio_value = account.net_liquidation
    existing_tickers = [p.symbol for p in existing_positions]

    decisions = []
    executed = []
    skipped = []

    for c in candidates:
        ticker = c.ticker

        # Sector check
        sector_count = sum(
            1 for p in existing_positions
            if _sector_of(p.symbol) == c.part1.sector
        )
        if sector_count >= RiskCfg.max_positions_per_sector:
            skipped.append((ticker, f"Sector limit ({c.part1.sector})"))
            continue

        # Position count check
        if len(existing_positions) >= RiskCfg.max_positions:
            skipped.append((ticker, "Max positions reached"))
            break

        # Correlation check
        if existing_tickers:
            corr = check_correlation(ticker, existing_tickers)
            if corr > RiskCfg.max_correlation:
                skipped.append((ticker, f"Corr {corr:.2f}"))
                continue

        # Generate decision
        decision = generate_trade_decision(c, portfolio_value, existing_tickers, market)
        decisions.append(decision)

        # Execute
        if decision.action in ('BUY', 'BUY_WEAK'):
            result = _execute_decision(decision, broker, account, dry_run)
            if result['executed']:
                executed.append(decision)
                existing_tickers.append(ticker)
            else:
                skipped.append((ticker, result['reason']))

    # Print summary
    logger.info(f"\n📋 Decisions: {len(decisions)}")
    for d in decisions:
        flags = f" ⚠️ {','.join(d.risk_flags)}" if d.risk_flags else ""
        logger.info(f"  {d.action:8s} {d.ticker:6s} {d.quantity:4d}sh "
                    f"@ ${d.entry_price:.2f} stop=${d.stop_loss:.2f} "
                    f"Q={d.confidence_score:.0%}{flags}")

    logger.info(f"\n✅ Executed: {len(executed)} | ⏭️ Skipped: {len(skipped)}")

    return {
        'decisions': decisions,
        'executed_count': len(executed),
        'skipped_count': len(skipped),
        'executed': [d.rationale for d in executed],
        'skipped': [{'ticker': t, 'reason': r} for t, r in skipped],
    }


def _execute_decision(decision, broker, account, dry_run: bool) -> Dict[str, Any]:
    """Execute a single trade decision with safety checks."""
    cost = decision.quantity * decision.entry_price

    if cost > account.buying_power * 0.85:
        return {'executed': False, 'reason': f"Cost ${cost:.0f} > 85% BP"}

    cash_after = account.cash - cost
    if cash_after < account.net_liquidation * 0.15:
        return {'executed': False, 'reason': "Breaks cash reserve"}

    if dry_run:
        return {'executed': True, 'reason': 'dry_run'}
    else:
        # Live execution via Tiger
        try:
            result = broker.place_order(
                symbol=decision.ticker,
                action='BUY',
                quantity=decision.quantity,
                order_type='LMT',
                limit_price=decision.entry_price,
            )
            if result.order_id > 0:
                # Place stop loss
                broker.place_order(
                    symbol=decision.ticker,
                    action='SELL',
                    quantity=decision.quantity,
                    order_type='STP',
                    stop_price=decision.stop_loss,
                )
                return {'executed': True, 'reason': 'live'}
            else:
                return {'executed': False, 'reason': f"Order rejected: {result.status}"}
        except Exception as e:
            return {'executed': False, 'reason': str(e)}


def _sector_of(ticker: str) -> str:
    """Quick sector lookup."""
    try:
        info = yf.Ticker(ticker).info
        return info.get('sector', 'Unknown')
    except Exception:
        return 'Unknown'


# ═══════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_full_pipeline(
    dry_run: bool = True,
    tickers: Optional[List[str]] = None,
    source: str = "sp500",
) -> Dict[str, Any]:
    """
    Run complete VMAA 2.0 two-stage pipeline:
    Part 1 (Quality) → Part 2 (MAGNA) → Risk → Execute
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info(f"VMAA 2.0 PIPELINE — {'DRY RUN' if dry_run else 'LIVE'} — {start_time}")
    logger.info("=" * 70)

    # ── Market Regime ──
    logger.info("\n[Market] Assessing regime...")
    market = get_market_regime()
    logger.info(f"  SPY: ${market.spy_price:.2f} | "
                f">MA50: {'✓' if market.above_ma50 else '✗'} | "
                f"Vol: {market.vol_regime} | "
                f"Scalar: {market.position_scalar}x | "
                f"OK: {'✓' if market.market_ok else '⚠️'}")

    # ── Universe ──
    universe = get_ticker_universe(source=source, custom=tickers)
    logger.info(f"\n[Universe] {len(universe)} stocks from '{source}'")

    # ── Stage 1: Part 1 Fundamentals ──
    quality_pool = run_stage1(universe)

    if not quality_pool:
        logger.info("\n⚠️ No stocks passed Part 1 quality screening.")
        return {
            'status': 'no_candidates',
            'timestamp': str(start_time),
            'market': _market_dict(market),
            'part1_scanned': len(universe),
            'part1_passed': 0,
        }

    # Save quality pool
    save_quality_pool(quality_pool)

    # ── Stage 2: Part 2 MAGNA ──
    quality_tickers = [p.ticker for p in quality_pool]
    _, signals, candidates = run_stage2(quality_tickers)

    # ── Stage 3: Risk + Execute ──
    from broker.tiger_broker import TigerBroker
    broker = TigerBroker()
    existing_positions = broker.get_positions()
    account = broker.get_account()

    logger.info(f"\n[Portfolio] ${account.net_liquidation:,.0f} | "
                f"{len(existing_positions)} positions | "
                f"BP: ${account.buying_power:,.0f}")

    exec_result = run_risk_and_execute(
        candidates, market, broker, existing_positions, dry_run
    )

    # ── Summary ──
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETE — {elapsed:.0f}s")
    logger.info(f"  Scanned:  {len(universe)}")
    logger.info(f"  Part 1:   {len(quality_pool)} passed quality")
    logger.info(f"  Part 2:   {len(signals)} MAGNA signals "
                f"({len([s for s in signals if s.entry_ready])} entry-ready)")
    logger.info(f"  Combined: {len(candidates)} candidates")
    logger.info(f"  Decisions:{len(exec_result['decisions'])}")
    logger.info(f"  Executed: {exec_result['executed_count']}")
    logger.info(f"  Skipped:  {exec_result['skipped_count']}")
    logger.info("=" * 70)

    return {
        'status': 'complete',
        'timestamp': str(start_time),
        'elapsed_seconds': elapsed,
        'market': _market_dict(market),
        'pipeline': {
            'universe_size': len(universe),
            'part1_passed': len(quality_pool),
            'part1_pass_rate': f"{len(quality_pool)/max(len(universe),1)*100:.1f}%",
            'part2_signals': len(signals),
            'part2_entry_ready': len([s for s in signals if s.entry_ready]),
            'candidates': len(candidates),
            'decisions': len(exec_result['decisions']),
            'executed': exec_result['executed_count'],
            'skipped': exec_result['skipped_count'],
        },
        'quality_pool_top20': [
            {
                'ticker': p.ticker,
                'quality_score': p.quality_score,
                'rationale': p.rationale,
            } for p in quality_pool[:20]
        ],
        'signals': [
            {
                'ticker': s.ticker,
                'magna_score': s.magna_score,
                'entry_ready': s.entry_ready,
                'triggers': s.trigger_signals,
            } for s in signals[:20]
        ],
        'decisions': [
            {
                'ticker': d.ticker,
                'action': d.action,
                'quantity': d.quantity,
                'entry': d.entry_price,
                'stop_loss': d.stop_loss,
                'confidence': d.confidence_score,
                'risk_flags': d.risk_flags,
                'rationale': d.rationale,
            } for d in exec_result['decisions']
        ],
        'execution': exec_result,
    }


# ═══════════════════════════════════════════════════════════════════
# Status Dashboard
# ═══════════════════════════════════════════════════════════════════

def show_status() -> Dict[str, Any]:
    """Portfolio status + risk dashboard."""
    from broker.tiger_broker import TigerBroker
    market = get_market_regime()
    broker = TigerBroker()
    account = broker.get_account()
    positions = broker.get_positions()
    orders = broker.get_orders(limit=10)

    return {
        'timestamp': str(datetime.now()),
        'market': _market_dict(market),
        'account': {
            'id': account.account_id,
            'value': account.net_liquidation,
            'cash': account.cash,
            'buying_power': account.buying_power,
            'invested': account.gross_position_value,
            'unrealized_pnl': account.unrealized_pnl,
            'realized_pnl': account.realized_pnl,
        },
        'positions': [
            {
                'ticker': p.symbol,
                'quantity': p.quantity,
                'avg_cost': p.average_cost,
                'current': p.market_price,
                'market_value': p.market_value,
                'pnl': p.unrealized_pnl,
                'pnl_pct': p.unrealized_pnl_pct,
            } for p in positions
        ],
        'active_orders': [
            {
                'id': o.order_id,
                'ticker': o.symbol,
                'action': o.action,
                'quantity': o.quantity,
                'price': o.limit_price,
                'status': o.status,
            } for o in orders if 'CANCEL' not in str(o.status).upper()
        ],
    }


def _market_dict(market) -> Dict[str, Any]:
    return {
        'spy_price': market.spy_price,
        'above_ma50': market.above_ma50,
        'vol_regime': market.vol_regime,
        'volatility_20d': market.volatility_20d,
        'market_ok': market.market_ok,
        'position_scalar': market.position_scalar,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import yfinance as yf  # noqa: F811

    parser = argparse.ArgumentParser(
        description="VMAA 2.0 — Two-Stage Value + Momentum Pipeline"
    )
    parser.add_argument('--scan-part1', action='store_true',
                        help='Run only Stage 1: Core Fundamentals')
    parser.add_argument('--scan-part2', action='store_true',
                        help='Run only Stage 2: MAGNA on saved quality pool')
    parser.add_argument('--full-scan', action='store_true',
                        help='Run complete two-stage pipeline')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Simulate execution (default)')
    parser.add_argument('--live', action='store_true',
                        help='LIVE trading mode')
    parser.add_argument('--status', action='store_true',
                        help='Show portfolio status')
    parser.add_argument('--tickers', nargs='*',
                        help='Specific tickers (comma or space separated)')
    parser.add_argument('--source', default='sp500',
                        help='Universe source: sp500, russell2000, custom')
    parser.add_argument('--output', default='output/pipeline_result.json',
                        help='Output file path')

    args = parser.parse_args()

    # Parse tickers
    tickers = None
    if args.tickers:
        tickers = []
        for t in args.tickers:
            for s in t.split(','):
                s = s.strip()
                if s:
                    tickers.append(s.upper())

    if args.status:
        result = show_status()
        print(f"\n📊 Portfolio: ${result['account']['value']:,.0f} | "
              f"Positions: {len(result['positions'])} | "
              f"P&L: ${result['account']['unrealized_pnl']:,.0f}")
        for p in result['positions']:
            pnl_sign = "🟢" if p['pnl'] >= 0 else "🔴"
            print(f"  {pnl_sign} {p['ticker']:6s} {p['quantity']:4d}sh "
                  f"@ ${p['current']:.2f} P&L: ${p['pnl']:,.0f} ({p['pnl_pct']:+.1%})")

    elif args.scan_part1:
        print("Stage 1: Core Financial Fundamentals Screening")
        universe = get_ticker_universe(source=args.source, custom=tickers)
        quality_pool = run_stage1(universe)
        save_quality_pool(quality_pool)
        print(f"\n✅ Part 1 complete: {len(quality_pool)} stocks passed quality screening")

    elif args.scan_part2:
        print("Stage 2: MAGNA 53/10 Momentum Screening")
        try:
            pool_tickers = load_quality_pool()
            if not pool_tickers:
                print("❌ No quality pool found. Run --scan-part1 first.")
                sys.exit(1)
            quality_pool, signals, candidates = run_stage2(pool_tickers)

            from broker.tiger_broker import TigerBroker
            market = get_market_regime()
            broker = TigerBroker()
            existing = broker.get_positions()
            exec_result = run_risk_and_execute(
                candidates, market, broker, existing,
                dry_run=not args.live
            )
        except FileNotFoundError:
            print("❌ No quality pool file found. Run --scan-part1 first.")
            sys.exit(1)
        result = {
            'status': 'complete',
            'part2_signals': len(signals),
            'candidates': len(candidates),
            'execution': exec_result,
        }

    elif args.full_scan:
        result = run_full_pipeline(
            dry_run=not args.live,
            tickers=tickers,
            source=args.source,
        )
    else:
        parser.print_help()
        sys.exit(1)

    # Save output
    if 'result' in locals():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n📁 Results saved to {output_path}")

        if args.full_scan or args.scan_part2:
            p = result.get('pipeline', {})
            print(f"\n🎯 Pipeline: {p.get('universe_size', '?')} scanned → "
                  f"{p.get('part1_passed', '?')} quality → "
                  f"{p.get('part2_signals', '?')} signals → "
                  f"{p.get('candidates', '?')} candidates → "
                  f"{p.get('executed', '?')} executed")
