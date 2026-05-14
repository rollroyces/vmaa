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
  python3 pipeline.py --full-scan --source combined  # S&P500+R2K+NASDAQ100
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

import yfinance as yf

from vmaa.models import VMAACandidate, TradeDecision
from vmaa.config import PC, P1C, P2C, RC as RiskCfg

from vmaa.part1_fundamentals import batch_screen as part1_batch
from vmaa.part2_magna import batch_screen_magna as part2_batch
from part2b_vcp import (
    batch_vcp_filter, apply_vcp_to_stop, apply_vcp_to_confidence,
    apply_vcp_to_position_size, get_vcp_entry_quality,
)
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
    Sources: sp500, russell2000, nasdaq100, combined, custom
    """
    if custom and len(custom) > 0:
        return [t.strip().upper() for t in custom if t.strip()]

    import pandas as pd

    def _fetch_sp500():
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
            logger.warning(f"Could not fetch S&P 500 list: {e}")
            return []

    def _fetch_russell2000():
        """Fetch Russell 2000 constituents from iShares IWM ETF holdings CSV."""
        try:
            import requests, csv, io as csvi
            url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM"
            headers_w = {
                "User-Agent": "Mozilla/5.0 (X11; Linux arm64) AppleWebKit/537.36",
                "Accept": "text/csv"
            }
            resp = requests.get(url, headers=headers_w, timeout=20)
            resp.raise_for_status()
            text = resp.text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            # Find header row
            start_idx = 0
            for i, line in enumerate(lines):
                if 'Ticker' in line and ('Name' in line or 'Market Value' in line):
                    start_idx = i
                    break
            data_section = '\n'.join(lines[start_idx:])
            reader = csv.reader(csvi.StringIO(data_section))
            tickers = []
            header = next(reader, None)
            if header:
                ticker_col = next((i for i, col in enumerate(header) if col and 'Ticker' in col), 0)
                for row in reader:
                    if not row or len(row) <= ticker_col:
                        continue
                    ticker = row[ticker_col].strip()
                    if (not ticker or ticker == '-' or len(ticker) < 2 or
                        ticker.startswith('The content') or
                        ticker.startswith('©') or
                        ticker.startswith('Holdings subject') or
                        'BlackRock' in ticker):
                        break
                    ticker = ticker.replace('.', '-')
                    tickers.append(ticker)
            tickers = sorted(set(tickers))
            logger.info(f"Loaded {len(tickers)} Russell 2000 tickers from iShares IWM")
            return tickers
        except Exception as e:
            logger.warning(f"Could not fetch Russell 2000: {e}")
            return []

    def _fetch_nasdaq100():
        """Fetch NASDAQ-100 constituents from Wikipedia."""
        try:
            import requests, io as ioi
            headers_w = {'User-Agent': 'Mozilla/5.0 (X11; Linux arm64) AppleWebKit/537.36'}
            resp = requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', headers=headers_w, timeout=15)
            tables = pd.read_html(ioi.StringIO(resp.text))
            # Table[5] = constituents: Ticker, Company, ICB Industry, ICB Subsector
            df = tables[5]
            tickers = df.iloc[:, 0].str.replace('.', '-', regex=False).tolist()
            tickers = [t.strip().upper() for t in tickers if str(t).strip()]
            logger.info(f"Loaded {len(tickers)} NASDAQ 100 tickers from Wikipedia")
            return tickers
        except Exception as e:
            logger.warning(f"Could not fetch NASDAQ 100: {e}")
            return []

    if source == "sp500":
        result = _fetch_sp500()
        if result:
            return result

    if source == "russell2000":
        result = _fetch_russell2000()
        if result:
            return result

    if source == "nasdaq100":
        result = _fetch_nasdaq100()
        if result:
            return result

    if source == "combined":
        sp500 = set(_fetch_sp500())
        r2k = set(_fetch_russell2000())
        nasdaq = set(_fetch_nasdaq100())
        combined = list(sp500 | r2k | nasdaq)
        combined = [t.strip().upper() for t in combined if t.strip()]
        logger.info(f"Combined universe: {len(sp500)} S&P500 + {len(r2k)} R2K + {len(nasdaq)} NASDAQ100 = {len(combined)} unique tickers")
        return combined

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

def run_stage1(tickers: List[str], workers: int = 15) -> List:
    """
    Stage 1: Core Financial Fundamentals Screening.
    Returns quality pool (list of Part1Result).
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Core Financial Fundamentals")
    logger.info("=" * 60)
    logger.info(f"Universe: {len(tickers)} stocks (workers={workers})")
    logger.info(f"Criteria: Cap($250M/$10B) | B/M≥{P1C.min_bm_ratio} | "
                f"FCF/Y≥{P1C.min_fcf_yield:.0%} | PTL≤{P1C.max_ptl_ratio}x | "
                f"ΔAssets<ΔEarnings | FCF/NI≥{P1C.min_fcf_conversion:.0%}")

    quality_pool = part1_batch(tickers, max_workers=workers)

    if quality_pool:
        # Print top 15
        logger.info(f"\n🏆 Top Quality Pool ({len(quality_pool)} total):")
        for i, r in enumerate(quality_pool[:15]):
            logger.info(f"  {i+1:2d}. {r.ticker:6s} {r.rationale}")
        if len(quality_pool) > 15:
            logger.info(f"  ... and {len(quality_pool)-15} more")

    return quality_pool


def run_stage2(quality_pool: List[Part1Result], workers: int = 12) -> tuple:
    """
    Stage 2: MAGNA 53/10 Momentum Screening on quality pool.
    Receives Part1Result objects directly from Stage 1.

    Returns: (part1_results, part2_signals, candidates)
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: MAGNA 53/10 Momentum Signals")
    logger.info("=" * 60)
    logger.info(f"Quality pool: {len(quality_pool)} stocks (workers={workers})")
    logger.info(f"MAGNA: M(EPS↑) | A(Sales↑) | G(Gap>4%) | N(Base) | "
                f"5(SI) | 3(Analyst) | Cap<$10B | IPO≤10yr")

    # Run Part 2 MAGNA using Part1Result objects directly (no re-fetch)
    signals = part2_batch(quality_pool, max_workers=workers)

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


def run_vcp_filter(
    candidates: List[VMAACandidate],
    part1_map: dict = None,
) -> List[VMAACandidate]:
    """
    Stage 2.5: VCP Precision Filter.

    Runs Volatility Contraction Pattern analysis on entry-ready candidates.
    VCP enhances — it never blocks entries. Non-VCP candidates proceed
    with standard WIDE_STOP parameters.

    Returns candidates with VCP metadata attached for Risk stage use.
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2.5: VCP Precision Filter (Minervini)")
    logger.info("=" * 60)

    entry_ready = [c for c in candidates if c.entry_triggered]
    logger.info(f"Entry-ready candidates: {len(entry_ready)}")

    if not entry_ready:
        logger.info("No entry-ready candidates — skipping VCP")
        return candidates

    vcp_results = batch_vcp_filter(entry_ready)

    vcp_confirmed = 0
    cache = {}  # Store VCP results on candidate objects via _vcp attribute

    for c in candidates:
        ticker = c.ticker
        vcp = vcp_results.get(ticker) if c.entry_triggered else None
        c._vcp = vcp  # Attach VCP result for Risk stage

        if vcp and vcp.vcp_detected:
            vcp_confirmed += 1
            logger.info(
                f"  ✅ {ticker:6s} VCP Q={vcp.vcp_quality:.0%} "
                f"waves={vcp.contractions} "
                f"pivot_ATR={vcp.pivot_volatility_pct:.1%} "
                f"dry_up={vcp.volume_dry_up_ratio:.0%} "
                f"[{get_vcp_entry_quality(vcp)}]"
            )
        elif vcp:
            logger.debug(
                f"  ❌ {ticker:6s} VCP Q={vcp.vcp_quality:.0%} "
                f"{vcp.rationale[:60]}"
            )
        else:
            logger.debug(f"  ⬜ {ticker:6s} no VCP data")

    logger.info(
        f"\nVCP Summary: {vcp_confirmed}/{len(entry_ready)} confirmed, "
        f"{len(entry_ready) - vcp_confirmed} proceed with standard params"
    )

    return candidates


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

    if broker is None:
        logger.info("  No broker available — skipping execution")
        return {
            'decisions': [],
            'executed_count': 0,
            'skipped_count': 0,
            'executed': [],
            'skipped': [{'ticker': c.ticker, 'reason': 'no_broker'} for c in candidates],
        }

    account = broker.get_account()
    portfolio_value = account.net_liquidation
    existing_tickers = [p.symbol for p in existing_positions]

    # Build sector cache for existing positions (avoid redundant yfinance calls)
    existing_sectors = {}
    for p in existing_positions:
        try:
            existing_sectors[p.symbol] = yf.Ticker(p.symbol).info.get('sector', 'Unknown')
        except Exception:
            existing_sectors[p.symbol] = 'Unknown'

    decisions = []
    executed = []
    skipped = []
    executed_count = 0

    for c in candidates:
        ticker = c.ticker

        # Sector check (uses candidate's sector + existing position sector cache)
        sector_count = sum(
            1 for p in existing_positions
            if existing_sectors.get(p.symbol, 'Unknown') == c.part1.sector
        )
        if sector_count >= RiskCfg.max_positions_per_sector:
            skipped.append((ticker, f"Sector limit ({c.part1.sector})"))
            continue

        # Position count check — track executed count independently
        if executed_count >= RiskCfg.max_positions:
            skipped.append((ticker, "Max positions reached"))
            break

        # Correlation check
        if existing_tickers:
            try:
                corr = check_correlation(ticker, existing_tickers)
            except Exception as e:
                logger.warning(f"  Correlation check failed for {ticker}: {e}, skipping check")
                corr = 0.0
            if corr > RiskCfg.max_correlation:
                skipped.append((ticker, f"Corr {corr:.2f}"))
                continue

        # Generate decision
        decision = generate_trade_decision(c, portfolio_value, existing_tickers, market)
        
        # Attach extra metadata for pipeline output
        decision._sentiment_label = c.sentiment.sentiment_label if c.sentiment else ''
        decision._sentiment_score = c.sentiment.composite_score if c.sentiment else 0
        decision._vcp_entry_quality = get_vcp_entry_quality(c._vcp) if hasattr(c, '_vcp') and c._vcp else 'NO_VCP'
        decision._quality_score = c.part1.quality_score
        decision._magna_score = c.part2.magna_score
        
        decisions.append(decision)

        # Execute
        if decision.action in ('BUY', 'BUY_WEAK'):
            result = _execute_decision(decision, broker, account, dry_run)
            if result.get('executed') or result.get('would_execute'):
                executed.append(decision)
                executed_count += 1
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
        return {'executed': False, 'would_execute': False,
                'reason': f"Cost ${cost:.0f} > 85% BP"}

    cash_after = account.cash - cost
    if cash_after < account.net_liquidation * 0.15:
        return {'executed': False, 'would_execute': False,
                'reason': "Breaks cash reserve"}

    if dry_run:
        return {'executed': True, 'would_execute': True, 'reason': 'dry_run'}

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
            # WARNING: Partial-fill risk. If the limit order is only partially
            # filled, placing a stop loss for the FULL quantity could leave
            # unfilled shares unprotected (if partially filled now, stop triggers
            # before the rest fills) or cause overselling. Query the actual
            # filled quantity and only stop-protect what was filled.
            filled_qty = getattr(result, 'filled_quantity', None) or getattr(result, 'quantity', None) or decision.quantity
            if filled_qty > 0:
                broker.place_order(
                    symbol=decision.ticker,
                    action='SELL',
                    quantity=filled_qty,
                    order_type='STP',
                    stop_price=decision.stop_loss,
                )
            return {'executed': True, 'would_execute': True, 'reason': 'live'}
        else:
            return {'executed': False, 'would_execute': True,
                    'reason': f"Order rejected: {result.status}"}
    except Exception as e:
        return {'executed': False, 'would_execute': True, 'reason': str(e)}


# ═══════════════════════════════════════════════════════════════════
# Stage 3: Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════

def run_sentiment(candidates: List[VMAACandidate]) -> List[VMAACandidate]:
    """
    Stage 3: Multi-source sentiment analysis on candidates.
    Attaches SentimentResult to each VMAACandidate.
    """
    from part3_sentiment import batch_sentiment

    if not candidates:
        return candidates

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: Sentiment Analysis")
    logger.info("=" * 60)
    logger.info(f"Analyzing sentiment for {len(candidates)} candidates")
    logger.info("Sources: Analyst | News (VADER) | Social | Technical | Insider")

    tickers = [c.ticker for c in candidates]
    sentiment_results = batch_sentiment(tickers, delay=0.15)

    for c in candidates:
        c.sentiment = sentiment_results.get(c.ticker)

    # Print summary
    if candidates:
        logger.info(f"\n😐 Sentiment Summary:")
        for c in candidates[:15]:
            s = c.sentiment
            if s:
                sig_str = f" [{','.join(s.signals[:2])}]" if s.signals else ""
                logger.info(
                    f"  {s.sentiment_label:20s} {c.ticker:6s} "
                    f"comp={s.composite_score:+.2f} "
                    f"A={s.analyst_score:+.2f} "
                    f"N={s.news_score:+.2f} "
                    f"S={s.social_score:+.2f} "
                    f"T={s.technical_score:+.2f}"
                    f"{sig_str}"
                )

        # Count sentiment labels
        labels = {}
        for c in candidates:
            if c.sentiment:
                lbl = c.sentiment.sentiment_label
                labels[lbl] = labels.get(lbl, 0) + 1
        label_summary = ", ".join(f"{k}:{v}" for k, v in sorted(labels.items()))
        logger.info(f"  Distribution: {label_summary}")

        # Contrarian opportunities
        contrarians = [c for c in candidates
                       if c.sentiment and "CONTRARIAN_BUY" in c.sentiment.signals]
        if contrarians:
            logger.info(f"\n  🔥 Contrarian Opportunities ({len(contrarians)}):")
            for c in contrarians[:5]:
                logger.info(f"    {c.ticker:6s} Q={c.part1.quality_score:.0%} "
                            f"MAGNA={c.part2.magna_score}/10 "
                            f"Sent={c.sentiment.composite_score:+.2f}")

    return candidates


# ═══════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_full_pipeline(
    dry_run: bool = True,
    tickers: Optional[List[str]] = None,
    source: str = "sp500",
    workers: int = 15,
) -> Dict[str, Any]:
    """
    Run complete VMAA 2.0 two-stage pipeline:
    Part 1 (Quality) → Part 2 (MAGNA) → Risk → Execute
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info(f"VMAA 2.0 PIPELINE — {'DRY RUN' if dry_run else 'LIVE'} — {start_time}")
    logger.info("=" * 70)

    # ── Data Source Health Check ──
    try:
        from vmaa.data.hybrid import yfinance_available
        if not yfinance_available():
            logger.warning("⚠️  yfinance rate-limited/unavailable — fallbacks: SEC EDGAR + Finnhub + Tiger")
        else:
            logger.info("✅  yfinance available")
    except Exception:
        pass

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
    quality_pool = run_stage1(universe, workers=workers)

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
    part2_workers = max(8, workers - 3)  # slightly fewer for Part 2
    _, signals, candidates = run_stage2(quality_pool, workers=part2_workers)

    # ── Stage 2.5: VCP Precision Filter ──
    candidates = run_vcp_filter(candidates)

    # ── Stage 3: Sentiment Analysis ──
    candidates = run_sentiment(candidates)

    # ── Stage 4: Risk + Execute ──
    try:
        from broker.tiger_broker import TigerBroker
        broker = TigerBroker()
        existing_positions = broker.get_positions()
        account = broker.get_account()
    except ImportError:
        logger.warning("Tiger Trade SDK not installed — running in analysis-only mode")
        broker = None
        existing_positions = []
        account = None

    if account:
        logger.info(f"\n[Portfolio] ${account.net_liquidation:,.0f} | "
                    f"{len(existing_positions)} positions | "
                    f"BP: ${account.buying_power:,.0f}")
    else:
        logger.info(f"\n[Portfolio] Analysis-only mode (no broker)")

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
        'timestamp': datetime.now().isoformat(),
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
                'trigger_signals': s.trigger_signals,
                'm_earnings_accel': s.m_earnings_accel,
                'a_sales_accel': s.a_sales_accel,
                'g_gap_up': s.g_gap_up,
                'n_neglect_base': s.n_neglect_base,
                'short_interest_score': s.short_interest_score,
                'analyst_upgrades': s.analyst_upgrades,
                'eps_acceleration': round(s.eps_acceleration, 4),
                'revenue_acceleration': round(s.revenue_acceleration, 4),
                'gap_pct': round(s.gap_pct, 4),
            }
            for s in signals
        ],
        'candidates': [
            {
                'ticker': c.ticker,
                'quality_score': c.quality_score,
                'magna_score': c.magna_score,
                'composite_rank': c.composite_rank,
                'entry_triggered': c.entry_triggered,
                'vcp_detected': getattr(c._vcp, 'vcp_detected', False) if hasattr(c, '_vcp') and c._vcp else False,
                'vcp_quality': c._vcp.vcp_quality if hasattr(c, '_vcp') and c._vcp else 0,
                'sentiment_label': c.sentiment.sentiment_label if c.sentiment else '',
                'sentiment_score': c.sentiment.composite_score if c.sentiment else 0,
            }
            for c in candidates
        ],
        'decisions': [
            {
                'ticker': d.ticker,
                'action': d.action,
                'quantity': d.quantity,
                'entry_price': d.entry_price,
                'stop_loss': d.stop_loss,
                'stop_type': d.stop_type,
                'take_profits': d.take_profits,
                'trailing_stop_pct': d.trailing_stop_pct,
                'trailing_activate_pct': d.trailing_activate_pct,
                'position_pct': d.position_pct,
                'risk_amount': d.risk_amount,
                'reward_ratio': d.reward_ratio,
                'confidence_score': d.confidence_score,
                'risk_flags': d.risk_flags,
                'entry_method': d.entry_method,
                'sentiment_label': getattr(d, '_sentiment_label', ''),
                'sentiment_score': getattr(d, '_sentiment_score', 0),
                'vcp_entry_quality': getattr(d, '_vcp_entry_quality', 'NO_VCP'),
                'rationale': d.rationale,
                'part1_quality_score': getattr(d, '_quality_score', 0),
                'part2_magna_score': getattr(d, '_magna_score', 0),
            }
            for d in exec_result['decisions']
        ],
        'execution': {
            'decisions': exec_result.get('decisions', []),
            'executed_count': exec_result.get('executed_count', 0),
            'skipped_count': exec_result.get('skipped_count', 0),
            'executed': exec_result.get('executed', []),
            'skipped': exec_result.get('skipped', []),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Status Dashboard
# ═══════════════════════════════════════════════════════════════════

def show_status() -> Dict[str, Any]:
    """Portfolio status + risk dashboard."""
    try:
        from broker.tiger_broker import TigerBroker
        broker = TigerBroker()
    except ImportError:
        logger.warning("Tiger Trade SDK not installed — status unavailable")
        return {
            'timestamp': str(datetime.now()),
            'market': {},
            'account': {'error': 'broker_unavailable'},
            'positions': [],
            'active_orders': [],
        }

    market = get_market_regime()
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
                        help='Universe source: sp500, russell2000, nasdaq100, combined, custom')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel workers for yfinance I/O (default: 15)')
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
        quality_pool = run_stage1(universe, workers=args.workers)
        save_quality_pool(quality_pool)
        print(f"\n✅ Part 1 complete: {len(quality_pool)} stocks passed quality screening")

    elif args.scan_part2:
        print("Stage 2: MAGNA 53/10 Momentum Screening")
        try:
            pool_tickers = load_quality_pool()
            if not pool_tickers:
                print("❌ No quality pool found. Run --scan-part1 first.")
                sys.exit(1)

            # Reconstruct Part1Result objects from saved quality pool
            from part1_fundamentals import screen_fundamentals
            part1_results = []
            print(f"Refreshing Part 1 data for {len(pool_tickers)} quality-pool stocks...")
            for ticker in pool_tickers:
                try:
                    p1 = screen_fundamentals(ticker)
                    if p1:
                        part1_results.append(p1)
                except Exception:
                    pass
            print(f"  {len(part1_results)}/{len(pool_tickers)} still pass Part 1")

            quality_pool, signals, candidates = run_stage2(part1_results, workers=args.workers)

            # Stage 2.5: VCP Filter
            candidates = run_vcp_filter(candidates)

            try:
                from broker.tiger_broker import TigerBroker
                broker = TigerBroker()
            except ImportError:
                logger.warning("Tiger Trade SDK not installed — analysis-only mode")
                broker = None

            market = get_market_regime()
            if broker:
                existing = broker.get_positions()
            else:
                existing = []
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
            workers=args.workers,
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
