#!/usr/bin/env python3
"""
VMAA Engine — Demo & Quickstart
================================
Run the full VMAA pipeline and see what each engine contributes.

Usage:
  python3 engine/demo.py                  # Quick scan (10 tickers)
  python3 engine/demo.py --full           # Full pipeline (50 tickers)
  python3 engine/demo.py --tickers AAPL,MSFT,GOOGL  # Custom tickers
  python3 engine/demo.py --telegram       # Telegram-friendly output format
  python3 engine/demo.py --status         # Engine status only
  python3 engine/demo.py --chip AAPL      # Chip analysis on single ticker
  python3 engine/demo.py --technical AAPL,MSFT  # Technical analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure VMAA root is importable
_vmaa_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_vmaa_root))

from engine import VMAAEngine, get_engine, get_engine_config
from engine.config import EngineConfig

# ── Setup ──

def setup_logging(level: str = "WARNING"):
    """Minimal logging — keep output clean for demos."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.WARNING),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ═══════════════════════════════════════════════════════════════════
# Formatters
# ═══════════════════════════════════════════════════════════════════

def format_telegram(result: Dict[str, Any]) -> str:
    """Format pipeline results as a Telegram-friendly message."""
    meta = result.get("pipeline_meta", {})
    market = result.get("market", {})
    summary = result.get("pipeline_summary", {})
    candidates = result.get("candidates", [])
    decisions = result.get("decisions", [])
    risk = result.get("risk", {})
    engines = result.get("engines", {})

    lines = [
        f"🦾 <b>VMAA v3 Pipeline</b> — {meta.get('timestamp', 'N/A')[:16].replace('T', ' ')}",
        "",
        f"📊 <b>Market:</b> SPY {market.get('spy', 'N/A')} | Regime: {market.get('regime', 'UNKNOWN')}",
        f"   VIX Proxy: {market.get('vix_proxy', 0):.2%} | Above 50MA: {market.get('above_50ma', False)}",
        "",
        f"📋 <b>Pipeline Summary:</b>",
        f"   Scanned: {summary.get('scanned', 0)}",
        f"   Quality Passed: {summary.get('quality', 0)}",
        f"   MAGNA Signals: {summary.get('signals', 0)}",
        f"   Entry Ready: {summary.get('entry_ready', 0)}",
    ]

    # Engine contributions
    lines.append("")
    lines.append("🔧 <b>Engine Contributions:</b>")

    tech = engines.get("technical", {})
    lines.append(f"   📐 Technical: {'✅' if tech.get('enabled') else '❌'} "
                 f"({tech.get('stocks_analyzed', 0)} stocks)")

    chip = engines.get("chip", {})
    lines.append(f"   🎯 Chip: {'✅' if chip.get('enabled') else '❌'} "
                 f"({chip.get('stocks_analyzed', 0)} stocks)")

    earnings = engines.get("earnings", {})
    lines.append(f"   💰 Earnings: {'✅' if earnings.get('enabled') else '❌'} "
                 f"({earnings.get('stocks_analyzed', 0)} stocks)")

    # Scoring weights
    cfg = get_engine_config()
    lines.append(f"\n⚖️ <b>Weights:</b> Q:{cfg.weight_quality:.0%} M:{cfg.weight_momentum:.0%} "
                 f"T:{cfg.weight_technical:.0%} E:{cfg.weight_earnings:.0%} C:{cfg.weight_chip:.0%}")

    # Top candidates
    if candidates:
        lines.append("")
        lines.append("🏆 <b>Top Candidates:</b>")
        for i, c in enumerate(candidates[:10], 1):
            stars = "⭐" if c["composite_score"] >= 0.75 else "🔹" if c["composite_score"] >= 0.55 else "▪️"
            lines.append(
                f"  {i:2d}. {stars} <code>{c['ticker']:6s}</code> "
                f"comp={c['composite_score']:.3f} "
                f"Q={c['quality_score']:.2f} M={c['magna_score']} "
                f"T={c['technical_score']:.2f} C={c['chip_score']:.2f}"
            )

    # Decisions
    if decisions:
        lines.append("")
        lines.append("💡 <b>Decisions:</b>")
        action_emoji = {
            "STRONG_BUY": "🟢",
            "BUY": "🟢",
            "ENTRY_READY": "🟡",
            "WATCH": "⚪",
            "HOLD_MARKET": "🔴",
            "HOLD_CIRCUIT": "🔴",
        }
        for d in decisions[:8]:
            emoji = action_emoji.get(d["action"], "⚪")
            lines.append(f"  {emoji} <code>{d['ticker']:6s}</code> {d['action']:14s} ${d.get('price', 0):.2f}")
            if len(lines[-1]) < 60:
                lines[-1] += f"  {d.get('reason', '')}"

    # Risk
    if risk.get("enabled"):
        lines.append("")
        lines.append(f"🛡️ <b>Risk:</b> Score={risk.get('risk_score', 'N/A')} | "
                     f"Level={risk.get('risk_level', 'UNKNOWN')} | "
                     f"CB={risk.get('circuit_breakers', '?')}")

    # Stage times
    stages = meta.get("stages", {})
    times = meta.get("stage_times", {})
    if times:
        lines.append(f"\n⏱️ Total: {meta.get('total_elapsed_seconds', 0):.1f}s")

    return "\n".join(lines)


def format_console(result: Dict[str, Any]) -> str:
    """Format results for console output."""
    return json.dumps(result, indent=2, default=str, ensure_ascii=False)


def format_engine_status(engine: VMAAEngine) -> str:
    """Format engine status for display."""
    s = engine.status()
    lines = [
        f"\n{'='*60}",
        "  VMAA Engine v3 — Integration Layer Status",
        f"{'='*60}",
        f"  Timestamp: {s['timestamp']}",
        f"  Mode: {s['mode']}",
        "",
        "  Engine Status:",
    ]

    for name, status in s["engines"].items():
        icon = "✅" if status == "loaded" else "⚠️" if "error" in status else "❌"
        lines.append(f"    {icon} {name:15s}: {status}")

    lines.append(f"\n  Scoring Weights:")
    for k, v in s["config"]["weights"].items():
        lines.append(f"    {k:12s}: {v:.0%}")

    lines.append(f"{'='*60}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VMAA Engine Demo — Run the unified pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--full", action="store_true",
                        help="Run full pipeline (50 tickers, all engines)")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers to scan")
    parser.add_argument("--telegram", action="store_true",
                        help="Output Telegram-friendly format")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON")
    parser.add_argument("--status", action="store_true",
                        help="Show engine status only")
    parser.add_argument("--chip", type=str, default=None,
                        help="Run chip analysis on a single ticker")
    parser.add_argument("--technical", type=str, default=None,
                        help="Run technical analysis on comma-separated tickers")
    parser.add_argument("--screen", type=str, default=None,
                        help="Run screening on comma-separated tickers")
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed logging")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    log_level = "INFO" if args.verbose else "WARNING"
    setup_logging(log_level)

    engine = get_engine()

    # ── Status Only ──
    if args.status:
        print(format_engine_status(engine))
        return

    # ── Chip Analysis ──
    if args.chip:
        ticker = args.chip.strip().upper()
        print(f"🎯 Chip Analysis: {ticker}")
        try:
            report = engine.chip.analyze(ticker) if engine.chip else None
            if report:
                import json as _json
                print(_json.dumps({
                    "ticker": report.ticker,
                    "price": report.current_price,
                    "cost_basis": report.cost_basis,
                    "support_resistance": report.support_resistance,
                    "concentration": report.concentration,
                }, indent=2, default=str))
            else:
                print("❌ Chip engine not loaded")
        except Exception as e:
            print(f"❌ Error: {e}")
        return

    # ── Technical Analysis ──
    if args.technical:
        tickers = [t.strip().upper() for t in args.technical.split(",")]
        print(f"📐 Technical Analysis: {len(tickers)} tickers")
        results = engine.analyze_technicals(tickers)
        for r in results:
            print(f"\n  {r.get('ticker', 'unknown')}:")
            signals = r.get("signals", {})
            print(f"    Signal: {signals.get('signal', 'N/A')} "
                  f"(strength: {signals.get('strength', 0):.2f})")
            indicators = r.get("indicators", {})
            if indicators:
                print(f"    RSI: {indicators.get('rsi_14', 'N/A')} | "
                      f"BB%b: {indicators.get('bb_pct_b', 'N/A')}")
        return

    # ── Screening Only ──
    if args.screen:
        tickers = [t.strip().upper() for t in args.screen.split(",")]
        print(f"🔍 Screening: {len(tickers)} tickers")
        result = engine.screen(tickers)
        print(f"  Passed: {result.get('passed', 0)}/{result.get('universe', 0)}")
        for c in result.get("candidates", [])[:10]:
            print(f"    {c['ticker']:6s}  {c['score']:.4f}")
        return

    # ── Pipeline Run ──
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        mode = "quick" if not args.full else "full"
    else:
        tickers = None
        mode = "full" if args.full else "quick"

    print(f"\n🦾 VMAA v3 Pipeline — {mode.upper()} mode")
    print(f"   Tickers: {'custom (' + str(len(tickers or [])) + ')' if tickers else 'default universe'}")
    print()

    try:
        if mode == "quick":
            result = engine.quick_scan(tickers=tickers)
        else:
            result = engine.full_pipeline(tickers=tickers)

        # Output
        if args.telegram:
            output = format_telegram(result)
        elif args.json:
            output = format_console(result)
        else:
            # Default: combined format
            output = format_telegram(result)
            print(output)
            print(f"\n{'─'*60}")
            print(f"JSON saved to engine/data/pipeline_result.json")

        # Save to file
        output_path = args.output or str(
            Path(__file__).resolve().parent / "data" / "pipeline_result.json"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        print(f"📁 Full results: {output_path}")

        if not args.telegram and not args.json:
            # Show the telegram-formatted version too
            pass  # Already printed above

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
