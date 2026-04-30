#!/usr/bin/env python3
"""VMAA Pipeline Report — Telegram-friendly summary from pipeline_result.json."""
import json
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).parent / 'output' / 'pipeline_result.json'

def generate_report(data: dict) -> str:
    lines = []
    def add(s=''):
        lines.append(s)

    market = data.get('market', {})
    pipeline = data.get('pipeline', {})
    decisions = data.get('decisions', [])

    # Header
    add('📊 VMAA Pipeline Report')
    add(f"🕐 {data.get('timestamp', '?')[:19]}")
    add(f"Market: {'✅ BULL' if market.get('market_ok') else '⚠️ CAUTION'} | "
        f"Vol: {market.get('vol_regime', '?')} | "
        f"SPY: ${market.get('spy_price', 0):.2f}")
    add(f"Candidates: {pipeline.get('candidates_found', 0)} | "
        f"Decisions: {pipeline.get('decisions_made', 0)} | "
        f"Executed: {pipeline.get('executed', 0)} | "
        f"Skipped: {pipeline.get('skipped', 0)}")
    add('')

    emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣']
    for i, d in enumerate(decisions):
        size = d['quantity'] * d.get('entry', d.get('entry_price', 0))
        tp = d.get('take_profits', [])
        flags = d.get('risk_flags', [])
        e = emoji[i] if i < len(emoji) else '▪️'

        add(f'{e} {d["ticker"]} — {d["action"]}')
        add(f'   Conf {d["confidence"]:.0%} · Size ${size:,.0f} ({d["position_pct"]:.0f}%) · '
            f'Risk ${d["risk_amount"]:,.0f} · R:R 1:{d["reward_ratio"]:.1f}')
        entry = d.get('entry', d.get('entry_price', 0))
        add(f'   Entry ${entry:.2f} · Stop ${d["stop_loss"]:.2f}')
        if tp:
            tp_str = ' / '.join(f'${t["level"]:.2f} ({t["sell_pct"]}%)' for t in tp)
            add(f'   TP: {tp_str}')
        if flags:
            add(f'   ⚠️  {" · ".join(flags)}')
        add('')

    total = sum(d['quantity'] * d.get('entry', d.get('entry_price', 0)) for d in decisions)
    add(f'💼 ${total:,.0f} deployed · R:R weighted avg')
    add('')
    add('⚙️ Entry: 52w-low + 0.5% · Stop: ATR 2x / Hard 10% / Structural')
    add('⚙️ Exit: 3-tier TP + 8% trailing + 60d time stop')
    add('⚙️ Size: Quarter-Kelly capped at $80K')
    add(f'⚙️ Mode: {data.get("execution", {}).get("mode", "DRY_RUN")}')

    return '\n'.join(lines)


if __name__ == '__main__':
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    with open(input_path) as f:
        data = json.load(f)
    report = generate_report(data)
    print(report)

    # Also save as .txt
    output_path = input_path.parent / 'pipeline_report.txt'
    with open(output_path, 'w') as f:
        f.write(report)
    print(f'\n📁 Report saved to {output_path}')
