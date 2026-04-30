#!/usr/bin/env python3
"""
VMAA → Tiger Trade Bridge
===========================
Connects VMAA's filter/price/ML pipeline to Tiger Trade paper/live trading.

Flow:
  1. filter_stocks.py → candidates.json (value + MAGNA screened)
  2. price_engine.py → prices.json (buy/sell/stop levels)
  3. THIS MODULE → executes trades via TigerBroker

Safety features:
  - Max position size (20% of portfolio per stock)
  - Max total positions (8)
  - Confirmation mode (dry-run flag)
  - Daily trade limit

Usage:
  python3 -m broker.vmaa_tiger_bridge \\
    --prices output/prices.json \\
    --dry-run          # simulate without real orders
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent dir for broker import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from broker.tiger_broker import TigerBroker, OrderResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("vmaa.bridge")

# ── Risk limits ─────────────────────────────────────────────
MAX_POSITION_PCT = 0.20   # Max 20% of portfolio per stock
MAX_POSITIONS = 8          # Max concurrent positions
MIN_ORDER_VALUE = 500.0    # Min $500 per trade
DAILY_TRADE_LIMIT = 10     # Max orders per day


def load_prices(path: str) -> List[Dict[str, Any]]:
    """Load VMAA price engine output."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Try common keys
        for key in ('trades', 'signals', 'plans', 'results'):
            if key in data:
                return data[key]
        # Return as single-item list if no known key
        return [data]
    return data


class VMAATigerBridge:
    """Bridge between VMAA signals and Tiger Trade execution."""

    def __init__(self, dry_run: bool = True):
        self.broker = TigerBroker()
        self.dry_run = dry_run
        self.results: List[OrderResult] = []
        self.skipped: List[Dict] = []
        self.daily_count = 0

        logger.info(f"Bridge initialized — {'DRY RUN' if dry_run else 'LIVE'} mode")

    def execute(self, prices_path: str) -> Dict[str, Any]:
        """Execute all trade signals from VMAA price engine output."""
        trades = load_prices(prices_path)
        logger.info(f"Loaded {len(trades)} trade signals from {prices_path}")

        for i, trade in enumerate(trades):
            if self.daily_count >= DAILY_TRADE_LIMIT:
                logger.warning(f"Daily trade limit ({DAILY_TRADE_LIMIT}) reached")
                self.skipped.append({**trade, 'reason': 'daily_limit'})
                continue

            result = self._process_trade(trade, i)
            if result:
                self.results.append(result)
                self.daily_count += 1

        return self._summary()

    def _process_trade(self, trade: Dict, index: int) -> Optional[OrderResult]:
        """Process a single trade signal with risk checks."""
        symbol = trade.get('symbol', trade.get('ticker', ''))
        action = trade.get('action', trade.get('direction', 'BUY'))
        price = trade.get('buy_price', trade.get('entry_price', trade.get('limit_price')))
        stop_price = trade.get('stop_loss', trade.get('stop_price'))
        take_profit = trade.get('take_profit', trade.get('target_price'))

        if not symbol:
            logger.warning(f"[{index}] Skipped: no symbol")
            return None

        # ── Risk check: position limit ──────────────────────
        positions = self.broker.get_positions()
        existing = next((p for p in positions if p.symbol.upper() == symbol.upper()), None)

        if existing and action.upper() == 'BUY':
            logger.info(f"[{index}] {symbol}: Already holding {existing.quantity} sh, skip BUY")
            self.skipped.append({**trade, 'reason': 'already_holding'})
            return None

        if len(positions) >= MAX_POSITIONS and action.upper() == 'BUY' and not existing:
            logger.warning(f"[{index}] {symbol}: Max positions ({MAX_POSITIONS}) reached")
            self.skipped.append({**trade, 'reason': 'max_positions'})
            return None

        # ── Risk check: position sizing ────────────────────
        bp = self.broker.get_buying_power()
        max_alloc = bp * MAX_POSITION_PCT
        position_value = price * trade.get('quantity', 1) if price else max_alloc
        quantity = trade.get('quantity', 0)

        if quantity <= 0 and price and price > 0:
            quantity = int(max_alloc / price)
            quantity = max(1, min(quantity, 10000))  # 1-10000 range
        elif quantity <= 0:
            quantity = 1  # fallback

        order_value = quantity * (price or 0)
        if price and order_value < MIN_ORDER_VALUE:
            logger.info(f"[{index}] {symbol}: Order value ${order_value:.0f} < min ${MIN_ORDER_VALUE}")
            self.skipped.append({**trade, 'reason': 'order_too_small'})
            return None

        # ── Execute ────────────────────────────────────────
        logger.info(f"[{index}] {symbol}: {action} {quantity} sh "
                    f"@ ${price:.2f}" + (f" | Stop: ${stop_price:.2f}" if stop_price else ""))

        if self.dry_run:
            logger.info(f"  [DRY RUN] Would place {action} {quantity} {symbol} @ ${price}")
            return OrderResult(
                order_id=-1, symbol=symbol, action=action,
                quantity=quantity, order_type='LMT',
                limit_price=price, status='DRY_RUN',
                filled_quantity=0, filled_price=0.0,
                timestamp=str(datetime.now()), reason='dry_run',
            )

        result = self.broker.place_order(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type='LMT',
            limit_price=price,
            stop_price=stop_price,
        )
        return result

    def _summary(self) -> Dict[str, Any]:
        """Generate execution summary."""
        return {
            'timestamp': str(datetime.now()),
            'mode': 'DRY_RUN' if self.dry_run else 'LIVE',
            'total_signals': len(self.results) + len(self.skipped),
            'executed': len(self.results),
            'skipped': len(self.skipped),
            'skipped_reasons': [s.get('reason', 'unknown') for s in self.skipped],
            'orders': [
                {
                    'order_id': r.order_id,
                    'symbol': r.symbol,
                    'action': r.action,
                    'quantity': r.quantity,
                    'price': r.limit_price,
                    'status': r.status,
                }
                for r in self.results
            ],
        }

    def cancel_all(self) -> int:
        """Cancel all open orders."""
        orders = self.broker.get_orders()
        cancelled = 0
        for o in orders:
            if o.status in ('Submitted', 'Pending', 'OrderStatus.HELD', 'OrderStatus.NEW'):
                if self.broker.cancel_order(o.order_id):
                    cancelled += 1
        logger.info(f"Cancelled {cancelled} open orders")
        return cancelled


# ── CLI ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VMAA → Tiger Trade Bridge")
    parser.add_argument('--prices', required=True, help='Path to VMAA prices.json')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='Simulate without real orders (default: True)')
    parser.add_argument('--live', action='store_true',
                        help='Execute real orders (overrides --dry-run)')
    parser.add_argument('--cancel-all', action='store_true',
                        help='Cancel all open orders')
    parser.add_argument('--status', action='store_true',
                        help='Show broker status only')
    args = parser.parse_args()

    bridge = VMAATigerBridge(dry_run=not args.live)

    if args.status:
        status = bridge.broker.status()
        print(json.dumps(status, indent=2))
        return

    if args.cancel_all:
        n = bridge.cancel_all()
        print(f"Cancelled {n} orders")
        return

    summary = bridge.execute(args.prices)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
