#!/usr/bin/env python3
"""
VMAA Tiger Broker — Paper/Live Trading Bridge
==============================================
Wraps Tiger Trade OpenAPI for VMAA execution:
  - Account & portfolio queries
  - Order placement (market, limit, stop)
  - Position tracking
  - Risk checks (buying power, margin)

Market data still uses yfinance (no Tiger quote subscription needed).
Tiger API only for trade execution.

Usage:
    from broker.tiger_broker import TigerBroker
    broker = TigerBroker()
    broker.place_order('AAPL', 'BUY', 10, order_type='LMT', limit_price=150.0)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.trade.trade_client import TradeClient
from tigeropen.quote.quote_client import QuoteClient
from tigeropen.common.consts import SecurityType, Currency, OrderType
from tigeropen.trade.domain.order import Order

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("vmaa.broker")

CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_PATH = str(CONFIG_DIR / 'tiger_openapi_config.properties')


@dataclass
class BrokerPosition:
    symbol: str
    name: str
    quantity: int
    average_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    currency: str


@dataclass
class BrokerAccount:
    account_id: str
    net_liquidation: float
    cash: float
    buying_power: float
    gross_position_value: float
    unrealized_pnl: float
    realized_pnl: float
    currency: str


@dataclass
class OrderResult:
    order_id: int
    symbol: str
    action: str
    quantity: int
    order_type: str
    limit_price: Optional[float]
    status: str
    filled_quantity: int
    filled_price: float
    timestamp: Optional[str]
    reason: Optional[str]


class TigerBroker:
    """Tiger Trade broker for VMAA execution layer."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.config = TigerOpenClientConfig(props_path=config_path)
        self.trade_client = TradeClient(self.config)
        self.quote_client = QuoteClient(self.config)
        self._account_cache = None
        self._cache_time = 0
        self.ACCOUNT_CACHE_TTL = 30  # seconds
        logger.info(f"TigerBroker initialized — Account: {self.config.account}")

    # ── Account ──────────────────────────────────────────────

    def get_account(self, use_cache: bool = True) -> BrokerAccount:
        """Fetch account summary with optional caching."""
        if use_cache and self._account_cache and (time.time() - self._cache_time) < self.ACCOUNT_CACHE_TTL:
            return self._account_cache

        assets = self.trade_client.get_assets()
        item = assets[0]
        summary = item.summary
        seg = item.segments.get('S', None)

        acct = BrokerAccount(
            account_id=item.account,
            net_liquidation=float(summary.net_liquidation or 0),
            cash=float(summary.cash or 0),
            buying_power=float(summary.buying_power or 0),
            gross_position_value=float(summary.gross_position_value or 0),
            unrealized_pnl=float(summary.unrealized_pnl or 0),
            realized_pnl=float(summary.realized_pnl or 0),
            currency=str(summary.currency or 'USD'),
        )
        self._account_cache = acct
        self._cache_time = time.time()
        return acct

    def get_buying_power(self) -> float:
        """Quick check: available buying power."""
        return self.get_account().buying_power

    def get_total_value(self) -> float:
        """Total portfolio value."""
        return self.get_account().net_liquidation

    # ── Positions ────────────────────────────────────────────

    def get_positions(self) -> List[BrokerPosition]:
        """Get all current positions."""
        raw = self.trade_client.get_positions()
        positions = []
        for pos in raw:
            positions.append(BrokerPosition(
                symbol=pos.contract.symbol,
                name=getattr(pos.contract, 'name', pos.contract.symbol),
                quantity=pos.quantity,
                average_cost=float(pos.average_cost or 0),
                market_price=float(pos.market_price or 0),
                market_value=float(pos.market_value or 0),
                unrealized_pnl=float(pos.unrealized_pnl or 0),
                unrealized_pnl_pct=float(getattr(pos, 'unrealized_pnl_pct', 0) or 0),
                currency=getattr(pos.contract, 'currency', 'USD'),
            ))
        return positions

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a specific symbol."""
        for pos in self.get_positions():
            if pos.symbol.upper() == symbol.upper():
                return pos
        return None

    # ── Orders ───────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        action: str,  # 'BUY' or 'SELL'
        quantity: int,
        order_type: str = 'LMT',  # 'MKT', 'LMT', 'STP', 'STP_LMT'
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        outside_rth: bool = False,
    ) -> OrderResult:
        """Place an order. Returns OrderResult with status."""
        contract = self.trade_client.get_contract(symbol, sec_type=SecurityType.STK)

        order = Order(
            account=self.config.account,
            contract=contract,
            action=action.upper(),
            order_type=order_type.upper(),
            quantity=quantity,
            limit_price=limit_price,
            aux_price=stop_price,
            outside_rth=outside_rth,
        )

        logger.info(f"PLACE {action} {quantity} {symbol} {order_type}"
                    f"{' @ $'+str(limit_price) if limit_price else ''}"
                    f"{' stop $'+str(stop_price) if stop_price else ''}")

        try:
            order_id = self.trade_client.place_order(order)
            # place_order returns order ID (int). Fetch status from get_orders.
            logger.info(f"  ✓ Order placed: ID={order_id}")
            
            # Try to get order status
            status = 'Submitted'
            filled_qty = 0
            filled_price = 0.0
            try:
                orders = self.trade_client.get_orders()
                for o in orders:
                    if o.id == order_id:
                        status = str(o.status)
                        filled_qty = o.filled or 0
                        filled_price = float(o.avg_fill_price or 0)
                        break
            except Exception:
                pass
            
            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status=status,
                filled_quantity=filled_qty,
                filled_price=filled_price,
                timestamp=str(datetime.now()),
                reason=None,
            )
        except Exception as e:
            logger.error(f"  ✗ Order failed: {e}")
            return OrderResult(
                order_id=-1,
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=order_type,
                limit_price=limit_price,
                status='REJECTED',
                filled_quantity=0,
                filled_price=0.0,
                timestamp=str(datetime.now()),
                reason=str(e),
            )

    def buy_limit(self, symbol: str, quantity: int, price: float) -> OrderResult:
        """Convenience: buy with limit order."""
        return self.place_order(symbol, 'BUY', quantity, 'LMT', limit_price=price)

    def sell_limit(self, symbol: str, quantity: int, price: float) -> OrderResult:
        """Convenience: sell with limit order."""
        return self.place_order(symbol, 'SELL', quantity, 'LMT', limit_price=price)

    def buy_market(self, symbol: str, quantity: int) -> OrderResult:
        """Convenience: buy with market order."""
        return self.place_order(symbol, 'BUY', quantity, 'MKT')

    def sell_market(self, symbol: str, quantity: int) -> OrderResult:
        """Convenience: sell with market order."""
        return self.place_order(symbol, 'SELL', quantity, 'MKT')

    def get_orders(self, limit: int = 20) -> List[OrderResult]:
        """Get recent orders."""
        raw = self.trade_client.get_orders()
        orders = []
        for o in raw[:limit]:
            orders.append(OrderResult(
                order_id=o.id,
                symbol=o.contract.symbol,
                action=o.action,
                quantity=o.quantity,
                order_type=o.order_type,
                limit_price=float(o.limit_price) if o.limit_price else None,
                status=o.status,
                filled_quantity=getattr(o, 'filled_quantity', 0) or 0,
                filled_price=float(getattr(o, 'filled_price', 0) or 0),
                timestamp=str(getattr(o, 'timestamp', None) or ''),
                reason=None,
            ))
        return orders

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order by global order ID."""
        try:
            result = self.trade_client.cancel_order(id=order_id)
            logger.info(f"Cancelled order {order_id}: {result}")
            return True
        except Exception as e:
            logger.error(f"Cancel order {order_id} failed: {e}")
            return False

    # ── Risk Checks ──────────────────────────────────────────

    def can_afford(self, amount: float) -> bool:
        """Check if account has enough buying power."""
        bp = self.get_buying_power()
        return bp >= amount

    def max_position_size(self, symbol: str, max_pct: float = 0.20) -> int:
        """
        Max shares for a single position (max_pct of portfolio).
        Uses account buying power, not just cash.
        """
        bp = self.get_buying_power()
        max_allocation = bp * max_pct
        # We'd need current price from yfinance — return allocation instead
        return int(max_allocation)

    def portfolio_risk_ok(self, max_positions: int = 8) -> Tuple[bool, str]:
        """Check portfolio risk limits."""
        positions = self.get_positions()
        if len(positions) >= max_positions:
            return False, f"Max positions ({max_positions}) reached"
        return True, "OK"

    # ── Contract Lookup ──────────────────────────────────────

    def get_contract(self, symbol: str):
        """Look up a contract by symbol."""
        return self.trade_client.get_contract(symbol, sec_type=SecurityType.STK)

    # ── Status ───────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Full broker status summary."""
        try:
            acct = self.get_account(use_cache=False)
            positions = self.get_positions()
            orders = self.get_orders(limit=10)

            return {
                'connected': True,
                'account_id': acct.account_id,
                'net_liquidation': acct.net_liquidation,
                'cash': acct.cash,
                'buying_power': acct.buying_power,
                'positions_count': len(positions),
                'positions_value': acct.gross_position_value,
                'unrealized_pnl': acct.unrealized_pnl,
                'realized_pnl': acct.realized_pnl,
                'active_orders': len([o for o in orders if o.status in ('Submitted', 'Pending', 'Filled')]),
                'currency': acct.currency,
            }
        except Exception as e:
            return {'connected': False, 'error': str(e)}


# ── Quick test ───────────────────────────────────────────────
if __name__ == '__main__':
    broker = TigerBroker()
    print("\n📊 Account Status")
    print("-" * 40)
    status = broker.status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print(f"\n📈 Positions: {len(broker.get_positions())}")
    for p in broker.get_positions():
        print(f"  {p.symbol}: {p.quantity} sh @ ${p.average_cost:.2f} | P&L ${p.unrealized_pnl:.2f}")

    print(f"\n📋 Recent Orders: {len(broker.get_orders())}")
    for o in broker.get_orders()[:5]:
        print(f"  #{o.order_id} {o.action} {o.quantity} {o.symbol} {o.status}")
