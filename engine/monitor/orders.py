#!/usr/bin/env python3
"""
VMAA 2.0 — Conditional Orders Engine
======================================
Staged order management with full lifecycle:
  - Order types: Limit Entry, Stop Entry, Trailing Stop, Take Profit, Time Stop
  - Lifecycle: CREATED → ACTIVE → TRIGGERED/FILLED/CANCELLED/EXPIRED
  - JSON persistence in engine/data/orders/
  - Fill history with slippage tracking
  - Optional Tiger broker integration for live execution
"""
from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("vmaa.monitor.orders")


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class OrderType(Enum):
    LIMIT_ENTRY = auto()     # Buy at specific price or better
    STOP_ENTRY = auto()      # Buy/sell when price breaks level
    TRAILING_STOP = auto()   # Follow price at configurable distance
    TAKE_PROFIT = auto()     # Partial/full at target levels
    TIME_STOP = auto()       # Auto-close after N days


class OrderStatus(Enum):
    CREATED = "created"       # Defined but not yet active
    ACTIVE = "active"         # Monitoring for trigger
    TRIGGERED = "triggered"   # Condition met, pending execution
    FILLED = "filled"         # Executed (fully or partially)
    CANCELLED = "cancelled"   # Manually cancelled
    EXPIRED = "expired"       # Time/date expiration reached


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Fill:
    """A fill record for an order execution."""
    timestamp: str
    quantity: int
    price: float
    slippage_pct: float = 0.0       # vs trigger/limit price
    order_id: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "quantity": self.quantity,
            "price": self.price,
            "slippage_pct": self.slippage_pct,
            "order_id": self.order_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fill":
        return cls(
            timestamp=d["timestamp"], quantity=d["quantity"],
            price=d["price"], slippage_pct=d.get("slippage_pct", 0),
            order_id=d.get("order_id", ""), notes=d.get("notes", ""),
        )


@dataclass
class OrderHistory:
    """Complete history for a single order ID."""
    order_id: str
    fills: List[Fill] = field(default_factory=list)
    total_filled: int = 0
    avg_fill_price: float = 0.0
    total_slippage: float = 0.0

    def record_fill(self, fill: Fill) -> None:
        self.fills.append(fill)
        self.total_filled = sum(f.quantity for f in self.fills)
        if self.total_filled > 0:
            self.avg_fill_price = sum(
                f.price * f.quantity for f in self.fills
            ) / self.total_filled
        self.total_slippage = (
            sum(f.slippage_pct for f in self.fills) / len(self.fills)
            if self.fills else 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "fills": [f.to_dict() for f in self.fills],
            "total_filled": self.total_filled,
            "avg_fill_price": self.avg_fill_price,
            "total_slippage": self.total_slippage,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OrderHistory":
        oh = cls(order_id=d["order_id"])
        for f in d.get("fills", []):
            oh.fills.append(Fill.from_dict(f))
        oh.total_filled = d.get("total_filled", 0)
        oh.avg_fill_price = d.get("avg_fill_price", 0)
        oh.total_slippage = d.get("total_slippage", 0)
        return oh


@dataclass
class ConditionalOrder:
    """A conditional order with trigger rules and lifecycle tracking."""

    # Identity
    id: str
    name: str = ""
    ticker: str = ""

    # Order type & direction
    order_type: OrderType = OrderType.LIMIT_ENTRY
    action: str = "BUY"                    # BUY or SELL
    quantity: int = 0
    remaining_quantity: int = 0            # For partial fills

    # Trigger conditions
    trigger_price: float = 0.0             # Price to trigger at
    limit_price: Optional[float] = None    # Limit price (if different from trigger)
    stop_price: Optional[float] = None     # Stop price for stop orders

    # Trailing stop
    trailing_distance_pct: float = 0.05    # Distance in %
    trailing_high: float = 0.0             # Tracked high for trailing stop
    trailing_low: float = 0.0              # Tracked low for trailing stop (sell)
    trailing_activated: bool = False

    # Take profit
    take_profit_tiers: List[Dict[str, Any]] = field(default_factory=list)
    # Each tier: {"pct": 15, "sell_pct": 30, "label": "TP1"}

    # Time stop
    time_stop_days: int = 90
    activation_date: Optional[str] = None  # ISO format

    # GTC / expiration
    good_till_cancelled: bool = True
    expires_at: Optional[str] = None       # ISO format

    # Lifecycle
    status: OrderStatus = OrderStatus.CREATED
    created_at: str = ""
    activated_at: Optional[str] = None
    triggered_at: Optional[str] = None
    filled_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    cancel_reason: str = ""

    # Execution metadata
    notes: str = ""
    source: str = ""                       # e.g. "manual", "pipeline", "auto-monitor"
    broker_order_id: Optional[int] = None  # Tiger broker order ID if executed

    # Related order (e.g., linked stop loss for an entry)
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "ticker": self.ticker,
            "order_type": self.order_type.name,
            "action": self.action,
            "quantity": self.quantity,
            "remaining_quantity": self.remaining_quantity,
            "trigger_price": self.trigger_price,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "trailing_distance_pct": self.trailing_distance_pct,
            "trailing_high": self.trailing_high,
            "trailing_low": self.trailing_low,
            "trailing_activated": self.trailing_activated,
            "take_profit_tiers": self.take_profit_tiers,
            "time_stop_days": self.time_stop_days,
            "activation_date": self.activation_date,
            "good_till_cancelled": self.good_till_cancelled,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "triggered_at": self.triggered_at,
            "filled_at": self.filled_at,
            "cancelled_at": self.cancelled_at,
            "cancel_reason": self.cancel_reason,
            "notes": self.notes,
            "source": self.source,
            "broker_order_id": self.broker_order_id,
            "parent_order_id": self.parent_order_id,
            "child_orders": self.child_orders,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConditionalOrder":
        return cls(
            id=d["id"], name=d.get("name", ""), ticker=d.get("ticker", ""),
            order_type=OrderType[d.get("order_type", "LIMIT_ENTRY")],
            action=d.get("action", "BUY"),
            quantity=d.get("quantity", 0),
            remaining_quantity=d.get("remaining_quantity", d.get("quantity", 0)),
            trigger_price=d.get("trigger_price", 0.0),
            limit_price=d.get("limit_price"),
            stop_price=d.get("stop_price"),
            trailing_distance_pct=d.get("trailing_distance_pct", 0.05),
            trailing_high=d.get("trailing_high", 0.0),
            trailing_low=d.get("trailing_low", 0.0),
            trailing_activated=d.get("trailing_activated", False),
            take_profit_tiers=d.get("take_profit_tiers", []),
            time_stop_days=d.get("time_stop_days", 90),
            activation_date=d.get("activation_date"),
            good_till_cancelled=d.get("good_till_cancelled", True),
            expires_at=d.get("expires_at"),
            status=OrderStatus(d.get("status", "created")),
            created_at=d.get("created_at", ""),
            activated_at=d.get("activated_at"),
            triggered_at=d.get("triggered_at"),
            filled_at=d.get("filled_at"),
            cancelled_at=d.get("cancelled_at"),
            cancel_reason=d.get("cancel_reason", ""),
            notes=d.get("notes", ""),
            source=d.get("source", ""),
            broker_order_id=d.get("broker_order_id"),
            parent_order_id=d.get("parent_order_id"),
            child_orders=d.get("child_orders", []),
        )


# ═══════════════════════════════════════════════════════════════════
# Order Manager
# ═══════════════════════════════════════════════════════════════════

class OrderManager:
    """
    Manages conditional orders through their full lifecycle.

    Usage:
        mgr = OrderManager(orders_dir="engine/data/orders")
        order_id = mgr.create_limit_entry("AAPL", 10, 145.0)
        triggered = mgr.check_orders()
    """

    def __init__(
        self,
        orders_dir: str = "engine/data/orders",
        history_file: str = "engine/data/orders/history.json",
        max_slippage_pct: float = 0.02,
        trailing_default_pct: float = 0.08,
        trailing_default_distance: float = 0.05,
        time_stop_default_days: int = 90,
        auto_execute: bool = False,
        broker=None,
    ):
        self.orders_dir = Path(orders_dir)
        self.history_file = Path(history_file)
        self.max_slippage_pct = max_slippage_pct
        self.trailing_default_pct = trailing_default_pct
        self.trailing_default_distance = trailing_default_distance
        self.time_stop_default_days = time_stop_default_days
        self.auto_execute = auto_execute
        self.broker = broker  # Optional TigerBroker instance

        self.orders: Dict[str, ConditionalOrder] = {}
        self.history: Dict[str, OrderHistory] = {}

        # Ensure directories
        self.orders_dir.mkdir(parents=True, exist_ok=True)

        # Data cache for price checks
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # ticker → (price, timestamp)
        self._price_cache_ttl: float = 30.0

    # ── Order Creation ────────────────────────────────────────

    def create_limit_entry(
        self,
        ticker: str,
        quantity: int,
        limit_price: float,
        name: str = "",
        action: str = "BUY",
    ) -> str:
        """Create a limit entry order."""
        order = ConditionalOrder(
            id=self._gen_id("limit", ticker),
            name=name or f"Limit {action} {quantity} {ticker} @ {limit_price}",
            ticker=ticker,
            order_type=OrderType.LIMIT_ENTRY,
            action=action,
            quantity=quantity,
            remaining_quantity=quantity,
            trigger_price=limit_price,
            limit_price=limit_price,
            status=OrderStatus.CREATED,
            created_at=datetime.now().isoformat(),
            source="auto-monitor",
        )
        return self._add_order(order)

    def create_stop_entry(
        self,
        ticker: str,
        quantity: int,
        stop_price: float,
        action: str = "BUY",
        name: str = "",
    ) -> str:
        """Create a stop entry order (buy/sell when price breaks stop_price)."""
        order = ConditionalOrder(
            id=self._gen_id("stop", ticker),
            name=name or f"Stop {action} {quantity} {ticker} @ {stop_price}",
            ticker=ticker,
            order_type=OrderType.STOP_ENTRY,
            action=action,
            quantity=quantity,
            remaining_quantity=quantity,
            trigger_price=stop_price,
            stop_price=stop_price,
            status=OrderStatus.CREATED,
            created_at=datetime.now().isoformat(),
            source="auto-monitor",
        )
        return self._add_order(order)

    def create_trailing_stop(
        self,
        ticker: str,
        quantity: int,
        entry_price: float,
        distance_pct: Optional[float] = None,
        action: str = "SELL",
        parent_order_id: Optional[str] = None,
        name: str = "",
    ) -> str:
        """Create a trailing stop order."""
        dist = distance_pct or self.trailing_default_distance
        # For SELL trailing: initial stop below current price
        initial_stop = entry_price * (1 - dist) if action == "SELL" else entry_price * (1 + dist)
        order = ConditionalOrder(
            id=self._gen_id("trail", ticker),
            name=name or f"Trailing {action} {quantity} {ticker} ({dist:.1%})",
            ticker=ticker,
            order_type=OrderType.TRAILING_STOP,
            action=action,
            quantity=quantity,
            remaining_quantity=quantity,
            trigger_price=initial_stop,
            trailing_distance_pct=dist,
            trailing_high=entry_price if action == "SELL" else 0.0,
            trailing_low=entry_price if action == "BUY" else float('inf'),
            trailing_activated=True,
            status=OrderStatus.CREATED,
            created_at=datetime.now().isoformat(),
            parent_order_id=parent_order_id,
            source="auto-monitor",
        )
        return self._add_order(order)

    def create_take_profit(
        self,
        ticker: str,
        quantity: int,
        entry_price: float,
        tiers: Optional[List[Dict[str, Any]]] = None,
        parent_order_id: Optional[str] = None,
        name: str = "",
    ) -> List[str]:
        """Create take profit orders at multiple tiers. Returns list of order IDs."""
        if tiers is None:
            tiers = [
                {"pct": 15, "sell_pct": 30, "label": "TP1"},
                {"pct": 25, "sell_pct": 30, "label": "TP2"},
                {"pct": 40, "sell_pct": 40, "label": "TP3"},
            ]
        order_ids = []
        for tier in tiers:
            tp_pct = tier["pct"] / 100
            sell_qty = int(quantity * tier["sell_pct"] / 100)
            if sell_qty <= 0:
                continue
            tp_price = round(entry_price * (1 + tp_pct), 2)
            order = ConditionalOrder(
                id=self._gen_id(f"tp_{tier['label']}", ticker),
                name=name or f"Take Profit {tier['label']} {ticker} @ {tp_price}",
                ticker=ticker,
                order_type=OrderType.TAKE_PROFIT,
                action="SELL",
                quantity=sell_qty,
                remaining_quantity=sell_qty,
                trigger_price=tp_price,
                limit_price=tp_price,
                take_profit_tiers=[tier],
                status=OrderStatus.CREATED,
                created_at=datetime.now().isoformat(),
                parent_order_id=parent_order_id,
                source="auto-monitor",
            )
            order_ids.append(self._add_order(order))
        return order_ids

    def create_time_stop(
        self,
        ticker: str,
        quantity: int,
        days: Optional[int] = None,
        parent_order_id: Optional[str] = None,
        name: str = "",
    ) -> str:
        """Create a time stop order (auto-close after N days)."""
        days = days or self.time_stop_default_days
        order = ConditionalOrder(
            id=self._gen_id("time", ticker),
            name=name or f"Time Stop {quantity} {ticker} ({days}d)",
            ticker=ticker,
            order_type=OrderType.TIME_STOP,
            action="SELL",
            quantity=quantity,
            remaining_quantity=quantity,
            trigger_price=0.0,  # Not price-triggered
            time_stop_days=days,
            activation_date=datetime.now().isoformat(),  # activated now
            expires_at=(datetime.now() + timedelta(days=days)).isoformat(),
            status=OrderStatus.CREATED,
            created_at=datetime.now().isoformat(),
            parent_order_id=parent_order_id,
            source="auto-monitor",
        )
        return self._add_order(order)

    # ── Order Lifecycle ───────────────────────────────────────

    def activate_order(self, order_id: str) -> bool:
        """Move order from CREATED to ACTIVE."""
        order = self.orders.get(order_id)
        if not order or order.status != OrderStatus.CREATED:
            return False
        order.status = OrderStatus.ACTIVE
        order.activated_at = datetime.now().isoformat()
        logger.info(f"Activated order {order.id} [{order.order_type.name}] {order.ticker}")
        return True

    def activate_all(self) -> int:
        """Activate all CREATED orders."""
        count = 0
        for oid in list(self.orders.keys()):
            if self.activate_order(oid):
                count += 1
        return count

    def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        if not order or order.status in (OrderStatus.FILLED, OrderStatus.EXPIRED, OrderStatus.CANCELLED):
            return False
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now().isoformat()
        order.cancel_reason = reason
        logger.info(f"Cancelled order {order.id}: {reason}")

        # Cancel child orders
        for child_id in order.child_orders:
            self.cancel_order(child_id, f"parent_cancelled:{order.id}")
        return True

    def expire_order(self, order_id: str) -> bool:
        """Mark an order as expired."""
        order = self.orders.get(order_id)
        if not order:
            return False
        order.status = OrderStatus.EXPIRED
        logger.info(f"Expired order {order.id} [{order.order_type.name}] {order.ticker}")
        return True

    # ── Order Checking ────────────────────────────────────────

    def _get_price(self, ticker: str) -> Optional[float]:
        """Get latest price with 30s cache."""
        now = time.time()
        if ticker in self._price_cache:
            price, ts = self._price_cache[ticker]
            if now - ts < self._price_cache_ttl:
                return price

        try:
            t = yf.Ticker(ticker)
            data = t.history(period="5d")
            if data.empty:
                return None
            price = float(data['Close'].iloc[-1])
            self._price_cache[ticker] = (price, now)
            return price
        except Exception as e:
            logger.debug(f"Price fetch failed for {ticker}: {e}")
            return None

    def check_order(self, order: ConditionalOrder) -> Optional[Dict[str, Any]]:
        """Check if an order's trigger conditions are met. Returns trigger event or None."""
        if order.status not in (OrderStatus.ACTIVE, OrderStatus.CREATED):
            return None

        now = datetime.now()

        # Check expiration
        if order.expires_at:
            try:
                expires = datetime.fromisoformat(order.expires_at)
                if now >= expires:
                    self.expire_order(order.id)
                    return {"type": "expired", "order": order}
            except (ValueError, TypeError):
                pass

        current_price = self._get_price(order.ticker)
        if current_price is None:
            return None

        if order.order_type == OrderType.LIMIT_ENTRY:
            return self._check_limit_entry(order, current_price)

        elif order.order_type == OrderType.STOP_ENTRY:
            return self._check_stop_entry(order, current_price)

        elif order.order_type == OrderType.TRAILING_STOP:
            return self._check_trailing_stop(order, current_price)

        elif order.order_type == OrderType.TAKE_PROFIT:
            return self._check_take_profit(order, current_price)

        elif order.order_type == OrderType.TIME_STOP:
            return self._check_time_stop(order, now)

        return None

    def _check_limit_entry(self, order: ConditionalOrder, price: float) -> Optional[Dict[str, Any]]:
        """Check limit entry: for BUY, price <= trigger; for SELL, price >= trigger."""
        limit = order.limit_price or order.trigger_price
        if limit <= 0:
            return None

        triggered = False
        if order.action == "BUY" and price <= limit:
            triggered = True
        elif order.action == "SELL" and price >= limit:
            triggered = True

        if triggered:
            order.status = OrderStatus.TRIGGERED
            order.triggered_at = datetime.now().isoformat()
            logger.info(f"LIMIT {order.action} {order.ticker} triggered @ ${price:.2f} (limit: ${limit:.2f})")
            return {"type": "triggered", "order": order, "price": price, "limit_price": limit}
        return None

    def _check_stop_entry(self, order: ConditionalOrder, price: float) -> Optional[Dict[str, Any]]:
        """Check stop entry: for BUY, price >= stop (breakout up); for SELL, price <= stop (breakdown)."""
        stop = order.stop_price or order.trigger_price
        if stop <= 0:
            return None

        triggered = False
        if order.action == "BUY" and price >= stop:
            triggered = True
        elif order.action == "SELL" and price <= stop:
            triggered = True

        if triggered:
            order.status = OrderStatus.TRIGGERED
            order.triggered_at = datetime.now().isoformat()
            logger.info(f"STOP {order.action} {order.ticker} triggered @ ${price:.2f} (stop: ${stop:.2f})")
            return {"type": "triggered", "order": order, "price": price, "stop_price": stop}
        return None

    def _check_trailing_stop(self, order: ConditionalOrder, price: float) -> Optional[Dict[str, Any]]:
        """Check trailing stop: update trailing level, trigger if breached."""
        if not order.trailing_activated:
            order.trailing_activated = True
            order.trailing_high = price
            order.trailing_low = price
            order.trigger_price = price * (1 - order.trailing_distance_pct)
            return None

        dist = order.trailing_distance_pct

        if order.action == "SELL":
            # Update trailing high, check breach
            if price > order.trailing_high:
                order.trailing_high = price
                new_stop = price * (1 - dist)
                if new_stop > order.trigger_price:
                    order.trigger_price = new_stop
                    logger.debug(f"Trailing stop updated: {order.ticker} → ${new_stop:.2f}")

            # Check if breached
            if price <= order.trigger_price:
                order.status = OrderStatus.TRIGGERED
                order.triggered_at = datetime.now().isoformat()
                logger.info(f"TRAILING STOP {order.ticker} triggered @ ${price:.2f} (stop: ${order.trigger_price:.2f})")
                return {"type": "triggered", "order": order, "price": price, "stop_price": order.trigger_price}

        elif order.action == "BUY":
            # Trailing buy stop — update low, trigger on breakout
            if price < order.trailing_low:
                order.trailing_low = price
                new_trigger = price * (1 + dist)
                if new_trigger < order.trigger_price:
                    order.trigger_price = new_trigger

            if price >= order.trigger_price:
                order.status = OrderStatus.TRIGGERED
                order.triggered_at = datetime.now().isoformat()
                return {"type": "triggered", "order": order, "price": price, "trigger_price": order.trigger_price}

        return None

    def _check_take_profit(self, order: ConditionalOrder, price: float) -> Optional[Dict[str, Any]]:
        """Check take profit: price >= target."""
        target = order.trigger_price
        if price >= target:
            order.status = OrderStatus.TRIGGERED
            order.triggered_at = datetime.now().isoformat()
            logger.info(f"TAKE PROFIT {order.ticker} triggered @ ${price:.2f} (target: ${target:.2f})")
            return {"type": "triggered", "order": order, "price": price, "target_price": target}
        return None

    def _check_time_stop(self, order: ConditionalOrder, now: datetime) -> Optional[Dict[str, Any]]:
        """Check time stop: expire after N days from activation."""
        if not order.activation_date:
            order.activation_date = order.created_at

        try:
            activated = datetime.fromisoformat(order.activation_date)
            elapsed = (now - activated).days
            if elapsed >= order.time_stop_days:
                order.status = OrderStatus.TRIGGERED
                order.triggered_at = now.isoformat()
                logger.info(f"TIME STOP {order.ticker} triggered ({elapsed}d ≥ {order.time_stop_days}d)")
                return {"type": "triggered", "order": order, "days_elapsed": elapsed}
        except (ValueError, TypeError):
            pass
        return None

    def check_orders(self) -> List[Dict[str, Any]]:
        """Check all active orders. Returns list of trigger events."""
        events: List[Dict[str, Any]] = []
        for order in list(self.orders.values()):
            event = self.check_order(order)
            if event:
                events.append(event)
        return events

    # ── Execution ────────────────────────────────────────────

    def execute_order(
        self,
        order_id: str,
        execution_price: float,
        quantity: Optional[int] = None,
    ) -> Optional[Fill]:
        """
        Record order execution. Optionally place via broker.
        Returns Fill record or None.
        """
        order = self.orders.get(order_id)
        if not order or order.status != OrderStatus.TRIGGERED:
            logger.warning(f"Cannot execute order {order_id}: not triggered (status={order.status.value if order else 'missing'})")
            return None

        qty = quantity or order.remaining_quantity

        # Calculate slippage
        trigger = order.trigger_price
        if order.limit_price:
            trigger = order.limit_price
        slippage = abs(execution_price - trigger) / trigger * 100 if trigger > 0 else 0

        # Check slippage limit
        if slippage > self.max_slippage_pct * 100:
            logger.warning(
                f"Slippage {slippage:.2f}% exceeds max {self.max_slippage_pct*100:.2f}% "
                f"for {order.ticker}. Order not filled."
            )
            return None

        fill = Fill(
            timestamp=datetime.now().isoformat(),
            quantity=qty,
            price=execution_price,
            slippage_pct=round(slippage, 4),
            order_id=order.id,
            notes="",
        )

        # Record in history
        if order.id not in self.history:
            self.history[order.id] = OrderHistory(order_id=order.id)
        self.history[order.id].record_fill(fill)

        # Update order
        order.remaining_quantity -= qty
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now().isoformat()
        else:
            order.status = OrderStatus.ACTIVE  # Partial fill, keep active
            order.triggered_at = None

        # Place broker order if configured
        if self.auto_execute and self.broker:
            self._broker_execute(order, qty, execution_price)

        logger.info(
            f"Filled {qty} {order.ticker} @ ${execution_price:.2f} "
            f"(slip: {slippage:.2f}%, remaining: {order.remaining_quantity})"
        )
        return fill

    def _broker_execute(self, order: ConditionalOrder, qty: int, price: float) -> None:
        """Place order via Tiger broker."""
        try:
            if order.action == "BUY":
                result = self.broker.buy_limit(order.ticker, qty, price)
            else:
                result = self.broker.sell_limit(order.ticker, qty, price)
            if result and result.order_id > 0:
                order.broker_order_id = result.order_id
                logger.info(f"Broker order placed: #{result.order_id} {order.ticker}")
            else:
                logger.error(f"Broker order failed: {result.reason if result else 'unknown'}")
        except Exception as e:
            logger.error(f"Broker execution error: {e}")

    # ── Order Bundle ─────────────────────────────────────────

    def create_entry_bundle(
        self,
        ticker: str,
        quantity: int,
        entry_price: float,
        stop_loss_pct: float = 0.08,
        take_profit_tiers: Optional[List[Dict[str, Any]]] = None,
        time_stop_days: int = 90,
        entry_type: str = "limit",  # "limit" or "stop"
    ) -> Dict[str, Any]:
        """
        Create a complete entry order bundle:
        - Entry order (limit or stop)
        - Trailing stop loss
        - Take profit tiers
        - Time stop

        Returns dict with all order IDs.
        """
        # Entry
        if entry_type == "limit":
            entry_id = self.create_limit_entry(ticker, quantity, entry_price)
        else:
            entry_id = self.create_stop_entry(ticker, quantity, entry_price)

        # Trailing stop
        stop_id = self.create_trailing_stop(
            ticker, quantity, entry_price,
            distance_pct=stop_loss_pct,
            parent_order_id=entry_id,
        )

        # Take profit
        tp_ids = self.create_take_profit(
            ticker, quantity, entry_price,
            tiers=take_profit_tiers,
            parent_order_id=entry_id,
        )

        # Time stop
        time_id = self.create_time_stop(
            ticker, quantity, days=time_stop_days,
            parent_order_id=entry_id,
        )

        # Link children to entry
        entry = self.orders.get(entry_id)
        if entry:
            entry.child_orders = [stop_id] + tp_ids + [time_id]

        return {
            "entry_order_id": entry_id,
            "trailing_stop_id": stop_id,
            "take_profit_ids": tp_ids,
            "time_stop_id": time_id,
        }

    # ── Persistence ───────────────────────────────────────────

    def save(self) -> None:
        """Save all orders and history to disk."""
        self.orders_dir.mkdir(parents=True, exist_ok=True)

        # Save each order individually
        for order in self.orders.values():
            filepath = self.orders_dir / f"{order.id}.json"
            with open(filepath, 'w') as f:
                json.dump(order.to_dict(), f, indent=2, default=str)

        # Save history
        history_data = {
            oid: h.to_dict() for oid, h in self.history.items()
        }
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)

        logger.info(f"Saved {len(self.orders)} orders + history to {self.orders_dir}")

    def load(self) -> bool:
        """Load all orders and history from disk."""
        if not self.orders_dir.exists():
            return False

        loaded = 0
        for filepath in self.orders_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                order = ConditionalOrder.from_dict(data)
                self.orders[order.id] = order
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load order {filepath}: {e}")

        # Load history
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    history_data = json.load(f)
                for oid, hdata in history_data.items():
                    self.history[oid] = OrderHistory.from_dict(hdata)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

        logger.info(f"Loaded {loaded} orders + {len(self.history)} history records")
        return True

    # ── Query ─────────────────────────────────────────────────

    def get_orders_by_status(self, *statuses: OrderStatus) -> List[ConditionalOrder]:
        """Get orders filtered by status."""
        return [o for o in self.orders.values() if o.status in statuses]

    def get_orders_by_ticker(self, ticker: str) -> List[ConditionalOrder]:
        """Get all orders for a ticker."""
        return [o for o in self.orders.values() if o.ticker.upper() == ticker.upper()]

    def get_orders_by_type(self, order_type: OrderType) -> List[ConditionalOrder]:
        """Get orders by type."""
        return [o for o in self.orders.values() if o.order_type == order_type]

    def get_active_orders(self) -> List[ConditionalOrder]:
        """Get all active orders."""
        return self.get_orders_by_status(OrderStatus.ACTIVE, OrderStatus.CREATED)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all orders."""
        status_counts = {s.value: 0 for s in OrderStatus}
        type_counts: Dict[str, int] = {}
        for o in self.orders.values():
            status_counts[o.status.value] = status_counts.get(o.status.value, 0) + 1
            tn = o.order_type.name
            type_counts[tn] = type_counts.get(tn, 0) + 1

        total_fills = sum(h.total_filled for h in self.history.values())
        avg_slippage = (
            sum(h.total_slippage for h in self.history.values() if h.fills) /
            sum(1 for h in self.history.values() if h.fills)
            if any(h.fills for h in self.history.values()) else 0
        )

        return {
            "total_orders": len(self.orders),
            "by_status": status_counts,
            "by_type": type_counts,
            "active_count": status_counts.get("active", 0) + status_counts.get("created", 0),
            "triggered_count": status_counts.get("triggered", 0),
            "filled_count": status_counts.get("filled", 0),
            "cancelled_count": status_counts.get("cancelled", 0),
            "expired_count": status_counts.get("expired", 0),
            "total_fills": total_fills,
            "avg_slippage_pct": round(avg_slippage, 4),
        }

    # ── Helpers ───────────────────────────────────────────────

    def _gen_id(self, prefix: str, ticker: str) -> str:
        """Generate a unique order ID."""
        ts = int(time.time() * 1000000) % 100000000
        return f"{prefix}_{ticker}_{ts}"

    def _add_order(self, order: ConditionalOrder) -> str:
        """Add order to registry and save."""
        self.orders[order.id] = order
        logger.info(f"Created order {order.id} [{order.order_type.name}] {order.action} {order.quantity} {order.ticker}")
        return order.id
