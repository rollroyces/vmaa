#!/usr/bin/env python3
"""
VMAA 2.0 — Price Alerts Engine
================================
Multi-type alerting system with:
  - Price level alerts (target / stop loss)
  - Daily % change threshold alerts
  - Volume spike detection
  - Breakout (MA50/MA200 crossover)
  - Technical indicators (MACD, RSI)
  - AND/OR condition combinations
  - Cooldown to prevent alert storms
  - Priority levels (CRITICAL/HIGH/MEDIUM/LOW)
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("vmaa.monitor.alerts")


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class AlertType(Enum):
    PRICE_LEVEL = auto()         # Stock hits target price / stop loss
    CHANGE_THRESHOLD = auto()    # Daily % change beyond limit
    VOLUME_SPIKE = auto()        # Volume > Nx 20d average
    BREAKOUT = auto()            # Price breaks above/below MA50/MA200
    TECHNICAL_CROSSOVER = auto() # MACD crossover, RSI oversold/overbought


class AlertPriority(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class AlertStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AlertCondition:
    """A single condition within an alert rule."""
    field: str                      # e.g. "price", "volume", "rsi", "macd_signal"
    operator: str                   # ">", "<", ">=", "<=", "==", "cross_above", "cross_below"
    value: float                    # Threshold value
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "operator": self.operator,
                "value": self.value, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlertCondition":
        return cls(field=d["field"], operator=d["operator"],
                   value=d["value"], enabled=d.get("enabled", True))


@dataclass
class Alert:
    """A configured alert rule."""
    id: str
    name: str
    alert_type: AlertType
    priority: AlertPriority = AlertPriority.MEDIUM
    ticker: str = ""
    tickers: List[str] = field(default_factory=list)

    # Conditions: list of lists — inner lists are AND, outer list is OR
    # e.g. [[a, b], [c]] → (a AND b) OR (c)
    conditions: List[List[AlertCondition]] = field(default_factory=list)

    # Trigger parameters
    trigger_count: int = 1          # Fire after N consecutive violations
    cooldown_hours: float = 4.0     # Min hours between repeated alerts
    enabled: bool = True

    # Runtime state
    status: AlertStatus = AlertStatus.ACTIVE
    last_triggered: Optional[float] = None  # epoch time
    trigger_history: List[float] = field(default_factory=list)
    trigger_count_current: int = 0

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name,
            "alert_type": self.alert_type.name,
            "priority": self.priority.name,
            "ticker": self.ticker,
            "tickers": self.tickers,
            "conditions": [
                [c.to_dict() for c in group]
                for group in self.conditions
            ],
            "trigger_count": self.trigger_count,
            "cooldown_hours": self.cooldown_hours,
            "enabled": self.enabled,
            "status": self.status.value,
            "last_triggered": self.last_triggered,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Alert":
        return cls(
            id=d["id"], name=d["name"],
            alert_type=AlertType[d["alert_type"]],
            priority=AlertPriority[d.get("priority", "MEDIUM")],
            ticker=d.get("ticker", ""),
            tickers=d.get("tickers", []),
            conditions=[
                [AlertCondition.from_dict(c) for c in group]
                for group in d.get("conditions", [])
            ],
            trigger_count=d.get("trigger_count", 1),
            cooldown_hours=d.get("cooldown_hours", 4.0),
            enabled=d.get("enabled", True),
            status=AlertStatus(d.get("status", "active")),
            last_triggered=d.get("last_triggered"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            notes=d.get("notes", ""),
        )


# ═══════════════════════════════════════════════════════════════════
# Alert Manager
# ═══════════════════════════════════════════════════════════════════

class AlertManager:
    """
    Manages alert definitions, evaluation, cooldowns, and history.

    Usage:
        mgr = AlertManager(cooldown_hours=4)
        mgr.add_price_alert("AAPL_stop", "AAPL", 145.0, "below", priority=AlertPriority.HIGH)
        triggered = mgr.check_all()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.alerts: Dict[str, Alert] = {}
        self.cooldown_default_hours: float = 4.0
        self.cooldown_by_priority: Dict[str, float] = {
            "CRITICAL": 0.5, "HIGH": 2.0, "MEDIUM": 6.0, "LOW": 24.0,
        }
        self.volume_spike_mult: float = 3.0
        self.volume_lookback: int = 20
        self.change_threshold_pct: float = 5.0
        self.price_buffer: float = 0.002
        self.rsi_period: int = 14
        self.rsi_overbought: float = 70
        self.rsi_oversold: float = 30
        self.ma_periods: List[int] = [50, 200]

        if config:
            self._apply_config(config)

        # Data cache: {ticker: DataFrame}
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl: float = 60.0  # seconds

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration from dict (usually config.json['alerts'])."""
        self.cooldown_default_hours = config.get("cooldown_default_hours", self.cooldown_default_hours)
        self.cooldown_by_priority = config.get("cooldown_by_priority", self.cooldown_by_priority)
        self.volume_spike_mult = config.get("volume_spike_multiplier", self.volume_spike_mult)
        self.volume_lookback = config.get("volume_lookback_days", self.volume_lookback)
        self.change_threshold_pct = config.get("change_threshold_pct", self.change_threshold_pct)
        self.price_buffer = config.get("price_level_buffer_pct", self.price_buffer)
        self.rsi_period = config.get("rsi_period", self.rsi_period)
        self.rsi_overbought = config.get("rsi_overbought", self.rsi_overbought)
        self.rsi_oversold = config.get("rsi_oversold", self.rsi_oversold)
        self.ma_periods = config.get("ma_periods", self.ma_periods)

    # ── Alert CRUD ────────────────────────────────────────────

    def add_alert(self, alert: Alert) -> str:
        """Add or update an alert. Returns alert ID."""
        if not alert.id:
            alert.id = hashlib.md5(
                f"{alert.alert_type.name}:{alert.ticker}:{time.time()}".encode()
            ).hexdigest()[:12]
        alert.created_at = alert.created_at or datetime.now().isoformat()
        alert.updated_at = datetime.now().isoformat()
        self.alerts[alert.id] = alert
        logger.info(f"Alert {alert.id} [{alert.alert_type.name}] {alert.name} — {alert.priority.name}")
        return alert.id

    def add_price_alert(
        self,
        name: str,
        ticker: str,
        target_price: float,
        direction: str = "above",  # "above" or "below"
        priority: AlertPriority = AlertPriority.MEDIUM,
        cooldown_hours: Optional[float] = None,
    ) -> str:
        """Convenience: create a simple price level alert."""
        operator = ">" if direction == "above" else "<"
        condition = AlertCondition(field="price", operator=operator, value=target_price)
        alert = Alert(
            id=f"price_{ticker}_{direction}_{int(target_price*100)}",
            name=name,
            alert_type=AlertType.PRICE_LEVEL,
            priority=priority,
            ticker=ticker,
            conditions=[[condition]],
            cooldown_hours=cooldown_hours or self.cooldown_default_hours,
        )
        return self.add_alert(alert)

    def add_change_alert(
        self,
        name: str,
        ticker: str,
        threshold_pct: float = 5.0,
        direction: str = "either",  # "up", "down", or "either"
        priority: AlertPriority = AlertPriority.MEDIUM,
    ) -> str:
        """Convenience: create a daily % change alert."""
        conditions = []
        if direction in ("up", "either"):
            conditions.append([AlertCondition(
                field="daily_change_pct", operator=">=", value=threshold_pct
            )])
        if direction in ("down", "either"):
            conditions.append([AlertCondition(
                field="daily_change_pct", operator="<=", value=-threshold_pct
            )])
        return self.add_alert(Alert(
            id=f"change_{ticker}_{threshold_pct}pct",
            name=name,
            alert_type=AlertType.CHANGE_THRESHOLD,
            priority=priority,
            ticker=ticker,
            conditions=conditions,
        ))

    def add_volume_alert(
        self,
        name: str,
        ticker: str,
        multiplier: Optional[float] = None,
        priority: AlertPriority = AlertPriority.LOW,
    ) -> str:
        """Convenience: create a volume spike alert."""
        mult = multiplier or self.volume_spike_mult
        return self.add_alert(Alert(
            id=f"volume_{ticker}_{mult}x",
            name=name,
            alert_type=AlertType.VOLUME_SPIKE,
            priority=priority,
            ticker=ticker,
            conditions=[[AlertCondition(
                field="volume_ratio", operator=">=", value=mult
            )]],
        ))

    def add_breakout_alert(
        self,
        name: str,
        ticker: str,
        ma_period: int = 50,
        direction: str = "above",  # "above" or "below"
        priority: AlertPriority = AlertPriority.HIGH,
    ) -> str:
        """Convenience: create an MA breakout alert."""
        field = f"ma_{ma_period}"
        operator = "cross_above" if direction == "above" else "cross_below"
        return self.add_alert(Alert(
            id=f"breakout_{ticker}_{ma_period}ma_{direction}",
            name=name,
            alert_type=AlertType.BREAKOUT,
            priority=priority,
            ticker=ticker,
            conditions=[[AlertCondition(
                field=field, operator=operator, value=0.0  # value unused for crossover
            )]],
        ))

    def add_technical_alert(
        self,
        name: str,
        ticker: str,
        indicator: str = "rsi",     # "rsi", "macd_cross"
        signal_type: str = "oversold",  # "oversold", "overbought", "bullish_cross", "bearish_cross"
        priority: AlertPriority = AlertPriority.MEDIUM,
    ) -> str:
        """Convenience: create a technical indicator alert."""
        if indicator == "rsi":
            field = "rsi"
            operator = "<=" if signal_type == "oversold" else ">="
            value = self.rsi_oversold if signal_type == "oversold" else self.rsi_overbought
        elif indicator == "macd_cross":
            field = "macd_signal"
            operator = "cross_above" if signal_type == "bullish_cross" else "cross_below"
            value = 0.0
        else:
            raise ValueError(f"Unknown indicator: {indicator}")

        return self.add_alert(Alert(
            id=f"tech_{ticker}_{indicator}_{signal_type}",
            name=name,
            alert_type=AlertType.TECHNICAL_CROSSOVER,
            priority=priority,
            ticker=ticker,
            conditions=[[AlertCondition(field=field, operator=operator, value=value)]],
        ))

    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert by ID."""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            return True
        return False

    def enable_alert(self, alert_id: str, enabled: bool = True) -> bool:
        """Enable or disable an alert."""
        alert = self.alerts.get(alert_id)
        if alert:
            alert.enabled = enabled
            if enabled and alert.status == AlertStatus.DISABLED:
                alert.status = AlertStatus.ACTIVE
            elif not enabled:
                alert.status = AlertStatus.DISABLED
            return True
        return False

    # ── Data Fetching ─────────────────────────────────────────

    def _get_data(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
        """Fetch OHLCV data with short-term caching."""
        now = time.time()
        if (ticker in self._data_cache
                and (now - self._cache_time.get(ticker, 0)) < self._cache_ttl):
            return self._data_cache[ticker]

        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period)
            if df.empty:
                logger.debug(f"No data for {ticker}")
                return pd.DataFrame()
            self._data_cache[ticker] = df
            self._cache_time[ticker] = now
            return df
        except Exception as e:
            logger.debug(f"Data fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    def _get_latest_price(self, ticker: str) -> Optional[float]:
        """Get most recent close price."""
        df = self._get_data(ticker, "5d")
        if df.empty:
            return None
        return float(df['Close'].iloc[-1])

    # ── Condition Evaluation ──────────────────────────────────

    def _evaluate_condition(self, condition: AlertCondition, data: pd.DataFrame) -> bool:
        """Evaluate a single condition against market data."""
        field = condition.field
        operator = condition.operator
        threshold = condition.value

        if data.empty or len(data) < 2:
            return False

        try:
            if field == "price":
                value = float(data['Close'].iloc[-1])
                return self._compare(value, operator, threshold)

            elif field == "daily_change_pct":
                if len(data) < 2:
                    return False
                prev_close = float(data['Close'].iloc[-2])
                curr_close = float(data['Close'].iloc[-1])
                if prev_close == 0:
                    return False
                value = (curr_close - prev_close) / prev_close * 100
                return self._compare(value, operator, threshold)

            elif field == "volume_ratio":
                if len(data) < self.volume_lookback + 1:
                    return False
                curr_vol = float(data['Volume'].iloc[-1])
                avg_vol = float(data['Volume'].iloc[-(self.volume_lookback+1):-1].mean())
                if avg_vol == 0:
                    return False
                ratio = curr_vol / avg_vol
                return self._compare(ratio, operator, threshold)

            elif field.startswith("ma_"):
                period = int(field.split("_")[1])
                if len(data) < period + 1:
                    return False
                ma = float(data['Close'].rolling(period).mean().iloc[-1])
                price = float(data['Close'].iloc[-1])

                if operator in ("cross_above", "cross_below"):
                    prev_ma = float(data['Close'].rolling(period).mean().iloc[-2])
                    prev_price = float(data['Close'].iloc[-2])
                    if operator == "cross_above":
                        return prev_price <= prev_ma and price > ma
                    else:
                        return prev_price >= prev_ma and price < ma
                return self._compare(price, operator, ma)

            elif field == "rsi":
                value = self._compute_rsi(data, self.rsi_period)
                return self._compare(value, operator, threshold)

            elif field == "macd_signal":
                macd, sig = self._compute_macd(data)
                if macd is None or sig is None:
                    return False
                if operator in ("cross_above", "cross_below"):
                    macd_prev, sig_prev = self._compute_macd(data.iloc[:-1])
                    if macd_prev is None or sig_prev is None:
                        return False
                    if operator == "cross_above":
                        return macd_prev <= sig_prev and macd > sig
                    else:
                        return macd_prev >= sig_prev and macd < sig
                value = macd - sig
                return self._compare(value, operator, threshold)

            return False
        except Exception as e:
            logger.debug(f"Condition eval error ({field}): {e}")
            return False

    def _evaluate_condition_group(
        self, conditions: List[AlertCondition], data: pd.DataFrame
    ) -> bool:
        """Evaluate AND group: all conditions must be true."""
        return all(
            self._evaluate_condition(c, data)
            for c in conditions
            if c.enabled
        )

    def evaluate_alert(self, alert: Alert) -> bool:
        """Evaluate an alert's conditions. Returns True if triggered."""
        if not alert.enabled or alert.status != AlertStatus.ACTIVE:
            return False

        # Check cooldown
        if alert.last_triggered:
            cd_hours = alert.cooldown_hours
            elapsed = (time.time() - alert.last_triggered) / 3600
            if elapsed < cd_hours:
                return False

        tickers = [alert.ticker] if alert.ticker else alert.tickers
        if not tickers:
            logger.warning(f"Alert {alert.id} has no tickers")
            return False

        for ticker in tickers:
            data = self._get_data(ticker, "6mo")
            if data.empty:
                continue

            # Evaluate conditions as DNF: (A AND B) OR (C AND D)
            triggered = False
            if not alert.conditions:
                logger.warning(f"Alert {alert.id} has no conditions")
                continue

            for and_group in alert.conditions:
                if self._evaluate_condition_group(and_group, data):
                    triggered = True
                    break

            if triggered:
                # Track consecutive triggers
                alert.trigger_count_current += 1
                logger.info(
                    f"Alert {alert.id} [{alert.alert_type.name}] {alert.name} "
                    f"triggered (count: {alert.trigger_count_current}/{alert.trigger_count})"
                )

                if alert.trigger_count_current >= alert.trigger_count:
                    alert.status = AlertStatus.TRIGGERED
                    alert.last_triggered = time.time()
                    alert.trigger_history.append(time.time())
                    alert.trigger_count_current = 0
                    return True
                return False

        # Reset trigger count if conditions not met
        alert.trigger_count_current = 0
        return False

    # ── Batch Checking ────────────────────────────────────────

    def check_all(self) -> List[Alert]:
        """Evaluate all active alerts. Returns list of triggered alerts."""
        triggered: List[Alert] = []
        for alert in self.alerts.values():
            if alert.status != AlertStatus.ACTIVE or not alert.enabled:
                continue
            if self.evaluate_alert(alert):
                triggered.append(alert)
        return triggered

    def reset_cooldown(self, alert_id: str) -> bool:
        """Manually reset an alert's cooldown."""
        alert = self.alerts.get(alert_id)
        if alert:
            alert.status = AlertStatus.ACTIVE
            alert.last_triggered = None
            alert.trigger_count_current = 0
            logger.info(f"Reset cooldown for alert {alert_id}")
            return True
        return False

    # ── Persistence ───────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alerts": {aid: a.to_dict() for aid, a in self.alerts.items()},
            "config": {
                "cooldown_default_hours": self.cooldown_default_hours,
                "cooldown_by_priority": self.cooldown_by_priority,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlertManager":
        mgr = cls()
        for alert_data in d.get("alerts", {}).values():
            alert = Alert.from_dict(alert_data)
            mgr.alerts[alert.id] = alert
        cfg = d.get("config", {})
        if cfg:
            mgr._apply_config(cfg)
        return mgr

    def save(self, filepath: str) -> None:
        """Save alerts to JSON file."""
        import json
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved {len(self.alerts)} alerts to {filepath}")

    def load(self, filepath: str) -> bool:
        """Load alerts from JSON file."""
        import json
        from pathlib import Path
        if not Path(filepath).exists():
            return False
        with open(filepath) as f:
            d = json.load(f)
        restored = AlertManager.from_dict(d)
        self.alerts = restored.alerts
        self._apply_config(d.get("config", {}))
        logger.info(f"Loaded {len(self.alerts)} alerts from {filepath}")
        return True

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _compare(value: float, operator: str, threshold: float) -> bool:
        ops = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: abs(v - t) < 1e-8,
        }
        return ops.get(operator, lambda v, t: False)(value, threshold)

    def _compute_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Compute RSI for the most recent bar."""
        if len(data) < period + 1:
            return 50.0
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _compute_macd(self, data: pd.DataFrame, fast: int = 12,
                       slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float]]:
        """Compute MACD line and signal line for the most recent bar."""
        if len(data) < slow + signal:
            return None, None
        close = data['Close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all alerts."""
        counts = {t.name: 0 for t in AlertType}
        status_counts = {s.value: 0 for s in AlertStatus}
        for a in self.alerts.values():
            counts[a.alert_type.name] = counts.get(a.alert_type.name, 0) + 1
            status_counts[a.status.value] = status_counts.get(a.status.value, 0) + 1
        return {
            "total_alerts": len(self.alerts),
            "by_type": counts,
            "by_status": status_counts,
            "active_count": status_counts.get("active", 0),
            "triggered_count": status_counts.get("triggered", 0),
        }
