#!/usr/bin/env python3
"""
VMAA 2.0 — Auto Monitoring Engine
==================================
Real-time monitoring, alerting, conditional orders, anomaly detection, and
push notifications for the VMAA trading system.

Components:
  - alerts:   Price alerts (level, change, volume, breakout, technical)
  - orders:   Conditional order templates with lifecycle management
  - anomaly:  Market, portfolio, and data anomaly detection
  - notify:   Multi-channel push notifications (Telegram, console, file)
  - engine:   Orchestrator with ONESHOT / DAEMON run modes
"""
from __future__ import annotations

from engine.monitor.alerts import (
    Alert, AlertCondition, AlertManager, AlertPriority, AlertStatus, AlertType,
)
from engine.monitor.anomaly import (
    Anomaly, AnomalyDetector, AnomalySeverity, AnomalyType,
)
from engine.monitor.engine import MonitorEngine, MonitorConfig, RunMode
from engine.monitor.notify import (
    Channel, MessageTemplate, Notifier, QuietHours,
)
from engine.monitor.orders import (
    ConditionalOrder, Fill, OrderHistory, OrderManager, OrderStatus, OrderType,
)

__all__ = [
    # Alerts
    "Alert", "AlertCondition", "AlertManager", "AlertPriority", "AlertStatus", "AlertType",
    # Orders
    "ConditionalOrder", "Fill", "OrderHistory", "OrderManager", "OrderStatus", "OrderType",
    # Anomaly
    "Anomaly", "AnomalyDetector", "AnomalySeverity", "AnomalyType",
    # Notify
    "Channel", "MessageTemplate", "Notifier", "QuietHours",
    # Engine
    "MonitorEngine", "MonitorConfig", "RunMode",
]
