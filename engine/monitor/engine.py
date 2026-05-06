#!/usr/bin/env python3
"""
VMAA 2.0 — Auto Monitoring Engine Orchestrator
================================================
Central orchestrator that ties together:
  - Price alerts (AlertManager)
  - Conditional orders (OrderManager)
  - Anomaly detection (AnomalyDetector)
  - Push notifications (Notifier)

Run modes:
  - ONESHOT:  Check once and exit
  - DAEMON:   Continuous loop at configurable intervals

Config: engine/monitor/config.json
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Setup path for VMAA imports
_vmaa_root = Path(__file__).resolve().parent.parent.parent
if str(_vmaa_root) not in sys.path:
    sys.path.insert(0, str(_vmaa_root))

from engine.monitor.alerts import (
    Alert, AlertCondition, AlertManager, AlertPriority, AlertStatus, AlertType,
)
from engine.monitor.anomaly import (
    Anomaly, AnomalyDetector, AnomalySeverity, AnomalyType,
)
from engine.monitor.notify import (
    Channel, MessageTemplate, Notifier, QuietHours,
)
from engine.monitor.orders import (
    ConditionalOrder, Fill, OrderHistory, OrderManager, OrderStatus, OrderType,
)

logger = logging.getLogger("vmaa.monitor.engine")


# ═══════════════════════════════════════════════════════════════════
# Enums & Config
# ═══════════════════════════════════════════════════════════════════

class RunMode(Enum):
    ONESHOT = "oneshot"
    DAEMON = "daemon"


@dataclass
class MonitorConfig:
    """Engine-level monitoring configuration."""
    mode: RunMode = RunMode.DAEMON
    check_interval_seconds: int = 300       # 5 minutes
    market_open_check: bool = False         # Only run during market hours
    log_level: str = "INFO"
    log_file: str = "engine/data/monitor.log"
    metrics_enabled: bool = True
    persist_state: bool = True
    state_file: str = "engine/data/monitor_state.json"

    # Module flags
    alerts_enabled: bool = True
    orders_enabled: bool = True
    anomaly_enabled: bool = True
    notify_enabled: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MonitorConfig":
        md = d.get("monitor", {})
        return cls(
            mode=RunMode(md.get("mode", "daemon")),
            check_interval_seconds=md.get("check_interval_seconds", 300),
            market_open_check=md.get("market_open_check", False),
            log_level=md.get("log_level", "INFO"),
            log_file=md.get("log_file", "engine/data/monitor.log"),
            metrics_enabled=md.get("metrics_enabled", True),
            persist_state=md.get("persist_state", True),
            state_file=md.get("state_file", "engine/data/monitor_state.json"),
            alerts_enabled=d.get("alerts", {}).get("enabled", True),
            orders_enabled=d.get("orders", {}).get("enabled", True),
            anomaly_enabled=d.get("anomaly", {}).get("enabled", True),
            notify_enabled=d.get("notify", {}).get("enabled", True),
        )


# ═══════════════════════════════════════════════════════════════════
# Monitor Engine
# ═══════════════════════════════════════════════════════════════════

class MonitorEngine:
    """
    Orchestrator for all monitoring subsystems.

    Usage:
        engine = MonitorEngine()
        engine.setup()                    # Load config, initialize modules
        engine.run_oneshot()              # Check once
        engine.run_daemon()               # Continuous loop (Ctrl+C to stop)

    Or use the unified interface:
        engine.run()  # Uses configured mode
    """

    def __init__(self, config_path: Optional[str] = None):
        # Resolve config path relative to this file
        if config_path is None:
            config_path = str(Path(__file__).resolve().parent / "config.json")
        self.config_path = Path(config_path)
        self.config: MonitorConfig = MonitorConfig()

        # Subsystems
        self.alert_manager: Optional[AlertManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.notifier: Optional[Notifier] = None

        # Runtime state
        self._running: bool = False
        self._stop_event = threading.Event()
        self._daemon_thread: Optional[threading.Thread] = None

        # Watchlist (tickers to monitor)
        self.watchlist: List[str] = []

        # Portfolio state (for anomaly detection)
        self.positions: List[Dict[str, Any]] = []
        self.pnl_history: List[float] = []

        # Metrics
        self.loop_count: int = 0
        self.start_time: Optional[float] = None
        self.last_check_time: Optional[float] = None
        self.total_alerts_triggered: int = 0
        self.total_anomalies_detected: int = 0
        self.total_orders_processed: int = 0

        # Telegram context (passed through from outer context)
        self.telegram_token: str = ""
        self.telegram_chat_id: str = ""

    # ── Setup ─────────────────────────────────────────────────

    def setup(self) -> None:
        """Load config and initialize all subsystems."""
        self._load_config()
        self._setup_logging()
        self._init_subsystems()
        self._load_state()
        self._setup_signal_handlers()
        logger.info(f"MonitorEngine ready — mode: {self.config.mode.value}, "
                    f"interval: {self.config.check_interval_seconds}s")

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            logger.warning(f"Config not found at {self.config_path}, using defaults")
            return

        with open(self.config_path) as f:
            raw = json.load(f)

        self.config = MonitorConfig.from_dict(raw)

        # Extract watchlist if present
        md = raw.get("monitor", {})
        self.watchlist = md.get("watchlist", [])

        # Telegram config
        self.telegram_token = md.get("telegram_token", "")
        self.telegram_chat_id = md.get("telegram_chat_id", "")

        logger.info(f"Loaded config: {self.config_path}")

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # File handler
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

        root_logger = logging.getLogger("vmaa.monitor")
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)

    def _init_subsystems(self) -> None:
        """Initialize all monitoring subsystems."""
        raw_config = self._load_raw_config()

        # --- Alert Manager ---
        if self.config.alerts_enabled:
            self.alert_manager = AlertManager(config=raw_config.get("alerts"))
            logger.info("AlertManager initialized")

        # --- Order Manager ---
        if self.config.orders_enabled:
            orders_cfg = raw_config.get("orders", {})
            self.order_manager = OrderManager(
                orders_dir=str(Path(orders_cfg.get("orders_dir", "engine/data/orders"))),
                history_file=str(Path(orders_cfg.get("history_file", "engine/data/orders/history.json"))),
                max_slippage_pct=orders_cfg.get("max_slippage_pct", 0.02),
                trailing_default_pct=orders_cfg.get("trailing_stop_default_pct", 0.08),
                trailing_default_distance=orders_cfg.get("trailing_stop_default_distance", 0.05),
                time_stop_default_days=orders_cfg.get("time_stop_default_days", 90),
                auto_execute=orders_cfg.get("auto_execute", False),
            )
            # Load existing orders
            self.order_manager.load()
            logger.info("OrderManager initialized")

        # --- Anomaly Detector ---
        if self.config.anomaly_enabled:
            self.anomaly_detector = AnomalyDetector(config=raw_config.get("anomaly"))
            logger.info("AnomalyDetector initialized")

        # --- Notifier ---
        if self.config.notify_enabled:
            notify_cfg = raw_config.get("notify", {})
            channels = []
            for ch in notify_cfg.get("channels", []):
                if ch == "telegram" and self.telegram_token:
                    channels.append(Channel.TELEGRAM)
                elif ch == "console":
                    channels.append(Channel.CONSOLE)
                elif ch == "file":
                    channels.append(Channel.FILE)

            self.notifier = Notifier(
                channels=channels or [Channel.CONSOLE, Channel.FILE],
                telegram_token=self.telegram_token or notify_cfg.get("telegram_token", ""),
                telegram_chat_id=self.telegram_chat_id or notify_cfg.get("telegram_chat_id", ""),
                file_log=notify_cfg.get("file_log", "engine/data/notifications.log"),
                rate_limit_per_hour=notify_cfg.get("rate_limit_per_hour", 60),
                quiet_hours=QuietHours(
                    **(notify_cfg.get("quiet_hours", {}))
                ),
                templates=notify_cfg.get("templates", {}),
            )

            # Also add a raw Telegram callback for direct messages when token is set
            if self.telegram_token and self.telegram_chat_id:
                self.notifier.add_callback(
                    "raw_telegram",
                    lambda msg, kw: self._raw_telegram_send(msg),
                )

            logger.info("Notifier initialized")

    def _load_raw_config(self) -> Dict[str, Any]:
        """Load raw config JSON dict."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {}

    def _load_state(self) -> None:
        """Restore persisted state from disk."""
        if not self.config.persist_state:
            return

        state_path = Path(self.config.state_file)
        if not state_path.exists():
            return

        try:
            with open(state_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load state: {e}")
            return

        # Restore watchlist
        saved_watchlist = state.get("watchlist", [])
        if saved_watchlist:
            self.watchlist = saved_watchlist

        # Restore metrics
        self.loop_count = state.get("loop_count", 0)
        self.total_alerts_triggered = state.get("total_alerts_triggered", 0)
        self.total_anomalies_detected = state.get("total_anomalies_detected", 0)
        self.total_orders_processed = state.get("total_orders_processed", 0)
        self.pnl_history = state.get("pnl_history", [])

        logger.info(f"Loaded state: loop={self.loop_count}, watchlist={len(self.watchlist)} tickers")

    def _save_state(self) -> None:
        """Persist current state to disk."""
        if not self.config.persist_state:
            return

        state = {
            "watchlist": self.watchlist,
            "loop_count": self.loop_count,
            "total_alerts_triggered": self.total_alerts_triggered,
            "total_anomalies_detected": self.total_anomalies_detected,
            "total_orders_processed": self.total_orders_processed,
            "pnl_history": self.pnl_history[-120:],  # Keep last 120 data points
            "last_check_time": self.last_check_time,
            "saved_at": datetime.now().isoformat(),
        }

        state_path = Path(self.config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        def handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._running = False
            self._stop_event.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    # ── Watchlist Management ──────────────────────────────────

    def set_watchlist(self, tickers: List[str]) -> None:
        """Set the tickers to monitor."""
        self.watchlist = [t.upper().strip() for t in tickers if t]
        logger.info(f"Watchlist set: {len(self.watchlist)} tickers")
        if self.notifier:
            self.notifier.notify(
                "PRICE_LEVEL",
                priority="LOW",
                ticker=f"WATCHLIST ({len(self.watchlist)})",
                current_price=0,
                direction="",
                target_price=0,
            )

    def add_to_watchlist(self, tickers: List[str]) -> None:
        """Add tickers to the watchlist."""
        for t in tickers:
            t = t.upper().strip()
            if t and t not in self.watchlist:
                self.watchlist.append(t)
        logger.debug(f"Watchlist: {len(self.watchlist)} tickers")

    def remove_from_watchlist(self, tickers: List[str]) -> None:
        """Remove tickers from the watchlist."""
        upper = {t.upper().strip() for t in tickers}
        self.watchlist = [t for t in self.watchlist if t not in upper]

    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update portfolio positions for anomaly detection."""
        self.positions = positions

        # Track P&L history
        total_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
        self.pnl_history.append(total_pnl)
        if len(self.pnl_history) > 120:
            self.pnl_history = self.pnl_history[-120:]

    # ── Check Cycle ───────────────────────────────────────────

    def check_alerts(self) -> List[Alert]:
        """Run alert checks. Returns triggered alerts."""
        if not self.alert_manager:
            return []

        triggered = self.alert_manager.check_all()
        if triggered:
            self.total_alerts_triggered += len(triggered)
            logger.info(f"Alerts triggered: {len(triggered)}")

            # Notify
            if self.notifier:
                for alert in triggered:
                    self.notifier.notify_alert(alert)

        return triggered

    def process_orders(self) -> List[Dict[str, Any]]:
        """Check and process conditional orders."""
        if not self.order_manager:
            return []

        events = self.order_manager.check_orders()
        if events:
            self.total_orders_processed += len(events)

            # Notify
            if self.notifier:
                for event in events:
                    event_type = event.get("type", "triggered")
                    order = event.get("order")
                    if order:
                        self.notifier.notify_order_event(
                            event_type,
                            order_id=order.id,
                            order_type=order.order_type.name.replace("_", " ").title(),
                            action=order.action,
                            quantity=order.quantity,
                            filled_quantity=getattr(order, 'remaining_quantity', 0),
                            ticker=order.ticker,
                            trigger_price=event.get('price', 0),
                            filled_price=event.get('price', 0),
                        )

            logger.info(f"Order events: {len(events)}")

        return events

    def scan_anomalies(self) -> List[Anomaly]:
        """Run anomaly detection scans."""
        if not self.anomaly_detector:
            return []

        anomalies = self.anomaly_detector.scan_all(
            watchlist=self.watchlist,
            positions=self.positions,
            pnl_history=self.pnl_history,
        )

        if anomalies:
            self.total_anomalies_detected += len(anomalies)

            # Notify
            if self.notifier:
                for anomaly in anomalies:
                    self.notifier.notify_anomaly(anomaly)

            logger.info(f"Anomalies detected: {len(anomalies)}")

        return anomalies

    def notify_cycle(self, alerts: List, orders: List, anomalies: List) -> None:
        """Send cycle summary if there are any events."""
        if not self.notifier:
            return

        if not any([alerts, orders, anomalies]):
            return  # Quiet cycle, skip notification

        self.notifier.send_summary(
            alerts=alerts,
            anomalies=anomalies,
            order_events=orders if orders else None,
        )

    # ── Run Modes ─────────────────────────────────────────────

    def run_oneshot(self) -> Dict[str, Any]:
        """
        Execute a single monitoring cycle.
        Returns a summary dict of what happened.
        """
        if self.start_time is None:
            self.start_time = time.time()

        logger.info(f"=== ONESHOT check #{self.loop_count + 1} ===")
        self.last_check_time = time.time()

        results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "mode": "oneshot",
            "alerts": {"triggered": [], "total": 0},
            "orders": {"events": [], "total": 0},
            "anomalies": {"detected": [], "total": 0},
        }

        # 1. Alerts
        if self.config.alerts_enabled:
            triggered_alerts = self.check_alerts()
            results["alerts"] = {
                "triggered": [a.to_dict() for a in triggered_alerts],
                "total": len(triggered_alerts),
            }

        # 2. Orders
        if self.config.orders_enabled:
            order_events = self.process_orders()
            results["orders"] = {
                "events": order_events,
                "total": len(order_events),
            }

        # 3. Anomalies
        if self.config.anomaly_enabled:
            detected_anomalies = self.scan_anomalies()
            results["anomalies"] = {
                "detected": [a.to_dict() for a in detected_anomalies],
                "total": len(detected_anomalies),
            }

        # 4. Summary notification
        if self.config.notify_enabled:
            self.notify_cycle(
                results["alerts"]["triggered"],
                results["orders"]["events"],
                results["anomalies"]["detected"],
            )

        # Persist
        self.loop_count += 1
        if self.config.persist_state:
            self._save_state()

        # Save orders
        if self.order_manager and self.config.orders_enabled:
            self.order_manager.save()

        elapsed = time.time() - (self.last_check_time or time.time())
        logger.info(f"ONESHOT complete in {elapsed:.2f}s")
        return results

    def run_daemon(self) -> None:
        """Run continuous monitoring loop until stopped."""
        if self._running:
            logger.warning("Daemon already running")
            return

        self._running = True
        self.start_time = time.time()
        logger.info(
            f"DAQMON started — watchlist: {len(self.watchlist)} tickers, "
            f"interval: {self.config.check_interval_seconds}s"
        )

        # Startup notification
        if self.notifier:
            self.notifier.notify(
                "PRICE_LEVEL",
                priority="LOW",
                force=True,
                ticker="VMAA_MONITOR",
                direction="STARTED",
                target_price=0,
                current_price=0,
            )

        try:
            while self._running and not self._stop_event.is_set():
                cycle_start = time.time()

                try:
                    results = self.run_oneshot()

                    # Print brief status
                    summary = (
                        f"Cycle #{self.loop_count}: "
                        f"{results['alerts']['total']} alerts, "
                        f"{results['orders']['total']} order events, "
                        f"{results['anomalies']['total']} anomalies"
                    )
                    logger.info(summary)

                except Exception as e:
                    logger.error(f"Monitor cycle error: {e}", exc_info=True)

                # Wait for next cycle (with early exit for shutdown)
                elapsed = time.time() - cycle_start
                wait_time = max(0, self.config.check_interval_seconds - elapsed)
                logger.debug(f"Next check in {wait_time:.0f}s")

                self._stop_event.wait(timeout=wait_time)

        except KeyboardInterrupt:
            logger.info("Daemon interrupted by user")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Graceful shutdown: save state, close connections."""
        self._running = False
        logger.info("Shutting down MonitorEngine...")

        if self.config.persist_state:
            self._save_state()

        if self.order_manager and self.config.orders_enabled:
            self.order_manager.save()

        elapsed = time.time() - (self.start_time or time.time())
        logger.info(
            f"MonitorEngine stopped. Uptime: {elapsed/3600:.1f}h, "
            f"Loops: {self.loop_count}, "
            f"Alerts: {self.total_alerts_triggered}, "
            f"Anomalies: {self.total_anomalies_detected}, "
            f"Orders: {self.total_orders_processed}"
        )

        if self.notifier:
            self.notifier.notify(
                "PRICE_LEVEL",
                priority="LOW",
                force=True,
                ticker="VMAA_MONITOR",
                direction="STOPPED",
                target_price=self.loop_count,
                current_price=self.total_alerts_triggered,
            )

    # ── Unified Interface ─────────────────────────────────────

    def run(self) -> Any:
        """Run the engine in the configured mode (oneshot or daemon)."""
        if not any([self.alert_manager, self.anomaly_detector, self.order_manager]):
            self.setup()

        if self.config.mode == RunMode.ONESHOT:
            return self.run_oneshot()
        elif self.config.mode == RunMode.DAEMON:
            return self.run_daemon()
        else:
            logger.error(f"Unknown run mode: {self.config.mode}")
            return None

    def run_async(self) -> threading.Thread:
        """Start daemon in a background thread. Returns the thread."""
        if not any([self.alert_manager, self.anomaly_detector, self.order_manager]):
            self.setup()

        self._daemon_thread = threading.Thread(
            target=self.run_daemon,
            name="vmaa-monitor-daemon",
            daemon=True,
        )
        self._daemon_thread.start()
        logger.info("MonitorEngine daemon thread started")
        return self._daemon_thread

    # ── Status & Metrics ──────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return comprehensive engine status."""
        uptime = time.time() - (self.start_time or time.time()) if self.start_time else 0

        status = {
            "running": self._running,
            "mode": self.config.mode.value,
            "uptime_seconds": round(uptime, 1),
            "loop_count": self.loop_count,
            "interval_seconds": self.config.check_interval_seconds,
            "watchlist_size": len(self.watchlist),
            "watchlist_sample": self.watchlist[:10],
            "total_alerts_triggered": self.total_alerts_triggered,
            "total_anomalies_detected": self.total_anomalies_detected,
            "total_orders_processed": self.total_orders_processed,
            "last_check_time": (
                datetime.fromtimestamp(self.last_check_time).isoformat()
                if self.last_check_time else None
            ),
        }

        # Subsystem summaries
        if self.alert_manager:
            status["alerts"] = self.alert_manager.summary()
        if self.order_manager:
            status["orders"] = self.order_manager.summary()
        if self.anomaly_detector:
            status["anomalies"] = self.anomaly_detector.summary()
        if self.notifier:
            status["notifications"] = self.notifier.stats()

        return status

    def print_status(self) -> None:
        """Pretty-print engine status."""
        s = self.status()
        print("\n" + "=" * 60)
        print("  VMAA Monitor Engine Status")
        print("=" * 60)
        print(f"  Mode:        {s['mode'].upper()}")
        print(f"  Running:     {'✅' if s['running'] else '⏸️'}")
        print(f"  Uptime:      {s['uptime_seconds']/3600:.1f}h ({s['loop_count']} cycles)")
        print(f"  Interval:    {s['interval_seconds']}s")
        print(f"  Watchlist:   {s['watchlist_size']} tickers")
        print(f"  Alerts:      {s['total_alerts_triggered']} triggered")
        print(f"  Anomalies:   {s['total_anomalies_detected']} detected")
        print(f"  Orders:      {s['total_orders_processed']} processed")

        if "alerts" in s:
            ac = s["alerts"]
            print(f"\n  Alerts:      {ac['total_alerts']} total, {ac['active_count']} active, {ac['triggered_count']} triggered")
        if "orders" in s:
            oc = s["orders"]
            print(f"  Orders:      {oc['total_orders']} total, {oc['active_count']} active, "
                  f"{oc['filled_count']} filled, {oc['cancelled_count']} cancelled")
        if "anomalies" in s:
            an = s["anomalies"]
            print(f"  Anomalies:   {an['total_detected']} total, {an['unresolved']} unresolved")
        if "notifications" in s:
            ns = s["notifications"]
            print(f"  Notify:      {ns['total_sent']} sent, {ns['suppressed']} suppressed ({ns['rate_limit']})")

        print("=" * 60)

    # ── Telegram Helpers ──────────────────────────────────────

    def _raw_telegram_send(self, message: str) -> None:
        """Direct Telegram send (used as callback when token is configured)."""
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            import urllib.request
            import urllib.parse

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            formatted = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            data = urllib.parse.urlencode({
                "chat_id": self.telegram_chat_id,
                "text": formatted,
                "parse_mode": "HTML",
            }).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=10)
        except Exception as e:
            logger.debug(f"Raw Telegram send failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# Quick Test / CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VMAA Auto Monitoring Engine")
    parser.add_argument("--mode", choices=["oneshot", "daemon"], default="oneshot",
                        help="Run mode: oneshot or daemon")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json")
    parser.add_argument("--watchlist", type=str, nargs="*", default=[],
                        help="Tickers to monitor")
    parser.add_argument("--status", action="store_true",
                        help="Print engine status and exit")
    parser.add_argument("--test-alert", type=str, default=None,
                        help="Create a test price alert for TICKER")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    engine = MonitorEngine(config_path=args.config)
    engine.setup()

    # Override mode from CLI
    if args.mode:
        engine.config.mode = RunMode(args.mode)

    # Set watchlist
    if args.watchlist:
        engine.set_watchlist(args.watchlist)

    # Test alert
    if args.test_alert:
        ticker = args.test_alert.upper()
        engine.add_to_watchlist([ticker])
        if engine.alert_manager:
            engine.alert_manager.add_price_alert(
                f"Test_{ticker}",
                ticker,
                target_price=200.0,
                direction="above",
                priority=AlertPriority.HIGH,
            )
            print(f"Created test alert for {ticker}")
        engine.run_oneshot()
        sys.exit(0)

    # Status
    if args.status:
        engine.print_status()
        sys.exit(0)

    # Run
    results = engine.run()
    if isinstance(results, dict):
        print(json.dumps(results, indent=2, default=str))
