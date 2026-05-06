#!/usr/bin/env python3
"""
VMAA 2.0 — Notification Engine
================================
Multi-channel push notifications with:
  - Telegram: formatted message delivery
  - Console: stdout/stderr logging
  - File: write to notification log
  - Rate limiting: max N notifications per hour
  - Scheduling: quiet hours with priority floor
  - Async delivery (non-blocking via threading)
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger("vmaa.monitor.notify")


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class Channel(Enum):
    TELEGRAM = auto()
    CONSOLE = auto()
    FILE = auto()


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class QuietHours:
    """Defines quiet hours when non-critical notifications are suppressed."""
    enabled: bool = False
    start_utc: str = "22:00"      # HH:MM UTC
    end_utc: str = "06:00"        # HH:MM UTC
    weekend_silence: bool = True
    quiet_priority_floor: str = "HIGH"  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"

    _priority_levels: Dict[str, int] = field(default_factory=lambda: {
        "CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1,
    })

    def is_quiet(self, priority: str = "MEDIUM") -> bool:
        """Check if we're in quiet hours for the given priority."""
        if not self.enabled:
            return False

        # Weekend check
        if self.weekend_silence:
            now = datetime.utcnow()
            if now.weekday() >= 5:  # Sat=5, Sun=6
                return True

        # Priority check — only block if priority is below floor
        floor = self._priority_levels.get(self.quiet_priority_floor, 3)
        pri = self._priority_levels.get(priority, 2)
        if pri >= floor:
            return False  # High-priority always through

        # Time window check
        now = datetime.utcnow()
        current_time = now.time()

        start_h, start_m = map(int, self.start_utc.split(":"))
        end_h, end_m = map(int, self.end_utc.split(":"))

        start = dtime(start_h, start_m)
        end = dtime(end_h, end_m)

        if start <= end:
            return start <= current_time <= end
        else:
            # Wraps midnight: e.g., 22:00 - 06:00
            return current_time >= start or current_time <= end


@dataclass
class MessageTemplate:
    """Format string for a notification type."""
    name: str
    template: str

    def format(self, **kwargs: Any) -> str:
        """Format the template with provided kwargs."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Template '{self.name}' missing key: {e}")
            # Fallback: show raw kwargs
            return f"[{self.name}] {json.dumps(kwargs, default=str)}"


# ═══════════════════════════════════════════════════════════════════
# Notification Engine
# ═══════════════════════════════════════════════════════════════════

class Notifier:
    """
    Multi-channel notification engine with rate limiting and quiet hours.

    Channels:
      - console: Prints to stdout (INFO) or stderr (WARNING+)
      - file:    Appends to a log file
      - telegram: Sends via Telegram bot (requires token + chat_id)

    Usage:
        notifier = Notifier(
            channels=[Channel.CONSOLE, Channel.FILE],
            telegram_token="...",
            telegram_chat_id="...",
        )
        notifier.notify(
            "PRICE_LEVEL", priority="HIGH",
            ticker="AAPL", direction="above", target_price=200.0, current_price=198.5
        )
    """

    def __init__(
        self,
        channels: Optional[List[Union[Channel, str]]] = None,
        telegram_token: str = "",
        telegram_chat_id: str = "",
        file_log: str = "engine/data/notifications.log",
        rate_limit_per_hour: int = 60,
        quiet_hours: Optional[QuietHours] = None,
        templates: Optional[Dict[str, str]] = None,
    ):
        # Parse channels
        self.channels: List[Channel] = []
        if channels:
            for ch in channels:
                if isinstance(ch, Channel):
                    self.channels.append(ch)
                elif ch == "telegram":
                    self.channels.append(Channel.TELEGRAM)
                elif ch == "console":
                    self.channels.append(Channel.CONSOLE)
                elif ch == "file":
                    self.channels.append(Channel.FILE)

        # Telegram config
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id

        # File logging
        self.file_log = Path(file_log)
        self.file_log.parent.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.rate_limit_per_hour = rate_limit_per_hour
        self._sent_timestamps: deque = deque()
        self._rate_lock = threading.Lock()

        # Quiet hours
        self.quiet_hours = quiet_hours or QuietHours()

        # Message templates
        self.templates: Dict[str, MessageTemplate] = {}
        if templates:
            for name, tmpl in templates.items():
                self.templates[name] = MessageTemplate(name=name, template=tmpl)

        # Delivery callbacks (for custom channels)
        self._callbacks: Dict[str, Callable] = {}

        # Stats
        self._total_sent: int = 0
        self._suppressed: int = 0
        self._errors: int = 0

        logger.info(
            f"Notifier initialized: channels={[c.name for c in self.channels]}, "
            f"rate={self.rate_limit_per_hour}/h, "
            f"quiet_hours={'on' if self.quiet_hours.enabled else 'off'}"
        )

    # ── Configuration ─────────────────────────────────────────

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply notification configuration from config dict."""
        # Channels
        cfg_channels = config.get("channels", [])
        if cfg_channels:
            self.channels = []
            for ch in cfg_channels:
                if ch == "telegram":
                    self.channels.append(Channel.TELEGRAM)
                elif ch == "console":
                    self.channels.append(Channel.CONSOLE)
                elif ch == "file":
                    self.channels.append(Channel.FILE)

        # Telegram
        self.telegram_token = config.get("telegram_token", self.telegram_token)
        self.telegram_chat_id = config.get("telegram_chat_id", self.telegram_chat_id)

        # Rate limit
        self.rate_limit_per_hour = config.get("rate_limit_per_hour", self.rate_limit_per_hour)

        # File log
        fp = config.get("file_log", "")
        if fp:
            self.file_log = Path(fp)

        # Quiet hours
        qh_cfg = config.get("quiet_hours", {})
        if qh_cfg:
            self.quiet_hours = QuietHours(
                enabled=qh_cfg.get("enabled", False),
                start_utc=qh_cfg.get("start_utc", "22:00"),
                end_utc=qh_cfg.get("end_utc", "06:00"),
                weekend_silence=qh_cfg.get("weekend_silence", True),
                quiet_priority_floor=qh_cfg.get("quiet_priority_floor", "HIGH"),
            )

        # Templates
        tmpl_cfg = config.get("templates", {})
        for name, tmpl in tmpl_cfg.items():
            self.templates[name] = MessageTemplate(name=name, template=tmpl)

    def add_callback(self, name: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a custom delivery callback. Called as callback(message, kwargs)."""
        self._callbacks[name] = callback

    # ── Rate Limiting ─────────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        """Check if we're within the hourly rate limit. Thread-safe."""
        with self._rate_lock:
            now = time.time()
            # Purge old entries (> 1 hour)
            while self._sent_timestamps and now - self._sent_timestamps[0] > 3600:
                self._sent_timestamps.popleft()

            if len(self._sent_timestamps) >= self.rate_limit_per_hour:
                self._suppressed += 1
                return False

            self._sent_timestamps.append(now)
            return True

    # ── Core Notification ─────────────────────────────────────

    def notify(
        self,
        template_name: str,
        priority: str = "MEDIUM",
        channel_override: Optional[List[Channel]] = None,
        force: bool = False,
        **kwargs: Any,
    ) -> bool:
        """
        Send a notification through all configured channels.

        Args:
            template_name: Name of the message template to use
            priority: CRITICAL, HIGH, MEDIUM, LOW
            channel_override: Override which channels to use
            force: Skip rate limiting and quiet hours
            **kwargs: Variables to format into the template

        Returns:
            True if notification was delivered, False if suppressed
        """
        # Check quiet hours
        if not force and self.quiet_hours.is_quiet(priority):
            self._suppressed += 1
            logger.debug(f"Suppressed {template_name} ({priority}) — quiet hours")
            return False

        # Check rate limit
        if not force and not self._check_rate_limit():
            logger.debug(f"Suppressed {template_name} ({priority}) — rate limit")
            return False

        # Format message
        tmpl = self.templates.get(template_name)
        if tmpl:
            message = tmpl.format(priority=priority, **kwargs)
        else:
            # Fallback: raw kwargs
            prefix_map = {
                "CRITICAL": "🚨 CRITICAL",
                "HIGH": "🔴 HIGH",
                "MEDIUM": "🟡 MEDIUM",
                "LOW": "🟢 LOW",
            }
            prefix = prefix_map.get(priority, f"[{priority}]")
            message = f"{prefix}: {template_name} — {json.dumps(kwargs, default=str, indent=2)}"

        # Determine channels
        channels = channel_override or self.channels

        # Deliver
        delivered = False
        for channel in channels:
            try:
                if channel == Channel.CONSOLE:
                    self._deliver_console(message, priority)
                    delivered = True
                elif channel == Channel.FILE:
                    self._deliver_file(message, priority, template_name)
                    delivered = True
                elif channel == Channel.TELEGRAM:
                    self._deliver_telegram_async(message, priority)
                    delivered = True
            except Exception as e:
                logger.error(f"Delivery failed [{channel.name}]: {e}")
                self._errors += 1

        # Custom callbacks
        for cb_name, callback in self._callbacks.items():
            try:
                callback(message, kwargs)
                delivered = True
            except Exception as e:
                logger.error(f"Callback '{cb_name}' failed: {e}")

        if delivered:
            self._total_sent += 1

        return delivered

    def notify_alert(self, alert: Any) -> bool:
        """Send notification for a triggered Alert."""
        from engine.monitor.alerts import Alert, AlertType

        template_map = {
            AlertType.PRICE_LEVEL: "PRICE_LEVEL",
            AlertType.CHANGE_THRESHOLD: "CHANGE_THRESHOLD",
            AlertType.VOLUME_SPIKE: "VOLUME_SPIKE",
            AlertType.BREAKOUT: "BREAKOUT",
            AlertType.TECHNICAL_CROSSOVER: "TECHNICAL_CROSSOVER",
        }

        tmpl_name = template_map.get(alert.alert_type, "PRICE_LEVEL")

        # Build kwargs based on alert type
        base_kwargs = {
            "ticker": alert.ticker,
            "priority": alert.priority.name,
        }

        # Try to get current price for context
        try:
            import yfinance as yf
            t = yf.Ticker(alert.ticker)
            data = t.history(period="5d")
            if not data.empty:
                base_kwargs["current_price"] = float(data['Close'].iloc[-1])
        except Exception:
            pass

        return self.notify(
            template_name=tmpl_name,
            priority=alert.priority.name,
            **base_kwargs,
            **{c.field: c.value for group in alert.conditions for c in group},
        )

    def notify_anomaly(self, anomaly: Any) -> bool:
        """Send notification for a detected Anomaly."""
        return self.notify(
            template_name="ANOMALY",
            priority=anomaly.severity.name,
            anomaly_type=anomaly.anomaly_type.name.replace("_", " ").title(),
            description=anomaly.description,
            ticker=anomaly.ticker,
            value=anomaly.value,
        )

    def notify_order_event(self, event_type: str, **kwargs: Any) -> bool:
        """Send notification for an order lifecycle event."""
        tmpl_map = {
            "triggered": "ORDER_TRIGGERED",
            "filled": "ORDER_FILLED",
            "expired": "ORDER_EXPIRED",
            "cancelled": "ORDER_EXPIRED",
        }
        return self.notify(
            template_name=tmpl_map.get(event_type, "ORDER_TRIGGERED"),
            priority="HIGH" if event_type in ("triggered", "filled") else "MEDIUM",
            **kwargs,
        )

    # ── Channel Delivery ──────────────────────────────────────

    def _deliver_console(self, message: str, priority: str) -> None:
        """Deliver to console with appropriate log level."""
        if priority in ("CRITICAL", "HIGH"):
            logger.warning(message)
        elif priority == "MEDIUM":
            logger.info(message)
        else:
            logger.debug(message)

    def _deliver_file(self, message: str, priority: str, template_name: str) -> None:
        """Append notification to log file."""
        try:
            timestamp = datetime.now().isoformat()
            line = f"[{timestamp}] [{priority}] [{template_name}] {message}\n"
            with open(self.file_log, 'a') as f:
                f.write(line)
        except Exception as e:
            logger.error(f"File notification failed: {e}")

    def _deliver_telegram_async(self, message: str, priority: str) -> None:
        """Deliver to Telegram asynchronously (non-blocking thread)."""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.debug(f"Telegram not configured; skipping: {message[:80]}...")
            return

        t = threading.Thread(
            target=self._send_telegram,
            args=(message, priority),
            daemon=True,
        )
        t.start()

    def _send_telegram(self, message: str, priority: str) -> None:
        """Send via Telegram Bot API (HTTP POST)."""
        try:
            import urllib.request
            import urllib.parse

            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            
            # Add emoji prefix based on priority
            emoji_map = {
                "CRITICAL": "🚨 ",
                "HIGH": "🔴 ",
                "MEDIUM": "🟡 ",
                "LOW": "🟢 ",
            }
            formatted = emoji_map.get(priority, "") + message

            # Escape HTML special chars
            formatted = formatted.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            data = urllib.parse.urlencode({
                "chat_id": self.telegram_chat_id,
                "text": formatted,
                "parse_mode": "HTML",
            }).encode()

            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                if not result.get("ok"):
                    logger.error(f"Telegram API error: {result}")
                else:
                    logger.debug(f"Telegram sent: {formatted[:80]}...")

        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            self._errors += 1

    # ── Bulk Send ─────────────────────────────────────────────

    def send_batch(
        self, items: List[Dict[str, Any]], force: bool = False
    ) -> int:
        """
        Send multiple notifications as a batch.
        Each item is a dict with keys matching notify() kwargs.
        """
        count = 0
        for item in items:
            if self.notify(**item, force=force):
                count += 1
        return count

    def send_summary(
        self,
        alerts: Optional[List[Any]] = None,
        anomalies: Optional[List[Any]] = None,
        order_events: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Send a consolidated summary of triggered items."""
        lines: List[str] = []
        now = datetime.utcnow().strftime("%H:%M UTC")
        lines.append(f"📊 VMAA Monitor Summary ({now})")
        lines.append("─" * 30)

        if alerts:
            lines.append(f"🔔 Alerts: {len(alerts)} triggered")
            for a in alerts[:5]:  # max 5
                lines.append(f"  • {a.ticker}: {a.name} [{a.priority.name}]")
            if len(alerts) > 5:
                lines.append(f"  ... and {len(alerts) - 5} more")

        if anomalies:
            lines.append(f"⚠️ Anomalies: {len(anomalies)} detected")
            for a in anomalies[:5]:
                lines.append(f"  • {a.description}")
            if len(anomalies) > 5:
                lines.append(f"  ... and {len(anomalies) - 5} more")

        if order_events:
            lines.append(f"📋 Order Events: {len(order_events)}")
            for e in order_events[:5]:
                lines.append(f"  • {e.get('type', '?')}: {e.get('order_id', '?')}")
            if len(order_events) > 5:
                lines.append(f"  ... and {len(order_events) - 5} more")

        if not any([alerts, anomalies, order_events]):
            return 0

        message = "\n".join(lines)
        return 1 if self.notify(
            template_name="PRICE_LEVEL",  # Fallback template
            priority="MEDIUM",
            force=True,
            ticker="SUMMARY",
            current_price=0,
            direction="",
            target_price=0,
            _raw_message=message,
        ) else 0

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return notification engine statistics."""
        with self._rate_lock:
            recent = sum(
                1 for ts in self._sent_timestamps
                if time.time() - ts <= 3600
            )
        return {
            "total_sent": self._total_sent,
            "suppressed": self._suppressed,
            "errors": self._errors,
            "channels": [c.name for c in self.channels],
            "recent_hour": recent,
            "rate_limit": f"{recent}/{self.rate_limit_per_hour}",
            "quiet_active": self.quiet_hours.is_quiet("LOW"),
        }

    def reset_stats(self) -> None:
        """Reset notification statistics."""
        self._total_sent = 0
        self._suppressed = 0
        self._errors = 0
        self._sent_timestamps.clear()
