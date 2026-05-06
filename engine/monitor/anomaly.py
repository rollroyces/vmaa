#!/usr/bin/env python3
"""
VMAA 2.0 — Anomaly Detection Engine
=====================================
Detects market, portfolio, and data anomalies:

Market Anomalies:
  - Overnight gap detection (>3%)
  - Flash crash detection (>5% drop in <5 min)
  - Abnormal volume (>10x normal)
  - Bid-ask spread widening (>2% of price)

Portfolio Anomalies:
  - Unusual P&L moves (>2x expected daily std)
  - Correlation break (diverge from historical correlation)
  - Unexpected sector rotation

Data Anomalies:
  - Missing data (>1 hour without price update)
  - Stale cache detection (yfinance returns old data)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("vmaa.monitor.anomaly")


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════

class AnomalyType(Enum):
    # Market anomalies
    GAP = auto()                # Overnight gap
    FLASH_CRASH = auto()        # Sudden sharp drop
    VOLUME_ANOMALY = auto()     # Volume spike >10x
    SPREAD_WIDENING = auto()    # Bid-ask spread abnormal

    # Portfolio anomalies
    PNL_MOVE = auto()           # Unusual P&L change
    CORRELATION_BREAK = auto()  # Correlation divergence
    SECTOR_ROTATION = auto()    # Unexpected sector shift

    # Data anomalies
    MISSING_DATA = auto()       # No price update for >1h
    STALE_PRICE = auto()        # Cached/old price data


class AnomalySeverity(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


# ═══════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Anomaly:
    """A detected anomaly event."""
    id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity = AnomalySeverity.MEDIUM
    ticker: str = ""
    tickers: List[str] = field(default_factory=list)
    
    # Metrics
    value: float = 0.0
    threshold: float = 0.0
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    detected_at: str = ""
    resolved_at: Optional[str] = None
    
    # Context
    source: str = "auto-monitor"
    notes: str = ""
    
    # Deduplication
    fingerprint: str = ""  # For dedup: type+ticker+value_hash
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "anomaly_type": self.anomaly_type.name,
            "severity": self.severity.name,
            "ticker": self.ticker,
            "tickers": self.tickers,
            "value": self.value,
            "threshold": self.threshold,
            "description": self.description,
            "details": self.details,
            "detected_at": self.detected_at,
            "resolved_at": self.resolved_at,
            "source": self.source,
            "notes": self.notes,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Anomaly":
        return cls(
            id=d["id"],
            anomaly_type=AnomalyType[d["anomaly_type"]],
            severity=AnomalySeverity[d.get("severity", "MEDIUM")],
            ticker=d.get("ticker", ""),
            tickers=d.get("tickers", []),
            value=d.get("value", 0),
            threshold=d.get("threshold", 0),
            description=d.get("description", ""),
            details=d.get("details", {}),
            detected_at=d.get("detected_at", ""),
            resolved_at=d.get("resolved_at"),
            source=d.get("source", "auto-monitor"),
            notes=d.get("notes", ""),
            fingerprint=d.get("fingerprint", ""),
        )


# ═══════════════════════════════════════════════════════════════════
# Anomaly Detector
# ═══════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Multi-faceted anomaly detection engine.

    Usage:
        detector = AnomalyDetector()
        anomalies = detector.scan_all(watchlist=["AAPL", "GOOGL"])
        portfolio_anomalies = detector.scan_portfolio(positions, pnl_history)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Market anomaly thresholds
        self.gap_threshold_pct: float = 3.0
        self.flash_crash_threshold_pct: float = 5.0
        self.flash_crash_window: int = 5  # minutes
        self.volume_anomaly_mult: float = 10.0
        self.spread_widening_pct: float = 2.0

        # Portfolio anomaly thresholds
        self.pnl_std_mult: float = 2.0
        self.pnl_lookback_days: int = 60
        self.correlation_break_threshold: float = 0.30
        self.correlation_lookback_days: int = 120
        self.correlation_recent_days: int = 20
        self.sector_lookback_days: int = 60

        # Data anomaly thresholds
        self.missing_data_hours: float = 1.0
        self.stale_cache_minutes: int = 15
        self.stale_cache_check_sample: int = 5

        if config:
            self._apply_config(config)

        # State
        self._last_data_time: Dict[str, float] = {}      # ticker → epoch
        self._known_anomalies: Dict[str, Anomaly] = {}   # fingerprint → anomaly
        self._correlation_cache: Dict[str, np.ndarray] = {}
        self._sector_performance_cache: Dict[str, Tuple[float, float]] = {}  # sector → (prev, current)

        # Counter
        self._scan_counter: int = 0

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply anomaly configuration."""
        self.gap_threshold_pct = config.get("gap_threshold_pct", self.gap_threshold_pct)
        self.flash_crash_threshold_pct = config.get("flash_crash_threshold_pct", self.flash_crash_threshold_pct)
        self.flash_crash_window = config.get("flash_crash_window_minutes", self.flash_crash_window)
        self.volume_anomaly_mult = config.get("volume_anomaly_multiplier", self.volume_anomaly_mult)
        self.spread_widening_pct = config.get("spread_widening_pct", self.spread_widening_pct)
        self.pnl_std_mult = config.get("pnl_std_multiplier", self.pnl_std_mult)
        self.pnl_lookback_days = config.get("pnl_lookback_days", self.pnl_lookback_days)
        self.correlation_break_threshold = config.get("correlation_break_threshold", self.correlation_break_threshold)
        self.correlation_lookback_days = config.get("correlation_lookback_days", self.correlation_lookback_days)
        self.correlation_recent_days = config.get("correlation_recent_days", self.correlation_recent_days)
        self.sector_lookback_days = config.get("sector_lookback_days", self.sector_lookback_days)
        self.missing_data_hours = config.get("missing_data_hours", self.missing_data_hours)
        self.stale_cache_minutes = config.get("stale_cache_minutes", self.stale_cache_minutes)
        self.stale_cache_check_sample = config.get("stale_cache_check_sample", self.stale_cache_check_sample)

    # ═══════════════════════════════════════════════════════════
    # Market Anomalies
    # ═══════════════════════════════════════════════════════════

    def detect_gaps(self, watchlist: List[str]) -> List[Anomaly]:
        """Detect overnight gaps > threshold."""
        anomalies: List[Anomaly] = []

        for ticker in watchlist:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if len(hist) < 2:
                    continue

                prev_close = float(hist['Close'].iloc[-2])
                curr_open = float(hist['Open'].iloc[-1])
                if prev_close == 0:
                    continue

                gap_pct = (curr_open - prev_close) / prev_close * 100
                if abs(gap_pct) >= self.gap_threshold_pct:
                    direction = "up" if gap_pct > 0 else "down"
                    severity = (
                        AnomalySeverity.CRITICAL if abs(gap_pct) > 10
                        else AnomalySeverity.HIGH if abs(gap_pct) > 5
                        else AnomalySeverity.MEDIUM
                    )
                    anomaly = Anomaly(
                        id=self._gen_id("gap", ticker),
                        anomaly_type=AnomalyType.GAP,
                        severity=severity,
                        ticker=ticker,
                        value=round(gap_pct, 2),
                        threshold=self.gap_threshold_pct,
                        description=f"{ticker} overnight gap {direction} {abs(gap_pct):.2f}%",
                        details={
                            "prev_close": prev_close,
                            "open": curr_open,
                            "direction": direction,
                            "gap_pct": round(gap_pct, 2),
                        },
                        detected_at=datetime.now().isoformat(),
                        fingerprint=f"gap:{ticker}:{int(gap_pct)}",
                    )
                    if self._is_new(anomaly):
                        anomalies.append(anomaly)
                        self._record(anomaly)
                        logger.info(f"GAP detected: {anomaly.description}")

            except Exception as e:
                logger.debug(f"Gap check failed for {ticker}: {e}")

        return anomalies

    def detect_flash_crash(self, watchlist: List[str]) -> List[Anomaly]:
        """
        Detect flash crashes: >5% drop within a short window.
        Uses 1-minute intraday data where available, falls back to daily range.
        """
        anomalies: List[Anomaly] = []

        for ticker in watchlist:
            try:
                t = yf.Ticker(ticker)

                # Try 1-min data
                intraday = t.history(period="1d", interval="1m")
                if len(intraday) >= self.flash_crash_window:
                    # Rolling min over window
                    high_in_window = intraday['High'].rolling(self.flash_crash_window).max()
                    low_in_window = intraday['Low'].rolling(self.flash_crash_window).min()
                    # Drop from rolling high
                    for i in range(self.flash_crash_window, len(intraday)):
                        hw = float(high_in_window.iloc[i])
                        lw = float(low_in_window.iloc[i])
                        if hw > 0:
                            drop = (hw - lw) / hw * 100
                            if drop >= self.flash_crash_threshold_pct:
                                anomaly = Anomaly(
                                    id=self._gen_id("flash", ticker),
                                    anomaly_type=AnomalyType.FLASH_CRASH,
                                    severity=AnomalySeverity.CRITICAL,
                                    ticker=ticker,
                                    value=round(drop, 2),
                                    threshold=self.flash_crash_threshold_pct,
                                    description=f"{ticker} flash crash: {drop:.2f}% drop in {self.flash_crash_window}min",
                                    details={
                                        "high": hw, "low": lw,
                                        "drop_pct": round(drop, 2),
                                        "window_minutes": self.flash_crash_window,
                                    },
                                    detected_at=datetime.now().isoformat(),
                                    fingerprint=f"flash:{ticker}:{int(drop*100)}",
                                )
                                if self._is_new(anomaly):
                                    anomalies.append(anomaly)
                                    self._record(anomaly)
                                    logger.warning(f"FLASH CRASH: {anomaly.description}")
                                break  # One per ticker
                else:
                    # Fallback to daily range
                    daily = t.history(period="5d")
                    if len(daily) >= 1:
                        today = daily.iloc[-1]
                        daily_range = (
                            (float(today['High']) - float(today['Low'])) / float(today['High']) * 100
                            if float(today['High']) > 0 else 0
                        )
                        if daily_range >= self.flash_crash_threshold_pct:
                            anomaly = Anomaly(
                                id=self._gen_id("flash_daily", ticker),
                                anomaly_type=AnomalyType.FLASH_CRASH,
                                severity=AnomalySeverity.HIGH,
                                ticker=ticker,
                                value=round(daily_range, 2),
                                threshold=self.flash_crash_threshold_pct,
                                description=f"{ticker} wide daily range: {daily_range:.2f}% (possible flash crash)",
                                details={"high": float(today['High']), "low": float(today['Low'])},
                                detected_at=datetime.now().isoformat(),
                                fingerprint=f"flash_daily:{ticker}:{int(daily_range*100)}",
                            )
                            if self._is_new(anomaly):
                                anomalies.append(anomaly)
                                self._record(anomaly)

            except Exception as e:
                logger.debug(f"Flash crash check failed for {ticker}: {e}")

        return anomalies

    def detect_volume_anomaly(self, watchlist: List[str]) -> List[Anomaly]:
        """Detect volume spikes >10x normal (20-day average)."""
        anomalies: List[Anomaly] = []

        for ticker in watchlist:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2mo")
                if len(hist) < 22:
                    continue

                curr_vol = float(hist['Volume'].iloc[-1])
                avg_vol = float(hist['Volume'].iloc[-21:-1].mean())
                if avg_vol <= 0:
                    continue

                ratio = curr_vol / avg_vol
                if ratio >= self.volume_anomaly_mult:
                    severity = (
                        AnomalySeverity.CRITICAL if ratio > 20
                        else AnomalySeverity.HIGH if ratio > 10
                        else AnomalySeverity.MEDIUM
                    )
                    anomaly = Anomaly(
                        id=self._gen_id("volume", ticker),
                        anomaly_type=AnomalyType.VOLUME_ANOMALY,
                        severity=severity,
                        ticker=ticker,
                        value=round(ratio, 1),
                        threshold=self.volume_anomaly_mult,
                        description=f"{ticker} volume {ratio:.1f}x normal ({curr_vol:,.0f} vs avg {avg_vol:,.0f})",
                        details={
                            "current_volume": curr_vol,
                            "avg_volume": avg_vol,
                            "ratio": round(ratio, 1),
                        },
                        detected_at=datetime.now().isoformat(),
                        fingerprint=f"volume:{ticker}:{int(ratio)}",
                    )
                    if self._is_new(anomaly):
                        anomalies.append(anomaly)
                        self._record(anomaly)
                        logger.info(f"VOLUME ANOMALY: {anomaly.description}")

            except Exception as e:
                logger.debug(f"Volume check failed for {ticker}: {e}")

        return anomalies

    def detect_spread_widening(self, watchlist: List[str]) -> List[Anomaly]:
        """Detect bid-ask spread > threshold % of price."""
        anomalies: List[Anomaly] = []

        for ticker in watchlist:
            try:
                t = yf.Ticker(ticker)
                info = t.info
                bid = info.get('bid')
                ask = info.get('ask')
                price = info.get('regularMarketPrice') or info.get('currentPrice')

                if not bid or not ask or not price or price <= 0:
                    continue

                spread_pct = (ask - bid) / price * 100
                if spread_pct >= self.spread_widening_pct:
                    severity = (
                        AnomalySeverity.HIGH if spread_pct > 5
                        else AnomalySeverity.MEDIUM
                    )
                    anomaly = Anomaly(
                        id=self._gen_id("spread", ticker),
                        anomaly_type=AnomalyType.SPREAD_WIDENING,
                        severity=severity,
                        ticker=ticker,
                        value=round(spread_pct, 2),
                        threshold=self.spread_widening_pct,
                        description=f"{ticker} spread widened to {spread_pct:.2f}% (bid: {bid:.2f}, ask: {ask:.2f})",
                        details={
                            "bid": bid, "ask": ask,
                            "price": price,
                            "spread_pct": round(spread_pct, 2),
                        },
                        detected_at=datetime.now().isoformat(),
                        fingerprint=f"spread:{ticker}:{int(spread_pct*100)}",
                    )
                    if self._is_new(anomaly):
                        anomalies.append(anomaly)
                        self._record(anomaly)
                        logger.info(f"SPREAD WIDENING: {anomaly.description}")

            except Exception as e:
                logger.debug(f"Spread check failed for {ticker}: {e}")

        return anomalies

    # ═══════════════════════════════════════════════════════════
    # Portfolio Anomalies
    # ═══════════════════════════════════════════════════════════

    def detect_pnl_anomaly(
        self,
        positions: List[Dict[str, Any]],
        pnl_history: Optional[List[float]] = None,
    ) -> List[Anomaly]:
        """
        Detect unusual P&L moves (>2x expected daily std).

        Args:
            positions: List of position dicts with at least {ticker, market_value, unrealized_pnl, unrealized_pnl_pct}
            pnl_history: Daily total P&L values for std calculation
        """
        anomalies: List[Anomaly] = []

        if not positions:
            return anomalies

        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in positions)
        total_value = sum(p.get('market_value', 0) for p in positions)
        total_pnl_pct = (total_unrealized / total_value * 100) if total_value > 0 else 0

        # If we have history, compare against std
        if pnl_history and len(pnl_history) >= 10:
            pnl_series = pd.Series(pnl_history)
            mean_pnl = float(pnl_series.mean())
            std_pnl = float(pnl_series.std())
            if std_pnl > 0:
                z_score = abs(total_unrealized - mean_pnl) / std_pnl
                if z_score >= self.pnl_std_mult:
                    severity = (
                        AnomalySeverity.CRITICAL if z_score > 4
                        else AnomalySeverity.HIGH if z_score > 3
                        else AnomalySeverity.MEDIUM
                    )
                    anomaly = Anomaly(
                        id=self._gen_id("pnl", "portfolio"),
                        anomaly_type=AnomalyType.PNL_MOVE,
                        severity=severity,
                        description=f"Portfolio P&L {total_pnl_pct:+.2f}% — {z_score:.1f}σ move (mean: {mean_pnl:.0f}, std: {std_pnl:.0f})",
                        value=round(z_score, 2),
                        threshold=self.pnl_std_mult,
                        details={
                            "total_unrealized": total_unrealized,
                            "total_value": total_value,
                            "pnl_pct": round(total_pnl_pct, 2),
                            "z_score": round(z_score, 2),
                            "mean_pnl": round(mean_pnl, 0),
                            "std_pnl": round(std_pnl, 0),
                        },
                        detected_at=datetime.now().isoformat(),
                        fingerprint=f"pnl:portfolio:{int(z_score*100)}",
                    )
                    if self._is_new(anomaly):
                        anomalies.append(anomaly)
                        self._record(anomaly)
                        logger.warning(f"PNL ANOMALY: {anomaly.description}")

        # Per-position check
        for pos in positions:
            pnl_pct = pos.get('unrealized_pnl_pct', 0)
            if abs(pnl_pct) > 20:
                ticker = pos.get('ticker', '?')
                anomaly = Anomaly(
                    id=self._gen_id("pnl_pos", ticker),
                    anomaly_type=AnomalyType.PNL_MOVE,
                    severity=AnomalySeverity.HIGH if abs(pnl_pct) > 30 else AnomalySeverity.MEDIUM,
                    ticker=ticker,
                    value=round(pnl_pct, 2),
                    threshold=20.0,
                    description=f"{ticker} position P&L {pnl_pct:+.1f}% — extreme move",
                    details={
                        "pnl_pct": pnl_pct,
                        "unrealized_pnl": pos.get('unrealized_pnl', 0),
                        "market_value": pos.get('market_value', 0),
                    },
                    detected_at=datetime.now().isoformat(),
                    fingerprint=f"pnl_pos:{ticker}:{int(pnl_pct)}",
                )
                if self._is_new(anomaly):
                    anomalies.append(anomaly)
                    self._record(anomaly)
                    logger.warning(f"POSITION PNL ANOMALY: {anomaly.description}")

        return anomalies

    def detect_correlation_break(
        self, tickers: List[str]
    ) -> List[Anomaly]:
        """
        Detect correlation break: pairs that used to be correlated but recently diverged.

        Compares long-term correlation (e.g., 120d) vs recent (20d).
        """
        anomalies: List[Anomaly] = []

        if len(tickers) < 2:
            return anomalies

        try:
            long_period = f"{self.correlation_lookback_days}d"
            recent_period = f"{self.correlation_recent_days}d"

            # Fetch long-term data
            data = yf.download(tickers, period="6mo", progress=False)
            if data.empty or 'Close' not in data.columns:
                return anomalies

            returns = data['Close'].pct_change().dropna()
            if returns.shape[1] < 2:
                return anomalies

            # Split into long-term and recent
            split_idx = max(1, len(returns) - self.correlation_recent_days)
            returns_long = returns.iloc[:split_idx]
            returns_recent = returns.iloc[split_idx:]

            if len(returns_long) < 20 or len(returns_recent) < 5:
                return anomalies

            # Correlation matrices
            corr_long = returns_long.corr()
            corr_recent = returns_recent.corr()

            # Find pairs with large divergence
            for i in range(len(tickers)):
                for j in range(i + 1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    if t1 not in corr_long.columns or t2 not in corr_long.columns:
                        continue

                    long_corr = float(corr_long.loc[t1, t2])
                    recent_corr = float(corr_recent.loc[t1, t2])
                    diff = abs(long_corr - recent_corr)

                    if diff >= self.correlation_break_threshold:
                        anomaly = Anomaly(
                            id=self._gen_id("corr", f"{t1}_{t2}"),
                            anomaly_type=AnomalyType.CORRELATION_BREAK,
                            severity=(
                                AnomalySeverity.HIGH if diff > 0.5
                                else AnomalySeverity.MEDIUM
                            ),
                            tickers=[t1, t2],
                            value=round(diff, 2),
                            threshold=self.correlation_break_threshold,
                            description=(
                                f"{t1}-{t2} correlation break: "
                                f"long {long_corr:.2f} → recent {recent_corr:.2f} (Δ{diff:.2f})"
                            ),
                            details={
                                "long_correlation": round(long_corr, 2),
                                "recent_correlation": round(recent_corr, 2),
                                "delta": round(diff, 2),
                                "long_days": self.correlation_lookback_days,
                                "recent_days": self.correlation_recent_days,
                            },
                            detected_at=datetime.now().isoformat(),
                            fingerprint=f"corr:{t1}:{t2}:{int(diff*100)}",
                        )
                        if self._is_new(anomaly):
                            anomalies.append(anomaly)
                            self._record(anomaly)
                            logger.info(f"CORRELATION BREAK: {anomaly.description}")

        except Exception as e:
            logger.warning(f"Correlation check failed: {e}")

        return anomalies

    def detect_sector_rotation(self, sector_etfs: Optional[Dict[str, str]] = None) -> List[Anomaly]:
        """
        Detect unexpected sector momentum shift.
        
        Args:
            sector_etfs: Dict mapping sector name → ETF ticker
                         Default: major SPDR sector ETFs
        """
        if sector_etfs is None:
            sector_etfs = {
                "Technology": "XLK",
                "Financials": "XLF",
                "Healthcare": "XLV",
                "Energy": "XLE",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Communication Services": "XLC",
            }

        anomalies: List[Anomaly] = []

        try:
            tickers = list(sector_etfs.values())
            period = f"{self.sector_lookback_days + 60}d"

            data = yf.download(tickers, period=period, progress=False)
            if data.empty or 'Close' not in data.columns:
                return anomalies

            prices = data['Close'].dropna(axis=1, how='all')
            sector_names = [s for s, t in sector_etfs.items() if t in prices.columns]
            relevant_tickers = [sector_etfs[s] for s in sector_names]

            if len(relevant_tickers) < 2:
                return anomalies

            # Compute 2-period performance
            total_len = len(prices)
            mid = total_len // 2

            perf_full = {
                t: (float(prices[t].iloc[-1]) / float(prices[t].iloc[0]) - 1) * 100
                for t in relevant_tickers
                if float(prices[t].iloc[0]) > 0
            }

            perf_recent = {
                t: (float(prices[t].iloc[-1]) / float(prices[t].iloc[mid]) - 1) * 100
                for t in relevant_tickers
                if float(prices[t].iloc[mid]) > 0
            }

            # Find sectors with >2σ shift in relative ranking
            if len(perf_full) >= 3:
                full_values = list(perf_full.values())
                recent_values = list(perf_recent.values())
                full_std = float(np.std(full_values)) if full_values else 0
                recent_std = float(np.std(recent_values)) if recent_values else 0

                full_mean = float(np.mean(full_values))
                recent_mean = float(np.mean(recent_values))

                for sector, ticker in zip(sector_names, relevant_tickers):
                    if ticker not in perf_full or ticker not in perf_recent:
                        continue

                    full_perf = perf_full[ticker]
                    recent_perf = perf_recent[ticker]

                    if full_std > 0:
                        full_z = (full_perf - full_mean) / full_std
                        recent_z = (recent_perf - recent_mean) / recent_std if recent_std > 0 else 0
                        shift = abs(recent_z - full_z)

                        if shift >= 1.5:  # 1.5σ shift in z-score ranking
                            direction = "strengthening" if recent_z > full_z else "weakening"
                            anomaly = Anomaly(
                                id=self._gen_id("sector", sector),
                                anomaly_type=AnomalyType.SECTOR_ROTATION,
                                severity=AnomalySeverity.MEDIUM,
                                ticker=ticker,
                                value=round(shift, 2),
                                threshold=1.5,
                                description=(
                                    f"Sector rotation: {sector} {direction} "
                                    f"(z: {full_z:.1f} → {recent_z:.1f}, shift={shift:.2f}σ)"
                                ),
                                details={
                                    "sector": sector,
                                    "ticker": ticker,
                                    "full_performance": round(full_perf, 2),
                                    "recent_performance": round(recent_perf, 2),
                                    "full_z": round(full_z, 2),
                                    "recent_z": round(recent_z, 2),
                                    "shift_sigma": round(shift, 2),
                                    "direction": direction,
                                },
                                detected_at=datetime.now().isoformat(),
                                fingerprint=f"sector:{sector}:{int(shift*100)}",
                            )
                            if self._is_new(anomaly):
                                anomalies.append(anomaly)
                                self._record(anomaly)
                                logger.info(f"SECTOR ROTATION: {anomaly.description}")

        except Exception as e:
            logger.warning(f"Sector rotation check failed: {e}")

        return anomalies

    # ═══════════════════════════════════════════════════════════
    # Data Anomalies
    # ═══════════════════════════════════════════════════════════

    def detect_missing_data(self, watchlist: List[str]) -> List[Anomaly]:
        """Detect tickers with no price update for > N hours."""
        anomalies: List[Anomaly] = []
        now = time.time()
        threshold_seconds = self.missing_data_hours * 3600

        for ticker in watchlist:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="2d")
                if hist.empty:
                    last_update = self._last_data_time.get(ticker, 0)
                    elapsed = now - last_update if last_update > 0 else threshold_seconds + 1
                else:
                    last_update = now
                    self._last_data_time[ticker] = now
                    elapsed = 0

                if elapsed > threshold_seconds:
                    hours = elapsed / 3600
                    anomaly = Anomaly(
                        id=self._gen_id("missing", ticker),
                        anomaly_type=AnomalyType.MISSING_DATA,
                        severity=AnomalySeverity.HIGH if hours > 4 else AnomalySeverity.MEDIUM,
                        ticker=ticker,
                        value=round(hours, 1),
                        threshold=self.missing_data_hours,
                        description=f"No price data for {ticker} in {hours:.1f}h",
                        details={
                            "last_update": (
                                datetime.fromtimestamp(last_update).isoformat()
                                if last_update > 0 else "unknown"
                            ),
                            "elapsed_hours": round(hours, 1),
                        },
                        detected_at=datetime.now().isoformat(),
                        fingerprint=f"missing:{ticker}:{int(hours)}",
                    )
                    if self._is_new(anomaly):
                        anomalies.append(anomaly)
                        self._record(anomaly)
                        logger.warning(f"MISSING DATA: {anomaly.description}")

            except Exception as e:
                # If we can't even reach the API, that's a data issue
                last_update = self._last_data_time.get(ticker, 0)
                elapsed = now - last_update if last_update > 0 else 0
                if elapsed > threshold_seconds:
                    anomaly = Anomaly(
                        id=self._gen_id("missing_api", ticker),
                        anomaly_type=AnomalyType.MISSING_DATA,
                        severity=AnomalySeverity.HIGH,
                        ticker=ticker,
                        value=round(elapsed / 3600, 1),
                        threshold=self.missing_data_hours,
                        description=f"API error fetching {ticker}: {e}",
                        detected_at=datetime.now().isoformat(),
                        fingerprint=f"missing_api:{ticker}:{int(elapsed/3600)}",
                    )
                    if self._is_new(anomaly):
                        anomalies.append(anomaly)
                        self._record(anomaly)

        return anomalies

    def detect_stale_price(self, watchlist: List[str]) -> List[Anomaly]:
        """
        Detect stale cached prices from yfinance.
        
        yfinance can return cached data. We check a sample of tickers
        by comparing the reported 'regularMarketTime' vs current time.
        """
        anomalies: List[Anomaly] = []
        now = datetime.now()
        threshold = timedelta(minutes=self.stale_cache_minutes)

        # Sample tickers to avoid excessive API calls
        import random
        sample = random.sample(
            watchlist, min(self.stale_cache_check_sample, len(watchlist))
        ) if watchlist else []

        for ticker in sample:
            try:
                t = yf.Ticker(ticker)
                info = t.info
                market_time_ts = info.get('regularMarketTime')

                if market_time_ts:
                    market_time = datetime.fromtimestamp(market_time_ts)
                    elapsed = now - market_time
                    if elapsed > threshold:
                        minutes = elapsed.total_seconds() / 60
                        anomaly = Anomaly(
                            id=self._gen_id("stale", ticker),
                            anomaly_type=AnomalyType.STALE_PRICE,
                            severity=(
                                AnomalySeverity.HIGH if minutes > 60
                                else AnomalySeverity.MEDIUM
                            ),
                            ticker=ticker,
                            value=round(minutes, 1),
                            threshold=self.stale_cache_minutes,
                            description=(
                                f"Stale price for {ticker}: "
                                f"last update {minutes:.0f}min ago ({market_time.isoformat()})"
                            ),
                            details={
                                "last_market_time": market_time.isoformat(),
                                "elapsed_minutes": round(minutes, 1),
                                "price": info.get('regularMarketPrice', 0),
                            },
                            detected_at=now.isoformat(),
                            fingerprint=f"stale:{ticker}:{int(minutes)}",
                        )
                        if self._is_new(anomaly):
                            anomalies.append(anomaly)
                            self._record(anomaly)
                            logger.info(f"STALE PRICE: {anomaly.description}")

            except Exception as e:
                logger.debug(f"Stale check failed for {ticker}: {e}")

        return anomalies

    # ═══════════════════════════════════════════════════════════
    # Comprehensive Scans
    # ═══════════════════════════════════════════════════════════

    def scan_market(self, watchlist: List[str]) -> List[Anomaly]:
        """Run all market anomaly checks."""
        anomalies: List[Anomaly] = []
        anomalies.extend(self.detect_gaps(watchlist))
        anomalies.extend(self.detect_flash_crash(watchlist))
        anomalies.extend(self.detect_volume_anomaly(watchlist))
        anomalies.extend(self.detect_spread_widening(watchlist))
        return anomalies

    def scan_portfolio(
        self,
        positions: List[Dict[str, Any]],
        pnl_history: Optional[List[float]] = None,
        tickers: Optional[List[str]] = None,
    ) -> List[Anomaly]:
        """Run all portfolio anomaly checks."""
        anomalies: List[Anomaly] = []
        anomalies.extend(self.detect_pnl_anomaly(positions, pnl_history))

        pos_tickers = tickers or [p.get('ticker', '') for p in positions if p.get('ticker')]
        if len(pos_tickers) >= 2:
            anomalies.extend(self.detect_correlation_break(pos_tickers))

        anomalies.extend(self.detect_sector_rotation())
        return anomalies

    def scan_data(self, watchlist: List[str]) -> List[Anomaly]:
        """Run all data quality anomaly checks."""
        anomalies: List[Anomaly] = []
        anomalies.extend(self.detect_missing_data(watchlist))
        anomalies.extend(self.detect_stale_price(watchlist))
        return anomalies

    def scan_all(
        self,
        watchlist: List[str],
        positions: Optional[List[Dict[str, Any]]] = None,
        pnl_history: Optional[List[float]] = None,
    ) -> List[Anomaly]:
        """Run all anomaly checks and return consolidated results."""
        self._scan_counter += 1
        all_anomalies: List[Anomaly] = []

        # Market
        all_anomalies.extend(self.detect_gaps(watchlist))
        all_anomalies.extend(self.detect_flash_crash(watchlist))
        all_anomalies.extend(self.detect_volume_anomaly(watchlist))
        all_anomalies.extend(self.detect_spread_widening(watchlist))

        # Portfolio
        if positions:
            all_anomalies.extend(self.detect_pnl_anomaly(positions, pnl_history))
            pos_tickers = [p.get('ticker', '') for p in positions if p.get('ticker')]
            if len(pos_tickers) >= 2:
                all_anomalies.extend(self.detect_correlation_break(pos_tickers))
            all_anomalies.extend(self.detect_sector_rotation())

        # Data
        all_anomalies.extend(self.detect_missing_data(watchlist))
        all_anomalies.extend(self.detect_stale_price(watchlist))

        logger.info(
            f"Scan #{self._scan_counter}: {len(all_anomalies)} anomalies from "
            f"{len(watchlist)} watched tickers"
        )
        return all_anomalies

    # ═══════════════════════════════════════════════════════════
    # Deduplication & State
    # ═══════════════════════════════════════════════════════════

    def _is_new(self, anomaly: Anomaly) -> bool:
        """Check if this anomaly is new (not previously detected)."""
        if anomaly.fingerprint and anomaly.fingerprint in self._known_anomalies:
            return False
        # Also check by type+ticker combo
        for existing in self._known_anomalies.values():
            if (existing.anomaly_type == anomaly.anomaly_type
                    and existing.ticker == anomaly.ticker
                    and existing.resolved_at is None):
                return False
        return True

    def _record(self, anomaly: Anomaly) -> None:
        """Record a detected anomaly."""
        if anomaly.fingerprint:
            self._known_anomalies[anomaly.fingerprint] = anomaly
        else:
            self._known_anomalies[anomaly.id] = anomaly

    def resolve_anomaly(self, anomaly_id: str) -> bool:
        """Mark an anomaly as resolved."""
        anomaly = self._known_anomalies.get(anomaly_id)
        if anomaly:
            anomaly.resolved_at = datetime.now().isoformat()
            logger.info(f"Resolved anomaly {anomaly_id}")
            return True
        return False

    def clear_resolved(self, max_age_hours: float = 24) -> int:
        """Remove resolved anomalies older than max_age_hours."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []
        for fp, anomaly in self._known_anomalies.items():
            if anomaly.resolved_at:
                try:
                    resolved = datetime.fromisoformat(anomaly.resolved_at)
                    if resolved < cutoff:
                        to_remove.append(fp)
                except (ValueError, TypeError):
                    pass
        for fp in to_remove:
            del self._known_anomalies[fp]
        logger.debug(f"Cleared {len(to_remove)} resolved anomalies")
        return len(to_remove)

    # ═══════════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════════

    def to_dict(self) -> Dict[str, Any]:
        return {
            "known_anomalies": {
                fp: a.to_dict() for fp, a in self._known_anomalies.items()
            },
            "last_data_time": self._last_data_time,
            "scan_counter": self._scan_counter,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> "AnomalyDetector":
        detector = cls(config=config)
        for fp, a_data in d.get("known_anomalies", {}).items():
            anomaly = Anomaly.from_dict(a_data)
            detector._known_anomalies[fp] = anomaly
        detector._last_data_time = d.get("last_data_time", {})
        detector._scan_counter = d.get("scan_counter", 0)
        return detector

    def save(self, filepath: str) -> None:
        """Save anomaly state to JSON."""
        import json
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.debug(f"Saved anomaly state ({len(self._known_anomalies)} known)")

    def load(self, filepath: str) -> bool:
        """Load anomaly state from JSON."""
        import json
        from pathlib import Path
        if not Path(filepath).exists():
            return False
        with open(filepath) as f:
            d = json.load(f)
        restored = AnomalyDetector.from_dict(d, config=None)
        self._known_anomalies = restored._known_anomalies
        self._last_data_time = restored._last_data_time
        self._scan_counter = restored._scan_counter
        logger.info(f"Loaded anomaly state ({len(self._known_anomalies)} known)")
        return True

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════

    def summary(self) -> Dict[str, Any]:
        """Return anomaly detection summary."""
        type_counts: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {}
        unresolved = 0
        for a in self._known_anomalies.values():
            tn = a.anomaly_type.name
            type_counts[tn] = type_counts.get(tn, 0) + 1
            sn = a.severity.name
            severity_counts[sn] = severity_counts.get(sn, 0) + 1
            if not a.resolved_at:
                unresolved += 1
        return {
            "total_detected": len(self._known_anomalies),
            "by_type": type_counts,
            "by_severity": severity_counts,
            "unresolved": unresolved,
            "resolved": len(self._known_anomalies) - unresolved,
            "scan_count": self._scan_counter,
        }

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _gen_id(prefix: str, ticker: str) -> str:
        ts = int(time.time() * 1000000) % 100000000
        return f"{prefix}_{ticker}_{ts}"
