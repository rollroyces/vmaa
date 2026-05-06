#!/usr/bin/env python3
"""
VMAA Engine — Integration Layer
=================================
Unified facade that lazily loads all sub-engines and orchestrates
the complete VMAA pipeline.

Sub-engines:
  selection/  → Smart Stock Screening (Part 1 Quality + Part 2 MAGNA)
  technical/  → Technical Analysis (40+ indicators + signals)
  chip/       → Chip Analysis (volume profile + S/R)
  risk/       → Risk Assessment (VaR, sizing, circuit breakers)
  monitor/    → Auto Monitoring (alerts, orders, anomaly detection)

Usage:
  from engine import VMAAEngine
  engine = VMAAEngine()
  result = engine.full_pipeline(["AAPL", "MSFT", "GOOGL"])
  scan = engine.quick_scan(tickers=["AAPL"])
  report = engine.portfolio_report([pos1, pos2, ...])
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure VMAA root is importable
_vmaa_root = Path(__file__).resolve().parent.parent
if str(_vmaa_root) not in sys.path:
    sys.path.insert(0, str(_vmaa_root))

from engine.config import EngineConfig, get_engine_config, update_engine_config

logger = logging.getLogger("vmaa.engine")


# ═══════════════════════════════════════════════════════════════════
# VMAA Engine — Main Facade
# ═══════════════════════════════════════════════════════════════════

class VMAAEngine:
    """
    VMAA 2.0 Unified Integration Layer.

    Lazy-initializes all available sub-engines and provides
    a single entry point for the full VMAA pipeline.

    Example:
        >>> engine = VMAAEngine()
        >>> result = engine.full_pipeline(["AAPL", "MSFT"])
        >>> candidates = result["candidates"]
        >>> decisions = result["decisions"]
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_engine_config()
        self._vmaa_root = Path(__file__).resolve().parent.parent

        # Lazy-init containers — populated on first access
        self.selection: Any = None       # SmartScreener
        self.risk: Any = None            # RiskEngine
        self.monitor: Any = None         # MonitorEngine
        self.technical: Any = None       # TechnicalEngine
        self.chip: Any = None            # ChipEngine

        # Engine status tracking
        self._engine_status: Dict[str, str] = {
            "selection": "not_loaded",
            "technical": "not_loaded",
            "chip": "not_loaded",
            "risk": "not_loaded",
            "monitor": "not_loaded",
        }

        # Pre-load engines based on config
        self._init_engines()

        logger.info(
            f"VMAAEngine initialized — mode: {self.config.mode} | "
            f"engines: {[k for k, v in self._engine_status.items() if v == 'loaded']}"
        )

    def _init_engines(self) -> None:
        """Lazy-init all enabled engines, catching ImportError gracefully."""
        if self.config.enable_selection:
            self._init_selection()
        if self.config.enable_technical:
            self._init_technical()
        if self.config.enable_chip:
            self._init_chip()
        if self.config.enable_risk:
            self._init_risk()
        if self.config.enable_monitor:
            self._init_monitor()

    def _init_selection(self) -> None:
        try:
            from engine.selection import SmartScreener
            self.selection = SmartScreener()
            self._engine_status["selection"] = "loaded"
            logger.info("Selection engine loaded")
        except ImportError as e:
            self._engine_status["selection"] = f"error: {e}"
            logger.warning(f"Selection engine unavailable: {e}")

    def _init_technical(self) -> None:
        try:
            from engine.technical.analysis import TechnicalEngine
            self.technical = TechnicalEngine()
            self._engine_status["technical"] = "loaded"
            logger.info("Technical engine loaded")
        except ImportError as e:
            self._engine_status["technical"] = f"error: {e}"
            logger.warning(f"Technical engine unavailable: {e}")

    def _init_chip(self) -> None:
        try:
            from engine.chip.engine import ChipEngine
            self.chip = ChipEngine()
            self._engine_status["chip"] = "loaded"
            logger.info("Chip engine loaded")
        except ImportError as e:
            self._engine_status["chip"] = f"error: {e}"
            logger.warning(f"Chip engine unavailable: {e}")

    def _init_risk(self) -> None:
        try:
            from engine.risk.engine import RiskEngine, Portfolio, Position
            self.risk = RiskEngine()
            self._engine_status["risk"] = "loaded"
            logger.info("Risk engine loaded")
        except ImportError as e:
            self._engine_status["risk"] = f"error: {e}"
            logger.warning(f"Risk engine unavailable: {e}")

    def _init_monitor(self) -> None:
        try:
            from engine.monitor.engine import MonitorEngine
            self.monitor = MonitorEngine()
            self.monitor.setup()
            self._engine_status["monitor"] = "loaded"
            logger.info("Monitor engine loaded")
        except ImportError as e:
            self._engine_status["monitor"] = f"error: {e}"
            logger.warning(f"Monitor engine unavailable: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Status
    # ═══════════════════════════════════════════════════════════════

    def status(self) -> Dict[str, Any]:
        """Return engine status overview."""
        return {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode": self.config.mode,
            "engines": self._engine_status,
            "config": {
                "weights": {
                    "quality": self.config.weight_quality,
                    "momentum": self.config.weight_momentum,
                    "technical": self.config.weight_technical,
                    "earnings": self.config.weight_earnings,
                    "chip": self.config.weight_chip,
                },
                "mode": self.config.mode,
                "universe": self.config.default_universe,
            },
        }

    # ═══════════════════════════════════════════════════════════════
    # Pipeline Orchestration
    # ═══════════════════════════════════════════════════════════════

    def full_pipeline(
        self,
        tickers: Optional[List[str]] = None,
        universe: str = "sp500",
        max_tickers: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete VMAA pipeline: data → selection → technical → chip → risk → decision.

        Args:
            tickers: Specific tickers to scan (overrides universe)
            universe: Universe source if tickers not provided
            max_tickers: Max tickers to process
            progress: Show progress output

        Returns:
            JSON-serializable dict with full pipeline results
        """
        from engine.vmaa_v3 import run_pipeline

        start_time = time.time()
        logger.info("=" * 60)
        logger.info("VMAA FULL PIPELINE — Starting")
        logger.info("=" * 60)

        result = run_pipeline(
            engine=self,
            tickers=tickers,
            universe=universe,
            max_tickers=max_tickers or self.config.max_tickers_full,
            progress=progress,
        )

        elapsed = time.time() - start_time
        result["pipeline_meta"]["total_elapsed_seconds"] = round(elapsed, 1)
        logger.info(f"Full pipeline complete in {elapsed:.1f}s")

        return result

    def quick_scan(
        self,
        tickers: Optional[List[str]] = None,
        max_tickers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Daily quick scan — faster, with cached data and subset of engines.

        Args:
            tickers: Specific tickers or None for default universe subset
            max_tickers: Cap on tickers

        Returns:
            Abridged result dict
        """
        from engine.vmaa_v3 import run_pipeline

        start_time = time.time()
        logger.info("VMAA QUICK SCAN — Starting")

        # Quick mode: limit tickers, skip heavy engines
        max_t = max_tickers or self.config.max_tickers_quick

        result = run_pipeline(
            engine=self,
            tickers=tickers,
            max_tickers=max_t,
            mode="quick",
            progress=False,
        )

        elapsed = time.time() - start_time
        result["pipeline_meta"]["total_elapsed_seconds"] = round(elapsed, 1)
        logger.info(f"Quick scan complete in {elapsed:.1f}s")

        return result

    def portfolio_report(self) -> Dict[str, Any]:
        """
        Full portfolio assessment — risk, exposure, circuit breakers.

        Returns:
            Portfolio status dict
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "portfolio": {},
            "risk": {},
            "pool_status": {},
            "warnings": [],
        }

        # Pool status from selection engine
        if self.selection:
            try:
                pool = self.selection.get_pool_status()
                report["pool_status"] = {
                    name: {
                        "size": s.size,
                        "max": s.max_size,
                        "avg_score": s.avg_composite_score,
                    }
                    for name, s in pool.items()
                }
            except Exception as e:
                report["warnings"].append(f"pool_status: {e}")

        # Risk assessment
        if self.risk:
            try:
                # Build a simple portfolio from pool data
                from engine.risk.engine import Portfolio, Position

                positions = []
                total_mv = 100_000.0  # default
                if self.selection:
                    pool_status = self.selection.get_pool_status()
                    # Use core pool positions
                    for pool_name in ["core", "tactical"]:
                        status = pool_status.get(pool_name)
                        if status and status.stocks:
                            # Can't access individual entries easily from PoolStatus
                            pass

                if not positions:
                    # Empty portfolio — just show risk framework status
                    cb = self.risk.check_circuit_breakers()
                    report["risk"] = {
                        "engine_loaded": True,
                        "circuit_breakers": cb.to_dict() if cb else {},
                        "note": "No positions loaded — showing framework status only",
                    }
                else:
                    port = Portfolio(total_value=total_mv, cash=0, positions=positions)
                    risk_report = self.risk.assess(port)
                    report["risk"] = {
                        "risk_score": risk_report.risk_score,
                        "risk_level": risk_report.risk_level,
                        "warnings": risk_report.warnings,
                    }
            except Exception as e:
                report["warnings"].append(f"risk: {e}")
        else:
            report["risk"] = {"engine_loaded": False}

        # Monitor status
        if self.monitor:
            try:
                report["monitor"] = self.monitor.status()
            except Exception as e:
                report["warnings"].append(f"monitor: {e}")

        return report

    # ═══════════════════════════════════════════════════════════════
    # Quick sub-engine access
    # ═══════════════════════════════════════════════════════════════

    def screen(self, tickers: List[str], template: str = "quality_value") -> Dict[str, Any]:
        """Quick screening with the selection engine."""
        if not self.selection:
            return {"error": "Selection engine not loaded"}
        result = self.selection.screen_with_template(tickers, template=template)
        return {
            "timestamp": result.timestamp,
            "universe": result.universe_size,
            "passed": result.passed_count,
            "candidates": [{"ticker": t, "score": s} for t, s in result.get_top(20)],
            "elapsed": result.elapsed_seconds,
        }

    def analyze_technicals(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Batch technical analysis."""
        if not self.technical:
            return [{"error": "Technical engine not loaded"}]
        return self.technical.batch_analysis(tickers)

    def analyze_chip(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Single ticker chip analysis."""
        if not self.chip:
            return {"error": "Chip engine not loaded"}
        report = self.chip.analyze(ticker, period)
        return self.chip.analyze_to_json(ticker, period) if hasattr(self.chip, 'analyze_to_json') else "no json method"

    def assess_risk(
        self,
        tickers: List[str],
        prices: List[float],
        quantities: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Quick risk assessment."""
        if not self.risk:
            return {"error": "Risk engine not loaded"}
        from engine.risk.engine import quick_assess
        report = quick_assess(tickers, prices, quantities)
        return report.to_dict()

    # ═══════════════════════════════════════════════════════════════
    # Convenience
    # ═══════════════════════════════════════════════════════════════

    def print_status(self) -> None:
        """Print engine status to console."""
        s = self.status()
        print("\n" + "=" * 60)
        print("  VMAA Engine v3 — Integration Layer Status")
        print("=" * 60)
        print(f"  Mode:      {s['mode']}")
        print(f"  Timestamp: {s['timestamp']}")
        print(f"\n  Engine Status:")
        for eng, status in s["engines"].items():
            icon = "✅" if status == "loaded" else "❌"
            print(f"    {icon} {eng}: {status}")
        print(f"\n  Scoring Weights:")
        for k, v in s["config"]["weights"].items():
            print(f"    {k}: {v:.0%}")
        print("=" * 60)

    def get_screening_engine(self):
        """Ensure selection engine is loaded and return it."""
        if not self.selection:
            self._init_selection()
        return self.selection

    def get_technical_engine(self):
        """Ensure technical engine is loaded and return it."""
        if not self.technical:
            self._init_technical()
        return self.technical

    def get_chip_engine(self):
        """Ensure chip engine is loaded and return it."""
        if not self.chip:
            self._init_chip()
        return self.chip

    def get_risk_engine(self):
        """Ensure risk engine is loaded and return it."""
        if not self.risk:
            self._init_risk()
        return self.risk

    def get_monitor_engine(self):
        """Ensure monitor engine is loaded and return it."""
        if not self.monitor:
            self._init_monitor()
        return self.monitor


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_VMAA_ENGINE: Optional[VMAAEngine] = None


def get_engine() -> VMAAEngine:
    """Get or create the singleton VMAAEngine."""
    global _VMAA_ENGINE
    if _VMAA_ENGINE is None:
        _VMAA_ENGINE = VMAAEngine()
    return _VMAA_ENGINE


# ═══════════════════════════════════════════════════════════════════
# Quick run helpers
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(
    tickers: Optional[List[str]] = None,
    mode: str = "full",
    **kwargs,
) -> Dict[str, Any]:
    """One-liner: run the pipeline."""
    engine = get_engine()
    engine.config.mode = mode
    if mode == "quick":
        return engine.quick_scan(tickers=tickers, **kwargs)
    else:
        return engine.full_pipeline(tickers=tickers, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    "VMAAEngine",
    "EngineConfig",
    "get_engine",
    "get_engine_config",
    "update_engine_config",
    "run_pipeline",
]
