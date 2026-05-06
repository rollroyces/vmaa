#!/usr/bin/env python3
"""
VMAA Engine — Global Configuration
====================================
Unified engine configuration with scoring weights,
pipeline modes, and engine enable/disable flags.

Usage:
    from engine.config import EngineConfig, get_engine_config
    cfg = get_engine_config()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


# ═══════════════════════════════════════════════════════════════════
# Engine Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EngineConfig:
    """Global engine configuration for VMAA integration layer."""

    # ── Engine toggles ──
    enable_selection: bool = True      # Part 1 Quality + Part 2 MAGNA screening
    enable_technical: bool = True      # Technical analysis indicators + signals
    enable_chip: bool = True           # Volume profile + chip analysis
    enable_risk: bool = True           # Risk assessment + sizing
    enable_monitor: bool = False       # Alert/order monitoring (daemon mode only)

    # ── Scoring weights for composite score ──
    weight_quality: float = 0.25       # Part 1 quality (value + FCF)
    weight_momentum: float = 0.25      # Part 2 MAGNA signals
    weight_technical: float = 0.15     # Technical indicators
    weight_earnings: float = 0.20      # Consensus earnings (from yfinance)
    weight_chip: float = 0.15          # Volume profile / chip analysis

    # ── Pipeline mode ──
    mode: str = "quick"                # "quick" | "full" | "backtest"

    # ── Data layer ──
    use_tiger_price: bool = True       # Tiger for prices (fallback to yfinance)
    use_sec_edgar: bool = True         # SEC EDGAR for fundamentals (fallback to yfinance)
    cache_ttl_minutes: int = 60        # Cache data TTL

    # ── Universe ──
    default_universe: str = "sp500"    # default universe source
    max_tickers_quick: int = 50        # max tickers in quick mode
    max_tickers_full: int = 500        # max tickers in full mode

    # ── Output ──
    output_dir: str = "engine/data"
    persist_results: bool = True

    # ── Composite thresholds ──
    min_composite_score: float = 0.35  # minimum composite to consider
    quality_pass_threshold: float = 0.40
    magna_pass_threshold: int = 3

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for fname in self.__dataclass_fields__:
            d[fname] = getattr(self, fname)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineConfig":
        valid = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in d.items() if k in valid}
        return cls(**kwargs)


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_ENGINE_CONFIG: EngineConfig | None = None


def get_engine_config() -> EngineConfig:
    """Get or create the singleton EngineConfig."""
    global _ENGINE_CONFIG
    if _ENGINE_CONFIG is None:
        _ENGINE_CONFIG = EngineConfig()
    return _ENGINE_CONFIG


def update_engine_config(**kwargs) -> EngineConfig:
    """Hot-reload engine config with kwargs."""
    global _ENGINE_CONFIG
    cfg = get_engine_config()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
