#!/usr/bin/env python3
"""
VMAA Chip Analysis Engine — Configuration
==========================================
Typed configuration for the capital distribution analysis engine.
Loads from config.json with sensible defaults.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vmaa.engine.chip.config")

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass
class ChipConfig:
    """Central configuration for the Chip Analysis Engine."""

    # ── Volume Profile Settings ──
    volume_profile_bins: int = 52          # Number of price buckets (~1 bin/week for 1y)
    volume_profile_lookback: int = 252      # Trading days for VP (default 1y)
    value_area_pct: float = 0.70            # Value Area percentage (68-70% typical)

    # ── Concentration Settings ──
    concentration_top_n: int = 5            # CR(N): top N price levels
    hhi_normalization: bool = True          # Normalize HHI to 0-1 range

    # ── Cost Basis Settings ──
    cost_lookback_days: int = 252           # Days for cost distribution calculation
    cost_impulse_days: int = 5              # Days for impulse cost (recent volume)
    cost_marginal_days: int = 20            # Days for marginal cost

    # ── Profitability Settings ──
    cmf_period: int = 21                    # Chaikin Money Flow period

    # ── Support/Resistance Settings ──
    sr_min_cluster_gap_pct: float = 0.02    # Min % gap between S/R clusters
    sr_top_clusters: int = 5                # Number of top S/R zones to return

    # ── Data Source Settings ──
    default_period: str = "1y"              # Default yfinance period
    min_data_points: int = 60               # Minimum data points required

    # ── Output Settings ──
    decimal_places: int = 4                 # Rounding precision for output


# ═══════════════════════════════════════════════════════════════════
# Config Manager
# ═══════════════════════════════════════════════════════════════════

class ChipConfigManager:
    """Loads and manages chip engine configuration from JSON."""

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = Path(config_path) if config_path else _CONFIG_PATH
        self._data: dict = {}
        self._load()

    def _load(self):
        if self._config_path.exists():
            try:
                with open(self._config_path, "r") as f:
                    self._data = json.load(f)
                logger.info(f"ChipConfig loaded from {self._config_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load chip config: {e}, using defaults")
                self._data = {}
        else:
            logger.info(f"ChipConfig not found at {self._config_path}, using defaults")
            self._data = {}

    def to_dataclass(self) -> ChipConfig:
        """Build a ChipConfig dataclass from loaded JSON or defaults."""
        vp = self._data.get("volume_profile", {})
        conc = self._data.get("concentration", {})
        cost_kw = self._data.get("cost_basis", {})
        prof = self._data.get("profitability", {})
        sr_kw = self._data.get("support_resistance", {})
        data_kw = self._data.get("data_source", {})
        out = self._data.get("output", {})

        return ChipConfig(
            volume_profile_bins=vp.get("bins", 52),
            volume_profile_lookback=vp.get("lookback_days", 252),
            value_area_pct=vp.get("value_area_pct", 0.70),
            concentration_top_n=conc.get("top_n", 5),
            hhi_normalization=conc.get("hhi_normalization", True),
            cost_lookback_days=cost_kw.get("lookback_days", 252),
            cost_impulse_days=cost_kw.get("impulse_days", 5),
            cost_marginal_days=cost_kw.get("marginal_days", 20),
            cmf_period=prof.get("cmf_period", 21),
            sr_min_cluster_gap_pct=sr_kw.get("min_cluster_gap_pct", 0.02),
            sr_top_clusters=sr_kw.get("top_clusters", 5),
            default_period=data_kw.get("default_period", "1y"),
            min_data_points=data_kw.get("min_data_points", 60),
            decimal_places=out.get("decimal_places", 4),
        )

    def reload(self):
        """Hot-reload configuration from disk."""
        self._data = {}
        self._load()


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_CHIP_CONFIG_MANAGER: Optional[ChipConfigManager] = None


def get_chip_config() -> ChipConfig:
    """Get or create the singleton chip configuration."""
    global _CHIP_CONFIG_MANAGER
    if _CHIP_CONFIG_MANAGER is None:
        _CHIP_CONFIG_MANAGER = ChipConfigManager()
    return _CHIP_CONFIG_MANAGER.to_dataclass()
