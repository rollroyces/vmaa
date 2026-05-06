#!/usr/bin/env python3
"""
VMAA Earnings Prediction Engine — Configuration
=================================================
Typed configuration for the earnings prediction engine.
Loads from config.json with sensible defaults.

Usage:
    from engine.earnings.config import EarningsConfig, get_earnings_config
    cfg = get_earnings_config()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vmaa.engine.earnings.config")

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass
class EarningsScoringWeights:
    """Scoring weight distribution for earnings quality score."""
    consensus_rating_weight: float = 0.30
    surprise_history_weight: float = 0.25
    estimate_revision_weight: float = 0.25
    dispersion_weight: float = 0.20

    def validate(self) -> None:
        total = (
            self.consensus_rating_weight
            + self.surprise_history_weight
            + self.estimate_revision_weight
            + self.dispersion_weight
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


@dataclass
class EarningsSurpriseConfig:
    """Surprise analysis configuration."""
    lookback_quarters: int = 8
    short_lookback_quarters: int = 4
    material_threshold_pct: float = 0.02


@dataclass
class EarningsConsensusConfig:
    """Consensus estimate configuration."""
    min_analysts_for_confidence: int = 3
    max_dispersion_for_confidence: float = 0.15


@dataclass
class RatingChangesConfig:
    """Rating change tracking configuration."""
    revision_window_days_1w: int = 7
    revision_window_days_1m: int = 30
    revision_window_days_3m: int = 90


@dataclass
class EarningsCacheConfig:
    """Cache configuration."""
    ttl_seconds: int = 21600  # 6 hours
    data_dir: str = "engine/data/earnings"


@dataclass
class EarningsOutputConfig:
    """Output formatting configuration."""
    decimal_places_score: int = 2
    decimal_places_pct: int = 4
    decimal_places_price: int = 2


@dataclass
class EarningsConfig:
    """Central configuration for the Earnings Prediction Engine."""

    scoring: EarningsScoringWeights = field(default_factory=EarningsScoringWeights)
    surprise: EarningsSurpriseConfig = field(default_factory=EarningsSurpriseConfig)
    consensus: EarningsConsensusConfig = field(default_factory=EarningsConsensusConfig)
    rating_changes: RatingChangesConfig = field(default_factory=RatingChangesConfig)
    cache: EarningsCacheConfig = field(default_factory=EarningsCacheConfig)
    output: EarningsOutputConfig = field(default_factory=EarningsOutputConfig)


# ═══════════════════════════════════════════════════════════════════
# Config Manager
# ═══════════════════════════════════════════════════════════════════

class EarningsConfigManager:
    """Loads and manages earnings engine configuration from JSON."""

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = Path(config_path) if config_path else _CONFIG_PATH
        self._data: dict = {}
        self._load()

    def _load(self):
        if self._config_path.exists():
            try:
                with open(self._config_path, "r") as f:
                    self._data = json.load(f)
                logger.info(f"EarningsConfig loaded from {self._config_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load earnings config: {e}, using defaults")
                self._data = {}
        else:
            logger.info(f"EarningsConfig not found at {self._config_path}, using defaults")
            self._data = {}

    def to_dataclass(self) -> EarningsConfig:
        """Build an EarningsConfig dataclass from loaded JSON or defaults."""
        scoring = self._data.get("scoring", {})
        surprise = self._data.get("surprise", {})
        consensus = self._data.get("consensus", {})
        rating = self._data.get("rating_changes", {})
        cache = self._data.get("cache", {})
        output = self._data.get("output", {})

        return EarningsConfig(
            scoring=EarningsScoringWeights(
                consensus_rating_weight=scoring.get("consensus_rating_weight", 0.30),
                surprise_history_weight=scoring.get("surprise_history_weight", 0.25),
                estimate_revision_weight=scoring.get("estimate_revision_weight", 0.25),
                dispersion_weight=scoring.get("dispersion_weight", 0.20),
            ),
            surprise=EarningsSurpriseConfig(
                lookback_quarters=surprise.get("lookback_quarters", 8),
                short_lookback_quarters=surprise.get("short_lookback_quarters", 4),
                material_threshold_pct=surprise.get("material_threshold_pct", 0.02),
            ),
            consensus=EarningsConsensusConfig(
                min_analysts_for_confidence=consensus.get("min_analysts_for_confidence", 3),
                max_dispersion_for_confidence=consensus.get("max_dispersion_for_confidence", 0.15),
            ),
            rating_changes=RatingChangesConfig(
                revision_window_days_1w=rating.get("revision_window_days_1w", 7),
                revision_window_days_1m=rating.get("revision_window_days_1m", 30),
                revision_window_days_3m=rating.get("revision_window_days_3m", 90),
            ),
            cache=EarningsCacheConfig(
                ttl_seconds=cache.get("ttl_seconds", 21600),
                data_dir=cache.get("data_dir", "engine/data/earnings"),
            ),
            output=EarningsOutputConfig(
                decimal_places_score=output.get("decimal_places_score", 2),
                decimal_places_pct=output.get("decimal_places_pct", 4),
                decimal_places_price=output.get("decimal_places_price", 2),
            ),
        )

    def reload(self):
        """Hot-reload configuration from disk."""
        self._data = {}
        self._load()


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_EARNINGS_CONFIG_MANAGER: Optional[EarningsConfigManager] = None


def get_earnings_config() -> EarningsConfig:
    """Get or create the singleton earnings configuration."""
    global _EARNINGS_CONFIG_MANAGER
    if _EARNINGS_CONFIG_MANAGER is None:
        _EARNINGS_CONFIG_MANAGER = EarningsConfigManager()
    return _EARNINGS_CONFIG_MANAGER.to_dataclass()
