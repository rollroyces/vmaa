#!/usr/bin/env python3
"""
VMAA Smart Selection Engine — Configuration Manager
=====================================================
Loads and manages all selection engine configuration from JSON files.
Provides typed access to factor weights, pool configs, and rotation params.

Usage:
  from engine.selection.config import SelectionConfig
  cfg = SelectionConfig()
  weights = cfg.get_factor_weights()
  pool_cfg = cfg.get_pool_config("core")
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vmaa.engine.selection.config")

# Config directory relative to this file
_CONFIG_DIR = Path(__file__).resolve().parent / "config"


# ═══════════════════════════════════════════════════════════════════
# Typed config dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FactorWeightsConfig:
    """Parsed factor weights configuration."""
    use_layer_weights: bool = True
    layer_weights: Dict[str, float] = field(default_factory=dict)
    factors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    factor_directions: Dict[str, int] = field(default_factory=dict)
    scoring_method: str = "percentile"
    zscore_clip: float = 3.0
    percentile_buckets: int = 100
    min_observations: int = 30


@dataclass
class PoolConfig:
    """Parsed pool configuration for a single pool type."""
    name: str = ""
    label: str = ""
    description: str = ""
    max_size: int = 30
    min_size: int = 10
    sector_limit: int = 5
    max_single_position_pct: float = 10.0
    entry_conditions: Dict[str, Any] = field(default_factory=dict)
    exit_conditions: Dict[str, Any] = field(default_factory=dict)
    rebalance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RotationConfig:
    """Parsed rotation configuration."""
    strategies: Dict[str, Any] = field(default_factory=dict)
    triggers: Dict[str, Any] = field(default_factory=dict)
    logging_config: Dict[str, Any] = field(default_factory=dict)
    slippage: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Configuration Manager
# ═══════════════════════════════════════════════════════════════════

class SelectionConfig:
    """
    Central configuration manager for the Smart Selection Engine.
    
    Loads configs from JSON files with validation and defaults.
    Supports hot-reload for config changes.
    
    Example:
        >>> cfg = SelectionConfig()
        >>> fw = cfg.factor_weights
        >>> print(fw.layer_weights['value'])
        0.20
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self._config_dir = Path(config_dir) if config_dir else _CONFIG_DIR
        self._cache: Dict[str, Any] = {}
        
        # Load all configs
        self.factor_weights = self._load_factor_weights()
        self.pool_configs = self._load_pool_configs()
        self.rotation_config = self._load_rotation_config()
        
        logger.info(
            f"SelectionConfig loaded: {len(self.pool_configs)} pools, "
            f"{len(self.factor_weights.factors)} factor categories, "
            f"{len(self.rotation_config.strategies)} rotation strategies"
        )

    # ── Loading ──────────────────────────────────────────────────

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """Load a JSON config file with caching."""
        if filename in self._cache:
            return self._cache[filename]
        
        path = self._config_dir / filename
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return {}
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self._cache[filename] = data
        return data

    def _load_factor_weights(self) -> FactorWeightsConfig:
        """Load and parse factor weights configuration."""
        data = self._load_json("factor_weights.json")
        
        if not data:
            return FactorWeightsConfig()
        
        return FactorWeightsConfig(
            use_layer_weights=data.get("use_layer_weights", True),
            layer_weights=data.get("layer_weights", {}),
            factors=data.get("factors", {}),
            factor_directions=data.get("factor_directions", {}),
            scoring_method=data.get("scoring", {}).get("method", "percentile"),
            zscore_clip=data.get("scoring", {}).get("zscore_clip", 3.0),
            percentile_buckets=data.get("scoring", {}).get("percentile_buckets", 100),
            min_observations=data.get("scoring", {}).get("min_observations", 30),
        )

    def _load_pool_configs(self) -> Dict[str, PoolConfig]:
        """Load pool configurations from JSON."""
        data = self._load_json("pool_config.json")
        
        pools = {}
        raw_pools = data.get("pools", {})
        
        # Priority ordering
        priority = data.get("priority", list(raw_pools.keys()))
        
        for name in priority:
            if name not in raw_pools:
                continue
            cfg = raw_pools[name]
            pools[name] = PoolConfig(
                name=name,
                label=cfg.get("label", name.title()),
                description=cfg.get("description", ""),
                max_size=cfg.get("max_size", 30),
                min_size=cfg.get("min_size", 0),
                sector_limit=cfg.get("sector_limit", 0),
                max_single_position_pct=cfg.get("max_single_position_pct", 10.0),
                entry_conditions=cfg.get("entry_conditions", {}),
                exit_conditions=cfg.get("exit_conditions", {}),
                rebalance=cfg.get("rebalance", {}),
            )
        
        return pools

    def _load_rotation_config(self) -> RotationConfig:
        """Load rotation configuration from JSON."""
        data = self._load_json("rotation_config.json")
        
        return RotationConfig(
            strategies=data.get("strategies", {}),
            triggers=data.get("triggers", {}),
            logging_config=data.get("logging", {}),
            slippage=data.get("slippage", {}),
        )

    # ── Public API ───────────────────────────────────────────────

    def get_factor_weights(self) -> FactorWeightsConfig:
        """Return factor weights configuration."""
        return self.factor_weights

    def get_pool_config(self, pool_name: str) -> Optional[PoolConfig]:
        """Get configuration for a specific pool type."""
        return self.pool_configs.get(pool_name)

    def get_all_pool_configs(self) -> Dict[str, PoolConfig]:
        """Return all pool configurations."""
        return deepcopy(self.pool_configs)

    def get_rotation_config(self) -> RotationConfig:
        """Return rotation configuration."""
        return self.rotation_config

    def get_factor_direction(self, factor_name: str) -> int:
        """Get scoring direction for a factor (1 or -1)."""
        return self.factor_weights.factor_directions.get(factor_name, 1)

    def get_layer_weight(self, layer_name: str) -> float:
        """Get weight for a factor layer."""
        return self.factor_weights.layer_weights.get(layer_name, 0.0)

    def get_factor_weight(self, factor_name: str) -> Optional[float]:
        """Get weight for a specific factor by searching across layers."""
        for layer, factors in self.factor_weights.factors.items():
            if factor_name in factors:
                return factors[factor_name]
        return None

    def reload(self):
        """Hot-reload all configurations from disk."""
        self._cache.clear()
        self.factor_weights = self._load_factor_weights()
        self.pool_configs = self._load_pool_configs()
        self.rotation_config = self._load_rotation_config()
        logger.info("SelectionConfig hot-reloaded")

    def validate(self) -> List[str]:
        """Validate configuration consistency and return any issues."""
        issues = []
        
        # Check layer weights sum
        if self.factor_weights.use_layer_weights:
            total = sum(self.factor_weights.layer_weights.values())
            if abs(total - 1.0) > 0.02:
                issues.append(f"Layer weights sum to {total:.3f}, expected ~1.0")
        
        # Check factor weights sum within each layer
        for layer, factors in self.factor_weights.factors.items():
            total = sum(factors.values())
            if abs(total - 1.0) > 0.02 and len(factors) > 0:
                issues.append(f"Factor weights in '{layer}' sum to {total:.3f}, expected ~1.0")
        
        # Check all factors have directions
        for layer, factors in self.factor_weights.factors.items():
            for fname in factors:
                if fname not in self.factor_weights.factor_directions:
                    issues.append(f"Missing direction for factor '{fname}'")
        
        # Check pool configs have required fields
        for name, cfg in self.pool_configs.items():
            if cfg.max_size < cfg.min_size:
                issues.append(f"Pool '{name}': max_size({cfg.max_size}) < min_size({cfg.min_size})")
        
        return issues


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_SELECTION_CONFIG: Optional[SelectionConfig] = None


def get_selection_config() -> SelectionConfig:
    """Get or create the singleton SelectionConfig instance."""
    global _SELECTION_CONFIG
    if _SELECTION_CONFIG is None:
        _SELECTION_CONFIG = SelectionConfig()
    return _SELECTION_CONFIG
