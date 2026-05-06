#!/usr/bin/env python3
"""
VMAA Risk Engine — Configuration Loader
========================================
Loads risk assessment configuration from YAML/JSON files,
with sensible defaults if file is missing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("vmaa.risk.engine.config")

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


@dataclass
class VolatilityConfig:
    windows: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    ewma_lambda: float = 0.94
    regime_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.12, "normal": 0.22, "high": 0.35, "extreme": 0.35,
    })
    term_structure: Dict[str, int] = field(default_factory=lambda: {
        "short_window": 20, "long_window": 60,
    })


@dataclass
class StressScenario:
    name: str
    description: str
    shock: Dict[str, float]


@dataclass
class VaRConfig:
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    historical_window: int = 252
    ewma_decay: float = 0.94
    monte_carlo: Dict[str, int] = field(default_factory=lambda: {
        "simulations": 2000, "horizon_days": 10,
    })
    stress_scenarios: List[StressScenario] = field(default_factory=list)


@dataclass
class ExposureConfig:
    factor_betas: Dict[str, str] = field(default_factory=lambda: {
        "market": "SPY", "size": "IWM", "value": "IWD", "growth": "IWF",
    })
    concentration: Dict[str, float] = field(default_factory=lambda: {
        "hhi_warning": 0.15, "hhi_danger": 0.25,
    })
    liquidity: Dict[str, float] = field(default_factory=lambda: {
        "max_position_pct_adv": 0.05, "adv_warning_threshold": 0.02,
    })
    correlation: Dict[str, float] = field(default_factory=lambda: {
        "max_pairwise_warning": 0.70, "max_pairwise_hard": 0.85,
    })
    tail_risk: Dict[str, float] = field(default_factory=lambda: {
        "skewness_warning": -0.50, "kurtosis_warning": 5.0,
    })


@dataclass
class FixedFractionalConfig:
    base_risk_pct: float = 0.015
    min_risk_pct: float = 0.005
    max_risk_pct: float = 0.03
    confidence_threshold: float = 0.35


@dataclass
class KellyConfig:
    fraction: float = 0.25
    min_fraction: float = 0.10
    max_fraction: float = 0.50
    min_win_probability: float = 0.30


@dataclass
class RiskParityConfig:
    target_risk_contribution: float = 1.0
    max_leverage: float = 1.5
    lookback_days: int = 60


@dataclass
class VolTargetingConfig:
    target_vol: float = 0.15
    max_scale_up: float = 2.0
    min_scale_down: float = 0.25


@dataclass
class CircuitBreakerConfig:
    daily_loss_limit: float = -0.03
    weekly_loss_limit: float = -0.06
    max_drawdown_limit: float = -0.15
    consecutive_losses: int = 5
    position_loss_limit: float = -0.20


@dataclass
class SizingConfig:
    method: str = "fixed_fractional"
    fixed_fractional: FixedFractionalConfig = field(default_factory=FixedFractionalConfig)
    kelly: KellyConfig = field(default_factory=KellyConfig)
    risk_parity: RiskParityConfig = field(default_factory=RiskParityConfig)
    volatility_targeting: VolTargetingConfig = field(default_factory=VolTargetingConfig)
    circuit_breakers: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


@dataclass
class EngineConfig:
    log_level: str = "INFO"
    report_format: str = "json"
    cache_ttl_seconds: int = 300


@dataclass
class RiskEngineConfig:
    """Master config for the Risk Assessment Engine."""
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    var: VaRConfig = field(default_factory=VaRConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)


def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: Optional[str] = None) -> RiskEngineConfig:
    """
    Load risk engine configuration.

    Priority:
      1. User-specified config_path (JSON or YAML)
      2. engine/risk/config/default.yaml
      3. Hardcoded defaults (dataclass defaults)
    """
    raw: Dict[str, Any] = {}

    # Try loading default YAML
    if DEFAULT_CONFIG.exists():
        try:
            import yaml
            with open(DEFAULT_CONFIG) as f:
                raw = yaml.safe_load(f) or {}
            logger.debug(f"Loaded default config from {DEFAULT_CONFIG}")
        except ImportError:
            # Fallback: try as JSON
            pass
        except Exception as e:
            logger.warning(f"Failed to load default config: {e}")

    # Override with user config if specified
    if config_path:
        user_path = Path(config_path)
        if user_path.exists():
            try:
                if user_path.suffix in ('.yaml', '.yml'):
                    import yaml
                    with open(user_path) as f:
                        user_raw = yaml.safe_load(f) or {}
                else:
                    with open(user_path) as f:
                        user_raw = json.load(f)
                raw = _merge_dicts(raw, user_raw)
                logger.info(f"Loaded user config from {user_path}")
            except Exception as e:
                logger.error(f"Failed to load user config {user_path}: {e}")

    return _build_config(raw)


def _build_config(raw: dict) -> RiskEngineConfig:
    """Build typed config from raw dict."""
    vol_raw = raw.get("volatility", {})
    vol_cfg = VolatilityConfig(
        windows=vol_raw.get("windows", [10, 20, 50, 100]),
        ewma_lambda=vol_raw.get("ewma_lambda", 0.94),
        regime_thresholds=vol_raw.get("regime_thresholds", {}),
        term_structure=vol_raw.get("term_structure", {}),
    )

    var_raw = raw.get("var", {})
    mc_raw = var_raw.get("monte_carlo", {})
    scenarios = []
    for s in var_raw.get("stress_scenarios", []):
        scenarios.append(StressScenario(
            name=s.get("name", ""),
            description=s.get("description", ""),
            shock=s.get("shock", {}),
        ))
    var_cfg = VaRConfig(
        confidence_levels=var_raw.get("confidence_levels", [0.95, 0.99]),
        historical_window=var_raw.get("historical_window", 252),
        ewma_decay=var_raw.get("ewma_decay", 0.94),
        monte_carlo={"simulations": mc_raw.get("simulations", 2000),
                      "horizon_days": mc_raw.get("horizon_days", 10)},
        stress_scenarios=scenarios,
    )

    exp_raw = raw.get("exposure", {})
    exp_cfg = ExposureConfig(
        factor_betas=exp_raw.get("factor_betas", {}),
        concentration=exp_raw.get("concentration", {}),
        liquidity=exp_raw.get("liquidity", {}),
        correlation=exp_raw.get("correlation", {}),
        tail_risk=exp_raw.get("tail_risk", {}),
    )

    siz_raw = raw.get("sizing", {})
    ff_raw = siz_raw.get("fixed_fractional", {})
    k_raw = siz_raw.get("kelly", {})
    rp_raw = siz_raw.get("risk_parity", {})
    vt_raw = siz_raw.get("volatility_targeting", {})
    cb_raw = siz_raw.get("circuit_breakers", {})

    siz_cfg = SizingConfig(
        method=siz_raw.get("method", "fixed_fractional"),
        fixed_fractional=FixedFractionalConfig(
            base_risk_pct=ff_raw.get("base_risk_pct", 0.015),
            min_risk_pct=ff_raw.get("min_risk_pct", 0.005),
            max_risk_pct=ff_raw.get("max_risk_pct", 0.03),
            confidence_threshold=ff_raw.get("confidence_threshold", 0.35),
        ),
        kelly=KellyConfig(
            fraction=k_raw.get("fraction", 0.25),
            min_fraction=k_raw.get("min_fraction", 0.10),
            max_fraction=k_raw.get("max_fraction", 0.50),
            min_win_probability=k_raw.get("min_win_probability", 0.30),
        ),
        risk_parity=RiskParityConfig(
            target_risk_contribution=rp_raw.get("target_risk_contribution", 1.0),
            max_leverage=rp_raw.get("max_leverage", 1.5),
            lookback_days=rp_raw.get("lookback_days", 60),
        ),
        volatility_targeting=VolTargetingConfig(
            target_vol=vt_raw.get("target_vol", 0.15),
            max_scale_up=vt_raw.get("max_scale_up", 2.0),
            min_scale_down=vt_raw.get("min_scale_down", 0.25),
        ),
        circuit_breakers=CircuitBreakerConfig(
            daily_loss_limit=cb_raw.get("daily_loss_limit", -0.03),
            weekly_loss_limit=cb_raw.get("weekly_loss_limit", -0.06),
            max_drawdown_limit=cb_raw.get("max_drawdown_limit", -0.15),
            consecutive_losses=cb_raw.get("consecutive_losses", 5),
            position_loss_limit=cb_raw.get("position_loss_limit", -0.20),
        ),
    )

    eng_raw = raw.get("engine", {})
    eng_cfg = EngineConfig(
        log_level=eng_raw.get("log_level", "INFO"),
        report_format=eng_raw.get("report_format", "json"),
        cache_ttl_seconds=eng_raw.get("cache_ttl_seconds", 300),
    )

    return RiskEngineConfig(
        volatility=vol_cfg,
        var=var_cfg,
        exposure=exp_cfg,
        sizing=siz_cfg,
        engine=eng_cfg,
    )
