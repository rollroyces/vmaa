#!/usr/bin/env python3
"""
VMAA Risk Assessment Engine
============================
Comprehensive portfolio risk analytics for the Value Mean-reversion
Algorithmic Advisor (VMAA).

Modules:
  config      — YAML/JSON configuration loader
  volatility  — Multi-method volatility estimation
  var         — Value at Risk (VaR) models
  exposure    — Risk exposure & factor analysis
  sizing      — Position sizing with circuit breakers
  engine      — Orchestrator (RiskEngine)

Quick Start:
  from vmaa.engine.risk import RiskEngine, Portfolio, Position

  engine = RiskEngine()

  # Full assessment
  report = engine.assess(portfolio, price_data)
  print(report.to_json())

  # VaR only
  var_breakdown = engine.get_var(positions, price_data)

  # Sizing
  sizing = engine.suggest_sizing(candidates, portfolio_value)

  # Circuit breakers
  cb = engine.check_circuit_breakers()

  # Stress test
  stress = engine.run_stress_test(portfolio)

Integration with existing vmaa modules:
  from vmaa.config import RC as RiskCfg
  from vmaa.risk import get_market_regime, generate_trade_decision
  # The risk engine extends (not replaces) these.
"""
from __future__ import annotations

from .config import (
    RiskEngineConfig,
    VolatilityConfig,
    VaRConfig,
    ExposureConfig,
    SizingConfig,
    FixedFractionalConfig,
    KellyConfig,
    RiskParityConfig,
    VolTargetingConfig,
    CircuitBreakerConfig,
    EngineConfig,
    StressScenario,
    load_config,
)

from .volatility import (
    VolatilityCalculator,
    VolatilityResult,
    portfolio_volatility,
    volatility_contribution,
)

from .var import (
    VaRCalculator,
    VaRResult,
)

from .exposure import (
    ExposureAnalyzer,
    ExposureResult,
    FactorExposure,
    ConcentrationMetrics,
    SectorExposure,
    LiquidityRisk,
    TailRisk,
    build_position_details,
)

from .sizing import (
    PositionSizer,
    SizeRecommendation,
    CircuitBreakerStatus,
    estimate_win_probability,
)

from .engine import (
    RiskEngine,
    Portfolio,
    Position,
    RiskReport,
    create_engine,
    quick_assess,
)

__all__ = [
    # Config
    "RiskEngineConfig",
    "VolatilityConfig",
    "VaRConfig",
    "ExposureConfig",
    "SizingConfig",
    "FixedFractionalConfig",
    "KellyConfig",
    "RiskParityConfig",
    "VolTargetingConfig",
    "CircuitBreakerConfig",
    "EngineConfig",
    "StressScenario",
    "load_config",
    # Volatility
    "VolatilityCalculator",
    "VolatilityResult",
    "portfolio_volatility",
    "volatility_contribution",
    # VaR
    "VaRCalculator",
    "VaRResult",
    # Exposure
    "ExposureAnalyzer",
    "ExposureResult",
    "FactorExposure",
    "ConcentrationMetrics",
    "SectorExposure",
    "LiquidityRisk",
    "TailRisk",
    "build_position_details",
    # Sizing
    "PositionSizer",
    "SizeRecommendation",
    "CircuitBreakerStatus",
    "estimate_win_probability",
    # Engine
    "RiskEngine",
    "Portfolio",
    "Position",
    "RiskReport",
    "create_engine",
    "quick_assess",
]
