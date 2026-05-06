#!/usr/bin/env python3
"""
VMAA Chip Engine — Capital Distribution Analysis
==================================================
籌碼分析引擎: Volume Profile, Concentration, Cost Basis, Profitability,
Support/Resistance, and Money Flow analysis.

Components:
  - distribution:  Volume Profile, Value Area (VA/VAL/VAH), POC, RVOL
  - concentration: CR(N), HHI, Volume Skew, S/R Detection
  - cost:          VWAP, Cost Distribution, Marginal/Impulse Cost
  - profitability: Floating P&L, Money Flow, CMF, Holder Health
  - engine:        ChipEngine orchestrator
  - config:        Configuration manager

Usage:
  from engine.chip import ChipEngine
  
  engine = ChipEngine()
  report = engine.analyze("AAPL", period="1y")
  print(engine.to_json(report))
"""
from __future__ import annotations

from engine.chip.config import (
    ChipConfig,
    ChipConfigManager,
    get_chip_config,
)
from engine.chip.distribution import (
    RVOLResult,
    ValueArea,
    VolumeDistributionResult,
    VolumeProfileBucket,
    analyze_distribution,
    build_time_weighted_profile,
    build_volume_profile,
    compute_rvol,
)
from engine.chip.concentration import (
    ConcentrationResult,
    SupportResistanceLevel,
    SupportResistanceResult,
    compute_concentration,
    detect_support_resistance,
)
from engine.chip.cost import (
    CostBasisResult,
    CostBucket,
    analyze_cost_basis,
    compute_vwap,
)
from engine.chip.profitability import (
    MoneyFlowResult,
    ProfitabilityResult,
    ProfitabilitySummary,
    compute_full_profitability,
    compute_money_flow,
)
from engine.chip.engine import (
    ChipEngine,
    ChipReport,
    get_chip_engine,
    quick_chip,
    quick_chip_json,
)

__all__ = [
    # Config
    "ChipConfig",
    "ChipConfigManager",
    "get_chip_config",
    # Distribution
    "VolumeProfileBucket",
    "ValueArea",
    "RVOLResult",
    "VolumeDistributionResult",
    "build_volume_profile",
    "compute_rvol",
    "build_time_weighted_profile",
    "analyze_distribution",
    # Concentration
    "ConcentrationResult",
    "SupportResistanceLevel",
    "SupportResistanceResult",
    "compute_concentration",
    "detect_support_resistance",
    # Cost
    "CostBucket",
    "CostBasisResult",
    "compute_vwap",
    "analyze_cost_basis",
    # Profitability
    "MoneyFlowResult",
    "ProfitabilitySummary",
    "ProfitabilityResult",
    "compute_money_flow",
    "compute_full_profitability",
    # Engine
    "ChipReport",
    "ChipEngine",
    "get_chip_engine",
    "quick_chip",
    "quick_chip_json",
]
