#!/usr/bin/env python3
"""
VMAA Technical Analysis Engine
===============================
Complete technical analysis framework for the VMAA trading system.

Modules:
    indicators  — 40+ technical indicators (trend, momentum, volatility, volume, composite)
    registry    — Indicator registry with metadata and plugin architecture
    custom      — Custom indicator builder with formula parser
    signals     — Signal generation from individual and multi-indicator combinations
    config      — Default parameters and thresholds
    analysis    — Orchestrator tying everything together

Quick Start:
    >>> from engine.technical import TechnicalEngine
    >>> engine = TechnicalEngine()
    >>> df = yf.download("AAPL", period="6mo")
    >>> result = engine.compute(df)          # All indicators
    >>> signals = engine.get_signals(df)      # Trading signals
    >>> custom = engine.custom(df, "MA(CLOSE,5) - MA(CLOSE,20)")  # Custom formula
"""

from engine.technical.config import TechnicalConfig, TC
from engine.technical.indicators import compute_all
from engine.technical.registry import (
    IndicatorRegistry,
    IndicatorMeta,
    IndicatorCategory,
    get_registry,
    create_default_registry,
)
from engine.technical.custom import (
    custom_indicator,
    validate_formula,
    add_custom_indicator,
    remove_custom_indicator,
    load_custom_indicators,
    list_available_functions,
    CustomIndicator,
)
from engine.technical.signals import (
    generate_all_signals,
    SignalResult,
    SignalType,
    SignalTracker,
    IndicatorSignal,
)
from engine.technical.analysis import TechnicalEngine

__all__ = [
    # Engine
    "TechnicalEngine",
    # Config
    "TechnicalConfig",
    "TC",
    # Indicators
    "compute_all",
    # Registry
    "IndicatorRegistry",
    "IndicatorMeta",
    "IndicatorCategory",
    "get_registry",
    "create_default_registry",
    # Custom
    "custom_indicator",
    "validate_formula",
    "add_custom_indicator",
    "remove_custom_indicator",
    "load_custom_indicators",
    "list_available_functions",
    "CustomIndicator",
    # Signals
    "generate_all_signals",
    "SignalResult",
    "SignalType",
    "SignalTracker",
    "IndicatorSignal",
]
