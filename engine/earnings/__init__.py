#!/usr/bin/env python3
"""
VMAA Earnings Prediction Engine — Full Package
================================================
業績預測引擎: Broker reports, forecast, consensus, ratings, and scoring.

Components:
  - broker_reports: Analyst data collection (yfinance info, recommendations, earnings_dates)
  - forecast:       Surprise analysis, accuracy, growth, estimate revisions
  - consensus:      Consensus rating, target, EPS/revenue, dispersion, confidence
  - ratings:        Upgrades/downgrades, target changes, momentum, calendar, sentiment
  - config:         Configuration manager
  - engine:         EarningsEngine orchestrator

Usage:
  from engine.earnings import EarningsEngine
  
  engine = EarningsEngine()
  consensus = engine.get_consensus("AAPL")
  score = engine.score("AAPL")
  batch = engine.batch_scan(["AAPL", "MSFT", "GOOGL"])
"""
from __future__ import annotations

from engine.earnings.config import (
    EarningsConfig,
    EarningsConfigManager,
    get_earnings_config,
)
from engine.earnings.broker_reports import (
    collect_analyst_info,
    collect_broker_reports,
    collect_earnings_estimates,
    collect_earnings_surprise_history,
    collect_recommendation_history,
    normalize_rating,
    rating_to_score,
    track_broker_coverage,
    build_rating_timeline,
)
from engine.earnings.forecast import (
    analyze_forecast,
    analyze_estimate_revisions,
    analyze_growth_metrics,
    analyze_revenue_vs_eps,
    compute_forecast_accuracy,
    compute_surprise_metrics,
)
from engine.earnings.consensus import (
    build_consensus,
    compute_confidence_score,
    compute_consensus_changes,
    compute_consensus_earnings,
    compute_consensus_rating,
    compute_consensus_target,
)
from engine.earnings.ratings import (
    analyze_ratings,
    compute_pre_earnings_sentiment,
    compute_rating_momentum,
    detect_target_price_changes,
    detect_upgrade_downgrade_activity,
    get_earnings_calendar,
)
from engine.earnings.engine import (
    EarningsEngine,
    get_earnings_engine,
    quick_score,
)

__all__ = [
    # Config
    "EarningsConfig",
    "EarningsConfigManager",
    "get_earnings_config",
    # Broker Reports
    "collect_analyst_info",
    "collect_broker_reports",
    "collect_earnings_estimates",
    "collect_earnings_surprise_history",
    "collect_recommendation_history",
    "normalize_rating",
    "rating_to_score",
    "track_broker_coverage",
    "build_rating_timeline",
    # Forecast
    "analyze_forecast",
    "analyze_estimate_revisions",
    "analyze_growth_metrics",
    "analyze_revenue_vs_eps",
    "compute_forecast_accuracy",
    "compute_surprise_metrics",
    # Consensus
    "build_consensus",
    "compute_confidence_score",
    "compute_consensus_changes",
    "compute_consensus_earnings",
    "compute_consensus_rating",
    "compute_consensus_target",
    # Ratings
    "analyze_ratings",
    "compute_pre_earnings_sentiment",
    "compute_rating_momentum",
    "detect_target_price_changes",
    "detect_upgrade_downgrade_activity",
    "get_earnings_calendar",
    # Engine
    "EarningsEngine",
    "get_earnings_engine",
    "quick_score",
]
