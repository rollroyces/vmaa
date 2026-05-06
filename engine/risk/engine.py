#!/usr/bin/env python3
"""
VMAA Risk Engine — Orchestrator
=================================
RiskEngine: Central risk assessment coordinator.

Integrates volatility, VaR, exposure, and sizing into a unified
risk assessment pipeline with JSON-serializable output.

Usage:
    engine = RiskEngine()
    report = engine.assess(portfolio)
    var_breakdown = engine.get_var(positions)
    sizing = engine.suggest_sizing(candidates)
    cb_status = engine.check_circuit_breakers()
    stress_results = engine.run_stress_test(portfolio, scenario_name)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RiskEngineConfig, load_config
from .volatility import VolatilityCalculator, VolatilityResult, portfolio_volatility
from .var import VaRCalculator, VaRResult
from .exposure import ExposureAnalyzer, ExposureResult, build_position_details
from .sizing import (
    PositionSizer, SizeRecommendation, CircuitBreakerStatus,
    estimate_win_probability,
)

logger = logging.getLogger("vmaa.risk.engine")


# ═══════════════════════════════════════════════════════════════════
# Portfolio data structure
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Simple position representation for the risk engine."""
    ticker: str
    quantity: int
    entry_price: float
    current_price: float
    sector: str = "Unknown"
    currency: str = "USD"
    avg_volume: float = 0.0          # Average daily share volume

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price / self.entry_price) - 1.0


@dataclass
class Portfolio:
    """Portfolio state for risk assessment."""
    total_value: float
    cash: float
    positions: List[Position]
    name: str = "Portfolio"

    @property
    def invested_value(self) -> float:
        return sum(p.market_value for p in self.positions)

    @property
    def tickers(self) -> List[str]:
        return [p.ticker for p in self.positions]

    @property
    def weights(self) -> Dict[str, float]:
        if self.total_value <= 0:
            return {}
        return {p.ticker: p.market_value / self.total_value for p in self.positions}

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "invested": round(self.invested_value, 2),
            "n_positions": self.n_positions,
            "positions": [
                {
                    "ticker": p.ticker,
                    "quantity": p.quantity,
                    "entry_price": round(p.entry_price, 2),
                    "current_price": round(p.current_price, 2),
                    "market_value": round(p.market_value, 2),
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(p.unrealized_pnl_pct, 4),
                    "weight": round(p.market_value / max(self.total_value, 1), 4),
                    "sector": p.sector,
                    "currency": p.currency,
                }
                for p in self.positions
            ],
        }


# ═══════════════════════════════════════════════════════════════════
# Risk Report
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RiskReport:
    """Complete risk assessment report."""
    timestamp: str = ""
    engine_version: str = "1.0.0"

    # Portfolio summary
    portfolio: Dict[str, Any] = field(default_factory=dict)

    # Volatility
    volatility: Optional[Dict[str, Any]] = None
    portfolio_vol: Optional[Dict[str, Any]] = None

    # VaR
    var: Optional[Dict[str, Any]] = None

    # Exposure
    exposure: Optional[Dict[str, Any]] = None

    # Sizing recommendations
    sizing_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    circuit_breakers: Optional[Dict[str, Any]] = None

    # Stress tests
    stress_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Warnings & summary
    warnings: List[str] = field(default_factory=list)
    risk_score: float = 0.0            # 0-100 composite risk score
    risk_level: str = "UNKNOWN"        # LOW | MODERATE | HIGH | CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "engine_version": self.engine_version,
            "portfolio": self.portfolio,
            "volatility": self.volatility,
            "portfolio_volatility": self.portfolio_vol,
            "value_at_risk": self.var,
            "exposure": self.exposure,
            "sizing_recommendations": self.sizing_recommendations,
            "circuit_breakers": self.circuit_breakers,
            "stress_tests": self.stress_tests,
            "warnings": self.warnings,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
        }

    def to_json(self, pretty: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=2 if pretty else None, default=str)


# ═══════════════════════════════════════════════════════════════════
# Risk Engine
# ═══════════════════════════════════════════════════════════════════

class RiskEngine:
    """
    Central risk assessment orchestrator.

    Integrates volatility, VaR, exposure, and sizing analysis
    for comprehensive portfolio risk assessment.

    Usage:
        engine = RiskEngine()
        engine.assess(portfolio)           # Full risk report
        engine.get_var(positions)          # VaR breakdown only
        engine.suggest_sizing(candidates)  # Position sizing
        engine.check_circuit_breakers()    # Circuit breaker status
        engine.run_stress_test(portfolio)  # Stress test
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[RiskEngineConfig] = None,
    ):
        """
        Initialize risk engine.

        Args:
            config_path: Path to custom config file
            config: Pre-built RiskEngineConfig
        """
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Initialize sub-components
        self.vol_calc = VolatilityCalculator(self.config.volatility)
        self.var_calc = VaRCalculator(self.config.var)
        self.exp_analyzer = ExposureAnalyzer(self.config.exposure)
        self.sizer = PositionSizer(self.config.sizing)

        # Configure logging
        log_level = getattr(logging, self.config.engine.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        # Trade history for circuit breakers (in-memory for now)
        self._trade_history: List[Dict[str, Any]] = []
        self._portfolio_history: Dict[str, float] = {}

    # ── Full Assessment ───────────────────────────────────────────

    def assess(
        self,
        portfolio: Portfolio,
        price_data: Optional[Dict[str, pd.DataFrame]] = None,
        market_data: Optional[pd.DataFrame] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
    ) -> RiskReport:
        """
        Full portfolio risk assessment.

        Args:
            portfolio: Portfolio with positions
            price_data: Dict of {ticker: OHLCV DataFrame}
            market_data: Market/benchmark price data (SPY)
            confidence_scores: {ticker: confidence} for sizing

        Returns:
            RiskReport with all metrics
        """
        report = RiskReport(
            timestamp=datetime.now().isoformat(),
            engine_version="1.0.0",
        )
        report.portfolio = portfolio.to_dict()
        warnings: List[str] = []

        if not portfolio.positions:
            report.warnings = ["Empty portfolio — no risk assessment possible"]
            report.risk_level = "UNKNOWN"
            return report

        # ── 1. Volatility Analysis ──
        if price_data:
            vol_results = {}
            for ticker, df in price_data.items():
                if df is not None and not df.empty:
                    try:
                        vr = self.vol_calc.compute(df, ticker=ticker)
                        vol_results[ticker] = vr.to_dict()
                    except Exception as e:
                        logger.warning(f"Volatility calc failed for {ticker}: {e}")
                        warnings.append(f"Volatility: {ticker} failed — {e}")

            report.volatility = {
                "positions": vol_results,
                "count": len(vol_results),
            }

        # Portfolio volatility (if we have multi-asset returns)
        if price_data and len(price_data) >= 2:
            try:
                returns_dict = {}
                for ticker, df in price_data.items():
                    if df is not None and 'Close' in df.columns:
                        returns_dict[ticker] = df['Close'].pct_change().dropna()

                if returns_dict:
                    returns_matrix = pd.DataFrame(returns_dict).dropna()
                    weights = np.array([
                        portfolio.weights.get(col, 0.0)
                        for col in returns_matrix.columns
                    ])
                    if weights.sum() > 0:
                        weights = weights / weights.sum()
                        port_vol = portfolio_volatility(returns_matrix, weights)

                        # Vol contribution
                        from .volatility import volatility_contribution
                        vol_contrib = volatility_contribution(returns_matrix, weights)

                        report.portfolio_vol = {
                            "annualized_volatility": port_vol,
                            "volatility_contributions": vol_contrib,
                        }
            except Exception as e:
                logger.warning(f"Portfolio vol calc failed: {e}")

        # ── 2. VaR Analysis ──
        if price_data and portfolio.total_value > 0:
            try:
                # Compute portfolio returns
                returns_dict = {}
                for ticker, df in price_data.items():
                    if df is not None and 'Close' in df.columns:
                        returns_dict[ticker] = df['Close'].pct_change().dropna()

                if returns_dict:
                    returns_matrix = pd.DataFrame(returns_dict).dropna()
                    weights = np.array([
                        portfolio.weights.get(col, 0.0)
                        for col in returns_matrix.columns
                    ])

                    if len(weights) > 0 and weights.sum() > 0:
                        weights = weights / weights.sum()

                    # Portfolio returns
                    if returns_matrix.shape[1] > 0:
                        port_returns = (returns_matrix.values @ weights) if len(weights) > 0 else returns_matrix.iloc[:, 0]
                        port_returns_series = pd.Series(port_returns, index=returns_matrix.index)

                        var_result = self.var_calc.compute(
                            returns=port_returns_series,
                            portfolio_value=portfolio.total_value,
                            label=portfolio.name,
                            returns_matrix=returns_matrix,
                            weights=weights,
                        )
                        report.var = var_result.to_dict()

                        # VaR warnings
                        var99 = var_result.hist_var_pct.get(0.99, 0)
                        if abs(var99) > 0.05:
                            warnings.append(f"High 99% VaR: {var99:.1%} daily")
            except Exception as e:
                logger.warning(f"VaR calc failed: {e}")
                warnings.append(f"VaR: calculation failed — {e}")

        # ── 3. Exposure Analysis ──
        try:
            sectors = {p.ticker: p.sector for p in portfolio.positions}
            weights = portfolio.weights
            currencies = {p.ticker: p.currency for p in portfolio.positions}

            # Build position details
            pd_dict = build_position_details(
                tickers=[p.ticker for p in portfolio.positions],
                market_values=[p.market_value for p in portfolio.positions],
                prices=[p.current_price for p in portfolio.positions],
                avg_volumes=[p.avg_volume for p in portfolio.positions],
            )

            # Asset returns matrix
            returns_dict = {}
            if price_data:
                for ticker, df in price_data.items():
                    if df is not None and 'Close' in df.columns and ticker in weights:
                        returns_dict[ticker] = df['Close'].pct_change().dropna()

            asset_returns = pd.DataFrame(returns_dict).dropna() if returns_dict else pd.DataFrame()

            # Market returns
            market_returns = None
            if market_data is not None and 'Close' in market_data.columns:
                market_returns = market_data['Close'].pct_change().dropna()

            exp_result = self.exp_analyzer.analyze(
                label=portfolio.name,
                portfolio_value=portfolio.total_value,
                weights=weights,
                asset_returns=asset_returns if not asset_returns.empty else pd.DataFrame(
                    {t: [0.0] for t in weights}, index=[0]
                ),
                sectors=sectors,
                market_returns=market_returns,
                position_details=pd_dict if any(p.avg_volume > 0 for p in portfolio.positions) else None,
                currencies=currencies,
            )
            report.exposure = exp_result.to_dict()
            warnings.extend(exp_result.warnings)
        except Exception as e:
            logger.warning(f"Exposure analysis failed: {e}")
            warnings.append(f"Exposure: analysis failed — {e}")

        # ── 4. Sizing Recommendations ──
        for pos in portfolio.positions:
            try:
                conf = (confidence_scores or {}).get(pos.ticker, 0.50)
                rec = self.sizer.suggest(
                    ticker=pos.ticker,
                    entry_price=pos.entry_price,
                    stop_loss=pos.entry_price * 0.90,  # Default 10% stop
                    portfolio_value=portfolio.total_value,
                    confidence=conf,
                )
                report.sizing_recommendations.append(rec.to_dict())
            except Exception as e:
                logger.warning(f"Sizing failed for {pos.ticker}: {e}")

        # ── 5. Circuit Breakers ──
        cb_status = self.check_circuit_breakers()
        report.circuit_breakers = cb_status.to_dict()
        if not cb_status.trading_allowed:
            warnings.append(f"Circuit breaker active: {cb_status.recommendation}")

        # ── 6. Stress Tests ──
        stress = self.run_stress_test(portfolio)
        report.stress_tests = stress

        # ── 7. Composite Risk Score ──
        report.risk_score, report.risk_level = self._compute_risk_score(report, warnings)

        report.warnings = warnings
        return report

    # ── VaR Breakdown ─────────────────────────────────────────────

    def get_var(
        self,
        positions: List[Position],
        price_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Compute VaR breakdown for positions.

        Returns per-position and portfolio-level VaR.
        """
        result: Dict[str, Any] = {
            "positions": {},
            "portfolio": None,
            "timestamp": datetime.now().isoformat(),
        }

        # Portfolio returns
        returns_dict = {}
        for ticker, df in price_data.items():
            if df is not None and 'Close' in df.columns:
                returns_dict[ticker] = df['Close'].pct_change().dropna()

        if not returns_dict:
            return result

        returns_matrix = pd.DataFrame(returns_dict).dropna()
        total_value = sum(p.market_value for p in positions)

        weights = []
        for col in returns_matrix.columns:
            pos = next((p for p in positions if p.ticker == col), None)
            weights.append(pos.market_value / max(total_value, 1) if pos else 0.0)

        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Per-position VaR
        for col in returns_matrix.columns:
            ret = returns_matrix[col].dropna()
            va_r = self.var_calc.compute(
                ret, portfolio_value=total_value * weights[list(returns_matrix.columns).index(col)],
                label=col,
            )
            result["positions"][col] = va_r.to_dict()

        # Portfolio VaR
        port_returns = (returns_matrix.values @ weights)
        port_series = pd.Series(port_returns, index=returns_matrix.index)
        port_var = self.var_calc.compute(
            port_series, total_value, label="Portfolio",
            returns_matrix=returns_matrix, weights=weights,
        )
        result["portfolio"] = port_var.to_dict()

        return result

    # ── Sizing Suggestions ────────────────────────────────────────

    def suggest_sizing(
        self,
        candidates: List[Dict[str, Any]],
        portfolio_value: float,
        market_ok: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate position sizing recommendations for candidates.

        Args:
            candidates: List of {ticker, entry_price, stop_loss, confidence, quality_score, magna_score}
            portfolio_value: Total portfolio value
            market_ok: Market regime favorable

        Returns:
            List of sizing recommendation dicts
        """
        results = []
        for c in candidates:
            ticker = c["ticker"]
            confidence = c.get("confidence", 0.50)
            quality_score = c.get("quality_score", 0.50)
            magna_score = c.get("magna_score", 5)

            win_prob = estimate_win_probability(
                quality_score, magna_score, confidence, market_ok
            )

            rec = self.sizer.suggest(
                ticker=ticker,
                entry_price=c.get("entry_price", 0),
                stop_loss=c.get("stop_loss", c.get("entry_price", 0) * 0.90),
                portfolio_value=portfolio_value,
                confidence=confidence,
                win_probability=win_prob,
                avg_win=c.get("avg_win", 0.15),
                avg_loss=c.get("avg_loss", 0.10),
            )
            results.append(rec.to_dict())

        return results

    # ── Circuit Breakers ──────────────────────────────────────────

    def check_circuit_breakers(
        self,
        trade_history: Optional[List[Dict[str, Any]]] = None,
    ) -> CircuitBreakerStatus:
        """
        Check all circuit breakers.

        Uses internal trade history unless overridden.
        """
        history = trade_history if trade_history is not None else self._trade_history
        return self.sizer.check_circuit_breakers(
            trade_history=history,
            portfolio_history=self._portfolio_history if self._portfolio_history else None,
        )

    def record_trade(self, ticker: str, pnl: float, close_date: str):
        """Record a completed trade for circuit breaker tracking."""
        self._trade_history.append({
            "ticker": ticker,
            "pnl": pnl,
            "close_date": close_date,
        })

    def record_portfolio_snapshot(self, value: float, date: Optional[str] = None):
        """Record portfolio value for drawdown tracking."""
        key = date or datetime.now().strftime("%Y-%m-%d")
        self._portfolio_history[key] = value

    # ── Stress Testing ────────────────────────────────────────────

    def run_stress_test(
        self,
        portfolio: Portfolio,
        scenario_name: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run stress tests on portfolio.

        Args:
            portfolio: Portfolio to stress test
            scenario_name: Specific scenario (None = all scenarios)

        Returns:
            Dict[scenario_name → stress_result]
        """
        scenarios = self.config.var.stress_scenarios

        if scenario_name:
            scenarios = [s for s in scenarios if s.name == scenario_name]
            if not scenarios:
                return {}

        # Build weights and approximate returns
        weights = np.array([portfolio.weights.get(p.ticker, 0.0) for p in portfolio.positions])
        if len(weights) == 0:
            weights = np.array([p.market_value / max(portfolio.total_value, 1) for p in portfolio.positions])

        # Dummy returns matrix for the stress test
        n_pos = len(portfolio.positions)
        dummy_returns = pd.DataFrame(
            np.zeros((10, n_pos)),
            columns=[p.ticker for p in portfolio.positions],
        )

        return self.var_calc.stress_test(
            portfolio_value=portfolio.total_value,
            returns_matrix=dummy_returns if n_pos > 0 else None,
            weights=weights if n_pos > 0 else None,
            scenarios=scenarios,
        )

    # ── Risk Score Computation ────────────────────────────────────

    def _compute_risk_score(
        self,
        report: RiskReport,
        warnings: List[str],
    ) -> Tuple[float, str]:
        """
        Compute composite risk score (0-100, higher = riskier).

        Components:
          - VaR (30%): How bad is the worst-case?
          - Concentration (20%): How diversified?
          - Volatility (20%): Current vol regime
          - Liquidity (15%): Can we exit positions?
          - Circuit breakers (10%): Are any tripped?
          - Warnings (5%): Count of warnings
        """
        score = 0.0

        # VaR contribution
        if report.var:
            var99_pct = abs(report.var.get("historical_var_pct", {}).get("var99", 0.02))
            # Map: var99_pct 0-10% → score 0-30
            var_score = min(var99_pct / 0.10, 1.0) * 30
            score += var_score

        # Concentration contribution
        if report.exposure and report.exposure.get("concentration"):
            hhi = report.exposure["concentration"].get("hhi", 0.1)
            # HHI 0-0.5 → score 0-20
            conc_score = min(hhi / 0.30, 1.0) * 20
            score += conc_score

        # Volatility contribution
        if report.volatility:
            for ticker, vol in report.volatility.get("positions", {}).items():
                curr_vol = vol.get("current_volatility", 0.15)
                vol_regime = vol.get("volatility_regime", "NORMAL")
                regime_scores = {"LOW": 0, "NORMAL": 5, "HIGH": 15, "EXTREME": 20}
                score += regime_scores.get(vol_regime, 5) / max(len(report.volatility["positions"]), 1)

        # Liquidity contribution
        if report.exposure and report.exposure.get("liquidity_risks"):
            liq_score = (1 - report.exposure.get("portfolio_liquidity_score", 1.0))
            score += liq_score * 15

        # Circuit breaker contribution
        if report.circuit_breakers:
            active = len(report.circuit_breakers.get("active_breakers", []))
            score += min(active * 3, 10)

        # Warnings contribution
        score += min(len(warnings) * 1, 5)

        score = round(min(score, 100.0), 1)

        # Risk level
        if score < 20:
            level = "LOW"
        elif score < 45:
            level = "MODERATE"
        elif score < 70:
            level = "HIGH"
        else:
            level = "CRITICAL"

        return score, level


# ═══════════════════════════════════════════════════════════════════
# Quick API: Factory functions
# ═══════════════════════════════════════════════════════════════════

def create_engine(config_path: Optional[str] = None) -> RiskEngine:
    """Create a RiskEngine instance with default or custom config."""
    return RiskEngine(config_path=config_path)


def quick_assess(
    tickers: List[str],
    prices: List[float],
    quantities: Optional[List[int]] = None,
    entry_prices: Optional[List[float]] = None,
    total_value: Optional[float] = None,
) -> RiskReport:
    """
    Quick risk assessment with minimal inputs.
    Assumes equal-weight, no OHLCV data available (VaR from price history only).
    """
    if quantities is None:
        quantities = [100] * len(tickers)
    if entry_prices is None:
        entry_prices = prices  # Assume at-cost

    positions = []
    for i, ticker in enumerate(tickers):
        positions.append(Position(
            ticker=ticker,
            quantity=quantities[i] if i < len(quantities) else 100,
            entry_price=entry_prices[i] if i < len(entry_prices) else prices[i],
            current_price=prices[i] if i < len(prices) else 0,
        ))

    port_value = total_value or sum(p.market_value for p in positions)

    portfolio = Portfolio(
        total_value=port_value,
        cash=port_value - sum(p.market_value for p in positions),
        positions=positions,
    )

    engine = RiskEngine()
    return engine.assess(portfolio)
