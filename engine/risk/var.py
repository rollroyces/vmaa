#!/usr/bin/env python3
"""
VMAA Risk Engine — Value at Risk (VaR) Models
==============================================
Comprehensive VaR estimation:
  - Historical Simulation VaR (95%, 99%)
  - Parametric VaR (normal distribution assumption)
  - EWMA VaR (volatility-weighted)
  - Expected Shortfall / CVaR (tail risk beyond VaR)
  - Monte Carlo VaR (GBM simulations)
  - Portfolio VaR with correlations
  - VaR decomposition (which positions contribute most)
  - Stress testing (predefined scenarios)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import VaRConfig, StressScenario, load_config

logger = logging.getLogger("vmaa.risk.engine.var")


@dataclass
class VaRResult:
    """Complete VaR breakdown for a portfolio or position."""
    # Identification
    label: str = ""                    # "Portfolio" or ticker
    portfolio_value: float = 0.0

    # Historical Simulation VaR
    hist_var: Dict[str, float] = field(default_factory=dict)   # conf_level → VaR_amount
    hist_var_pct: Dict[str, float] = field(default_factory=dict)  # conf_level → VaR_pct

    # Parametric VaR
    param_var: Dict[str, float] = field(default_factory=dict)
    param_var_pct: Dict[str, float] = field(default_factory=dict)

    # EWMA VaR
    ewma_var: Dict[str, float] = field(default_factory=dict)
    ewma_var_pct: Dict[str, float] = field(default_factory=dict)

    # Expected Shortfall (CVaR)
    expected_shortfall: Dict[str, float] = field(default_factory=dict)
    expected_shortfall_pct: Dict[str, float] = field(default_factory=dict)

    # Monte Carlo VaR
    mc_var: Dict[str, float] = field(default_factory=dict)
    mc_var_pct: Dict[str, float] = field(default_factory=dict)

    # VaR Decomposition
    var_decomposition: Dict[str, float] = field(default_factory=dict)  # ticker → contrib_pct

    # Stress Test Results
    stress_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # scenario_name → {pct_loss, dollar_loss, portfolio_value_after}

    # Metadata
    n_observations: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "portfolio_value": round(self.portfolio_value, 2),
            "historical_var": {
                f"var{int(k*100)}": round(v, 2)
                for k, v in self.hist_var.items()
            },
            "historical_var_pct": {
                f"var{int(k*100)}": round(v, 6)
                for k, v in self.hist_var_pct.items()
            },
            "parametric_var": {
                f"var{int(k*100)}": round(v, 2)
                for k, v in self.param_var.items()
            },
            "parametric_var_pct": {
                f"var{int(k*100)}": round(v, 6)
                for k, v in self.param_var_pct.items()
            },
            "ewma_var": {
                f"var{int(k*100)}": round(v, 2)
                for k, v in self.ewma_var.items()
            },
            "ewma_var_pct": {
                f"var{int(k*100)}": round(v, 6)
                for k, v in self.ewma_var_pct.items()
            },
            "expected_shortfall": {
                f"es{int(k*100)}": round(v, 2)
                for k, v in self.expected_shortfall.items()
            },
            "expected_shortfall_pct": {
                f"es{int(k*100)}": round(v, 6)
                for k, v in self.expected_shortfall_pct.items()
            },
            "monte_carlo_var": {
                f"var{int(k*100)}": round(v, 2)
                for k, v in self.mc_var.items()
            },
            "monte_carlo_var_pct": {
                f"var{int(k*100)}": round(v, 6)
                for k, v in self.mc_var_pct.items()
            },
            "var_decomposition": {
                k: round(v, 4) for k, v in self.var_decomposition.items()
            },
            "stress_tests": {
                name: {
                    "pct_loss": round(d["pct_loss"], 4),
                    "dollar_loss": round(d["dollar_loss"], 2),
                    "value_after": round(d["portfolio_value_after"], 2),
                }
                for name, d in self.stress_tests.items()
            },
            "n_observations": self.n_observations,
            "warnings": self.warnings,
        }


class VaRCalculator:
    """
    Multi-method Value-at-Risk calculator.

    Usage:
        calc = VaRCalculator(config)
        result = calc.compute(returns, portfolio_value, label="AAPL")
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        if config is None:
            full_cfg = load_config()
            config = full_cfg.var
        self.config = config
        self.conf_levels = config.confidence_levels

    # ── Historical Simulation VaR ─────────────────────────────────

    def historical_var(
        self,
        returns: pd.Series,
        confidence_levels: Optional[List[float]] = None,
    ) -> Tuple[Dict[float, float], Dict[float, float]]:
        """
        Historical simulation VaR.

        Simply takes the (1-confidence) percentile of historical returns.
        Non-parametric, no distribution assumption.

        Returns:
            (var_absolute_values, var_percentages) keyed by confidence level
        """
        if confidence_levels is None:
            confidence_levels = self.conf_levels

        returns = returns.dropna()
        if len(returns) < 20:
            return {}, {}

        var_abs: Dict[float, float] = {}
        var_pct: Dict[float, float] = {}

        for cl in confidence_levels:
            percentile = (1 - cl) * 100
            var_pct_val = float(np.percentile(returns, percentile))
            var_abs[cl] = abs(var_pct_val)    # Store as positive number
            var_pct[cl] = var_pct_val         # Store actual (negative) return

        return var_abs, var_pct

    # ── Parametric VaR ────────────────────────────────────────────

    def parametric_var(
        self,
        returns: pd.Series,
        confidence_levels: Optional[List[float]] = None,
    ) -> Tuple[Dict[float, float], Dict[float, float]]:
        """
        Parametric VaR (assumes normal distribution).

        VaR_α = μ - z_α · σ      (for normally distributed returns)
        Where z_α is the z-score for confidence level α.

        Returns:
            (var_absolute_values, var_percentages)
        """
        if confidence_levels is None:
            confidence_levels = self.conf_levels

        returns = returns.dropna()
        if len(returns) < 10:
            return {}, {}

        mu = float(returns.mean())
        sigma = float(returns.std())

        var_abs: Dict[float, float] = {}
        var_pct: Dict[float, float] = {}

        for cl in confidence_levels:
            z_score = stats.norm.ppf(1 - cl)   # e.g., -1.645 for 95%, -2.326 for 99%
            var_pct_val = mu + z_score * sigma  # Daily VaR (%)
            var_abs[cl] = abs(var_pct_val)
            var_pct[cl] = var_pct_val

        return var_abs, var_pct

    # ── EWMA VaR ──────────────────────────────────────────────────

    def ewma_var(
        self,
        returns: pd.Series,
        lambda_: Optional[float] = None,
        confidence_levels: Optional[List[float]] = None,
    ) -> Tuple[Dict[float, float], Dict[float, float]]:
        """
        EWMA-based VaR with time-varying volatility.

        VaR_t = z_α · σ_t (EWMA)
        Where σ_t is the current EWMA volatility estimate.

        This better captures volatility clustering than constant-var parametric VaR.
        """
        if lambda_ is None:
            lambda_ = self.config.ewma_decay
        if confidence_levels is None:
            confidence_levels = self.conf_levels

        returns = returns.dropna()
        n = len(returns)
        if n < 10:
            return {}, {}

        # EWMA variance
        r2 = returns.values ** 2
        ewma_var_arr = np.zeros(n)
        ewma_var_arr[0] = float(r2.mean())
        for t in range(1, n):
            ewma_var_arr[t] = lambda_ * ewma_var_arr[t - 1] + (1 - lambda_) * r2[t - 1]

        current_sigma = float(np.sqrt(ewma_var_arr[-1]))

        var_abs: Dict[float, float] = {}
        var_pct: Dict[float, float] = {}

        for cl in confidence_levels:
            z_score = stats.norm.ppf(1 - cl)
            ewma_var_pct = z_score * current_sigma  # No mean term (assumed zero for daily)
            var_abs[cl] = abs(float(ewma_var_pct))
            var_pct[cl] = float(ewma_var_pct)

        return var_abs, var_pct

    # ── Expected Shortfall / CVaR ─────────────────────────────────

    def expected_shortfall(
        self,
        returns: pd.Series,
        confidence_levels: Optional[List[float]] = None,
    ) -> Dict[float, float]:
        """
        Expected Shortfall (CVaR) — average loss beyond VaR.

        ES_α = E[r | r ≤ VaR_α]

        This captures tail risk that VaR misses. If VaR tells you the
        threshold, ES tells you how bad it gets beyond that threshold.

        Returns:
            Dict[confidence_level, ES_as_pct] (negative values)
        """
        if confidence_levels is None:
            confidence_levels = self.conf_levels

        returns = returns.dropna().values
        if len(returns) < 20:
            return {}

        es_dict: Dict[float, float] = {}
        for cl in confidence_levels:
            var_threshold = np.percentile(returns, (1 - cl) * 100)
            tail = returns[returns <= var_threshold]
            if len(tail) > 0:
                es_dict[cl] = float(tail.mean())
            else:
                es_dict[cl] = float(var_threshold)

        return es_dict

    # ── Monte Carlo VaR (GBM) ─────────────────────────────────────

    def monte_carlo_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
        n_simulations: Optional[int] = None,
        horizon_days: Optional[int] = None,
        confidence_levels: Optional[List[float]] = None,
        seed: int = 42,
    ) -> Tuple[Dict[float, float], Dict[float, float]]:
        """
        Monte Carlo VaR using Geometric Brownian Motion.

        Simulates n_simulations paths over horizon_days, then computes
        VaR from the simulated terminal distribution.

        dS/S = μ·dt + σ·dW, where dW ~ N(0, dt)

        Returns:
            (var_absolute_values, var_percentages)
        """
        if n_simulations is None:
            n_simulations = self.config.monte_carlo.get("simulations", 2000)
        if horizon_days is None:
            horizon_days = self.config.monte_carlo.get("horizon_days", 10)
        if confidence_levels is None:
            confidence_levels = self.conf_levels

        returns = returns.dropna()
        if len(returns) < 10:
            return {}, {}

        mu = float(returns.mean())
        sigma = float(returns.std())

        rng = np.random.default_rng(seed)

        # Simulate GBM paths
        dt = 1.0 / 252  # Daily steps
        total_steps = horizon_days
        returns_sim = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Cumulative return over horizon
            z = rng.normal(0, 1, total_steps)
            path_returns = mu * dt + sigma * np.sqrt(dt) * z
            returns_sim[i] = np.sum(path_returns)

        var_abs: Dict[float, float] = {}
        var_pct: Dict[float, float] = {}

        for cl in confidence_levels:
            pct_var = float(np.percentile(returns_sim, (1 - cl) * 100))
            var_pct[cl] = pct_var
            var_abs[cl] = abs(pct_var) * portfolio_value

        return var_abs, var_pct

    # ── Portfolio VaR ─────────────────────────────────────────────

    def portfolio_var(
        self,
        returns_matrix: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float,
        confidence_levels: Optional[List[float]] = None,
    ) -> Tuple[Dict[float, float], Dict[float, float]]:
        """
        Portfolio VaR considering correlation between positions.

        For parametric (normal):
          σ_p = sqrt(w' · Σ · w)
          VaR_α = -z_α · σ_p · portfolio_value

        For historical: applies weights to historical returns, then takes percentile.

        Args:
            returns_matrix: T x N matrix of asset returns
            weights: N-length array of portfolio weights
            portfolio_value: Total portfolio value
            confidence_levels: List of confidence levels

        Returns:
            (var_dollar_amounts, var_percentages) keyed by confidence level
        """
        if confidence_levels is None:
            confidence_levels = self.conf_levels

        returns_matrix = returns_matrix.dropna()
        if returns_matrix.empty:
            return {}, {}

        # Align weights to columns
        aligned_cols = returns_matrix.columns.tolist()
        if len(weights) != len(aligned_cols):
            logger.warning(f"Weight/column mismatch: {len(weights)} vs {len(aligned_cols)}")
            return {}, {}

        # Historical approach: compute portfolio returns
        port_returns = (returns_matrix.values @ weights)
        port_returns_series = pd.Series(port_returns, index=returns_matrix.index)

        # Parametric approach
        cov = returns_matrix.cov().values
        port_var = weights.T @ cov @ weights
        port_sigma = float(np.sqrt(max(port_var, 0)))

        var_abs: Dict[float, float] = {}
        var_pct: Dict[float, float] = {}

        for cl in confidence_levels:
            # Historical
            hist_var_pct = float(np.percentile(port_returns, (1 - cl) * 100))
            # Parametric
            z_score = stats.norm.ppf(1 - cl)
            param_var_pct = z_score * port_sigma

            # Blend: use historical with parametric adjustment for small samples
            var_pct[cl] = float(np.mean([hist_var_pct, param_var_pct]))
            var_abs[cl] = abs(var_pct[cl]) * portfolio_value

        return var_abs, var_pct

    # ── VaR Decomposition ─────────────────────────────────────────

    def var_decomposition(
        self,
        returns_matrix: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Decompose portfolio VaR into position-level contributions.

        Component VaR = w_i · ∂VaR/∂w_i  (marginal VaR · weight)

        For parametric VaR:
          CVaR_i = w_i · (cov·w)_i / σ_p · z_α · portfolio_value

        Returns dict of {ticker: contribution_as_fraction_of_total_VaR}.
        """
        returns_matrix = returns_matrix.dropna()
        cov = returns_matrix.cov().values
        port_var = weights.T @ cov @ weights
        port_sigma = float(np.sqrt(max(port_var, 0)))

        if port_sigma < 1e-10:
            return {col: 0.0 for col in returns_matrix.columns}

        z_score = stats.norm.ppf(1 - confidence)
        total_var = abs(z_score) * port_sigma * portfolio_value

        if total_var < 0.01:
            return {col: 0.0 for col in returns_matrix.columns}

        # Marginal VaR = (cov · w) / σ_p * |z_α| * portfolio_value
        marginal_var = (cov @ weights) / port_sigma * abs(z_score) * portfolio_value
        component_var = weights * marginal_var

        # Normalize to fractions
        total_component_var = np.sum(np.abs(component_var))
        if total_component_var < 0.01:
            return {col: 0.0 for col in returns_matrix.columns}

        return {
            col: round(float(component_var[i] / total_component_var), 4)
            for i, col in enumerate(returns_matrix.columns)
        }

    # ── Stress Testing ────────────────────────────────────────────

    def stress_test(
        self,
        portfolio_value: float,
        returns_matrix: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
        scenarios: Optional[List[StressScenario]] = None,
        base_vol: float = 0.20,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run stress tests under predefined scenarios.

        If returns_matrix + weights provided, computes scenario impact
        using factor shock approach. Otherwise uses simplified portfolio-level
        shocks.

        Returns:
            Dict[scenario_name → {pct_loss, dollar_loss, portfolio_value_after}]
        """
        if scenarios is None:
            scenarios = self.config.stress_scenarios

        if not scenarios:
            return {}

        results: Dict[str, Dict[str, float]] = {}

        for scenario in scenarios:
            shock = scenario.shock
            equity_shock = shock.get("equity", -0.30)

            if returns_matrix is not None and weights is not None and not returns_matrix.empty:
                # Asset-level simulation
                vol_mult = shock.get("volatility_mult", 1.0)

                # Scale each asset's return by the equity shock * beta proxy
                asset_var = returns_matrix.var()
                market_var = asset_var.mean()  # Simple proxy

                scenario_returns = []
                for col in returns_matrix.columns:
                    # Approximate β as relative variance
                    beta_i = np.sqrt(asset_var[col] / max(market_var, 1e-6))
                    scenario_ret = beta_i * equity_shock * vol_mult
                    scenario_returns.append(scenario_ret)

                scenario_returns = np.array(scenario_returns)
                portfolio_shock = float(np.dot(scenario_returns, weights))
            else:
                portfolio_shock = equity_shock

            pct_loss = portfolio_shock
            dollar_loss = portfolio_value * abs(pct_loss)
            value_after = portfolio_value * (1 + pct_loss)

            results[scenario.name] = {
                "pct_loss": round(pct_loss, 4),
                "dollar_loss": round(dollar_loss, 2),
                "portfolio_value_after": round(value_after, 2),
            }

        return results

    # ── Full Computation ─────────────────────────────────────────

    def compute(
        self,
        returns: pd.Series,
        portfolio_value: float,
        label: str = "",
        returns_matrix: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None,
    ) -> VaRResult:
        """
        Compute complete VaR profile.

        Args:
            returns: Daily return series for the instrument/portfolio
            portfolio_value: Current value in dollars
            label: Identifier (ticker or "Portfolio")
            returns_matrix: Multi-asset returns for portfolio VaR & decomposition
            weights: Portfolio weights for multi-asset analysis

        Returns:
            VaRResult with all VaR metrics
        """
        result = VaRResult(label=label, portfolio_value=portfolio_value)
        returns = returns.dropna()
        result.n_observations = len(returns)

        if len(returns) < 20:
            result.warnings.append(f"Insufficient data: {len(returns)} observations")
            return result

        n_obs_for_var = min(len(returns), self.config.historical_window)
        window_returns = returns.tail(n_obs_for_var)

        # 1. Historical Simulation VaR
        var_abs, var_pct = self.historical_var(window_returns)
        result.hist_var = {cl: v * portfolio_value for cl, v in var_abs.items()}
        result.hist_var_pct = var_pct

        # 2. Parametric VaR
        var_abs, var_pct = self.parametric_var(window_returns)
        result.param_var = {cl: v * portfolio_value for cl, v in var_abs.items()}
        result.param_var_pct = var_pct

        # 3. EWMA VaR
        var_abs, var_pct = self.ewma_var(window_returns)
        result.ewma_var = {cl: v * portfolio_value for cl, v in var_abs.items()}
        result.ewma_var_pct = var_pct

        # 4. Expected Shortfall
        es = self.expected_shortfall(window_returns)
        result.expected_shortfall = {
            cl: abs(v) * portfolio_value for cl, v in es.items()
        }
        result.expected_shortfall_pct = {
            cl: abs(v) for cl, v in es.items()
        }

        # 5. Monte Carlo VaR
        var_abs, var_pct = self.monte_carlo_var(window_returns, portfolio_value)
        result.mc_var = var_abs
        result.mc_var_pct = var_pct

        # 6. Portfolio VaR & Decomposition (if multi-asset)
        if returns_matrix is not None and weights is not None and not returns_matrix.empty:
            # Portfolio-level VaR considering correlations
            port_var_abs, _ = self.portfolio_var(
                returns_matrix, weights, portfolio_value
            )
            # Blend portfolio VaR into result
            for cl in self.conf_levels:
                if cl in port_var_abs:
                    result.hist_var[cl] = port_var_abs[cl]

            # VaR decomposition
            result.var_decomposition = self.var_decomposition(
                returns_matrix, weights, portfolio_value
            )

        # 7. Stress Tests
        result.stress_tests = self.stress_test(
            portfolio_value, returns_matrix, weights
        )

        return result
