#!/usr/bin/env python3
"""
VMAA Risk Engine — Volatility Calculation
==========================================
Comprehensive volatility estimation:
  - Historical close-to-close (10d, 20d, 50d, 100d)
  - EWMA (RiskMetrics lambda=0.94)
  - GARCH(1,1) — simple implementation
  - Parkinson's estimator (OHLC-based, ~5.2x more efficient)
  - Realized volatility (intraday proxy from daily range)
  - Volatility regime classification
  - Volatility term structure
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import VolatilityConfig, load_config

logger = logging.getLogger("vmaa.risk.engine.volatility")

ANNUAL_FACTOR = np.sqrt(252)


@dataclass
class VolatilityResult:
    """Complete volatility profile for a single instrument."""
    ticker: str
    # Historical volatility (annualized)
    hist_vol: Dict[int, float] = field(default_factory=dict)    # window_days → vol
    # EWMA volatility
    ewma_vol: float = 0.0
    ewma_lambda: float = 0.94
    # GARCH(1,1)
    garch_vol: float = 0.0
    garch_omega: float = 0.0
    garch_alpha: float = 0.0
    garch_beta: float = 0.0
    # Parkinson
    parkinson_vol: float = 0.0
    # Realized volatility (daily range proxy)
    realized_vol: float = 0.0
    # Regime
    vol_regime: str = "UNKNOWN"
    current_vol: float = 0.0            # Primary vol measure (20d hist)
    # Volatility term structure
    term_structure_ratio: float = 0.0   # short-term / long-term vol
    term_structure_status: str = "UNKNOWN"  # CONTANGO | BACKWARDATION | FLAT
    # Metadata
    n_observations: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "historical_volatility": {str(k): round(v, 6) for k, v in self.hist_vol.items()},
            "ewma_volatility": round(self.ewma_vol, 6),
            "ewma_lambda": self.ewma_lambda,
            "garch_volatility": round(self.garch_vol, 6),
            "parkinson_volatility": round(self.parkinson_vol, 6),
            "realized_volatility": round(self.realized_vol, 6),
            "volatility_regime": self.vol_regime,
            "current_volatility": round(self.current_vol, 6),
            "term_structure_ratio": round(self.term_structure_ratio, 4),
            "term_structure": self.term_structure_status,
            "n_observations": self.n_observations,
            "warnings": self.warnings,
        }


class VolatilityCalculator:
    """
    Multi-method volatility estimation.

    Usage:
        calc = VolatilityCalculator(config)
        result = calc.compute(price_data, ticker="AAPL")
    """

    def __init__(self, config: Optional[VolatilityConfig] = None):
        if config is None:
            full_cfg = load_config()
            config = full_cfg.volatility
        self.config = config
        self.regime_thresholds = config.regime_thresholds

    # ── Historical Volatility ────────────────────────────────────

    def historical_volatility(
        self,
        returns: pd.Series,
        windows: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """
        Compute annualized historical volatility over specified windows.

        Args:
            returns: Daily log or simple returns
            windows: Trading-day windows (default from config)

        Returns:
            Dict[window_days, annualized_vol]
        """
        if windows is None:
            windows = self.config.windows

        result: Dict[int, float] = {}
        for w in windows:
            if len(returns) >= w:
                vol = float(returns.tail(w).std() * ANNUAL_FACTOR)
                result[w] = vol
            else:
                result[w] = float('nan')
        return result

    # ── EWMA Volatility (RiskMetrics) ─────────────────────────────

    def ewma_volatility(
        self,
        returns: pd.Series,
        lambda_: Optional[float] = None,
    ) -> Tuple[float, pd.Series]:
        """
        Exponentially Weighted Moving Average volatility.

        RiskMetrics standard: lambda = 0.94

        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

        Args:
            returns: Daily returns
            lambda_: Decay factor (default from config)

        Returns:
            (current_annualized_vol, ewma_variance_series)
        """
        if lambda_ is None:
            lambda_ = self.config.ewma_lambda

        returns = returns.dropna()
        if len(returns) < 2:
            return 0.0, pd.Series(dtype=float)

        # Initialize with sample variance
        var0 = float(returns.var())
        ewma_var = pd.Series(index=returns.index, dtype=float)
        ewma_var.iloc[0] = var0

        for t in range(1, len(returns)):
            ewma_var.iloc[t] = (
                lambda_ * ewma_var.iloc[t - 1] +
                (1 - lambda_) * returns.iloc[t - 1] ** 2
            )

        current_daily_var = float(ewma_var.iloc[-1])
        return float(np.sqrt(current_daily_var) * ANNUAL_FACTOR), ewma_var

    # ── GARCH(1,1) ────────────────────────────────────────────────

    def garch_volatility(
        self,
        returns: pd.Series,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> Tuple[float, float, float, float, pd.Series]:
        """
        Simple GARCH(1,1) estimation via MLE-like iterative approach.

        Uses a variance-targeting simplified method:
          σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
          with ω = unconditional_variance * (1 - α - β)

        Parameters estimated via simple grid/log-likelihood search.

        Returns:
            (annualized_vol, omega, alpha, beta, conditional_var_series)
        """
        returns = returns.dropna()
        n = len(returns)
        if n < 60:
            logger.warning("GARCH: insufficient data (<60 obs), returning EWMA fallback")
            ewma_vol, _ = self.ewma_volatility(returns)
            return ewma_vol, 0.0, 0.05, 0.90, pd.Series(dtype=float)

        # Unconditional variance
        var_uncond = float(returns.var())

        # Grid search for α, β
        best_loglik = float('inf')
        best_alpha, best_beta = 0.05, 0.90

        # Coarse grid
        alphas = np.linspace(0.01, 0.30, 15)
        betas = np.linspace(0.60, 0.97, 19)

        r2 = (returns.values ** 2)

        for a in alphas:
            for b in betas:
                if a + b >= 0.999:
                    continue
                omega = var_uncond * (1.0 - a - b)
                if omega <= 0:
                    continue

                # Compute conditional variances
                h = np.zeros(n)
                h[0] = var_uncond
                for t in range(1, n):
                    h[t] = omega + a * r2[t - 1] + b * h[t - 1]

                # Log-likelihood: -0.5 * Σ[log(h_t) + r²_t / h_t]
                loglik = 0.5 * np.sum(np.log(h[1:] + 1e-12) + r2[1:] / (h[1:] + 1e-12))

                if loglik < best_loglik:
                    best_loglik = loglik
                    best_alpha, best_beta = a, b

        omega = var_uncond * (1.0 - best_alpha - best_beta)

        # Recompute final conditional variance series
        h_final = np.zeros(n)
        h_final[0] = var_uncond
        for t in range(1, n):
            h_final[t] = omega + best_alpha * r2[t - 1] + best_beta * h_final[t - 1]

        current_daily_var = float(h_final[-1])
        current_vol = float(np.sqrt(current_daily_var) * ANNUAL_FACTOR)

        # Persistence check — warn if near unit root
        persistence = best_alpha + best_beta
        if persistence > 0.99:
            logger.warning(f"GARCH: high persistence α+β={persistence:.4f} — near unit root")

        h_series = pd.Series(h_final, index=returns.index)

        return current_vol, omega, best_alpha, best_beta, h_series

    # ── Parkinson's Estimator (OHLC-based) ────────────────────────

    def parkinson_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
    ) -> float:
        """
        Parkinson's extreme value volatility estimator.

        σ²_park = (1 / 4·ln(2)·N) · Σ [ln(H_t / L_t)]²

        About 5.2x more efficient than close-to-close estimation.

        Args:
            high: Daily high prices
            low: Daily low prices
            window: Rolling window for estimation

        Returns:
            Annualized Parkinson volatility
        """
        if len(high) < window or len(low) < window:
            return float('nan')

        # Align indices
        common_idx = high.index.intersection(low.index)
        h = high.loc[common_idx].tail(window)
        l = low.loc[common_idx].tail(window)

        # Avoid division by zero
        mask = (h > 0) & (l > 0) & (h >= l)
        h, l = h[mask], l[mask]

        if len(h) < 5:
            return float('nan')

        log_hl = np.log(h / l)
        n = len(log_hl)
        parkinson_var = (1.0 / (4.0 * np.log(2.0))) * (1.0 / n) * np.sum(log_hl ** 2)

        return float(np.sqrt(parkinson_var) * ANNUAL_FACTOR)

    # ── Realized Volatility (Daily Range Proxy) ───────────────────

    def realized_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        open_: Optional[pd.Series] = None,
        window: int = 20,
    ) -> float:
        """
        Realized volatility proxy using daily range and overnight gaps.

        When intraday data isn't available, uses:
        - Parkinson (daily range)
        - Garman-Klass (OHLC)
        - Rogers-Satchell (drift-independent)

        This implements a blended estimator:
          RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)  (Rogers-Satchell)
          PK = (ln(H/L))² / (4·ln(2))             (Parkinson component)

        Blended: σ² = PK * 0.6 + RS * 0.4 (weighted average)

        Returns:
            Annualized realized volatility
        """
        if len(close) < window:
            return float('nan')

        h = high.tail(window)
        l = low.tail(window)
        c = close.tail(window)

        if open_ is not None and len(open_) >= window:
            o = open_.tail(window)
        else:
            o = c.shift(1).fillna(c)  # Proxy: previous close as open

        # Align all series
        common_idx = h.index.intersection(l.index).intersection(c.index).intersection(o.index)
        h, l, c, o = h[common_idx], l[common_idx], c[common_idx], o[common_idx]

        # Filter valid data
        mask = (h > 0) & (l > 0) & (c > 0) & (o > 0) & (h >= l)
        h, l, c, o = h[mask], l[mask], c[mask], o[mask]

        n = len(h)
        if n < 3:
            return float('nan')

        # Parkinson component
        log_hl = np.log(h.values / l.values)
        park_var = (1.0 / (4.0 * np.log(2.0))) * np.mean(log_hl ** 2)

        # Rogers-Satchell component
        log_hc = np.log(h.values / c.values)
        log_ho = np.log(h.values / o.values)
        log_lc = np.log(l.values / c.values)
        log_lo = np.log(l.values / o.values)
        rs_var = np.mean(log_hc * log_ho + log_lc * log_lo)

        # Blended variance
        blended_var = 0.6 * park_var + 0.4 * max(rs_var, 0)

        return float(np.sqrt(blended_var) * ANNUAL_FACTOR)

    # ── Regime Classification ────────────────────────────────────

    def classify_regime(self, volatility: float) -> str:
        """
        Classify volatility regime.

        Returns one of: LOW, NORMAL, HIGH, EXTREME
        """
        if volatility < self.regime_thresholds.get("low", 0.12):
            return "LOW"
        elif volatility < self.regime_thresholds.get("normal", 0.22):
            return "NORMAL"
        elif volatility < self.regime_thresholds.get("high", 0.35):
            return "HIGH"
        else:
            return "EXTREME"

    # ── Term Structure ───────────────────────────────────────────

    def term_structure(
        self,
        returns: pd.Series,
        short_window: Optional[int] = None,
        long_window: Optional[int] = None,
    ) -> Tuple[float, str]:
        """
        Volatility term structure: short-term vs long-term vol ratio.

        Returns:
            (ratio, status) where status ∈ {CONTANGO, BACKWARDATION, FLAT}
            - CONTANGO: short > long (near-term uncertainty high)
            - BACKWARDATION: short < long (near-term calm)
            - FLAT: roughly equal
        """
        if short_window is None:
            short_window = self.config.term_structure.get("short_window", 20)
        if long_window is None:
            long_window = self.config.term_structure.get("long_window", 60)

        if len(returns) < max(short_window, long_window):
            return 0.0, "UNKNOWN"

        short_vol = float(returns.tail(short_window).std() * ANNUAL_FACTOR)
        long_vol = float(returns.tail(long_window).std() * ANNUAL_FACTOR)

        if long_vol < 0.001:
            return 0.0, "FLAT"

        ratio = short_vol / long_vol

        if ratio > 1.15:
            status = "CONTANGO"         # Near-term vol elevated
        elif ratio < 0.85:
            status = "BACKWARDATION"    # Near-term vol depressed
        else:
            status = "FLAT"

        return round(ratio, 4), status

    # ── Full Computation ─────────────────────────────────────────

    def compute(
        self,
        prices: pd.DataFrame,
        ticker: str = "",
    ) -> VolatilityResult:
        """
        Compute complete volatility profile from OHLCV data.

        Args:
            prices: DataFrame with columns [Open, High, Low, Close, Volume]
            ticker: Instrument identifier

        Returns:
            VolatilityResult with all metrics
        """
        result = VolatilityResult(ticker=ticker)

        if prices.empty or 'Close' not in prices.columns:
            result.warnings.append("No price data provided")
            return result

        close = prices['Close']
        returns = close.pct_change().dropna()
        result.n_observations = len(returns)

        if len(returns) < 10:
            result.warnings.append(f"Insufficient data: {len(returns)} observations")
            return result

        # 1. Historical volatility
        result.hist_vol = self.historical_volatility(returns)

        # 2. EWMA
        result.ewma_vol, _ = self.ewma_volatility(returns)
        result.ewma_lambda = self.config.ewma_lambda

        # 3. GARCH(1,1)
        try:
            garch_vol, omega, alpha, beta, _ = self.garch_volatility(returns)
            result.garch_vol = garch_vol
            result.garch_omega = omega
            result.garch_alpha = alpha
            result.garch_beta = beta
        except Exception as e:
            logger.debug(f"GARCH estimation failed for {ticker}: {e}")
            result.garch_vol = result.ewma_vol  # Fallback to EWMA
            result.warnings.append(f"GARCH failed, using EWMA fallback: {e}")

        # 4. Parkinson (OHLC)
        if 'High' in prices.columns and 'Low' in prices.columns:
            result.parkinson_vol = self.parkinson_volatility(
                prices['High'], prices['Low']
            )
        else:
            # Parkinson from close-only: use daily range approximation
            hi = close.rolling(2).max()
            lo = close.rolling(2).min()
            result.parkinson_vol = self.parkinson_volatility(hi, lo)

        # 5. Realized volatility
        if 'High' in prices.columns and 'Low' in prices.columns:
            open_ = prices.get('Open', None)
            result.realized_vol = self.realized_volatility(
                prices['High'], prices['Low'], prices['Close'], open_
            )
        else:
            result.realized_vol = result.hist_vol.get(20, 0.0)

        # 6. Current volatility (20d historical)
        result.current_vol = result.hist_vol.get(20, result.ewma_vol)

        # 7. Regime
        result.vol_regime = self.classify_regime(result.current_vol)

        # 8. Term structure
        result.term_structure_ratio, result.term_structure_status = self.term_structure(returns)

        return result


# ═══════════════════════════════════════════════════════════════════
# Portfolio-Level Volatility
# ═══════════════════════════════════════════════════════════════════

def portfolio_volatility(
    returns_matrix: pd.DataFrame,
    weights: np.ndarray,
    annualize: bool = True,
) -> float:
    """
    Compute portfolio volatility from return series and weights.

    σ_p = sqrt(w' · Σ · w)

    Args:
        returns_matrix: DataFrame of returns (assets x time)
        weights: Array of portfolio weights (must sum to ~1)
        annualize: If True, multiply by sqrt(252)

    Returns:
        Portfolio volatility (annualized if annualize=True)
    """
    cov = returns_matrix.cov().values
    port_var = weights.T @ cov @ weights
    port_vol = float(np.sqrt(max(port_var, 0)))

    if annualize:
        port_vol *= ANNUAL_FACTOR

    return round(port_vol, 6)


def volatility_contribution(
    returns_matrix: pd.DataFrame,
    weights: np.ndarray,
) -> Dict[str, float]:
    """
    Decompose portfolio volatility into marginal contributions.

    Marginal contribution of asset i:
      MC_i = w_i * (Σ·w)_i / σ_p

    Returns dict of {ticker: pct_contribution} mapping.
    """
    cov = returns_matrix.cov().values
    port_var = weights.T @ cov @ weights
    port_vol = float(np.sqrt(max(port_var, 0)))

    if port_vol < 1e-10:
        return {col: 0.0 for col in returns_matrix.columns}

    marginal = cov @ weights
    contributions = (weights * marginal) / port_vol

    return {
        col: round(float(contributions[i]), 6)
        for i, col in enumerate(returns_matrix.columns)
    }
