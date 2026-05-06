#!/usr/bin/env python3
"""
VMAA Risk Engine — Risk Exposure Analysis
==========================================
Comprehensive exposure analytics:
  - Factor exposure (beta to market, sector, size, value)
  - Concentration risk (HHI, max position weight)
  - Sector exposure (% allocation vs benchmark)
  - Correlation matrix (pairwise correlations within portfolio)
  - Liquidity risk (position size / ADV)
  - FX risk (currency exposure for non-USD stocks)
  - Tail risk (skewness, kurtosis of portfolio returns)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ExposureConfig, load_config

logger = logging.getLogger("vmaa.risk.engine.exposure")


@dataclass
class FactorExposure:
    """Beta exposures to common risk factors."""
    market_beta: float = 0.0
    size_beta: float = 0.0          # Small-cap factor loading
    value_beta: float = 0.0         # Value factor loading
    growth_beta: float = 0.0        # Growth factor loading
    r_squared: float = 0.0          # Model fit
    residual_vol: float = 0.0       # Idiosyncratic risk

    def to_dict(self) -> Dict[str, float]:
        return {
            "market_beta": round(self.market_beta, 4),
            "size_beta": round(self.size_beta, 4),
            "value_beta": round(self.value_beta, 4),
            "growth_beta": round(self.growth_beta, 4),
            "r_squared": round(self.r_squared, 4),
            "residual_volatility": round(self.residual_vol, 4),
        }


@dataclass
class ConcentrationMetrics:
    """Portfolio concentration analysis."""
    hhi: float = 0.0                        # Herfindahl-Hirschman Index
    hhi_status: str = "OK"                   # OK | WARNING | DANGER
    max_position_weight: float = 0.0
    max_position_ticker: str = ""
    top3_weight: float = 0.0                # Sum of top 3 positions
    top5_weight: float = 0.0                # Sum of top 5 positions
    effective_n: float = 0.0                # 1/HHI (effective number of equal-weight positions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hhi": round(self.hhi, 4),
            "hhi_status": self.hhi_status,
            "max_position_weight": round(self.max_position_weight, 4),
            "max_position_ticker": self.max_position_ticker,
            "top3_weight": round(self.top3_weight, 4),
            "top5_weight": round(self.top5_weight, 4),
            "effective_n": round(self.effective_n, 1),
        }


@dataclass
class SectorExposure:
    """Sector allocation breakdown."""
    sector_weights: Dict[str, float] = field(default_factory=dict)
    sector_deviations: Dict[str, float] = field(default_factory=dict)  # vs benchmark
    max_sector_weight: float = 0.0
    max_sector_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector_weights": {k: round(v, 4) for k, v in self.sector_weights.items()},
            "sector_deviations": {k: round(v, 4) for k, v in self.sector_deviations.items()},
            "max_sector_weight": round(self.max_sector_weight, 4),
            "max_sector_name": self.max_sector_name,
        }


@dataclass
class LiquidityRisk:
    """Liquidity risk per position."""
    ticker: str = ""
    position_size: float = 0.0               # Dollar value
    avg_daily_volume: float = 0.0             # Dollar ADV
    pct_of_adv: float = 0.0                   # position / ADV
    days_to_liquidate: float = 0.0            # Estimated days at 20% of ADV
    status: str = "OK"                        # OK | WARNING | DANGER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "position_size": round(self.position_size, 2),
            "avg_daily_volume_dollar": round(self.avg_daily_volume, 0),
            "pct_of_adv": round(self.pct_of_adv, 4),
            "days_to_liquidate": round(self.days_to_liquidate, 1),
            "status": self.status,
        }


@dataclass
class TailRisk:
    """Tail risk statistics."""
    skewness: float = 0.0
    kurtosis: float = 0.0                    # Excess kurtosis
    max_drawdown: float = 0.0
    max_drawdown_days: int = 0
    worst_day: float = 0.0
    worst_day_date: str = ""
    best_day: float = 0.0
    best_day_date: str = ""
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall_95: float = 0.0
    calmar_ratio: float = 0.0                # Ann.Return / MaxDD
    sterling_ratio: float = 0.0              # Ann.Return / (AvgDD + 10%)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skewness": round(self.skewness, 4),
            "excess_kurtosis": round(self.kurtosis, 4),
            "max_drawdown_pct": round(self.max_drawdown, 4),
            "max_drawdown_days": self.max_drawdown_days,
            "worst_day_pct": round(self.worst_day, 4),
            "worst_day_date": self.worst_day_date,
            "best_day_pct": round(self.best_day, 4),
            "best_day_date": self.best_day_date,
            "var_95": round(self.var_95, 4),
            "var_99": round(self.var_99, 4),
            "expected_shortfall_95": round(self.expected_shortfall_95, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "sterling_ratio": round(self.sterling_ratio, 4),
        }


@dataclass
class ExposureResult:
    """Complete exposure analysis."""
    label: str = ""
    portfolio_value: float = 0.0
    n_positions: int = 0

    # Factor exposure
    factor_exposure: Optional[FactorExposure] = None

    # Concentration
    concentration: Optional[ConcentrationMetrics] = None

    # Sector
    sector_exposure: Optional[SectorExposure] = None

    # Correlation
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    max_corr_pair: str = ""

    # Liquidity (per position)
    liquidity_risks: List[LiquidityRisk] = field(default_factory=list)
    portfolio_liquidity_score: float = 0.0   # 0-1, higher = more liquid

    # FX
    fx_exposure: Dict[str, float] = field(default_factory=dict)
    fx_risk_pct: float = 0.0                 # % of portfolio in non-USD

    # Tail risk
    tail_risk: Optional[TailRisk] = None

    # Warnings
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "label": self.label,
            "portfolio_value": round(self.portfolio_value, 2),
            "n_positions": self.n_positions,
            "avg_correlation": round(self.avg_correlation, 4),
            "max_correlation": round(self.max_correlation, 4),
            "max_corr_pair": self.max_corr_pair,
            "portfolio_liquidity_score": round(self.portfolio_liquidity_score, 4),
            "fx_exposure": {k: round(v, 4) for k, v in self.fx_exposure.items()},
            "fx_risk_pct": round(self.fx_risk_pct, 4),
            "warnings": self.warnings,
        }
        if self.factor_exposure:
            result["factor_exposure"] = self.factor_exposure.to_dict()
        if self.concentration:
            result["concentration"] = self.concentration.to_dict()
        if self.sector_exposure:
            result["sector_exposure"] = self.sector_exposure.to_dict()
        if self.correlation_matrix:
            result["correlation_matrix"] = {
                k: {kk: round(vv, 4) for kk, vv in v.items()}
                for k, v in self.correlation_matrix.items()
            }
        if self.liquidity_risks:
            result["liquidity_risks"] = [lr.to_dict() for lr in self.liquidity_risks]
        if self.tail_risk:
            result["tail_risk"] = self.tail_risk.to_dict()
        return result


class ExposureAnalyzer:
    """
    Multi-dimensional risk exposure analyzer.

    Usage:
        analyzer = ExposureAnalyzer(config)
        result = analyzer.analyze(positions_dict, returns_matrix, sectors_dict)
    """

    def __init__(self, config: Optional[ExposureConfig] = None):
        if config is None:
            full_cfg = load_config()
            config = full_cfg.exposure
        self.config = config

    # ── Factor Exposure ───────────────────────────────────────────

    def factor_exposure(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        factor_returns: Optional[Dict[str, pd.Series]] = None,
    ) -> FactorExposure:
        """
        Compute factor betas via multi-factor regression.

        r_i = α + β_m·r_m + β_size·r_size + β_value·r_value + β_growth·r_growth + ε

        Args:
            asset_returns: Target asset return series
            market_returns: Market factor returns (e.g., SPY)
            factor_returns: Optional dict of {factor_name: return_series}

        Returns:
            FactorExposure with betas
        """
        result = FactorExposure()

        # Align data
        common = asset_returns.index.intersection(market_returns.index)
        if len(common) < 30:
            return result

        y = asset_returns[common].values
        X = market_returns[common].values.reshape(-1, 1)

        if factor_returns:
            for name, series in factor_returns.items():
                if len(series[common]) == len(common):
                    X_add = series[common].values.reshape(-1, 1)
                    X = np.hstack([X, X_add])

        # OLS regression
        try:
            X_with_const = np.column_stack([np.ones(len(X)), X])
            coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

            result.market_beta = float(coeffs[1])
            if factor_returns and len(coeffs) > 2:
                keys = list(factor_returns.keys())
                for i, name in enumerate(keys):
                    if i + 2 < len(coeffs):
                        if name == "size":
                            result.size_beta = float(coeffs[i + 2])
                        elif name == "value":
                            result.value_beta = float(coeffs[i + 2])
                        elif name == "growth":
                            result.growth_beta = float(coeffs[i + 2])

            # R² and residual vol
            y_pred = X_with_const @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            result.r_squared = float(1 - ss_res / max(ss_tot, 1e-10))
            result.residual_vol = float(np.std(y - y_pred))
        except Exception as e:
            logger.warning(f"Factor regression failed: {e}")

        return result

    # ── Concentration Risk ────────────────────────────────────────

    def concentration(
        self,
        weights: Dict[str, float],
    ) -> ConcentrationMetrics:
        """
        Compute concentration metrics.

        HHI = Σ w_i²  (Herfindahl-Hirschman Index)
        Effective N = 1 / HHI
        """
        result = ConcentrationMetrics()

        if not weights:
            return result

        w = np.array(list(weights.values()))
        tickers = list(weights.keys())

        # HHI
        result.hhi = float(np.sum(w ** 2))

        hhi_warn = self.config.concentration.get("hhi_warning", 0.15)
        hhi_danger = self.config.concentration.get("hhi_danger", 0.25)
        if result.hhi > hhi_danger:
            result.hhi_status = "DANGER"
        elif result.hhi > hhi_warn:
            result.hhi_status = "WARNING"
        else:
            result.hhi_status = "OK"

        # Max position
        max_idx = int(np.argmax(w))
        result.max_position_weight = float(w[max_idx])
        result.max_position_ticker = tickers[max_idx]

        # Top 3 and Top 5
        sorted_w = np.sort(w)[::-1]
        result.top3_weight = float(np.sum(sorted_w[:3]))
        result.top5_weight = float(np.sum(sorted_w[:5]))

        # Effective N
        result.effective_n = float(1.0 / max(result.hhi, 1e-10))

        return result

    # ── Sector Exposure ───────────────────────────────────────────

    def sector_exposure(
        self,
        sector_weights: Dict[str, float],
        benchmark_sectors: Optional[Dict[str, float]] = None,
    ) -> SectorExposure:
        """
        Analyze sector allocation vs benchmark.

        Args:
            sector_weights: {sector_name: weight_in_portfolio}
            benchmark_sectors: {sector_name: weight_in_benchmark} (e.g., SPY sectors)

        Returns:
            SectorExposure with weights and deviations
        """
        result = SectorExposure()
        result.sector_weights = sector_weights

        if sector_weights:
            max_sector = max(sector_weights, key=sector_weights.get)
            result.max_sector_weight = sector_weights[max_sector]
            result.max_sector_name = max_sector

            # Deviations vs benchmark
            if benchmark_sectors:
                all_sectors = set(sector_weights.keys()) | set(benchmark_sectors.keys())
                deviations = {}
                for s in all_sectors:
                    port_w = sector_weights.get(s, 0.0)
                    bench_w = benchmark_sectors.get(s, 0.0)
                    deviations[s] = port_w - bench_w
                result.sector_deviations = deviations

        return result

    # ── Correlation Matrix ────────────────────────────────────────

    def correlation_matrix(
        self,
        returns_matrix: pd.DataFrame,
    ) -> Tuple[Dict[str, Dict[str, float]], float, float, str]:
        """
        Compute pairwise correlations within portfolio.

        Returns:
            (correlation_dict, avg_correlation, max_correlation, max_corr_pair)
        """
        if returns_matrix.empty or len(returns_matrix.columns) < 2:
            return {}, 0.0, 0.0, ""

        corr = returns_matrix.corr()
        corr_dict: Dict[str, Dict[str, float]] = {}
        avg_corr = 0.0
        max_corr = 0.0
        max_pair = ""
        pair_count = 0

        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            corr_dict[c1] = {}
            for j, c2 in enumerate(cols):
                val = float(corr.iloc[i, j])
                corr_dict[c1][c2] = round(val, 4)
                if i < j:
                    avg_corr += val
                    pair_count += 1
                    if val > max_corr:
                        max_corr = val
                        max_pair = f"{c1}-{c2}"

        avg_corr = avg_corr / max(pair_count, 1)

        return corr_dict, round(avg_corr, 4), round(max_corr, 4), max_pair

    # ── Liquidity Risk ────────────────────────────────────────────

    def liquidity_risk(
        self,
        positions: Dict[str, Dict[str, float]],
    ) -> List[LiquidityRisk]:
        """
        Assess liquidity risk per position.

        Args:
            positions: {
                ticker: {
                    "value": dollar_position_size,
                    "avg_volume_shares": daily_avg_shares,
                    "price": current_price,
                }
            }

        Returns:
            List of LiquidityRisk per position
        """
        results = []
        pct_adv_max = self.config.liquidity.get("max_position_pct_adv", 0.05)
        pct_adv_warn = self.config.liquidity.get("adv_warning_threshold", 0.02)

        for ticker, pos in positions.items():
            position_value = pos.get("value", 0.0)
            avg_vol_shares = pos.get("avg_volume_shares", 0)
            price = pos.get("price", 0.0)
            adv_dollar = avg_vol_shares * price if avg_vol_shares > 0 and price > 0 else 0

            pct_of_adv = position_value / adv_dollar if adv_dollar > 0 else float('inf')
            # Days to liquidate assuming 20% of daily volume
            d2l = position_value / (adv_dollar * 0.20) if adv_dollar > 0 else float('inf')

            if pct_of_adv > pct_adv_max:
                status = "DANGER"
            elif pct_of_adv > pct_adv_warn:
                status = "WARNING"
            else:
                status = "OK"

            results.append(LiquidityRisk(
                ticker=ticker,
                position_size=position_value,
                avg_daily_volume=adv_dollar,
                pct_of_adv=round(pct_of_adv, 4),
                days_to_liquidate=round(d2l, 1),
                status=status,
            ))

        return results

    def portfolio_liquidity_score(self, liquidity_risks: List[LiquidityRisk]) -> float:
        """
        Compute aggregate portfolio liquidity score (0-1, higher = more liquid).

        Score = 1 - weighted_average(pct_of_adv / max_acceptable)
        """
        if not liquidity_risks:
            return 1.0

        total_value = sum(lr.position_size for lr in liquidity_risks)
        if total_value < 1:
            return 1.0

        pct_adv_max = self.config.liquidity.get("max_position_pct_adv", 0.05)
        score = 0.0
        for lr in liquidity_risks:
            weight = lr.position_size / total_value
            normalized_risk = min(lr.pct_of_adv / pct_adv_max, 1.0) if pct_adv_max > 0 else 0
            score += weight * (1.0 - normalized_risk)

        return round(max(score, 0.0), 4)

    # ── FX Risk ───────────────────────────────────────────────────

    def fx_exposure(
        self,
        currency_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], float]:
        """
        Analyze currency exposure.

        Args:
            currency_weights: {currency_code: portfolio_weight}

        Returns:
            (currency_exposures, fx_risk_pct)
        """
        non_usd = sum(
            w for curr, w in currency_weights.items()
            if curr.upper() != "USD"
        )
        return currency_weights, round(non_usd, 4)

    # ── Tail Risk ─────────────────────────────────────────────────

    def tail_risk(
        self,
        returns: pd.Series,
    ) -> TailRisk:
        """
        Compute tail risk statistics.

        Args:
            returns: Daily return series

        Returns:
            TailRisk with skewness, kurtosis, drawdowns, etc.
        """
        result = TailRisk()
        returns = returns.dropna()
        if len(returns) < 20:
            return result

        r = returns.values

        # Skewness & kurtosis
        result.skewness = float(pd.Series(r).skew())
        result.kurtosis = float(pd.Series(r).kurtosis())  # Excess kurtosis

        # Max drawdown
        cum = (1 + pd.Series(r, index=returns.index)).cumprod()
        rolling_max = cum.expanding().max()
        drawdowns = cum / rolling_max - 1
        result.max_drawdown = float(drawdowns.min())
        result.max_drawdown_days = int(drawdowns.idxmin() - drawdowns[:drawdowns.idxmin()].idxmin().days if hasattr(drawdowns, 'idxmin') else 0) if result.max_drawdown < 0 else 0

        # Max drawdown duration
        dd_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        for i, dd in enumerate(drawdowns.values):
            if dd < 0:
                if dd_start is None:
                    dd_start = i
                current_dd_duration = i - dd_start
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                dd_start = None

        result.max_drawdown_days = max_dd_duration

        # Worst and best days
        worst_idx = int(np.argmin(r))
        best_idx = int(np.argmax(r))
        result.worst_day = float(r[worst_idx])
        result.best_day = float(r[best_idx])
        if hasattr(returns, 'index'):
            result.worst_day_date = str(returns.index[worst_idx])[:10]
            result.best_day_date = str(returns.index[best_idx])[:10]

        # VaR & ES
        result.var_95 = float(np.percentile(r, 5))
        result.var_99 = float(np.percentile(r, 1))
        tail_95 = r[r <= result.var_95]
        result.expected_shortfall_95 = float(tail_95.mean()) if len(tail_95) > 0 else result.var_95

        # Calmar & Sterling ratios
        annual_return = float(np.mean(r) * 252)
        result.calmar_ratio = annual_return / abs(result.max_drawdown) if abs(result.max_drawdown) > 0.001 else 0
        avg_dd = float(drawdowns[drawdowns < 0].mean()) if (drawdowns < 0).any() else 0
        result.sterling_ratio = annual_return / (abs(avg_dd) + 0.10) if avg_dd != 0 else annual_return / 0.10

        return result

    # ── Full Analysis ─────────────────────────────────────────────

    def analyze(
        self,
        label: str,
        portfolio_value: float,
        weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        sectors: Optional[Dict[str, str]] = None,
        market_returns: Optional[pd.Series] = None,
        factor_returns: Optional[Dict[str, pd.Series]] = None,
        position_details: Optional[Dict[str, Dict[str, float]]] = None,
        currencies: Optional[Dict[str, str]] = None,
        benchmark_sectors: Optional[Dict[str, float]] = None,
    ) -> ExposureResult:
        """
        Complete exposure analysis.

        Args:
            label: Portfolio or instrument name
            portfolio_value: Total value
            weights: {ticker: weight}
            asset_returns: DataFrame of asset returns (assets x time)
            sectors: {ticker: sector_name} mapping
            market_returns: Market factor returns
            factor_returns: Additional factor return series
            position_details: {ticker: {value, avg_volume_shares, price}}
            currencies: {ticker: currency_code}
            benchmark_sectors: {sector: benchmark_weight}

        Returns:
            ExposureResult
        """
        result = ExposureResult(label=label, portfolio_value=portfolio_value)
        result.n_positions = len(weights)
        warnings: List[str] = []

        # Portfolio returns
        aligned_returns = asset_returns[list(weights.keys())].dropna()
        w_array = np.array([weights.get(col, 0.0) for col in aligned_returns.columns])
        w_array = w_array / w_array.sum() if w_array.sum() > 0 else w_array
        port_returns = (aligned_returns.values @ w_array)
        port_returns_series = pd.Series(port_returns, index=aligned_returns.index)

        # 1. Factor exposure (portfolio level, using portfolio returns)
        if market_returns is not None:
            result.factor_exposure = self.factor_exposure(
                port_returns_series, market_returns, factor_returns
            )

        # 2. Concentration
        result.concentration = self.concentration(weights)
        if result.concentration.hhi_status != "OK":
            warnings.append(f"Concentration {result.concentration.hhi_status}: HHI={result.concentration.hhi:.3f}")

        # 3. Sector exposure
        if sectors:
            sector_weights: Dict[str, float] = {}
            for ticker, w in weights.items():
                sector = sectors.get(ticker, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0.0) + w
            result.sector_exposure = self.sector_exposure(sector_weights, benchmark_sectors)

        # 4. Correlation matrix
        corr_matrix, avg_corr, max_corr, max_pair = self.correlation_matrix(aligned_returns)
        result.correlation_matrix = corr_matrix
        result.avg_correlation = avg_corr
        result.max_correlation = max_corr
        result.max_corr_pair = max_pair

        corr_hard = self.config.correlation.get("max_pairwise_hard", 0.85)
        corr_warn = self.config.correlation.get("max_pairwise_warning", 0.70)
        if max_corr > corr_hard:
            warnings.append(f"Extreme correlation: {max_pair}={max_corr:.2f}")
        elif max_corr > corr_warn:
            warnings.append(f"High correlation: {max_pair}={max_corr:.2f}")

        # 5. Liquidity risk
        if position_details:
            result.liquidity_risks = self.liquidity_risk(position_details)
            result.portfolio_liquidity_score = self.portfolio_liquidity_score(
                result.liquidity_risks
            )
            for lr in result.liquidity_risks:
                if lr.status != "OK":
                    warnings.append(f"Liquidity {lr.status}: {lr.ticker} @ {lr.pct_of_adv:.2%} ADV")

        # 6. FX risk
        if currencies:
            fx_weights: Dict[str, float] = {}
            for ticker, w in weights.items():
                curr = currencies.get(ticker, "USD")
                fx_weights[curr] = fx_weights.get(curr, 0.0) + w
            result.fx_exposure, result.fx_risk_pct = self.fx_exposure(fx_weights)
            if result.fx_risk_pct > 0.25:
                warnings.append(f"High FX exposure: {result.fx_risk_pct:.0%} non-USD")

        # 7. Tail risk
        result.tail_risk = self.tail_risk(port_returns_series)

        skew_warn = self.config.tail_risk.get("skewness_warning", -0.5)
        kurt_warn = self.config.tail_risk.get("kurtosis_warning", 5.0)
        if result.tail_risk.skewness < skew_warn:
            warnings.append(f"Negative skew: {result.tail_risk.skewness:.2f}")
        if result.tail_risk.kurtosis > kurt_warn:
            warnings.append(f"High kurtosis (fat tails): {result.tail_risk.kurtosis:.1f}")

        result.warnings = warnings
        return result


# ═══════════════════════════════════════════════════════════════════
# Utility: build position details from simple data
# ═══════════════════════════════════════════════════════════════════

def build_position_details(
    tickers: List[str],
    market_values: List[float],
    prices: List[float],
    avg_volumes: Optional[List[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience helper: construct position_details dict.

    Args:
        tickers: List of tickers
        market_values: Position values in dollars
        prices: Current prices
        avg_volumes: Average daily share volumes (optional)

    Returns:
        Dict suitable for ExposureAnalyzer.liquidity_risk()
    """
    details: Dict[str, Dict[str, float]] = {}
    for i, ticker in enumerate(tickers):
        details[ticker] = {
            "value": market_values[i] if i < len(market_values) else 0,
            "price": prices[i] if i < len(prices) else 0,
            "avg_volume_shares": avg_volumes[i] if avg_volumes and i < len(avg_volumes) else 0,
        }
    return details
