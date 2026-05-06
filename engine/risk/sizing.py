#!/usr/bin/env python3
"""
VMAA Risk Engine — Position Sizing
====================================
Multiple sizing methodologies with circuit breakers:
  - Fixed Fractional (1-3% risk per trade)
  - Kelly Criterion (optimal f, fractional Kelly)
  - Risk Parity (equal risk contribution)
  - Volatility Targeting (scale size to target volatility)
  - Drawdown-based sizing (reduce in drawdown)
  - Circuit breakers (daily/weekly/max loss limits)
  - Sizing override (manual override possible)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import SizingConfig, load_config

logger = logging.getLogger("vmaa.risk.engine.sizing")


@dataclass
class SizeRecommendation:
    """Position size recommendation for a single candidate."""
    ticker: str
    entry_price: float
    stop_loss: float
    risk_per_share: float = 0.0

    # Fixed Fractional
    ff_shares: int = 0
    ff_position_pct: float = 0.0
    ff_risk_capital: float = 0.0

    # Kelly
    kelly_fraction: float = 0.0
    kelly_shares: int = 0
    kelly_position_pct: float = 0.0

    # Risk Parity
    rp_weight: float = 0.0
    rp_shares: int = 0

    # Volatility Targeting
    vt_scale: float = 1.0
    vt_shares: int = 0

    # Chosen (method specified in config)
    method: str = "fixed_fractional"
    recommended_shares: int = 0
    recommended_position_pct: float = 0.0
    recommended_risk_capital: float = 0.0

    # Override
    override_shares: Optional[int] = None
    override_reason: str = ""

    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "risk_per_share": round(self.risk_per_share, 2),
            "fixed_fractional": {
                "shares": self.ff_shares,
                "position_pct": round(self.ff_position_pct, 4),
                "risk_capital": round(self.ff_risk_capital, 2),
            },
            "kelly": {
                "fraction": round(self.kelly_fraction, 4),
                "shares": self.kelly_shares,
                "position_pct": round(self.kelly_position_pct, 4),
            },
            "risk_parity": {
                "weight": round(self.rp_weight, 4),
                "shares": self.rp_shares,
            },
            "volatility_targeting": {
                "scale": round(self.vt_scale, 4),
                "shares": self.vt_shares,
            },
            "recommended": {
                "method": self.method,
                "shares": self.recommended_shares,
                "position_pct": round(self.recommended_position_pct, 4),
                "risk_capital": round(self.recommended_risk_capital, 2),
            },
            "override": {
                "shares": self.override_shares,
                "reason": self.override_reason,
            } if self.override_shares is not None else None,
            "reasoning": self.reasoning,
        }


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker check result."""
    trading_allowed: bool = True
    daily_loss: float = 0.0
    daily_limit_hit: bool = False
    weekly_loss: float = 0.0
    weekly_limit_hit: bool = False
    max_drawdown: float = 0.0
    drawdown_limit_hit: bool = False
    consecutive_losses: int = 0
    consecutive_loss_limit_hit: bool = False
    active_breakers: List[str] = field(default_factory=list)
    recommendation: str = "CONTINUE"   # CONTINUE | REDUCE | STOP | FULL_STOP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trading_allowed": self.trading_allowed,
            "daily_loss_pct": round(self.daily_loss, 4),
            "daily_limit_hit": self.daily_limit_hit,
            "weekly_loss_pct": round(self.weekly_loss, 4),
            "weekly_limit_hit": self.weekly_limit_hit,
            "max_drawdown_pct": round(self.max_drawdown, 4),
            "drawdown_limit_hit": self.drawdown_limit_hit,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_loss_limit_hit": self.consecutive_loss_limit_hit,
            "active_breakers": self.active_breakers,
            "recommendation": self.recommendation,
        }


class PositionSizer:
    """
    Multi-method position sizing with circuit breakers.

    Methods:
      - Fixed Fractional: risk fixed % of capital per trade
      - Kelly Criterion: optimal f based on win probability
      - Risk Parity: equal risk contribution
      - Volatility Targeting: scale to target portfolio vol

    Usage:
        sizer = PositionSizer(config)
        rec = sizer.suggest("AAPL", entry=100, stop=95, portfolio=100000, ...)
        cb = sizer.check_circuit_breakers(trade_history, portfolio_history)
    """

    def __init__(self, config: Optional[SizingConfig] = None):
        if config is None:
            full_cfg = load_config()
            config = full_cfg.sizing
        self.config = config
        self.method = config.method

    # ── Fixed Fractional ──────────────────────────────────────────

    def fixed_fractional(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
        confidence: float = 0.50,
    ) -> Tuple[int, float, float, List[str]]:
        """
        Fixed fractional position sizing.

        Risk capital = portfolio_value * base_risk_pct * confidence_scalar
        Shares = risk_capital / risk_per_share

        Args:
            ticker: Instrument ticker
            entry_price: Entry price per share
            stop_loss: Stop loss price
            portfolio_value: Total portfolio value
            confidence: Trade confidence score (0-1)

        Returns:
            (shares, position_pct, risk_capital, reasoning)
        """
        reasoning: List[str] = []
        ff = self.config.fixed_fractional

        # Confidence threshold
        if confidence < ff.confidence_threshold:
            reasoning.append(f"Confidence {confidence:.0%} < threshold {ff.confidence_threshold:.0%}")
            return 0, 0.0, 0.0, reasoning

        # Risk per share
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.05
            reasoning.append(f"Risk/share negative, using fallback 5%")

        # Risk percentage scaled by confidence
        confidence_scalar = max(0.35, min(confidence, 1.0))
        risk_pct = ff.base_risk_pct * confidence_scalar
        risk_pct = max(ff.min_risk_pct, min(risk_pct, ff.max_risk_pct))
        risk_capital = portfolio_value * risk_pct

        # Shares
        raw_shares = risk_capital / risk_per_share if risk_per_share > 0 else 0
        shares = int(raw_shares)

        # Cap at 5000 shares (sanity limit)
        shares = min(shares, 5000)

        position_pct = (shares * entry_price / portfolio_value) if portfolio_value > 0 else 0.0

        reasoning.append(
            f"Fixed fractional: {risk_pct:.2%} risk * confidence={confidence_scalar:.2f}x "
            f"= ${risk_capital:,.0f} risk capital, {shares} shares ({position_pct:.1%})"
        )

        return shares, round(position_pct, 4), round(risk_capital, 2), reasoning

    # ── Kelly Criterion ───────────────────────────────────────────

    def kelly_criterion(
        self,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        entry_price: float,
        portfolio_value: float,
    ) -> Tuple[float, int, float]:
        """
        Kelly Criterion for optimal position sizing.

        f* = (p·W - (1-p)·L) / (W·L)   or simplified: f* = p - (1-p)/(W/L)

        Where:
          p = win probability
          W = average win (as fraction of capital)
          L = average loss (as fraction of capital)
          W/L = win/loss ratio (reward-to-risk)

        Returns fractional Kelly: f = fraction_of_kelly * f*

        Args:
            win_probability: Estimated probability of winning
            avg_win: Average winning return (decimal)
            avg_loss: Average losing return (decimal, positive)

        Returns:
            (kelly_fraction, shares, position_pct)
        """
        k = self.config.kelly

        if win_probability < k.min_win_probability:
            return 0.0, 0, 0.0

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        if win_loss_ratio <= 0:
            return 0.0, 0, 0.0

        # Kelly formula: f* = (p * W/L - (1-p)) / (W/L)  simplified
        # f* = p - (1-p) / (W/L)
        f_star = win_probability - (1.0 - win_probability) / win_loss_ratio

        # Clamp
        f_star = max(0.0, min(f_star, 1.0))

        # Fractional Kelly
        f = f_star * k.fraction
        f = max(k.min_fraction, min(f, k.max_fraction))

        # Position size
        position_value = portfolio_value * f
        shares = int(position_value / entry_price) if entry_price > 0 else 0
        shares = min(shares, 5000)
        position_pct = (shares * entry_price / portfolio_value) if portfolio_value > 0 else 0.0

        return round(f, 4), shares, round(position_pct, 4)

    # ── Risk Parity ───────────────────────────────────────────────

    def risk_parity(
        self,
        returns_matrix: pd.DataFrame,
        portfolio_value: float,
        prices: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Risk Parity: equal risk contribution from each position.

        Target weights w* such that w_i · ∂σ_p/∂w_i = constant

        Uses simple inverse-volatility heuristic:
          w_i ∝ 1/σ_i  (normalized)

        Args:
            returns_matrix: DataFrame of asset returns
            portfolio_value: Total portfolio value
            prices: {ticker: current_price}

        Returns:
            Dict of {ticker: target_weight}
        """
        import pandas as pd

        rp = self.config.risk_parity
        lookback = rp.lookback_days

        if returns_matrix.empty:
            return {}

        recent = returns_matrix.tail(lookback).dropna(axis=1)
        if recent.empty:
            return {}

        # Inverse volatility weights
        vols = recent.std()
        inv_vols = 1.0 / (vols + 1e-10)
        raw_weights = inv_vols / inv_vols.sum()

        # Clamp to max leverage
        total_w = float(raw_weights.sum())
        if total_w > rp.max_leverage:
            raw_weights = raw_weights / total_w * rp.max_leverage

        return {col: round(float(raw_weights[col]), 4) for col in raw_weights.index}

    def rp_shares(
        self,
        rp_weight: float,
        portfolio_value: float,
        entry_price: float,
    ) -> int:
        """Convert risk parity weight to share count."""
        position_value = portfolio_value * rp_weight
        return int(position_value / entry_price) if entry_price > 0 else 0

    # ── Volatility Targeting ──────────────────────────────────────

    def volatility_targeting(
        self,
        current_vol: float,
        entry_price: float,
        base_shares: int,
    ) -> Tuple[float, int]:
        """
        Scale position size to target portfolio volatility.

        scale_factor = target_vol / current_vol
        adjusted_shares = base_shares * scale_factor

        Args:
            current_vol: Current annualized vol of the instrument
            entry_price: Entry price
            base_shares: Base position size before scaling

        Returns:
            (scale_factor, adjusted_shares)
        """
        vt = self.config.volatility_targeting

        if current_vol < 0.01:
            return 1.0, base_shares

        scale = vt.target_vol / current_vol
        scale = max(vt.min_scale_down, min(scale, vt.max_scale_up))

        adj_shares = int(base_shares * scale)

        return round(scale, 4), adj_shares

    # ── Drawdown-Based Sizing ─────────────────────────────────────

    def drawdown_adjustment(
        self,
        current_drawdown: float,
        base_shares: int,
    ) -> Tuple[float, int, str]:
        """
        Reduce position size when in drawdown.

        Adjustment schedule:
          0-5% DD:    1.0x (no reduction)
          5-10% DD:   0.75x
          10-15% DD:  0.50x
          15%+ DD:    0.25x (or stop)

        Args:
            current_drawdown: Current drawdown (negative)
            base_shares: Base position size

        Returns:
            (multiplier, adjusted_shares, reason)
        """
        dd = abs(current_drawdown)  # Make positive for comparison
        if dd < 0.05:
            return 1.0, base_shares, "No drawdown adjustment"
        elif dd < 0.10:
            mult = 0.75
            reason = f"Moderate drawdown ({dd:.1%}), 0.75x sizing"
        elif dd < 0.15:
            mult = 0.50
            reason = f"Significant drawdown ({dd:.1%}), 0.50x sizing"
        else:
            mult = 0.25
            reason = f"Deep drawdown ({dd:.1%}), 0.25x sizing"

        return mult, int(base_shares * mult), reason

    # ── Circuit Breakers ──────────────────────────────────────────

    def check_circuit_breakers(
        self,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        portfolio_history: Optional[Dict[str, float]] = None,
    ) -> CircuitBreakerStatus:
        """
        Check all circuit breakers.

        Args:
            trade_history: List of completed trades with {pnl, close_date}
            portfolio_history: Dict of {date: portfolio_value} for drawdown calc

        Returns:
            CircuitBreakerStatus
        """
        cb = self.config.circuit_breakers
        status = CircuitBreakerStatus()

        if not trade_history and not portfolio_history:
            return status

        today = datetime.now().date()

        # ── Daily loss limit ──
        daily_pnl = 0.0
        for t in (trade_history or []):
            trade_date = t.get("close_date", "")
            if isinstance(trade_date, str):
                trade_date = trade_date[:10]
            if str(trade_date) == str(today):
                daily_pnl += t.get("pnl", 0.0)
        status.daily_loss = daily_pnl

        if portfolio_history:
            # Use portfolio history for daily loss
            dates = sorted(portfolio_history.keys())
            if dates:
                today_str = str(today)
                yesterday_str = str(today - timedelta(days=1))
                if today_str in portfolio_history and yesterday_str in portfolio_history:
                    daily_return = (portfolio_history[today_str] / portfolio_history[yesterday_str]) - 1
                    if daily_return < 0:
                        status.daily_loss = min(status.daily_loss, daily_return)

        if status.daily_loss <= cb.daily_loss_limit:
            status.daily_limit_hit = True
            status.active_breakers.append("daily_loss")

        # ── Weekly loss limit ──
        weekly_pnl = 0.0
        week_ago = today - timedelta(days=7)
        for t in (trade_history or []):
            trade_date = t.get("close_date", "")
            if isinstance(trade_date, str):
                trade_date = trade_date[:10]
            try:
                td = datetime.strptime(str(trade_date), "%Y-%m-%d").date()
                if td >= week_ago:
                    weekly_pnl += t.get("pnl", 0.0)
            except (ValueError, TypeError):
                pass
        status.weekly_loss = weekly_pnl

        if status.weekly_loss <= cb.weekly_loss_limit:
            status.weekly_limit_hit = True
            status.active_breakers.append("weekly_loss")

        # ── Max drawdown limit ──
        if portfolio_history:
            values = sorted(portfolio_history.items())
            if len(values) > 1:
                peak = values[0][1]
                max_dd = 0.0
                for _, val in values:
                    if val > peak:
                        peak = val
                    dd = (val / peak) - 1.0 if peak > 0 else 0.0
                    max_dd = min(max_dd, dd)
                status.max_drawdown = max_dd

                if status.max_drawdown <= cb.max_drawdown_limit:
                    status.drawdown_limit_hit = True
                    status.active_breakers.append("max_drawdown")

        # ── Consecutive losses ──
        if trade_history:
            sorted_trades = sorted(
                trade_history,
                key=lambda t: t.get("close_date", ""),
                reverse=True,
            )
            consecutive = 0
            for t in sorted_trades:
                pnl = t.get("pnl", 0.0)
                if pnl < 0:
                    consecutive += 1
                else:
                    break
            status.consecutive_losses = consecutive

            if consecutive >= cb.consecutive_losses:
                status.consecutive_loss_limit_hit = True
                status.active_breakers.append("consecutive_losses")

        # ── Recommendation ──
        if status.drawdown_limit_hit:
            status.recommendation = "FULL_STOP"
            status.trading_allowed = False
        elif status.daily_limit_hit:
            status.recommendation = "STOP"          # Stop for the day
            status.trading_allowed = False
        elif status.weekly_limit_hit:
            status.recommendation = "STOP"          # Stop for the week
            status.trading_allowed = False
        elif status.consecutive_loss_limit_hit:
            status.recommendation = "REDUCE"        # Reduce sizing
            status.trading_allowed = True
        elif len(status.active_breakers) > 0:
            status.recommendation = "REDUCE"
            status.trading_allowed = True
        else:
            status.recommendation = "CONTINUE"
            status.trading_allowed = True

        return status

    # ── Position Loss Check ───────────────────────────────────────

    def check_position_loss(
        self,
        ticker: str,
        entry_price: float,
        current_price: float,
    ) -> Tuple[bool, str]:
        """
        Check if a position has hit its stop-loss limit.

        Returns:
            (should_exit, reason)
        """
        cb = self.config.circuit_breakers
        pnl_pct = (current_price / entry_price) - 1.0 if entry_price > 0 else 0

        if pnl_pct <= cb.position_loss_limit:
            return True, f"Position loss limit hit: {pnl_pct:.1%} ≤ {cb.position_loss_limit:.0%}"

        return False, "OK"

    # ── Full Suggestion ───────────────────────────────────────────

    def suggest(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
        confidence: float = 0.50,
        win_probability: Optional[float] = None,
        avg_win: float = 0.15,
        avg_loss: float = 0.10,
        returns: Optional[np.ndarray] = None,
        current_vol: Optional[float] = None,
        rp_weight: float = 0.0,
        current_drawdown: float = 0.0,
        override_shares: Optional[int] = None,
        override_reason: str = "",
    ) -> SizeRecommendation:
        """
        Generate position size recommendation across all methods.

        Args:
            ticker: Instrument ticker
            entry_price: Entry price per share
            stop_loss: Stop loss price
            portfolio_value: Total portfolio value in dollars
            confidence: Trade confidence (0-1)
            win_probability: Estimated win rate for Kelly
            avg_win: Average winning return for Kelly
            avg_loss: Average losing return for Kelly
            returns: Historical returns for vol targeting
            current_vol: Current volatility for vol targeting
            rp_weight: Risk parity target weight
            current_drawdown: Current portfolio drawdown
            override_shares: Manual override share count
            override_reason: Reason for override

        Returns:
            SizeRecommendation
        """
        rec = SizeRecommendation(
            ticker=ticker,
            entry_price=entry_price,
            stop_loss=stop_loss,
            risk_per_share=round(entry_price - stop_loss, 2),
            method=self.method,
            override_shares=override_shares,
            override_reason=override_reason,
        )

        # ── Fixed Fractional ──
        ff_shares, ff_pct, ff_risk, ff_reasoning = self.fixed_fractional(
            ticker, entry_price, stop_loss, portfolio_value, confidence
        )
        rec.ff_shares = ff_shares
        rec.ff_position_pct = ff_pct
        rec.ff_risk_capital = ff_risk
        rec.reasoning.extend(ff_reasoning)

        # ── Kelly Criterion ──
        if win_probability is not None:
            k_fraction, k_shares, k_pct = self.kelly_criterion(
                win_probability, avg_win, avg_loss,
                entry_price, portfolio_value
            )
            rec.kelly_fraction = k_fraction
            rec.kelly_shares = k_shares
            rec.kelly_position_pct = k_pct
            rec.reasoning.append(
                f"Kelly: f*={k_fraction:.4f}, {k_shares} shares ({k_pct:.1%})"
                f" (win_prob={win_probability:.0%}, W/L={avg_win/avg_loss:.1f}x)"
            )

        # ── Risk Parity ──
        rec.rp_weight = rp_weight
        rec.rp_shares = self.rp_shares(rp_weight, portfolio_value, entry_price)

        # ── Volatility Targeting ──
        if current_vol is not None and ff_shares > 0:
            vt_scale, vt_shares = self.volatility_targeting(current_vol, entry_price, ff_shares)
            rec.vt_scale = vt_scale
            rec.vt_shares = vt_shares

        # ── Drawdown Adjustment ──
        dd_mult, dd_shares, dd_reason = self.drawdown_adjustment(
            current_drawdown, ff_shares
        )
        rec.reasoning.append(dd_reason)

        # ── Select Method ──
        if self.method == "fixed_fractional":
            base_shares = ff_shares
        elif self.method == "kelly":
            base_shares = rec.kelly_shares
        elif self.method == "risk_parity":
            base_shares = rec.rp_shares
        elif self.method == "vol_target":
            base_shares = rec.vt_shares
        else:
            base_shares = ff_shares

        # Apply drawdown adjustment
        final_shares = int(base_shares * dd_mult)
        final_shares = max(0, min(final_shares, 5000))

        rec.recommended_shares = final_shares
        rec.recommended_position_pct = round(
            (final_shares * entry_price / portfolio_value), 4
        ) if portfolio_value > 0 else 0.0
        rec.recommended_risk_capital = round(
            final_shares * (entry_price - stop_loss), 2
        ) if final_shares > 0 else 0.0

        # Override if specified
        if override_shares is not None:
            rec.reasoning.append(f"OVERRIDE: {override_shares} shares — {override_reason}")

        return rec


def estimate_win_probability(
    quality_score: float,
    magna_score: int,
    confidence: float,
    market_ok: bool,
) -> float:
    """
    Estimate win probability from quality, momentum, and market signals.

    Simple heuristic:
      base = 0.40 (random chance for screened stocks)
      + quality_score * 0.20
      + (magna_score / 10) * 0.15
      + confidence * 0.15
      + market_ok * 0.10

    Returns:
        Estimated win probability (0.0-1.0)
    """
    base = 0.40
    prob = (
        base
        + quality_score * 0.20
        + (magna_score / 10.0) * 0.15
        + confidence * 0.15
        + (0.10 if market_ok else 0.0)
    )
    return round(min(max(prob, 0.05), 0.85), 4)
