#!/usr/bin/env python3
"""
VMAA Smart Selection Engine — Auto Rotation Engine
====================================================
Rotation strategies for dynamic pool management:
  - Score-Based:     Replace lowest-ranked with highest-ranked candidates
  - Sector-Balance:  Maintain sector allocation limits
  - Volatility-Adjusted: Reduce positions during high volatility

Triggers:
  - Scheduled (daily/weekly/monthly)
  - Event-driven (earnings, price moves, volume spikes)
  - Performance-driven (drawdown, Sharpe, turnover)

Outputs:
  - Rotation log (what entered, left, why)
  - Impact analysis (slippage estimation, turnover rate)

Usage:
  from engine.selection.rotation import RotationEngine
  re = RotationEngine(pool_manager)
  result = re.execute_rotation(strategy="score_based")
  print(result.summary)
"""
from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from engine.selection.config import RotationConfig, get_selection_config
from engine.selection.pool import PoolManager, PoolStatus, PoolEntry
from engine.selection.factors import FactorEngine, FactorScores, extract_factors

logger = logging.getLogger("vmaa.engine.selection.rotation")


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RotationDecision:
    """A single rotation decision (enter/exit/replace)."""
    action: str                        # "add", "remove", "replace", "reduce", "hold"
    ticker: str
    from_pool: str = ""
    to_pool: str = ""
    reason: str = ""
    score: float = 0.0
    estimated_slippage_bps: float = 0.0
    estimated_cost: float = 0.0


@dataclass
class RotationResult:
    """Complete rotation execution result."""
    strategy: str
    timestamp: str
    decisions: List[RotationDecision] = field(default_factory=list)
    
    # Summary stats
    total_replaced: int = 0
    total_added: int = 0
    total_removed: int = 0
    turnover_rate_pct: float = 0.0
    estimated_total_slippage_bps: float = 0.0
    
    # Impact
    score_improvement: float = 0.0    # Improvement in avg composite score
    sector_rebalancing: bool = False
    
    def summarize(self) -> str:
        lines = [
            f"Rotation [{self.strategy}] @ {self.timestamp}",
            f"  Replaced: {self.total_replaced} | Added: {self.total_added} | Removed: {self.total_removed}",
            f"  Turnover: {self.turnover_rate_pct:.1f}% | Slippage: {self.estimated_total_slippage_bps:.0f} bps",
            f"  Score Δ: {self.score_improvement:+.4f}",
        ]
        if self.decisions:
            lines.append("  Decisions:")
            for d in self.decisions[:10]:
                lines.append(f"    {d.action:7s} {d.ticker:6s} — {d.reason}")
            if len(self.decisions) > 10:
                lines.append(f"    ... and {len(self.decisions)-10} more")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Rotation Engine
# ═══════════════════════════════════════════════════════════════════

class RotationEngine:
    """
    Auto rotation engine for the Smart Selection system.
    
    Implements three rotation strategies:
    1. Score-Based: Replace lowest-ranked with highest-ranked
    2. Sector-Balance: Enforce sector allocation limits
    3. Volatility-Adjusted: Scale positions based on volatility regime
    
    Example:
        >>> from engine.selection.pool import PoolManager
        >>> pm = PoolManager()
        >>> re = RotationEngine(pm)
        >>> result = re.execute_rotation(strategy="score_based", pool="core")
        >>> print(result.summarize())
    """
    
    def __init__(self, 
                 pool_manager: Optional[PoolManager] = None,
                 factor_engine: Optional[FactorEngine] = None):
        self.pm = pool_manager or PoolManager()
        self.fe = factor_engine or FactorEngine()
        self.config = get_selection_config().get_rotation_config()
        
        # Rotation cooldown tracking
        self._last_rotation_time: Dict[str, float] = {}
        self._event_rotation_count: Dict[str, int] = defaultdict(int)
        self._event_rotation_reset = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        logger.info(
            f"RotationEngine initialized: "
            f"{sum(1 for s in self.config.strategies.values() if s.get('enabled'))} "
            f"active strategies"
        )

    # ── Strategy: Score-Based Replacement ────────────────────────

    def rotate_score_based(self, 
                           pool: str = "core",
                           candidate_pool: str = "watchlist",
                           dry_run: bool = False,
                           ) -> RotationResult:
        """
        Replace lowest-ranked stocks in a pool with highest-ranked candidates
        from the candidate pool.
        
        Args:
            pool: Pool to rotate (e.g., "core")
            candidate_pool: Pool to draw candidates from (e.g., "watchlist")
            dry_run: If True, only compute decisions without executing
        
        Returns:
            RotationResult with all decisions
        """
        strategy_cfg = self.config.strategies.get("score_based", {})
        if not strategy_cfg.get("enabled", True):
            return RotationResult(strategy="score_based", timestamp="disabled")
        
        replacement_ratio = strategy_cfg.get("replacement_ratio", 0.20)
        min_score_gap = strategy_cfg.get("min_score_gap", 0.10)
        max_replacements = strategy_cfg.get("max_replacements_per_cycle", 5)
        
        pool_status = self.pm.get_pool_status(pool)
        candidate_status = self.pm.get_pool_status(candidate_pool)
        
        decisions = []
        
        # Get pool entries sorted by score (lowest first)
        pool_entries = [
            self.pm.get_stock_entry(t)
            for t in pool_status.tickers
            if self.pm.get_stock_entry(t)
        ]
        pool_entries.sort(key=lambda e: e.composite_score if e else 0)
        
        # Get candidate entries sorted by score (highest first)
        candidate_entries = [
            self.pm.get_stock_entry(t)
            for t in candidate_status.tickers
            if self.pm.get_stock_entry(t)
        ]
        candidate_entries.sort(key=lambda e: e.composite_score if e else 0, reverse=True)
        
        # Calculate how many to replace
        num_to_replace = min(
            max_replacements,
            int(len(pool_entries) * replacement_ratio),
            len(candidate_entries),
        )
        
        if num_to_replace == 0:
            logger.info(f"No score-based replacements needed for {pool}")
            return RotationResult(strategy="score_based",
                               timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        
        for i in range(num_to_replace):
            if i >= len(pool_entries) or i >= len(candidate_entries):
                break
            
            weak_stock = pool_entries[i]
            strong_candidate = candidate_entries[-(i+1)]  # Best from candidates
            
            if not weak_stock or not strong_candidate:
                continue
            
            score_gap = strong_candidate.composite_score - weak_stock.composite_score
            
            if score_gap < min_score_gap:
                logger.debug(
                    f"Score gap {score_gap:.3f} < {min_score_gap}, "
                    f"skipping {weak_stock.ticker}/{strong_candidate.ticker}"
                )
                continue
            
            slippage = self._estimate_slippage(strong_candidate.ticker)
            
            decisions.append(RotationDecision(
                action="replace",
                ticker=f"{weak_stock.ticker}→{strong_candidate.ticker}",
                from_pool=pool,
                to_pool=pool,
                reason=f"Score {weak_stock.composite_score:.3f} → {strong_candidate.composite_score:.3f} (gap: {score_gap:.3f})",
                score=score_gap,
                estimated_slippage_bps=slippage,
            ))
        
        # Execute if not dry run
        if not dry_run:
            for d in decisions:
                # Swap: remove weak, promote strong
                tickers = d.ticker.split("→")
                weak_ticker = tickers[0].strip()
                strong_ticker = tickers[1].strip()
                
                self.pm.move_stock(weak_ticker, candidate_pool, d.reason)
                self.pm.move_stock(strong_ticker, pool, d.reason)
        
        # Compute summary
        pool_after = self.pm.get_pool_status(pool)
        score_improvement = pool_after.avg_composite_score - pool_status.avg_composite_score
        
        result = RotationResult(
            strategy="score_based",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            decisions=decisions,
            total_replaced=len(decisions),
            total_added=len(decisions),
            total_removed=len(decisions),
            turnover_rate_pct=(len(decisions) / max(pool_status.size, 1)) * 100,
            estimated_total_slippage_bps=sum(d.estimated_slippage_bps for d in decisions),
            score_improvement=round(score_improvement, 4),
        )
        
        logger.info(f"Score-based rotation: {len(decisions)} replacements")
        return result

    # ── Strategy: Sector Balance ─────────────────────────────────

    def rotate_sector_balance(self,
                              pool: str = "core",
                              dry_run: bool = False,
                              ) -> RotationResult:
        """
        Enforce sector allocation limits within a pool.
        
        Rotates out overweight sectors, rotates in underweight sectors.
        
        Args:
            pool: Pool to balance
            dry_run: If True, only compute decisions
        """
        strategy_cfg = self.config.strategies.get("sector_balance", {})
        if not strategy_cfg.get("enabled", True):
            return RotationResult(strategy="sector_balance", timestamp="disabled")
        
        max_sector_pct = strategy_cfg.get("max_sector_weight_pct", 25.0)
        tolerance_pct = strategy_cfg.get("rebalance_tolerance_pct", 5.0)
        target_sectors = strategy_cfg.get("target_sectors", [])
        
        pool_status = self.pm.get_pool_status(pool)
        decisions = []
        
        if pool_status.size == 0:
            return RotationResult(strategy="sector_balance",
                               timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        
        sectors = pool_status.sector_distribution
        total = pool_status.size
        
        # Identify overweight sectors
        overweight: List[Tuple[str, float, int]] = []
        for sector, count in sectors.items():
            pct = (count / total) * 100
            if pct > max_sector_pct + tolerance_pct:
                overweight.append((sector, pct, count))
        
        if not overweight:
            logger.debug(f"No sector imbalances in {pool}")
            return RotationResult(strategy="sector_balance",
                               timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        
        # Find stocks in overweight sectors to move out
        for sector, pct, count in overweight:
            excess = int(count - (max_sector_pct / 100 * total))
            if excess <= 0:
                continue
            
            # Get stocks in this sector, sorted by lowest composite score
            sector_stocks = []
            for ticker in pool_status.tickers:
                entry = self.pm.get_stock_entry(ticker)
                if entry and entry.metadata.get("sector") == sector:
                    sector_stocks.append((ticker, entry.composite_score))
            
            sector_stocks.sort(key=lambda x: x[1])
            
            for ticker, score in sector_stocks[:excess]:
                decisions.append(RotationDecision(
                    action="reduce",
                    ticker=ticker,
                    from_pool=pool,
                    to_pool="watchlist",
                    reason=f"Overweight {sector} ({pct:.0f}% > {max_sector_pct:.0f}%)",
                    score=score,
                ))
        
        # Execute
        if not dry_run:
            for d in decisions:
                if d.action == "reduce":
                    self.pm.move_stock(d.ticker, "watchlist", d.reason)
        
        result = RotationResult(
            strategy="sector_balance",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            decisions=decisions,
            total_removed=len(decisions),
            sector_rebalancing=True,
        )
        
        logger.info(f"Sector balance: {len(decisions)} stocks moved out")
        return result

    # ── Strategy: Volatility-Adjusted Rotation ────────────────────

    def rotate_volatility_adjusted(self,
                                    pool: str = "core",
                                    dry_run: bool = False,
                                    ) -> RotationResult:
        """
        Adjust pool composition based on market volatility regime.
        
        High volatility → move high-beta stocks to watchlist
        Low volatility → can increase exposure to growth/momentum names
        
        Args:
            pool: Pool to adjust
            dry_run: If True, only compute decisions
        """
        strategy_cfg = self.config.strategies.get("volatility_adjusted", {})
        if not strategy_cfg.get("enabled", True):
            return RotationResult(strategy="volatility_adjusted", timestamp="disabled")
        
        high_vol_threshold = strategy_cfg.get("high_vol_threshold", 0.30)
        normal_vol_threshold = strategy_cfg.get("normal_vol_threshold", 0.20)
        vol_lookback = strategy_cfg.get("vol_lookback_days", 20)
        
        # Determine current volatility regime
        try:
            vol_regime = self._get_volatility_regime()
        except Exception as e:
            logger.warning(f"Could not determine volatility regime: {e}")
            return RotationResult(strategy="volatility_adjusted",
                               timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        
        decisions = []
        pool_status = self.pm.get_pool_status(pool)
        
        if vol_regime == "HIGH":
            # Reduce high-beta exposure
            for ticker in pool_status.tickers:
                entry = self.pm.get_stock_entry(ticker)
                if not entry:
                    continue
                
                try:
                    profile = extract_factors(ticker)
                    if profile.beta > 1.5 or profile.volatility_20d > high_vol_threshold:
                        decisions.append(RotationDecision(
                            action="reduce",
                            ticker=ticker,
                            from_pool=pool,
                            to_pool="watchlist",
                            reason=f"High vol regime — beta={profile.beta:.1f}, vol20d={profile.volatility_20d:.2f}",
                            score=profile.beta,
                        ))
                except Exception:
                    pass
        
        elif vol_regime == "NORMAL" or vol_regime == "LOW":
            # Can consider adding more names
            pass
        
        scale = strategy_cfg.get("high_vol_position_scalar", 0.50) if vol_regime == "HIGH" else 1.0
        
        # Execute
        if not dry_run:
            for d in decisions:
                if d.action == "reduce":
                    self.pm.move_stock(d.ticker, "watchlist", d.reason)
        
        result = RotationResult(
            strategy="volatility_adjusted",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            decisions=decisions,
            total_removed=len(decisions),
        )
        
        logger.info(f"Volatility-adjusted ({vol_regime}): {len(decisions)} adjustments")
        return result

    # ── Combined Rotation ────────────────────────────────────────

    def execute_rotation(self,
                        pool: str = "core",
                        strategies: Optional[List[str]] = None,
                        dry_run: bool = False,
                        ) -> List[RotationResult]:
        """
        Execute all enabled rotation strategies for a pool.
        
        Args:
            pool: Pool to rotate
            strategies: Specific strategies to run (default: all enabled)
            dry_run: If True, only compute decisions
        
        Returns:
            List of RotationResult for each strategy executed
        """
        if strategies is None:
            strategies = [
                s for s, cfg in self.config.strategies.items()
                if cfg.get("enabled", True)
            ]
        
        results = []
        
        for strategy in strategies:
            try:
                if strategy == "score_based":
                    result = self.rotate_score_based(pool=pool, dry_run=dry_run)
                elif strategy == "sector_balance":
                    result = self.rotate_sector_balance(pool=pool, dry_run=dry_run)
                elif strategy == "volatility_adjusted":
                    result = self.rotate_volatility_adjusted(pool=pool, dry_run=dry_run)
                else:
                    logger.warning(f"Unknown rotation strategy: {strategy}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Rotation strategy '{strategy}' failed: {e}")
        
        # Log
        total_decisions = sum(r.total_replaced + r.total_added + r.total_removed for r in results)
        logger.info(
            f"Rotation complete for {pool}: {len(results)} strategies, "
            f"{total_decisions} total decisions"
        )
        
        return results

    # ── Trigger Checking ─────────────────────────────────────────

    def should_rotate(self, pool: str) -> Tuple[bool, str]:
        """
        Check if any rotation triggers have fired for a pool.
        
        Returns:
            (should_rotate, reason)
        """
        pool_cfg = self.pm.pool_configs.get(pool)
        if not pool_cfg:
            return False, f"Unknown pool: {pool}"
        
        status = self.pm.get_pool_status(pool)
        
        # Check if pool is full and has candidates waiting
        if status.is_full:
            candidate_status = self.pm.get_pool_status("watchlist")
            if candidate_status.size > 0:
                return True, "Pool full + watchlist candidates available"
        
        # Check sector imbalance
        sectors = status.sector_distribution
        max_sector = self.config.strategies.get("sector_balance", {}).get("max_sector_weight_pct", 25.0)
        total = status.size
        
        for sector, count in sectors.items():
            if total > 0:
                pct = (count / total) * 100
                if pct > max_sector + 5:
                    return True, f"Sector overweight: {sector} at {pct:.0f}%"
        
        # Check scheduled triggers
        triggers_cfg = self.config.triggers
        
        # Performance trigger
        perf_cfg = triggers_cfg.get("performance_driven", {})
        if perf_cfg.get("enabled", True):
            dd_threshold = perf_cfg.get("triggers", {}).get("pool_drawdown_pct", 10.0)
            # Would need actual P&L tracking for this
        
        return False, "No triggers fired"

    # ── Slippage Estimation ──────────────────────────────────────

    def _estimate_slippage(self, ticker: str) -> float:
        """
        Estimate slippage for trading a stock.
        
        Uses spread-based method with volume impact factor.
        Returns slippage in basis points.
        """
        slippage_cfg = self.config.slippage
        default_bps = slippage_cfg.get("default_spread_bps", 10.0)
        vol_factor = slippage_cfg.get("volume_impact_factor", 0.1)
        
        try:
            profile = extract_factors(ticker)
            if profile.extraction_ok and profile.market_cap > 0:
                # Larger caps = less slippage
                cap_b = profile.market_cap / 1e9
                if cap_b > 100:
                    return default_bps * 0.3
                elif cap_b > 10:
                    return default_bps * 0.6
                elif cap_b > 1:
                    return default_bps
                else:
                    return default_bps * (1 + vol_factor * (1 / max(cap_b, 0.1)))
        except Exception:
            pass
        
        return default_bps

    def _get_volatility_regime(self) -> str:
        """
        Determine current market volatility regime.
        
        Returns:
            "HIGH", "NORMAL", or "LOW"
        """
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period="3mo")
            if hist is None or len(hist) < 20:
                return "NORMAL"
            
            returns = hist["Close"].pct_change().dropna()
            vol20 = float(returns.iloc[-20:].std() * (252 ** 0.5))
            
            high_threshold = self.config.strategies.get("volatility_adjusted", {}).get("high_vol_threshold", 0.30)
            normal_threshold = self.config.strategies.get("volatility_adjusted", {}).get("normal_vol_threshold", 0.20)
            
            if vol20 > high_threshold:
                return "HIGH"
            elif vol20 > normal_threshold:
                return "NORMAL"
            else:
                return "LOW"
        except Exception:
            return "NORMAL"

    # ── Impact Analysis ──────────────────────────────────────────

    def analyze_rotation_impact(self, 
                                 pool: str,
                                 decisions: List[RotationDecision],
                                 ) -> Dict[str, Any]:
        """
        Analyze the impact of proposed rotation decisions.
        
        Returns dict with:
          - turnover_rate: % of pool rotated
          - estimated_slippage_total: total slippage in bps
          - estimated_cost: approximate cost in dollars
          - score_impact: expected change in avg composite score
          - sector_impact: sector composition changes
        """
        pool_status = self.pm.get_pool_status(pool)
        
        analysis = {
            "pool": pool,
            "pool_size": pool_status.size,
            "num_decisions": len(decisions),
            "turnover_rate_pct": (len(decisions) / max(pool_status.size, 1)) * 100,
            "total_slippage_bps": sum(d.estimated_slippage_bps for d in decisions),
            "avg_slippage_bps": (
                sum(d.estimated_slippage_bps for d in decisions) / max(len(decisions), 1)
            ),
            "actions": defaultdict(int),
            "sectors_affected": set(),
            "warnings": [],
        }
        
        for d in decisions:
            analysis["actions"][d.action] += 1
            # Get sector info
            tickers = d.ticker.split("→")
            for t in tickers:
                entry = self.pm.get_stock_entry(t.strip())
                if entry:
                    sector = entry.metadata.get("sector", "Unknown")
                    analysis["sectors_affected"].add(sector)
        
        analysis["sectors_affected"] = list(analysis["sectors_affected"])
        analysis["actions"] = dict(analysis["actions"])
        
        # Warnings
        if analysis["turnover_rate_pct"] > 40:
            analysis["warnings"].append("High turnover — potential tax/transaction cost impact")
        if analysis["total_slippage_bps"] > 100:
            analysis["warnings"].append("Significant slippage expected")
        
        return analysis


# ═══════════════════════════════════════════════════════════════════
# Unit Tests (as docstring examples)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from engine.selection.pool import PoolManager
    
    print("=" * 60)
    print("Rotation Engine Test")
    print("=" * 60)
    
    pm = PoolManager()
    
    # Seed pools with test data
    pm.add_stock("AAPL", pool="core", entry_price=190, composite_score=0.85,
                 quality_score=0.90, metadata={"sector": "Technology"})
    pm.add_stock("MSFT", pool="core", entry_price=410, composite_score=0.82,
                 quality_score=0.88, metadata={"sector": "Technology"})
    pm.add_stock("GOOGL", pool="core", entry_price=185, composite_score=0.78,
                 quality_score=0.84, metadata={"sector": "Technology"})
    pm.add_stock("JNJ", pool="core", entry_price=155, composite_score=0.70,
                 quality_score=0.72, metadata={"sector": "Healthcare"})
    pm.add_stock("WMT", pool="core", entry_price=82, composite_score=0.65,
                 quality_score=0.68, metadata={"sector": "Consumer Defensive"})
    
    pm.add_stock("NVDA", pool="watchlist", composite_score=0.92,
                 metadata={"sector": "Technology"})
    pm.add_stock("LLY", pool="watchlist", composite_score=0.88,
                 metadata={"sector": "Healthcare"})
    
    re = RotationEngine(pm)
    
    # Score-based rotation
    print("\n🔍 Score-Based Rotation:")
    result = re.rotate_score_based(pool="core", candidate_pool="watchlist", dry_run=True)
    print(result.summarize())
    print(f"\nImpact Analysis:")
    impact = re.analyze_rotation_impact("core", result.decisions)
    for k, v in impact.items():
        print(f"  {k}: {v}")
    
    # Sector balance
    print("\n🔍 Sector Balance:")
    result = re.rotate_sector_balance(pool="core", dry_run=True)
    print(result.summarize())
    
    # Should rotate?
    should, reason = re.should_rotate("core")
    print(f"\n🔍 Should rotate core? {should} — {reason}")
    
    # Volatility regime
    regime = re._get_volatility_regime()
    print(f"\n📊 Current vol regime: {regime}")
