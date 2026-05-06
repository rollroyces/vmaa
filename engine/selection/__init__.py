#!/usr/bin/env python3
"""
VMAA Smart Stock Selection Engine
===================================
Orchestration layer that integrates all selection engine components:

  Factors   → Multi-factor extraction + scoring
  Conditions → Boolean screening with AND/OR/NOT
  Pool      → Dynamic pool management
  Rotation  → Auto rotation strategies

The `SmartScreener` class provides the main interface:
  - screen(universe, conditions)    → ranked candidates
  - update_pool(candidates)         → update dynamic pool
  - rotate()                        → execute rotation if triggered
  - get_pool_status()               → pool stats

Usage:
  from engine.selection import SmartScreener
  screener = SmartScreener()
  
  # Screen a universe
  candidates = screener.screen(["AAPL", "MSFT", "GOOGL"])
  for t, score in candidates[:5]:
      print(f"{t}: {score:.3f}")
  
  # Update pools with screened candidates
  screener.update_pool(candidates)
  
  # Execute rotation
  result = screener.rotate(pool="core")
  
  # Get status
  status = screener.get_pool_status()
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure VMAA root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from engine.selection.factors import (
    FactorEngine, FactorScores, FactorProfile, extract_factors,
)
from engine.selection.conditions import (
    ConditionNode, Condition, And, Or, Not,
    ConditionTemplates, ConditionEvaluator,
)
from engine.selection.pool import PoolManager, PoolStatus, PoolEntry
from engine.selection.rotation import RotationEngine, RotationResult
from engine.selection.config import SelectionConfig, get_selection_config

logger = logging.getLogger("vmaa.engine.selection")


# ═══════════════════════════════════════════════════════════════════
# Screening Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ScreenResult:
    """Complete screening result."""
    timestamp: str
    universe_size: int
    screened_count: int                     # Successfully extracted factors
    passed_count: int                       # Passed all conditions
    failed_count: int
    
    # Ranked results
    ranked_candidates: List[Tuple[str, float, Dict[str, float]]] = field(default_factory=list)
    # (ticker, composite_score, layer_scores)
    
    # Condition results
    condition_pass: Dict[str, bool] = field(default_factory=dict)
    condition_fail_reasons: Dict[str, List[str]] = field(default_factory=dict)
    
    # Factor scores
    factor_scores: Optional[FactorScores] = None
    
    # Performance
    elapsed_seconds: float = 0.0
    
    def get_top(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return top N candidates by composite score."""
        return [(t, s) for t, s, _ in self.ranked_candidates[:n]]
    
    def summarize(self) -> str:
        lines = [
            f"Screening Result @ {self.timestamp}",
            f"  Universe: {self.universe_size} → Extracted: {self.screened_count} → Passed: {self.passed_count} ({self.passed_count/max(self.universe_size,1)*100:.1f}%)",
            f"  Elapsed: {self.elapsed_seconds:.1f}s",
        ]
        if self.ranked_candidates:
            lines.append(f"  Top Candidates:")
            for i, (t, s, layers) in enumerate(self.ranked_candidates[:10]):
                lines.append(f"    {i+1:2d}. {t:6s}  {s:.4f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# SmartScreener — Main Orchestrator
# ═══════════════════════════════════════════════════════════════════

class SmartScreener:
    """
    Smart Stock Selection Engine — main orchestrator.
    
    Integrates factor screening, condition evaluation, pool management,
    and auto rotation into a single facade.
    
    Example:
        >>> screener = SmartScreener()
        >>> 
        >>> # Screen with default quality-value condition
        >>> result = screener.screen(
        ...     universe=["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
        ...     conditions=ConditionTemplates.quality_value(),
        ... )
        >>> print(result.summarize())
        >>> 
        >>> # Update pools
        >>> screener.update_pool(result.ranked_candidates)
        >>> 
        >>> # Check pool status
        >>> status = screener.get_pool_status()
        >>> for pool, s in status.items():
        ...     print(f"{pool}: {s.size} stocks")
    """

    def __init__(self, 
                 config: Optional[SelectionConfig] = None,
                 data_dir: Optional[Path] = None):
        """
        Initialize the Smart Screener.
        
        Args:
            config: SelectionConfig (uses singleton if None)
            data_dir: Directory for pool state persistence
        """
        self.config = config or get_selection_config()
        
        # Initialize components
        self.factor_engine = FactorEngine(config=self.config)
        self.condition_evaluator = ConditionEvaluator()
        self.pool_manager = PoolManager(data_dir=data_dir)
        self.rotation_engine = RotationEngine(
            pool_manager=self.pool_manager,
            factor_engine=self.factor_engine,
        )
        
        # Screening history
        self._screen_history: List[Dict[str, Any]] = []
        
        logger.info(
            "SmartScreener initialized — "
            f"{sum(len(p) for p in self.pool_manager._pools.values())} stocks in pools"
        )

    # ── Screening ────────────────────────────────────────────────

    def screen(self,
               universe: List[str],
               conditions: Optional[ConditionNode] = None,
               min_composite_score: float = 0.0,
               max_candidates: int = 100,
               progress: bool = False,
               ) -> ScreenResult:
        """
        Screen a universe of stocks through factor extraction + condition filtering.
        
        Args:
            universe: List of ticker symbols
            conditions: Optional condition tree for filtering
                        (uses quality_value template if None)
            min_composite_score: Minimum composite score to include
            max_candidates: Maximum candidates to return
            progress: Show progress output
        
        Returns:
            ScreenResult with ranked candidates and details
        
        Example:
            >>> screener = SmartScreener()
            >>> # Screen with deep value template
            >>> from engine.selection.conditions import ConditionTemplates
            >>> result = screener.screen(
            ...     ["AAPL", "MSFT", "GOOGL"],
            ...     conditions=ConditionTemplates.deep_value()
            ... )
        """
        start_time = datetime.now(timezone.utc)
        universe_size = len(universe)
        
        logger.info(f"Starting screening: {universe_size} stocks")
        
        # Default condition: quality value
        if conditions is None:
            conditions = ConditionTemplates.quality_value()
            logger.info(f"Using default condition: {conditions.explain()}")
        
        # Step 1: Extract factors and score
        logger.info("Extracting factors...")
        profiles = self.factor_engine.extract_universe(
            universe,
            progress_callback=(
                (lambda cur, total, t: print(f"  [{cur}/{total}] {t}")) 
                if progress else None
            ),
        )
        
        # Step 2: Score universe
        logger.info("Computing factor scores...")
        factor_scores = self.factor_engine.score_universe(profiles=profiles)
        
        # Step 3: Apply composite score filter
        candidates = [
            (t, s, factor_scores.get_layer_breakdown(t))
            for t, s in factor_scores.composite_scores.items()
            if s >= min_composite_score
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Step 4: Apply condition filter
        passed: Dict[str, bool] = {}
        fail_reasons: Dict[str, List[str]] = {}
        
        if conditions:
            # Convert profiles to dicts for condition evaluation
            stock_dicts = [
                {
                    "ticker": t,
                    **profiles[t].to_dict(),
                    "sector": profiles[t].sector,
                }
                for t in profiles
            ]
            
            passing_ids = set()
            for stock in stock_dicts:
                ticker = stock["ticker"]
                try:
                    if conditions.evaluate(stock):
                        passed[ticker] = True
                        passing_ids.add(ticker)
                    else:
                        passed[ticker] = False
                        fail_reasons[ticker] = self.condition_evaluator._diagnose(stock, conditions)
                except Exception as e:
                    passed[ticker] = False
                    fail_reasons[ticker] = [f"Error: {e}"]
            
            # Filter to only condition-passing candidates
            candidates = [
                (t, s, layers)
                for t, s, layers in candidates
                if t in passing_ids
            ]
        
        # Step 5: Limit results
        candidates = candidates[:max_candidates]
        
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        result = ScreenResult(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            universe_size=universe_size,
            screened_count=len(profiles),
            passed_count=len(candidates),
            failed_count=len(profiles) - len(candidates),
            ranked_candidates=candidates,
            condition_pass=passed,
            condition_fail_reasons=fail_reasons,
            factor_scores=factor_scores,
            elapsed_seconds=round(elapsed, 1),
        )
        
        # Log to history
        self._screen_history.append({
            "timestamp": result.timestamp,
            "universe_size": universe_size,
            "passed": len(candidates),
            "condition": conditions.label if hasattr(conditions, 'label') else str(conditions)[:80],
            "elapsed": elapsed,
        })
        
        logger.info(
            f"Screening complete: {len(candidates)}/{universe_size} passed "
            f"in {elapsed:.1f}s"
        )
        
        return result

    def screen_with_template(self,
                             universe: List[str],
                             template: str = "quality_value",
                             **template_kwargs,
                             ) -> ScreenResult:
        """
        Screen using a named condition template.
        
        Args:
            universe: List of tickers
            template: Template name:
                "deep_value", "growth_at_reasonable_price", "high_momentum",
                "quality_value", "turnaround", "deep_value_or_growth",
                "dividend_quality"
            **template_kwargs: Override template parameters
        
        Example:
            >>> result = screener.screen_with_template(
            ...     ["AAPL", "MSFT"],
            ...     template="deep_value",
            ...     max_market_cap=1e9,
            ... )
        """
        templates = {
            "deep_value": ConditionTemplates.deep_value,
            "growth_at_reasonable_price": ConditionTemplates.growth_at_reasonable_price,
            "high_momentum": ConditionTemplates.high_momentum,
            "quality_value": ConditionTemplates.quality_value,
            "turnaround": ConditionTemplates.turnaround,
            "deep_value_or_growth": ConditionTemplates.deep_value_or_growth,
            "dividend_quality": ConditionTemplates.dividend_quality,
        }
        
        if template not in templates:
            raise ValueError(
                f"Unknown template: {template}. Available: {list(templates.keys())}"
            )
        
        conditions = templates[template](**template_kwargs)
        return self.screen(universe, conditions=conditions)

    # ── Pool Management ──────────────────────────────────────────

    def update_pool(self,
                    candidates: List[Tuple[str, float, Dict[str, float]]],
                    pool: str = "watchlist",
                    clear_existing: bool = False,
                    ) -> int:
        """
        Update a pool with screened candidates.
        
        Args:
            candidates: List of (ticker, composite_score, layer_scores)
            pool: Target pool
            clear_existing: If True, clear pool before adding
        
        Returns:
            Number of stocks added
        
        Example:
            >>> result = screener.screen(universe)
            >>> added = screener.update_pool(result.ranked_candidates, pool="watchlist")
            >>> print(f"Added {added} stocks to watchlist")
        """
        if clear_existing:
            self.pool_manager.clear_pool(pool)
        
        added = 0
        for ticker, comp_score, layer_scores in candidates:
            metadata = {
                "sector": "",  # Will be filled from profile
                "composite_score": comp_score,
            }
            
            # Add layer scores
            for layer, score in layer_scores.items():
                metadata[f"layer_{layer}"] = score
            
            success, msg = self.pool_manager.add_stock(
                ticker=ticker,
                pool=pool,
                entry_reason="screen_result",
                composite_score=comp_score,
                quality_score=layer_scores.get("quality", 0),
                momentum_score=layer_scores.get("momentum", 0),
                metadata=metadata,
            )
            
            if success:
                added += 1
        
        logger.info(f"Updated {pool} pool: {added} added")
        return added

    def promote_to_pool(self,
                        tickers: List[str],
                        target_pool: str = "core",
                        ) -> int:
        """
        Promote stocks from watchlist to a higher pool.
        
        Args:
            tickers: Tickers to promote
            target_pool: Target pool ("core" or "tactical")
        
        Returns:
            Number successfully promoted
        """
        promoted = 0
        for ticker in tickers:
            success, _ = self.pool_manager.move_stock(
                ticker, target_pool, reason="manual_promotion"
            )
            if success:
                promoted += 1
        return promoted

    def demote_stock(self, ticker: str, reason: str = "") -> str:
        """Demote a stock from core/tactical to watchlist."""
        entry = self.pool_manager.get_stock_entry(ticker)
        if not entry:
            return f"{ticker} not found in any pool"
        
        success, msg = self.pool_manager.move_stock(ticker, "watchlist", reason or "manual_demotion")
        return msg

    # ── Rotation ─────────────────────────────────────────────────

    def rotate(self,
               pool: str = "core",
               strategies: Optional[List[str]] = None,
               dry_run: bool = False,
               ) -> List[RotationResult]:
        """
        Execute rotation for a pool.
        
        Args:
            pool: Pool to rotate
            strategies: Specific strategies (default: all enabled)
            dry_run: If True, only compute decisions without executing
        
        Returns:
            List of RotationResult
        
        Example:
            >>> results = screener.rotate(pool="core", dry_run=True)
            >>> for r in results:
            ...     print(r.summarize())
        """
        return self.rotation_engine.execute_rotation(
            pool=pool,
            strategies=strategies,
            dry_run=dry_run,
        )

    def auto_rotate(self, dry_run: bool = False) -> Dict[str, List[RotationResult]]:
        """
        Check all pools for rotation triggers and execute if needed.
        
        Returns:
            {pool_name: [RotationResult, ...]}
        """
        results = {}
        
        for pool_name in self.pool_manager.POOL_PRIORITY:
            if pool_name == "remove":
                continue
            
            should, reason = self.rotation_engine.should_rotate(pool_name)
            if should:
                logger.info(f"Auto-rotating {pool_name}: {reason}")
                pool_results = self.rotate(pool=pool_name, dry_run=dry_run)
                results[pool_name] = pool_results
        
        return results

    # ── Status and Monitoring ────────────────────────────────────

    def get_pool_status(self) -> Dict[str, PoolStatus]:
        """Get status for all pools."""
        return self.pool_manager.get_all_statuses()

    def get_pool_summary(self) -> str:
        """Get human-readable pool summary."""
        return self.pool_manager.summarize()

    def get_rotation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent rotation log."""
        return self.pool_manager.get_rotation_log(limit)

    def get_screening_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent screening history."""
        return self._screen_history[-limit:]

    def refresh_prices(self) -> int:
        """
        Refresh current prices for all stocks in pools.
        
        Returns:
            Number of stocks updated
        """
        updated = 0
        for pool_name in self.pool_manager.POOL_PRIORITY:
            for ticker, entry in list(self.pool_manager._pools[pool_name].items()):
                try:
                    from data.hybrid import get_price
                    price, _, _, _ = get_price(ticker)
                    if price > 0:
                        entry.current_price = price
                        if entry.entry_price > 0:
                            entry.current_return = (price - entry.entry_price) / entry.entry_price
                        entry.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                        updated += 1
                except Exception:
                    pass
        
        if updated > 0:
            self.pool_manager._save_state()
            logger.info(f"Refreshed prices for {updated} stocks")
        
        return updated

    # ── Combined Workflow ────────────────────────────────────────

    def full_cycle(self,
                   universe: List[str],
                   conditions: Optional[ConditionNode] = None,
                   target_pool: str = "watchlist",
                   dry_run: bool = True,
                   ) -> Dict[str, Any]:
        """
        Run a complete screening + pool update + rotation cycle.
        
        Args:
            universe: Stock universe to screen
            conditions: Screening conditions
            target_pool: Pool to add candidates to
            dry_run: If True, rotations are simulated
        
        Returns:
            Dict with full cycle results
        
        Example:
            >>> screener = SmartScreener()
            >>> result = screener.full_cycle(
            ...     universe=["AAPL", "MSFT", "GOOGL", "NVDA"],
            ...     conditions=ConditionTemplates.quality_value(),
            ... )
        """
        logger.info("=" * 60)
        logger.info(f"SMART SCREENER FULL CYCLE — {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info("=" * 60)
        
        # 1. Screen
        screen_result = self.screen(universe, conditions=conditions)
        logger.info(f"Step 1 — Screening: {screen_result.passed_count} passed")
        
        # 2. Update pool
        added = self.update_pool(
            screen_result.ranked_candidates, pool=target_pool
        )
        logger.info(f"Step 2 — Pool update: {added} added to {target_pool}")
        
        # 3. Rotate
        rotation_results = self.auto_rotate(dry_run=dry_run)
        total_rotations = sum(len(r) for r in rotation_results.values())
        logger.info(f"Step 3 — Rotation: {total_rotations} strategy results")
        
        # 4. Summary
        pool_status = self.get_pool_status()
        status_summary = {
            name: {
                "size": s.size,
                "max": s.max_size,
                "avg_score": s.avg_composite_score,
                "full": s.is_full,
            }
            for name, s in pool_status.items()
        }
        
        return {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode": "DRY_RUN" if dry_run else "LIVE",
            "screening": {
                "universe_size": screen_result.universe_size,
                "screened": screen_result.screened_count,
                "passed": screen_result.passed_count,
                "top_candidates": screen_result.get_top(5),
                "elapsed": screen_result.elapsed_seconds,
            },
            "pool_update": {
                "added": added,
                "target_pool": target_pool,
            },
            "rotation": {
                "pools_rotated": list(rotation_results.keys()),
                "total_decisions": total_rotations,
            },
            "pool_status": status_summary,
        }

    # ── Utility ──────────────────────────────────────────────────

    def reload_config(self):
        """Hot-reload configuration."""
        self.config.reload()
        self.factor_engine = FactorEngine(config=self.config)
        logger.info("SmartScreener config reloaded")

    def validate_setup(self) -> List[str]:
        """Validate the entire setup and return issues."""
        issues = []
        
        # Config issues
        issues.extend(self.config.validate())
        
        # Pool issues
        for name, status in self.get_pool_status().items():
            if status.size > status.max_size:
                issues.append(f"Pool '{name}' over capacity: {status.size}/{status.max_size}")
        
        # Check data directory
        data_dir = self.pool_manager._data_dir
        if not data_dir.exists():
            issues.append(f"Data directory missing: {data_dir}")
        
        return issues


# ═══════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════

def quick_screen(tickers: List[str],
                 template: str = "quality_value",
                 top_n: int = 10,
                 **template_kwargs,
                 ) -> ScreenResult:
    """
    Quick one-liner screening with template.
    
    Example:
        >>> result = quick_screen(["AAPL", "MSFT", "GOOGL"], template="deep_value")
        >>> for t, s in result.get_top(5):
        ...     print(f"{t}: {s:.3f}")
    """
    screener = SmartScreener()
    return screener.screen_with_template(
        tickers, template=template, **template_kwargs
    )


def get_smart_screener() -> SmartScreener:
    """Get or create a singleton SmartScreener instance."""
    global _SMART_SCREENER
    if _SMART_SCREENER is None:
        _SMART_SCREENER = SmartScreener()
    return _SMART_SCREENER


_SMART_SCREENER: Optional[SmartScreener] = None


# ═══════════════════════════════════════════════════════════════════
# Unit Tests (as docstring examples)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Smart Stock Selection Engine — Integration Test")
    print("=" * 60)
    
    screener = SmartScreener()
    
    # Validate setup
    issues = screener.validate_setup()
    if issues:
        print("⚠️ Setup issues:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("✅ Setup validated")
    
    # Quick screen
    print("\n🔍 Quick Screen (Deep Value):")
    result = screener.screen_with_template(
        universe=["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA"],
        template="deep_value",
    )
    print(result.summarize())
    
    # Full cycle (dry run)
    print("\n🔄 Full Cycle (dry run):")
    cycle_result = screener.full_cycle(
        universe=["AAPL", "MSFT", "GOOGL", "NVDA", "META", "JNJ", "WMT"],
        conditions=ConditionTemplates.quality_value(),
        dry_run=True,
    )
    
    print(f"\nCycle Summary:")
    print(f"  Screening: {cycle_result['screening']['passed']}/{cycle_result['screening']['universe_size']} passed")
    print(f"  Pool Update: {cycle_result['pool_update']['added']} added")
    print(f"  Rotation: {cycle_result['rotation']}")
    print(f"  Pool Status:")
    for pool, s in cycle_result['pool_status'].items():
        print(f"    {pool}: {s['size']}/{s['max']} stocks, avg score {s['avg_score']:.3f}")
    
    # Pool summary
    print(f"\n{screener.get_pool_summary()}")
