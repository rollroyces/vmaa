#!/usr/bin/env python3
"""
VMAA Smart Selection Engine — Dynamic Stock Pool
==================================================
Manages four pool types with automatic entry/exit rules:
  - Core Pool:     Long-term holdings with strong fundamentals
  - Tactical Pool: Short-term opportunities with momentum triggers
  - Watchlist:     Stocks to monitor for potential entry
  - Remove Pool:   Blacklisted stocks (explicitly excluded)

Features:
  - Pool entry/exit conditions with factor thresholds
  - Auto rotation rules (time-based, event-based, performance-based)
  - Pool metrics: size, sector distribution, average scores
  - Pool history tracking for performance analysis
  - State persistence to JSON

Usage:
  from engine.selection.pool import PoolManager
  pm = PoolManager()
  pm.add_stock("AAPL", pool="core", metadata={"quality_score": 0.85})
  pm.get_pool_status("core")
  pm.rotate_pools()  # Execute rotation if triggered
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from engine.selection.config import PoolConfig, get_selection_config
from engine.selection.conditions import ConditionNode, And, Or, Not, Condition
from engine.selection.factors import FactorProfile, extract_factors

logger = logging.getLogger("vmaa.engine.selection.pool")


# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PoolEntry:
    """A single stock entry in a pool."""
    ticker: str
    pool_name: str                     # "core", "tactical", "watchlist", "remove"
    added_date: str = ""
    last_updated: str = ""
    entry_price: float = 0.0
    entry_reason: str = ""
    current_price: float = 0.0
    
    # Scores at entry time
    composite_score: float = 0.0
    quality_score: float = 0.0
    momentum_score: float = 0.0
    
    # Tracking
    days_in_pool: int = 0
    max_drawdown: float = 0.0
    current_return: float = 0.0
    num_rotations: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Source: factor profile (cached)
    _profile: Optional[FactorProfile] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "pool_name": self.pool_name,
            "added_date": self.added_date,
            "last_updated": self.last_updated,
            "entry_price": self.entry_price,
            "entry_reason": self.entry_reason,
            "current_price": self.current_price,
            "composite_score": self.composite_score,
            "quality_score": self.quality_score,
            "momentum_score": self.momentum_score,
            "days_in_pool": self.days_in_pool,
            "max_drawdown": self.max_drawdown,
            "current_return": self.current_return,
            "num_rotations": self.num_rotations,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoolEntry':
        return cls(**{k: data.get(k, v.default if v.default is not field(default) else v.default_factory() if v.default_factory is not field(default) else None)
                      for k, v in cls.__dataclass_fields__.items()  # type: ignore
                      if k != '_profile'})


@dataclass
class PoolState:
    """Serializable state for all pools."""
    version: str = "1.0.0"
    timestamp: str = ""
    pools: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    rotation_log: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolStatus:
    """Current status/metrics for a pool."""
    name: str
    label: str
    size: int
    max_size: int
    min_size: int
    sector_limit: int
    
    # Composition
    tickers: List[str]
    
    # Scores
    avg_composite_score: float = 0.0
    avg_quality_score: float = 0.0
    avg_momentum_score: float = 0.0
    score_std: float = 0.0
    
    # Sector distribution
    sector_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Performance
    avg_return: float = 0.0
    avg_drawdown: float = 0.0
    avg_days_in_pool: float = 0.0
    
    # Capacity
    is_full: bool = False
    needs_rebalance: bool = False
    available_slots: int = 0


# ═══════════════════════════════════════════════════════════════════
# Pool Manager
# ═══════════════════════════════════════════════════════════════════

class PoolManager:
    """
    Dynamic stock pool manager.
    
    Manages four pools (core, tactical, watchlist, remove) with:
    - Entry conditions based on factor thresholds
    - Exit conditions for underperforming stocks
    - Auto-rebalance rules
    - Pool history and metrics
    
    State is persisted to JSON for recovery across sessions.
    
    Example:
        >>> pm = PoolManager()
        >>> pm.add_stock("AAPL", pool="core", metadata={"quality_score": 0.85})
        >>> status = pm.get_pool_status("core")
        >>> print(f"Core pool: {status.size}/{status.max_size} stocks")
    """
    
    POOL_PRIORITY = ["core", "tactical", "watchlist", "remove"]
    
    def __init__(self, data_dir: Optional[Path] = None):
        self._data_dir = Path(data_dir) if data_dir else (
            Path(__file__).resolve().parent.parent.parent.parent / "engine" / "data"
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = get_selection_config()
        self.pool_configs = self.config.get_all_pool_configs()
        
        # In-memory pool storage: pool_name → {ticker → PoolEntry}
        self._pools: Dict[str, Dict[str, PoolEntry]] = {
            name: {} for name in self.POOL_PRIORITY
        }
        
        # History and logs
        self._history: List[Dict[str, Any]] = []
        self._rotation_log: List[Dict[str, Any]] = []
        self._stats: Dict[str, Any] = {}
        
        # Rotation state
        self._last_rotation: Dict[str, str] = {}
        self._event_cooldown_until: Optional[float] = None
        
        # Load saved state
        self._load_state()
        
        logger.info(
            f"PoolManager initialized: "
            f"{sum(len(p) for p in self._pools.values())} total stocks across {len(self._pools)} pools"
        )

    # ── Stock Management ─────────────────────────────────────────

    def add_stock(self, 
                  ticker: str,
                  pool: str = "watchlist",
                  entry_price: float = 0.0,
                  entry_reason: str = "",
                  composite_score: float = 0.0,
                  quality_score: float = 0.0,
                  momentum_score: float = 0.0,
                  metadata: Optional[Dict[str, Any]] = None,
                  force: bool = False,
                  ) -> Tuple[bool, str]:
        """
        Add a stock to a pool.
        
        Args:
            ticker: Stock symbol
            pool: Target pool ("core", "tactical", "watchlist", "remove")
            entry_price: Price at entry
            entry_reason: Why this stock was added
            composite_score: Overall factor score
            quality_score: Quality factor score
            momentum_score: Momentum factor score
            metadata: Additional metadata
            force: Skip validation checks
        
        Returns:
            (success, message)
        """
        ticker = ticker.upper()
        pool = pool.lower()
        
        # Validate pool
        if pool not in self._pools:
            return False, f"Unknown pool: {pool}"
        
        # Check remove pool — block unless explicit
        if pool == "remove" and not force:
            return False, "Use force=True to add to remove pool"
        
        # Check if already in another pool
        current_pool = self._find_stock_pool(ticker)
        if current_pool and current_pool != pool:
            if not force:
                return False, f"{ticker} already in '{current_pool}' pool. Remove first or use force=True."
            self.remove_stock(ticker, reason=f"Moving to {pool}")
        
        # Check pool capacity
        pool_cfg = self.pool_configs.get(pool)
        if pool_cfg and len(self._pools[pool]) >= pool_cfg.max_size and not force:
            return False, f"Pool '{pool}' is full ({pool_cfg.max_size} max)"
        
        # Check sector limits
        if pool_cfg and pool_cfg.sector_limit > 0:
            sector = self._get_sector(ticker)
            sector_count = sum(
                1 for e in self._pools[pool].values()
                if self._get_sector(e.ticker) == sector
            )
            if sector_count >= pool_cfg.sector_limit and not force:
                return False, f"Sector limit reached for '{sector}' in '{pool}'"
        
        # Check entry conditions
        if not force and pool_cfg:
            passes, reasons = self._check_entry_conditions(ticker, pool_cfg)
            if not passes:
                return False, f"Entry conditions not met: {'; '.join(reasons)}"
        
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        
        entry = PoolEntry(
            ticker=ticker,
            pool_name=pool,
            added_date=now,
            last_updated=now,
            entry_price=entry_price,
            entry_reason=entry_reason or "manual_add",
            current_price=entry_price,
            composite_score=composite_score,
            quality_score=quality_score,
            momentum_score=momentum_score,
            days_in_pool=0,
            max_drawdown=0.0,
            current_return=0.0,
            tags=[],
            metadata=metadata or {},
        )
        
        self._pools[pool][ticker] = entry
        self._save_state()
        
        logger.info(f"✅ {ticker} → {pool} pool ({entry_reason})")
        return True, f"Added {ticker} to {pool} pool"

    def remove_stock(self, ticker: str, reason: str = "") -> bool:
        """
        Remove a stock from its current pool.
        
        Returns True if found and removed.
        """
        ticker = ticker.upper()
        for pool_name in self.POOL_PRIORITY:
            if ticker in self._pools[pool_name]:
                entry = self._pools[pool_name].pop(ticker)
                # Log to history
                self._log_history({
                    "action": "remove",
                    "ticker": ticker,
                    "from_pool": pool_name,
                    "reason": reason,
                    "days_in_pool": entry.days_in_pool,
                    "final_return": entry.current_return,
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                })
                self._save_state()
                logger.info(f"❌ {ticker} removed from {pool_name}: {reason}")
                return True
        return False

    def move_stock(self, ticker: str, target_pool: str, reason: str = "") -> Tuple[bool, str]:
        """
        Move a stock between pools.
        
        Removes from current pool and adds to target.
        """
        ticker = ticker.upper()
        target_pool = target_pool.lower()
        
        if target_pool not in self._pools:
            return False, f"Unknown target pool: {target_pool}"
        
        current = self._find_stock_pool(ticker)
        if not current:
            return False, f"{ticker} not found in any pool"
        
        if current == target_pool:
            return False, f"{ticker} already in '{target_pool}'"
        
        # Get existing entry data
        entry = self._pools[current].pop(ticker)
        
        # Update for new pool
        entry.pool_name = target_pool
        entry.entry_reason = reason or f"moved_from_{current}"
        entry.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        entry.num_rotations += 1
        
        self._pools[target_pool][ticker] = entry
        
        self._log_history({
            "action": "move",
            "ticker": ticker,
            "from_pool": current,
            "to_pool": target_pool,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        })
        
        self._save_state()
        logger.info(f"🔄 {ticker}: {current} → {target_pool} ({reason})")
        return True, f"Moved {ticker} from {current} to {target_pool}"

    def get_stock_entry(self, ticker: str) -> Optional[PoolEntry]:
        """Get a stock's pool entry if it exists."""
        ticker = ticker.upper()
        for pool in self._pools.values():
            if ticker in pool:
                return pool[ticker]
        return None

    def is_in_pool(self, ticker: str, pool: Optional[str] = None) -> bool:
        """Check if a ticker is in any pool (or a specific pool)."""
        ticker = ticker.upper()
        if pool:
            return pool.lower() in self._pools and ticker in self._pools[pool.lower()]
        return self._find_stock_pool(ticker) is not None

    def _find_stock_pool(self, ticker: str) -> Optional[str]:
        """Find which pool a ticker belongs to."""
        for pool_name in self.POOL_PRIORITY:
            if ticker in self._pools[pool_name]:
                return pool_name
        return None

    def _get_sector(self, ticker: str) -> str:
        """Get sector for a ticker (from cached profile or fallback)."""
        entry = self.get_stock_entry(ticker)
        if entry and entry.metadata.get("sector"):
            return entry.metadata["sector"]
        try:
            profile = extract_factors(ticker)
            if profile.extraction_ok and profile.sector:
                return profile.sector
        except Exception:
            pass
        return "Unknown"

    # ── Pool Status and Metrics ──────────────────────────────────

    def get_pool_status(self, pool: str) -> PoolStatus:
        """Get comprehensive status for a pool."""
        pool = pool.lower()
        cfg = self.pool_configs.get(pool)
        
        if pool not in self._pools:
            return PoolStatus(name=pool, label="Unknown", size=0, max_size=0, min_size=0, 
                              sector_limit=0, tickers=[])
        
        entries = list(self._pools[pool].values())
        tickers = [e.ticker for e in entries]
        
        # Scores
        comp_scores = [e.composite_score for e in entries if e.composite_score > 0]
        qual_scores = [e.quality_score for e in entries if e.quality_score > 0]
        mom_scores = [e.momentum_score for e in entries if e.momentum_score > 0]
        
        # Sector distribution
        sectors: Dict[str, int] = defaultdict(int)
        for e in entries:
            sector = e.metadata.get("sector", "Unknown")
            sectors[sector] += 1
        
        # Performance
        returns = [e.current_return for e in entries if e.days_in_pool > 0]
        drawdowns = [e.max_drawdown for e in entries]
        days = [e.days_in_pool for e in entries]
        
        available = cfg.max_size - len(entries) if cfg else 0
        
        return PoolStatus(
            name=pool,
            label=cfg.label if cfg else pool.title(),
            size=len(entries),
            max_size=cfg.max_size if cfg else 30,
            min_size=cfg.min_size if cfg else 0,
            sector_limit=cfg.sector_limit if cfg else 0,
            tickers=tickers,
            avg_composite_score=np.mean(comp_scores) if comp_scores else 0.0,
            avg_quality_score=np.mean(qual_scores) if qual_scores else 0.0,
            avg_momentum_score=np.mean(mom_scores) if mom_scores else 0.0,
            score_std=np.std(comp_scores) if len(comp_scores) > 1 else 0.0,
            sector_distribution=dict(sectors),
            avg_return=np.mean(returns) if returns else 0.0,
            avg_drawdown=np.mean(drawdowns) if drawdowns else 0.0,
            avg_days_in_pool=np.mean(days) if days else 0.0,
            is_full=available <= 0,
            needs_rebalance=len(entries) < (cfg.min_size if cfg else 10),
            available_slots=max(0, available),
        )

    def get_all_statuses(self) -> Dict[str, PoolStatus]:
        """Get status for all pools."""
        return {name: self.get_pool_status(name) for name in self.POOL_PRIORITY}

    def get_all_tickers(self) -> Dict[str, List[str]]:
        """Get all tickers grouped by pool."""
        return {
            name: list(entries.keys())
            for name, entries in self._pools.items()
        }

    def get_pool_tickers(self, pool: str) -> List[str]:
        """Get tickers in a specific pool."""
        pool = pool.lower()
        if pool in self._pools:
            return list(self._pools[pool].keys())
        return []

    # ── Entry/Exit Condition Checking ────────────────────────────

    def _check_entry_conditions(self, ticker: str, 
                                 pool_cfg: PoolConfig) -> Tuple[bool, List[str]]:
        """Check if a ticker meets pool entry conditions."""
        reasons = []
        conditions = pool_cfg.entry_conditions
        
        if not conditions:
            return True, []
        
        try:
            profile = extract_factors(ticker)
            if not profile.extraction_ok:
                return False, ["Data unavailable"]
            
            stock_dict = profile.to_dict()
            stock_dict["ticker"] = ticker
            stock_dict["sector"] = profile.sector
            
            # Min composite score
            min_score = conditions.get("min_composite_score")
            if min_score is not None:
                # Need to compute composite score from factors
                from engine.selection.factors import FactorEngine
                fe = FactorEngine()
                scores = fe.score_universe(profiles={ticker: profile})
                composite = scores.composite_scores.get(ticker, 0)
                if composite < min_score:
                    reasons.append(f"Composite score {composite:.3f} < {min_score}")
            
            # Min quality score
            min_quality = conditions.get("min_quality_score")
            if min_quality is not None:
                quality = stock_dict.get("roe", 0) * 0.5 + stock_dict.get("profit_margin", 0) * 0.3 + stock_dict.get("fcf_yield", 0) * 0.2
                quality = max(0, min(1, quality))
                if quality < min_quality:
                    reasons.append(f"Quality score {quality:.3f} < {min_quality}")
            
            # Min momentum score
            min_momentum = conditions.get("min_momentum_score")
            if min_momentum is not None:
                momentum = stock_dict.get("return_3m", 0)
                momentum = max(0, min(1, (momentum + 0.5)))  # Scale to 0-1
                if momentum < min_momentum:
                    reasons.append(f"Momentum score {momentum:.3f} < {min_momentum}")
            
            # Profitable
            if conditions.get("require_profitable"):
                if profile.roe <= 0 and profile.roa <= 0:
                    reasons.append("Not profitable")
            
            # Market cap minimum
            min_cap_b = conditions.get("min_market_cap_b")
            if min_cap_b is not None:
                cap_b = profile.market_cap / 1e9 if profile.market_cap > 0 else 0
                if cap_b < min_cap_b:
                    reasons.append(f"Market cap ${cap_b:.1f}B < ${min_cap_b:.1f}B")
            
            # Volume minimum
            min_vol = conditions.get("min_volume_1m")
            if min_vol is not None:
                # Volume check (simplified)
                pass
            
            # Near 52w low
            near_52w = conditions.get("near_52w_low_pct")
            if near_52w is not None:
                try:
                    import yfinance as yf
                    t = yf.Ticker(ticker)
                    info = t.info
                    low = info.get("fiftyTwoWeekLow", 0)
                    high = info.get("fiftyTwoWeekHigh", 0)
                    if low > 0 and profile.price > 0:
                        pct_above_low = (profile.price - low) / low
                        if pct_above_low > near_52w / 100:
                            reasons.append(f"Price {pct_above_low:.1%} above 52w low > {near_52w}%")
                except Exception:
                    pass
            
            return len(reasons) == 0, reasons
            
        except Exception as e:
            return False, [f"Error checking conditions: {e}"]

    def check_exit_conditions(self, ticker: str, pool: str) -> Tuple[bool, List[str]]:
        """Check if a stock should be removed from its pool."""
        pool = pool.lower()
        cfg = self.pool_configs.get(pool)
        if not cfg:
            return False, []
        
        reasons = []
        conditions = cfg.exit_conditions
        entry = self.get_stock_entry(ticker)
        
        if not entry:
            return False, [f"{ticker} not in {pool}"]
        
        # Drawdown
        max_dd_pct = conditions.get("max_drawdown_pct")
        if max_dd_pct is not None:
            dd = abs(entry.max_drawdown) * 100 if entry.max_drawdown < 0 else 0
            if dd > max_dd_pct:
                reasons.append(f"Drawdown {dd:.1f}% > {max_dd_pct}%")
        
        # Min composite score
        min_score = conditions.get("min_composite_score")
        if min_score is not None and entry.composite_score > 0:
            if entry.composite_score < min_score:
                reasons.append(f"Composite score {entry.composite_score:.3f} < {min_score}")
        
        # Max hold days
        max_days = conditions.get("max_hold_days")
        if max_days is not None:
            if entry.days_in_pool > max_days:
                reasons.append(f"Held {entry.days_in_pool}d > max {max_days}d")
        
        # Cooling period
        cooling_days = conditions.get("momentum_cooling_period_days")
        if cooling_days is not None:
            if entry.momentum_score > 0 and entry.days_in_pool > cooling_days:
                if entry.momentum_score < 0.3:
                    reasons.append("Momentum cooled below threshold")
        
        # No signal days (watchlist)
        no_signal_days = conditions.get("days_no_signal")
        if no_signal_days is not None and entry.days_in_pool > no_signal_days:
            reasons.append(f"No signal for {entry.days_in_pool}d > {no_signal_days}d")
        
        return len(reasons) > 0, reasons

    # ── Pool Rotation ────────────────────────────────────────────

    def rotate_pools(self) -> Dict[str, Any]:
        """
        Execute pool rotation: check all pools for exit candidates and rebalance.
        
        Returns rotation summary.
        """
        summary = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "rotations": [],
            "exits": [],
            "rebalances": [],
        }
        
        for pool_name in self.POOL_PRIORITY:
            if pool_name == "remove":
                continue
            
            cfg = self.pool_configs.get(pool_name)
            if not cfg:
                continue
            
            # Check exit conditions for each stock in pool
            for ticker, entry in list(self._pools[pool_name].items()):
                should_exit, reasons = self.check_exit_conditions(ticker, pool_name)
                if should_exit:
                    target = self._determine_exit_target(pool_name)
                    success, msg = self.move_stock(ticker, target, f"Exit: {'; '.join(reasons)}")
                    if success:
                        summary["exits"].append({
                            "ticker": ticker,
                            "from_pool": pool_name,
                            "to_pool": target,
                            "reasons": reasons,
                        })
        
        # Log rotation
        if summary["exits"]:
            self._rotation_log.append(summary)
            self._save_state()
            logger.info(f"Rotation complete: {len(summary['exits'])} exits")
        
        return summary

    def _determine_exit_target(self, source_pool: str) -> str:
        """Determine where to move a stock when it exits a pool."""
        if source_pool == "tactical":
            return "watchlist"
        elif source_pool == "core":
            return "watchlist"
        return "watchlist"

    # ── Bulk Operations ──────────────────────────────────────────

    def import_candidates(self, 
                          tickers: List[str],
                          pool: str = "watchlist",
                          metadata: Optional[Dict[str, Dict[str, Any]]] = None,
                          ) -> Dict[str, Any]:
        """
        Import a batch of candidate tickers into a pool.
        
        Args:
            tickers: List of ticker symbols
            pool: Target pool
            metadata: Optional per-ticker metadata {ticker: {key: val}}
        
        Returns:
            {"added": int, "skipped": int, "failed": List[str], "details": Dict}
        """
        result = {"added": 0, "skipped": 0, "failed": [], "details": {}}
        meta = metadata or {}
        
        for ticker in tickers:
            t = ticker.upper()
            ticker_meta = meta.get(t, {})
            
            success, msg = self.add_stock(
                ticker=t,
                pool=pool,
                entry_reason=ticker_meta.get("entry_reason", "batch_import"),
                composite_score=ticker_meta.get("composite_score", 0),
                quality_score=ticker_meta.get("quality_score", 0),
                momentum_score=ticker_meta.get("momentum_score", 0),
                metadata=ticker_meta,
            )
            
            if success:
                result["added"] += 1
            else:
                result["skipped"] += 1
                result["failed"].append(t)
                result["details"][t] = msg
        
        logger.info(f"Batch import: {result['added']} added, {result['skipped']} skipped")
        return result

    def clear_pool(self, pool: str) -> int:
        """Clear all entries from a pool. Returns number removed."""
        pool = pool.lower()
        if pool not in self._pools:
            return 0
        count = len(self._pools[pool])
        self._pools[pool].clear()
        self._save_state()
        logger.info(f"Cleared {count} entries from {pool} pool")
        return count

    # ── History and Logging ──────────────────────────────────────

    def _log_history(self, event: Dict[str, Any]):
        """Log a history event."""
        self._history.append(event)
        # Trim if needed
        if len(self._history) > 10000:
            self._history = self._history[-5000:]

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent pool history events."""
        return self._history[-limit:] if self._history else []

    def get_rotation_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent rotation log entries."""
        return self._rotation_log[-limit:] if self._rotation_log else []

    def get_pool_history_for_analysis(self) -> List[Dict[str, Any]]:
        """
        Get pool history suitable for performance analysis.
        
        Returns list of snapshots with pool composition and metrics.
        """
        snapshots = []
        for event in self._history:
            if event.get("action") in ("add", "remove", "move"):
                snapshots.append(event)
        return snapshots

    # ── Persistence ──────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self._data_dir / "pool_state.json"

    def _save_state(self):
        """Save pool state to JSON."""
        state = PoolState(
            version="1.0.0",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            pools={
                name: [e.to_dict() for e in entries.values()]
                for name, entries in self._pools.items()
            },
            history=self._history[-500:],  # Keep last 500 events
            rotation_log=self._rotation_log[-100:],
            stats={
                "total_stocks": sum(len(p) for p in self._pools.values()),
                "pool_sizes": {n: len(p) for n, p in self._pools.items()},
            },
        )
        
        try:
            with open(self._state_path(), 'w') as f:
                json.dump(state.__dict__, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save pool state: {e}")

    def _load_state(self):
        """Load pool state from JSON."""
        path = self._state_path()
        if not path.exists():
            logger.info("No saved pool state found, starting fresh")
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Restore pools
            pools_data = data.get("pools", {})
            for pool_name, entries_data in pools_data.items():
                if pool_name in self._pools:
                    for entry_data in entries_data:
                        entry = PoolEntry.from_dict(entry_data)
                        # Ensure pool_name is correct
                        entry.pool_name = pool_name
                        self._pools[pool_name][entry.ticker] = entry
            
            # Restore history and logs
            self._history = data.get("history", [])
            self._rotation_log = data.get("rotation_log", [])
            self._stats = data.get("stats", {})
            
            total = sum(len(p) for p in self._pools.values())
            logger.info(f"Loaded pool state: {total} stocks across {len(pools_data)} pools")
            
        except Exception as e:
            logger.error(f"Failed to load pool state: {e}, starting fresh")

    # ── Utility ──────────────────────────────────────────────────

    def summarize(self) -> str:
        """Generate a human-readable summary of all pools."""
        lines = ["═" * 50, "POOL SUMMARY", "═" * 50]
        
        for name in self.POOL_PRIORITY:
            status = self.get_pool_status(name)
            tickers_str = ", ".join(status.tickers[:10])
            if len(status.tickers) > 10:
                tickers_str += f", ... (+{len(status.tickers)-10})"
            
            lines.append(
                f"\n[{status.label}] {status.size}/{status.max_size} stocks"
            )
            if status.size > 0:
                lines.append(f"  Avg Score: {status.avg_composite_score:.3f}")
                lines.append(f"  Avg Return: {status.avg_return:+.1%}")
                lines.append(f"  Sectors: {status.sector_distribution}")
                lines.append(f"  Tickers: {tickers_str}")
            else:
                lines.append(f"  (empty)")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Unit Tests (as docstring examples)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Dynamic Stock Pool Test")
    print("=" * 60)
    
    pm = PoolManager()
    
    # Add stocks
    pm.add_stock("AAPL", pool="core", entry_price=190.0, composite_score=0.75,
                 quality_score=0.85, metadata={"sector": "Technology"})
    pm.add_stock("MSFT", pool="core", entry_price=410.0, composite_score=0.82,
                 quality_score=0.90, metadata={"sector": "Technology"})
    pm.add_stock("JNJ", pool="core", entry_price=155.0, composite_score=0.68,
                 quality_score=0.72, metadata={"sector": "Healthcare"})
    
    pm.add_stock("PLTR", pool="tactical", entry_price=85.0, composite_score=0.60,
                 momentum_score=0.80, metadata={"sector": "Technology"})
    
    pm.add_stock("SOFI", pool="watchlist", composite_score=0.50,
                 metadata={"sector": "Financial Services"})
    
    # Explicit remove
    pm.add_stock("SCAM", pool="remove", force=True, entry_reason="penny_stock_fraud")
    
    print(pm.summarize())
    
    # Pool status
    print("\n📊 Core Pool Status:")
    status = pm.get_pool_status("core")
    print(f"  Tickers: {status.tickers}")
    print(f"  Avg Score: {status.avg_composite_score:.3f}")
    print(f"  Full: {status.is_full}")
    print(f"  Available: {status.available_slots}")
    
    # Move
    print("\n🔄 Moving PLTR to core:")
    pm.move_stock("PLTR", "core", reason="upgraded_to_core")
    print(pm.summarize())
    
    # Check
    print(f"\n🔍 AAPL in core: {pm.is_in_pool('AAPL', 'core')}")
    print(f"🔍 MISSING in any pool: {pm.is_in_pool('MISSING')}")
    
    # Rotation
    print("\n🔄 Rotation:")
    result = pm.rotate_pools()
    print(f"  Exits: {len(result['exits'])}")
