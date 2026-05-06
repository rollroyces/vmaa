#!/usr/bin/env python3
"""
VMAA Earnings Prediction Engine — Orchestrator
================================================
EarningsEngine: central orchestrator for earnings prediction,
consensus aggregation, surprise analysis, and scoring.

Usage:
  from engine.earnings.engine import EarningsEngine
  engine = EarningsEngine()
  consensus = engine.get_consensus("AAPL")
  score = engine.score("AAPL")
  batch = engine.batch_scan(["AAPL", "MSFT", "GOOGL"])
  revisions = engine.get_revisions("AAPL")
  calendar = engine.get_earnings_calendar()
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.earnings.config import (
    EarningsConfig,
    EarningsConfigManager,
    get_earnings_config,
)
from engine.earnings.broker_reports import collect_broker_reports
from engine.earnings.forecast import analyze_forecast
from engine.earnings.consensus import build_consensus
from engine.earnings.ratings import analyze_ratings

logger = logging.getLogger("vmaa.engine.earnings.engine")


# ═══════════════════════════════════════════════════════════════════
# Simple Cache
# ═══════════════════════════════════════════════════════════════════

class EarningsCache:
    """In-memory cache with TTL-based expiration."""

    def __init__(self, ttl_seconds: int = 21600):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        ts, data = entry
        if time.time() - ts > self._ttl:
            del self._cache[key]
            return None
        return data

    def set(self, key: str, data: Any) -> None:
        self._cache[key] = (time.time(), data)

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        self._evict_expired()
        return len(self._cache)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, (ts, _) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]


# ═══════════════════════════════════════════════════════════════════
# Earnings Engine
# ═══════════════════════════════════════════════════════════════════

class EarningsEngine:
    """
    Earnings Prediction Engine — aggregates broker reports, forecasts,
    consensus, and ratings to produce earnings quality scores.

    Example:
        >>> engine = EarningsEngine()
        >>> consensus = engine.get_consensus("AAPL")
        >>> print(consensus["consensus_rating"]["rating_label"])
        >>> score = engine.score("AAPL")
        >>> print(f"Earnings quality: {score['score']}/100")
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_manager = EarningsConfigManager(config_path)
        self._cfg = self._config_manager.to_dataclass()
        self._cache = EarningsCache(ttl_seconds=self._cfg.cache.ttl_seconds)

        # Ensure data directory exists
        _data_dir = Path(__file__).resolve().parent.parent / self._cfg.cache.data_dir
        if _data_dir.exists():
            logger.info(f"Earnings data dir: {_data_dir}")

        logger.info(
            f"EarningsEngine initialized — "
            f"cache_ttl: {self._cfg.cache.ttl_seconds}s | "
            f"scoring weights: consensus={self._cfg.scoring.consensus_rating_weight:.0%} "
            f"surprise={self._cfg.scoring.surprise_history_weight:.0%} "
            f"revisions={self._cfg.scoring.estimate_revision_weight:.0%} "
            f"dispersion={self._cfg.scoring.dispersion_weight:.0%}"
        )

    @property
    def config(self) -> EarningsConfig:
        return self._cfg

    def reload_config(self):
        """Hot-reload configuration."""
        self._config_manager.reload()
        self._cfg = self._config_manager.to_dataclass()
        self._cache = EarningsCache(ttl_seconds=self._cfg.cache.ttl_seconds)
        logger.info("EarningsEngine config reloaded")

    # ── Internal: Data Collection ────────────────────────────────

    def _get_broker_data(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get broker data for a ticker with caching."""
        cache_key = f"broker:{ticker}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        result = collect_broker_reports(
            [ticker],
            surprise_lookback=self._cfg.surprise.lookback_quarters,
        )
        data = result.get(ticker, {"ticker": ticker, "error": "no_data"})

        self._cache.set(cache_key, data)
        return data

    def _get_forecast_data(
        self,
        ticker: str,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Get forecast analysis for a ticker with caching."""
        cache_key = f"forecast:{ticker}"
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        broker_data = self._get_broker_data(ticker, use_cache)
        data = analyze_forecast(
            ticker,
            broker_data,
            material_threshold=self._cfg.surprise.material_threshold_pct,
            window_days_1w=self._cfg.rating_changes.revision_window_days_1w,
            window_days_1m=self._cfg.rating_changes.revision_window_days_1m,
            window_days_3m=self._cfg.rating_changes.revision_window_days_3m,
        )

        self._cache.set(cache_key, data)
        return data

    def _get_ticker_object(self, ticker: str):
        """Get yfinance Ticker object for earnings calendar queries."""
        try:
            import yfinance as yf
            return yf.Ticker(ticker)
        except ImportError:
            return None

    # ── Public API ───────────────────────────────────────────────

    def get_consensus(self, ticker: str) -> Dict[str, Any]:
        """
        Get full consensus report for a ticker.

        Returns dict with:
          - consensus_rating (label, score, normalized)
          - consensus_target (mean/median/high/low, implied return)
          - consensus_earnings (EPS/revenue estimates, dispersion)
          - consensus_changes (1w/1m revision trends)
          - confidence (score, level, factor breakdown)

        Args:
            ticker: Ticker symbol (e.g., 'AAPL', '0700.HK')

        Returns:
            Full consensus report dict
        """
        start = time.time()

        broker_data = self._get_broker_data(ticker)
        if broker_data.get("error"):
            return {"ticker": ticker, "error": broker_data["error"]}

        forecast_data = self._get_forecast_data(ticker)
        if forecast_data.get("error"):
            return {"ticker": ticker, "error": forecast_data["error"]}

        consensus = build_consensus(
            ticker,
            broker_data,
            forecast_data,
            min_analysts=self._cfg.consensus.min_analysts_for_confidence,
            max_dispersion=self._cfg.consensus.max_dispersion_for_confidence,
        )

        consensus["_meta"] = {
            "elapsed_seconds": round(time.time() - start, 2),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"[{ticker}] Consensus: rating={consensus.get('consensus_rating', {}).get('rating_label', '?')} | "
            f"return={consensus.get('consensus_target', {}).get('implied_return', '?')} | "
            f"confidence={consensus.get('confidence', {}).get('confidence_score', '?')}"
        )

        return consensus

    def get_revisions(self, ticker: str) -> Dict[str, Any]:
        """
        Get recent estimate revision changes for a ticker.

        Returns revision data across 1w/1m/3m windows with net
        upgrade/downgrade counts and momentum signals.

        Args:
            ticker: Ticker symbol

        Returns:
            Revision analysis dict
        """
        broker_data = self._get_broker_data(ticker)
        if broker_data.get("error"):
            return {"ticker": ticker, "error": broker_data["error"]}

        forecast_data = self._get_forecast_data(ticker)
        return {
            "ticker": ticker,
            "revisions": forecast_data.get("revisions", {}),
            "_meta": {
                "collected_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def get_earnings_calendar(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get upcoming earnings dates for tickers.

        Args:
            tickers: List of ticker symbols, or None for empty

        Returns:
            Dict with earnings calendar entries
        """
        result: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "upcoming": [],
        }

        if not tickers:
            return result

        for ticker in tickers:
            try:
                broker_data = self._get_broker_data(ticker)
                if broker_data.get("error"):
                    continue

                t = self._get_ticker_object(ticker)
                ratings_data = analyze_ratings(
                    ticker, broker_data, t,
                    window_days_1w=self._cfg.rating_changes.revision_window_days_1w,
                    window_days_1m=self._cfg.rating_changes.revision_window_days_1m,
                    window_days_3m=self._cfg.rating_changes.revision_window_days_3m,
                )

                calendar = ratings_data.get("earnings_calendar", {})
                if calendar.get("next_earnings_date"):
                    result["upcoming"].append({
                        "ticker": ticker,
                        "earnings_date": calendar["next_earnings_date"],
                        "days_to_earnings": calendar.get("days_to_earnings"),
                        "is_soon": calendar.get("is_earnings_soon", False),
                        "quarterly_eps_growth": calendar.get("earnings_quarterly_growth"),
                        "sentiment": ratings_data.get("pre_earnings_sentiment", {}).get("pre_earnings_sentiment"),
                    })
            except Exception as e:
                logger.warning(f"Earnings calendar failed for {ticker}: {e}")
                continue

        # Sort by days to earnings (soonest first)
        result["upcoming"].sort(key=lambda x: x.get("days_to_earnings", 999) or 999)

        return result

    def analyze_surprise(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze beat/miss history for a ticker.

        Returns surprise metrics: beat rates, consecutive streaks,
        surprise momentum, and material beat/miss counts.

        Args:
            ticker: Ticker symbol

        Returns:
            Surprise analysis dict
        """
        forecast_data = self._get_forecast_data(ticker)
        if forecast_data.get("error"):
            return {"ticker": ticker, "error": forecast_data["error"]}

        surprise = forecast_data.get("surprise", {})
        accuracy = forecast_data.get("accuracy", {})

        return {
            "ticker": ticker,
            "surprise": surprise,
            "accuracy": accuracy,
            "_meta": {
                "collected_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def score(self, ticker: str) -> Dict[str, Any]:
        """
        Compute earnings quality score (0-100) for a ticker.

        Scoring methodology:
          - 30%: consensus rating (higher = better)
          - 25%: surprise history (more beats = better)
          - 25%: estimate revision momentum (upgrades > downgrades)
          - 20%: consistency (low estimate dispersion)

        Args:
            ticker: Ticker symbol

        Returns:
            Dict with score, grade, and sub-scores
        """
        consensus = self.get_consensus(ticker)
        if consensus.get("error"):
            return {"ticker": ticker, "score": None, "grade": "N/A", "error": consensus["error"]}

        # Sub-score: Consensus Rating (30%)
        rating_score_raw = consensus.get("consensus_rating", {}).get("rating_normalized", 0.5)
        sub_rating = rating_score_raw * 100  # 0-1 → 0-100
        weight_rating = self._cfg.scoring.consensus_rating_weight

        # Sub-score: Surprise History (25%)
        # Get forecast data for surprise metrics
        forecast_data = self._get_forecast_data(ticker)
        surprise = forecast_data.get("surprise", {})

        beat_rate_4q = surprise.get("beat_rate_4q") or 0.5
        beat_rate_8q = surprise.get("beat_rate_8q") or 0.5
        momentum = surprise.get("surprise_momentum", "neutral")
        streak_boost = min(0.1, surprise.get("consecutive_beats", 0) * 0.025)

        # Weighted beat rate (recent > older) + streak boost
        combined_beat = beat_rate_4q * 0.6 + beat_rate_8q * 0.4 + streak_boost
        sub_surprise = min(100, combined_beat * 100)
        weight_surprise = self._cfg.scoring.surprise_history_weight

        # Sub-score: Estimate Revision Momentum (25%)
        revisions = forecast_data.get("revisions", {})
        rev_1m = revisions.get("revisions_1m", {})
        rev_3m = revisions.get("revisions_3m", {})

        net_1m = rev_1m.get("net", 0)
        total_1m = rev_1m.get("total_changes", 0)
        net_3m = rev_3m.get("net", 0)
        total_3m = rev_3m.get("total_changes", 0)

        rev_score_1m = ((net_1m / max(1, total_1m)) + 1) / 2 if total_1m > 0 else 0.5
        rev_score_3m = ((net_3m / max(1, total_3m)) + 1) / 2 if total_3m > 0 else 0.5
        sub_revisions = (rev_score_1m * 0.6 + rev_score_3m * 0.4) * 100
        weight_revisions = self._cfg.scoring.estimate_revision_weight

        # Sub-score: Consistency / Low Dispersion (20%)
        cons_earnings = consensus.get("consensus_earnings", {})
        disp_score = cons_earnings.get("dispersion_score")
        if disp_score is not None:
            sub_dispersion = disp_score * 100
        else:
            sub_dispersion = 50  # neutral if unknown
        weight_dispersion = self._cfg.scoring.dispersion_weight

        # Composite score
        composite = (
            sub_rating * weight_rating
            + sub_surprise * weight_surprise
            + sub_revisions * weight_revisions
            + sub_dispersion * weight_dispersion
        )

        # Grade
        if composite >= 80:
            grade = "A"
        elif composite >= 65:
            grade = "B"
        elif composite >= 50:
            grade = "C"
        elif composite >= 35:
            grade = "D"
        else:
            grade = "F"

        return {
            "ticker": ticker,
            "score": round(composite, self._cfg.output.decimal_places_score),
            "grade": grade,
            "sub_scores": {
                "consensus_rating": {
                    "score": round(sub_rating, 1),
                    "weight": weight_rating,
                    "contribution": round(sub_rating * weight_rating, 1),
                },
                "surprise_history": {
                    "score": round(sub_surprise, 1),
                    "weight": weight_surprise,
                    "contribution": round(sub_surprise * weight_surprise, 1),
                },
                "estimate_revisions": {
                    "score": round(sub_revisions, 1),
                    "weight": weight_revisions,
                    "contribution": round(sub_revisions * weight_revisions, 1),
                },
                "dispersion_consistency": {
                    "score": round(sub_dispersion, 1),
                    "weight": weight_dispersion,
                    "contribution": round(sub_dispersion * weight_dispersion, 1),
                },
            },
            "consensus_summary": {
                "rating": consensus.get("consensus_rating", {}).get("rating_label", "N/A"),
                "target_premium": consensus.get("consensus_target", {}).get("implied_return"),
                "analysts": consensus.get("consensus_target", {}).get("num_analysts", 0),
                "confidence": consensus.get("confidence", {}).get("confidence_level", "N/A"),
            },
            "_meta": {
                "collected_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def batch_scan(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Run consensus analysis on a batch of tickers.

        Returns summary with overall stats and per-ticker consensus
        scores sorted by quality.

        Args:
            tickers: List of ticker symbols

        Returns:
            Batch scan result dict
        """
        if not tickers:
            return {"tickers_scanned": 0, "scores": [], "summary": {}}

        start = time.time()
        scores = []
        errors = []

        for ticker in tickers:
            try:
                score_result = self.score(ticker)
                if score_result.get("score") is not None:
                    scores.append(score_result)
                else:
                    errors.append(score_result)
            except Exception as e:
                logger.warning(f"Batch scan failed for {ticker}: {e}")
                errors.append({"ticker": ticker, "error": str(e)})

        # Sort by score descending
        scores.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Summary stats
        all_scores = [s["score"] for s in scores if s.get("score") is not None]
        import numpy as np

        summary = {
            "tickers_scanned": len(tickers),
            "tickers_with_data": len(scores),
            "tickers_with_errors": len(errors),
            "avg_score": round(float(np.mean(all_scores)), 1) if all_scores else None,
            "median_score": round(float(np.median(all_scores)), 1) if all_scores else None,
            "max_score": round(float(np.max(all_scores)), 1) if all_scores else None,
            "min_score": round(float(np.min(all_scores)), 1) if all_scores else None,
            "grade_distribution": {
                "A": sum(1 for s in scores if s.get("grade") == "A"),
                "B": sum(1 for s in scores if s.get("grade") == "B"),
                "C": sum(1 for s in scores if s.get("grade") == "C"),
                "D": sum(1 for s in scores if s.get("grade") == "D"),
                "F": sum(1 for s in scores if s.get("grade") == "F"),
            },
            "elapsed_seconds": round(time.time() - start, 1),
        }

        result: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tickers_scanned": len(tickers),
            "scores": scores,
            "errors": errors,
            "summary": summary,
        }

        logger.info(
            f"Batch scan complete: {len(scores)} scored / {len(tickers)} total "
            f"in {summary['elapsed_seconds']}s | "
            f"avg={summary['avg_score']} | "
            f"A:{summary['grade_distribution']['A']} "
            f"B:{summary['grade_distribution']['B']} "
            f"C:{summary['grade_distribution']['C']}"
        )

        return result

    # ── Persistence ─────────────────────────────────────────────

    def save_batch_result(self, result: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save batch scan result to JSON file."""
        data_dir = Path(__file__).resolve().parent.parent / self._cfg.cache.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"earnings_scan_{ts}.json"

        path = data_dir / filename
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"Batch result saved to {path}")
        return str(path)

    # ── Convenience ─────────────────────────────────────────────

    def get_rating_momentum(self, ticker: str) -> Dict[str, Any]:
        """Get rating change momentum summary for a ticker."""
        broker_data = self._get_broker_data(ticker)
        if broker_data.get("error"):
            return {"ticker": ticker, "error": broker_data["error"]}

        ratings_data = analyze_ratings(
            ticker, broker_data,
            window_days_1w=self._cfg.rating_changes.revision_window_days_1w,
            window_days_1m=self._cfg.rating_changes.revision_window_days_1m,
            window_days_3m=self._cfg.rating_changes.revision_window_days_3m,
        )

        return {
            "ticker": ticker,
            "momentum": ratings_data.get("rating_momentum", {}),
            "activity": {
                k: v for k, v in ratings_data.get("rating_activity", {}).items()
            },
            "_meta": {
                "collected_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def clear_cache(self) -> None:
        """Clear the earnings cache."""
        self._cache.clear()
        logger.info("Earnings cache cleared")


# ═══════════════════════════════════════════════════════════════════
# Quick Functions (convenience)
# ═══════════════════════════════════════════════════════════════════

def quick_score(ticker: str) -> Dict[str, Any]:
    """One-liner: compute earnings quality score for a ticker."""
    engine = EarningsEngine()
    return engine.score(ticker)


# ═══════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════

_EARNINGS_ENGINE: Optional[EarningsEngine] = None


def get_earnings_engine() -> EarningsEngine:
    """Get or create the singleton EarningsEngine instance."""
    global _EARNINGS_ENGINE
    if _EARNINGS_ENGINE is None:
        _EARNINGS_ENGINE = EarningsEngine()
    return _EARNINGS_ENGINE
