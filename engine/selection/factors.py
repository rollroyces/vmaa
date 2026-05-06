#!/usr/bin/env python3
"""
VMAA Smart Selection Engine — Multi-Factor Screening
======================================================
Factor extraction, Z-score normalization, percentile ranking, and
weighted composite scoring for stock selection.

Supported factor types:
  - Value:     P/E, P/B, P/S, EV/EBITDA, FCF Yield
  - Quality:   ROE, ROA, Profit Margin, Debt/Equity, FCF Conversion
  - Growth:    Revenue Growth (YoY/QoQ), EPS Growth (YoY/QoQ), EBITDA Growth
  - Momentum:  1M/3M/6M/12M return, RS Ratio, MA crossover
  - Volatility: Beta, 20d/50d volatility
  - Size:      Market Cap, Enterprise Value

Scoring: Z-score normalization + percentile ranking → composite score

Usage:
  from engine.selection.factors import FactorEngine
  fe = FactorEngine()
  scores = fe.score_universe(tickers=["AAPL", "MSFT", "GOOGL"])
  # scores: {"AAPL": {"value": 0.72, "quality": 0.85, ...}, ...}
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure VMAA root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.hybrid import get_snapshot
from engine.selection.config import SelectionConfig, get_selection_config

logger = logging.getLogger("vmaa.engine.selection.factors")


# ═══════════════════════════════════════════════════════════════════
# Factor Extraction
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FactorProfile:
    """Extracted factor values for a single stock."""
    ticker: str
    sector: str = ""
    industry: str = ""
    name: str = ""
    price: float = 0.0
    
    # Value factors
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    ps_ratio: float = 0.0
    ev_ebitda: float = 0.0
    fcf_yield: float = 0.0
    
    # Quality factors
    roe: float = 0.0
    roa: float = 0.0
    profit_margin: float = 0.0
    debt_to_equity: float = 0.0
    fcf_conversion: float = 0.0
    
    # Growth factors
    revenue_growth_yoy: float = 0.0
    revenue_growth_qoq: float = 0.0
    eps_growth_yoy: float = 0.0
    eps_growth_qoq: float = 0.0
    ebitda_growth: float = 0.0
    
    # Momentum factors
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_6m: float = 0.0
    return_12m: float = 0.0
    rs_ratio: float = 0.0
    ma_crossover: float = 0.0
    
    # Volatility
    beta: float = 0.0
    volatility_20d: float = 0.0
    volatility_50d: float = 0.0
    
    # Size
    market_cap: float = 0.0
    enterprise_value: float = 0.0
    
    # Source metadata
    data_source: str = ""
    data_date: str = ""
    extraction_ok: bool = False
    
    def to_dict(self) -> Dict[str, float]:
        """Convert numeric factor values to dict (skip metadata)."""
        return {
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'ps_ratio': self.ps_ratio,
            'ev_ebitda': self.ev_ebitda,
            'fcf_yield': self.fcf_yield,
            'roe': self.roe,
            'roa': self.roa,
            'profit_margin': self.profit_margin,
            'debt_to_equity': self.debt_to_equity,
            'fcf_conversion': self.fcf_conversion,
            'revenue_growth_yoy': self.revenue_growth_yoy,
            'revenue_growth_qoq': self.revenue_growth_qoq,
            'eps_growth_yoy': self.eps_growth_yoy,
            'eps_growth_qoq': self.eps_growth_qoq,
            'ebitda_growth': self.ebitda_growth,
            'return_1m': self.return_1m,
            'return_3m': self.return_3m,
            'return_6m': self.return_6m,
            'return_12m': self.return_12m,
            'rs_ratio': self.rs_ratio,
            'ma_crossover': self.ma_crossover,
            'beta': self.beta,
            'volatility_20d': self.volatility_20d,
            'volatility_50d': self.volatility_50d,
            'market_cap': self.market_cap,
            'enterprise_value': self.enterprise_value,
        }


def extract_factors(ticker: str) -> FactorProfile:
    """
    Extract all factor values for a single ticker.
    
    Uses HybridData (SEC + yfinance + Tiger) for data retrieval.
    Returns FactorProfile with all numeric values populated.
    
    Example:
        >>> profile = extract_factors("AAPL")
        >>> print(f"P/E: {profile.pe_ratio:.1f}, FCF Yield: {profile.fcf_yield:.1%}")
    """
    profile = FactorProfile(ticker=ticker)
    
    try:
        snap = get_snapshot(ticker)
        
        # Metadata
        profile.name = snap.get("name", ticker)
        profile.sector = snap.get("sector", "Unknown")
        profile.industry = snap.get("industry", "Unknown")
        profile.price = snap.get("price", 0.0)
        profile.data_source = snap.get("data_source", "unknown")
        profile.data_date = snap.get("price_date", "")
        
        # Market cap
        profile.market_cap = snap.get("market_cap", 0) or 0
        
        # Value factors
        profile.pe_ratio = snap.get("trailing_pe", 0) or 0
        bv = snap.get("book_value", 0) or 0
        profile.pb_ratio = profile.price / bv if bv > 0 and profile.price > 0 else 0
        profile.fcf_yield = snap.get("fcf_yield", 0) or 0
        
        # EV/EBITDA
        ebitda = snap.get("ebitda", 0) or (snap.get("ebitda_margin", 0) * snap.get("latest_revenue", 0))
        if ebitda == 0:
            ebitda = snap.get("ebitda_margin", 0) * snap.get("latest_revenue", 0) if snap.get("ebitda_margin", 0) and snap.get("latest_revenue", 0) else 0
        debt = snap.get("debt_to_equity", 0) * bv if bv > 0 and snap.get("debt_to_equity", 0) else 0
        ev = profile.market_cap + debt if profile.market_cap > 0 else 0
        profile.ev_ebitda = ev / ebitda if ebitda > 0 else 0
        profile.enterprise_value = ev
        
        # P/S ratio
        rev = snap.get("latest_revenue", 0) or 0
        profile.ps_ratio = profile.market_cap / rev if rev > 0 and profile.market_cap > 0 else 0
        
        # Quality factors
        profile.roe = snap.get("roe", 0) or 0
        profile.roa = snap.get("roa", 0) or 0
        profile.profit_margin = snap.get("profit_margin", 0) or 0
        profile.debt_to_equity = snap.get("debt_to_equity", 0) or 0
        profile.fcf_conversion = snap.get("fcf_conversion", 0) or 0
        
        # Growth factors
        profile.revenue_growth_yoy = snap.get("revenue_growth", 0) or 0
        profile.eps_growth_yoy = snap.get("earnings_growth", 0) or 0
        profile.ebitda_growth = _compute_ebitda_growth(snap)
        
        # Momentum factors
        from data.hybrid import get_price
        profile.return_1m, profile.return_3m, profile.return_6m, profile.return_12m = (
            _compute_returns(ticker, profile.price)
        )
        profile.rs_ratio = _compute_rs_ratio(ticker)
        profile.ma_crossover = _compute_ma_crossover(ticker, profile.price)
        
        # Volatility
        profile.beta = snap.get("beta", 0) or 0
        profile.volatility_20d, profile.volatility_50d = _compute_volatility(ticker)
        
        profile.extraction_ok = True
        
    except Exception as e:
        logger.debug(f"Factor extraction failed for {ticker}: {e}")
    
    return profile


def _compute_returns(ticker: str, current_price: float) -> Tuple[float, float, float, float]:
    """Compute 1M, 3M, 6M, 12M price returns from yfinance history."""
    returns = (0.0, 0.0, 0.0, 0.0)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if hist is None or len(hist) < 2:
            return returns
        
        closes = hist["Close"]
        # Use current price if available, otherwise last close
        latest = current_price if current_price > 0 else float(closes.iloc[-1])
        
        periods = {
            'return_1m': 21,
            'return_3m': 63,
            'return_6m': 126,
            'return_12m': 252,
        }
        
        results = []
        for _, days in periods.items():
            if len(closes) >= days:
                prev = float(closes.iloc[-(days + 1)]) if len(closes) > days else float(closes.iloc[0])
                ret = (latest - prev) / prev if prev > 0 else 0.0
            else:
                ret = 0.0
            results.append(ret)
        
        return tuple(results)
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


def _compute_rs_ratio(ticker: str) -> float:
    """Compute relative strength ratio (stock return / market return over 12M)."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        spy = yf.Ticker("SPY")
        
        hist_t = t.history(period="1y")
        hist_spy = spy.history(period="1y")
        
        if hist_t is None or hist_spy is None:
            return 0.0
        if len(hist_t) < 2 or len(hist_spy) < 2:
            return 0.0
        
        stock_ret = float(hist_t["Close"].iloc[-1] / hist_t["Close"].iloc[0] - 1)
        spy_ret = float(hist_spy["Close"].iloc[-1] / hist_spy["Close"].iloc[0] - 1)
        
        if spy_ret == 0:
            return 0.0
        return stock_ret / spy_ret
    except Exception:
        return 0.0


def _compute_ma_crossover(ticker: str, price: float) -> float:
    """Compute MA crossover signal: (price - MA50) / MA50."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="6mo")
        if hist is None or len(hist) < 50:
            return 0.0
        
        ma50 = float(hist["Close"].rolling(window=50).mean().iloc[-1])
        if ma50 <= 0:
            return 0.0
        
        p = price if price > 0 else float(hist["Close"].iloc[-1])
        return (p - ma50) / ma50
    except Exception:
        return 0.0


def _compute_volatility(ticker: str) -> Tuple[float, float]:
    """Compute 20d and 50d annualized volatility."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="6mo")
        if hist is None or len(hist) < 50:
            return (0.0, 0.0)
        
        returns = hist["Close"].pct_change().dropna()
        
        vol20 = float(returns.iloc[-20:].std() * np.sqrt(252)) if len(returns) >= 20 else 0.0
        vol50 = float(returns.iloc[-50:].std() * np.sqrt(252)) if len(returns) >= 50 else 0.0
        
        return vol20, vol50
    except Exception:
        return (0.0, 0.0)


def _compute_ebitda_growth(snap: dict) -> float:
    """Estimate EBITDA growth from available data."""
    try:
        # Use revenue growth as proxy if direct EBITDA trend unavailable
        rev_trend = snap.get("revenue_trend", [])
        if len(rev_trend) >= 5:
            recent = rev_trend[0]["val"]
            year_ago = rev_trend[4]["val"]
            if year_ago > 0:
                return (recent - year_ago) / year_ago
        return snap.get("revenue_growth", 0) or 0
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# Scoring Engine
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FactorScores:
    """Scored results for a universe of stocks."""
    profiles: Dict[str, FactorProfile] = field(default_factory=dict)
    raw_factors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    z_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    composite_scores: Dict[str, float] = field(default_factory=dict)
    layer_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ranks: Dict[str, int] = field(default_factory=dict)
    
    def get_top(self, n: int = 20) -> List[Tuple[str, float]]:
        """Return top N tickers by composite score."""
        sorted_items = sorted(self.composite_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]
    
    def get_layer_breakdown(self, ticker: str) -> Dict[str, float]:
        """Get per-layer scores for a ticker."""
        return {layer: scores.get(ticker, 0.0) for layer, scores in self.layer_scores.items()}


class FactorEngine:
    """
    Multi-factor screening engine.
    
    Extracts factors from data sources, normalizes with Z-scores,
    converts to percentiles, and computes weighted composite scores.
    
    Config is loaded from external JSON (factor_weights.json).
    
    Example:
        >>> fe = FactorEngine()
        >>> scores = fe.score_universe(["AAPL", "MSFT", "GOOGL"])
        >>> print(scores.composite_scores["AAPL"])
        0.723
    """

    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or get_selection_config()
        self.fw = self.config.get_factor_weights()
        
        # Cache: ticker → FactorProfile
        self._profile_cache: Dict[str, FactorProfile] = {}
        
        logger.info(
            f"FactorEngine initialized: {len(self.fw.factors)} categories, "
            f"scoring={self.fw.scoring_method}"
        )

    # ── Extraction ──────────────────────────────────────────────

    def extract_universe(self, tickers: List[str], 
                         progress_callback: Optional[Callable[[int, int, str], None]] = None
                         ) -> Dict[str, FactorProfile]:
        """
        Extract factors for a universe of tickers.
        
        Args:
            tickers: List of ticker symbols
            progress_callback: Optional fn(current, total, ticker) for progress
        
        Returns:
            Dict[ticker → FactorProfile]
        """
        profiles = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers):
            if ticker in self._profile_cache:
                profiles[ticker] = self._profile_cache[ticker]
            else:
                profile = extract_factors(ticker)
                if profile.extraction_ok:
                    self._profile_cache[ticker] = profile
                    profiles[ticker] = profile
            
            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, total, ticker)
            
            # Small delay to avoid rate limits
            if i > 0 and i % 20 == 0:
                import time
                time.sleep(0.05)
        
        if progress_callback:
            progress_callback(len(tickers), total, "done")
        
        logger.info(f"Extracted factors for {len(profiles)}/{total} tickers")
        return profiles

    # ── Scoring ─────────────────────────────────────────────────

    def score_universe(self, 
                       tickers: Optional[List[str]] = None,
                       profiles: Optional[Dict[str, FactorProfile]] = None,
                       ) -> FactorScores:
        """
        Score a universe of stocks.
        
        Args:
            tickers: List of tickers (extracts factors if profiles not provided)
            profiles: Pre-extracted FactorProfiles (skip extraction)
        
        Returns:
            FactorScores with all scoring layers
        """
        if profiles is None and tickers is not None:
            profiles = self.extract_universe(tickers)
        elif profiles is None:
            raise ValueError("Either tickers or profiles must be provided")
        
        if len(profiles) < self.fw.min_observations:
            logger.warning(
                f"Universe size ({len(profiles)}) below min_observations "
                f"({self.fw.min_observations}), scores may be unreliable"
            )
        
        # Step 1: Build raw factor matrix
        raw_factors: Dict[str, Dict[str, float]] = {}
        for ticker, profile in profiles.items():
            raw_factors[ticker] = profile.to_dict()
        
        # Step 2: Z-score normalization
        z_scores = self._compute_z_scores(raw_factors)
        
        # Step 3: Percentile ranking
        if self.fw.scoring_method == "percentile":
            percentiles = self._compute_percentiles(z_scores)
        else:
            percentiles = {t: dict(z_scores[t]) for t in z_scores}
        
        # Step 4: Composite scores
        composite, layer_scores = self._compute_composite(percentiles)
        
        # Step 5: Ranking
        ranks = {
            t: i + 1 
            for i, (t, _) in enumerate(
                sorted(composite.items(), key=lambda x: x[1], reverse=True)
            )
        }
        
        return FactorScores(
            profiles=profiles,
            raw_factors=raw_factors,
            z_scores=z_scores,
            percentiles=percentiles,
            composite_scores=composite,
            layer_scores=layer_scores,
            ranks=ranks,
        )

    def _compute_z_scores(self, raw: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Compute Z-scores for all factors across tickers.
        Z = (x - μ) / σ, clipped to ±zscore_clip.
        Sign flipped based on factor_directions for consistent interpretation
        (higher = better).
        
        Example:
            >>> engine = FactorEngine()
            >>> raw = {"AAPL": {"pe_ratio": 28}, "MSFT": {"pe_ratio": 35}}
            >>> z = engine._compute_z_scores(raw)
            >>> # P/E direction is -1, so lower P/E → higher Z-score
        """
        if not raw:
            return {}
        
        # Collect all factor names
        factor_names = set()
        for ticker_factors in raw.values():
            factor_names.update(ticker_factors.keys())
        
        # Compute mean and std for each factor
        stats: Dict[str, Tuple[float, float]] = {}
        for fname in factor_names:
            values = [
                raw[t].get(fname, 0.0)
                for t in raw
                if fname in raw[t]
            ]
            if len(values) < 3:
                stats[fname] = (0.0, 1.0)  # Not enough data, skip normalization
                continue
            
            arr = np.array(values, dtype=np.float64)
            mean = float(np.nanmean(arr))
            std = float(np.nanstd(arr))
            if std == 0 or np.isnan(std):
                std = 1.0
            stats[fname] = (mean, std)
        
        # Compute Z-scores
        z_scores: Dict[str, Dict[str, float]] = {}
        clip = self.fw.zscore_clip
        directions = self.fw.factor_directions
        
        for ticker in raw:
            z_scores[ticker] = {}
            for fname, value in raw[ticker].items():
                mean, std = stats[fname]
                if std > 0:
                    z = (value - mean) / std
                else:
                    z = 0.0
                
                # Apply direction sign flip
                direction = directions.get(fname, 1)
                z *= direction
                
                # Clip
                z_scores[ticker][fname] = max(-clip, min(clip, z))
        
        return z_scores

    def _compute_percentiles(self, z_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Convert Z-scores to percentile ranks [0, 1].
        Uses empirical CDF: (rank - 1) / (N - 1).
        
        Example:
            >>> z = {"AAPL": {"value": 1.5}, "MSFT": {"value": 0.5}}
            >>> p = engine._compute_percentiles(z)
            >>> # AAPL value percentile > MSFT value percentile
        """
        if not z_scores:
            return {}
        
        # Collect all factor names
        factor_names = set()
        for ticker_scores in z_scores.values():
            factor_names.update(ticker_scores.keys())
        
        # Compute percentiles per factor
        tickers = list(z_scores.keys())
        n = len(tickers)
        
        percentiles: Dict[str, Dict[str, float]] = {t: {} for t in tickers}
        
        for fname in factor_names:
            # Get all values for this factor
            pairs = [(t, z_scores[t].get(fname, 0.0)) for t in tickers]
            pairs.sort(key=lambda x: x[1])
            
            # Assign percentile ranks
            for rank, (ticker, _) in enumerate(pairs):
                if n > 1:
                    pct = rank / (n - 1)
                else:
                    pct = 0.5
                percentiles[ticker][fname] = pct
        
        return percentiles

    def _compute_composite(self, 
                           scores: Dict[str, Dict[str, float]]
                           ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute composite scores using configured weights.
        
        Returns:
            (composite_scores, layer_scores)
            composite_scores: {ticker: overall_score}
            layer_scores: {layer_name: {ticker: layer_score}}
        """
        layer_weights = self.fw.layer_weights
        factor_weights = self.fw.factors
        use_layers = self.fw.use_layer_weights
        
        composite = {}
        layer_scores: Dict[str, Dict[str, float]] = {
            layer: {} for layer in factor_weights
        }
        
        for ticker, ticker_scores in scores.items():
            total = 0.0
            
            for layer, factors in factor_weights.items():
                # Compute layer score
                layer_sum = 0.0
                layer_weight_sum = 0.0
                
                for fname, weight in factors.items():
                    if fname in ticker_scores:
                        layer_sum += ticker_scores[fname] * weight
                        layer_weight_sum += weight
                
                layer_score = layer_sum / layer_weight_sum if layer_weight_sum > 0 else 0.0
                layer_scores[layer][ticker] = layer_score
                
                # Add to composite
                if use_layers:
                    total += layer_score * layer_weights.get(layer, 0.0)
                else:
                    total += layer_score / len(factor_weights)
            
            composite[ticker] = round(total, 4)
        
        return composite, layer_scores

    # ── Utility ─────────────────────────────────────────────────

    def clear_cache(self):
        """Clear the profile extraction cache."""
        self._profile_cache.clear()
        logger.debug("Factor profile cache cleared")

    def get_profile(self, ticker: str, use_cache: bool = True) -> FactorProfile:
        """Get factor profile for a single ticker."""
        if use_cache and ticker in self._profile_cache:
            return self._profile_cache[ticker]
        profile = extract_factors(ticker)
        if profile.extraction_ok:
            self._profile_cache[ticker] = profile
        return profile


# ═══════════════════════════════════════════════════════════════════
# Unit Tests (as docstring examples)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test extraction
    print("=" * 60)
    print("Factor Extraction Test")
    print("=" * 60)
    profile = extract_factors("AAPL")
    print(f"Ticker: {profile.ticker}")
    print(f"Sector: {profile.sector}")
    print(f"Price: ${profile.price:.2f}")
    print(f"P/E: {profile.pe_ratio:.1f}")
    print(f"FCF Yield: {profile.fcf_yield:.4f}")
    print(f"ROE: {profile.roe:.4f}")
    print(f"Revenue Growth: {profile.revenue_growth_yoy:.4f}")
    print(f"Return 3M: {profile.return_3m:.4f}")
    print(f"Beta: {profile.beta:.1f}")
    print(f"Vol 20d: {profile.volatility_20d:.4f}")
    print(f"Market Cap: ${profile.market_cap/1e9:.1f}B")
    
    # Test scoring
    print("\n" + "=" * 60)
    print("Multi-Factor Scoring Test")
    print("=" * 60)
    fe = FactorEngine()
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA"]
    scores = fe.score_universe(tickers=tickers)
    
    print(f"\nComposite Scores (top → bottom):")
    for i, (t, s) in enumerate(scores.get_top(7)):
        print(f"  {i+1}. {t:6s}  {s:.4f}")
    
    # Layer breakdown for top stock
    top_ticker = scores.get_top(1)[0][0]
    print(f"\nLayer breakdown for {top_ticker}:")
    for layer, layer_score in scores.get_layer_breakdown(top_ticker).items():
        print(f"  {layer:12s}: {layer_score:.4f}")
