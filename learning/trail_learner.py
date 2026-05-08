#!/usr/bin/env python3
"""
VMAA v3.2.1 — Trail Learner (Reinforcement Learning for Trailing Stop)
======================================================================
Learns optimal per-stock trailing stop parameters from backtest data.

Approach: Supervised Learning on Backtest Optimal Parameters
  1. For each trade in the backtest, simulate what trailing (W, A) pair
     would have maximized P&L given the actual price path.
  2. Extract per-trade features: ATR, market_cap, price, beta, sector.
  3. Train regression models: W_opt = f(features), A_opt = f(features).
  4. Export learned weights to replace hand-crafted compute_trailing_stop().

This is Phase 1 (offline learning from historical data). Phase 2 would
be online RL using paper trading feedback for continuous improvement.

Usage:
  python3 learning/trail_learner.py --trades backtest/output_adaptive_trail/trade_log_*.csv
  python3 learning/trail_learner.py --trades trade_log.csv --train
  python3 learning/trail_learner.py --predict  # test learned model
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("vmaa.learning.trail_learner")

# Output paths
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Grid Search: Optimal (W, A) per trade
# ═══════════════════════════════════════════════════════════════════

def simulate_trade_with_trail(
    entry_price: float,
    daily_prices: np.ndarray,
    hard_stop_pct: float,
    tp1_pct: float,
    trail_width: float,
    trail_activate: float,
) -> Tuple[float, str, float]:
    """
    Simulate a single trade with given trailing stop parameters.
    
    Args:
        entry_price: Entry price
        daily_prices: Array of daily closing prices (including entry day)
        hard_stop_pct: Hard stop % (e.g., 0.15)
        tp1_pct: Take profit % (e.g., 0.20)
        trail_width: Trailing stop width (e.g., 0.12)
        trail_activate: P&L % at which trailing activates (e.g., 0.15)
    
    Returns: (exit_price, exit_reason, peak_price)
    """
    stop_price = entry_price * (1 - hard_stop_pct)
    tp_price = entry_price * (1 + tp1_pct)
    trail_active = False
    trail_high = entry_price
    trail_stop = 0.0
    
    peak = entry_price
    
    for price in daily_prices[1:]:  # skip entry day
        peak = max(peak, price)
        pnl_pct = (price / entry_price - 1)
        
        # Check trailing activation
        if not trail_active and pnl_pct >= trail_activate:
            trail_active = True
            trail_high = price
        
        if trail_active:
            trail_high = max(trail_high, price)
            trail_stop = trail_high * (1 - trail_width)
        
        # Check exits (hard stop is checked first — most conservative)
        if price <= stop_price:
            return price, "hard_stop", peak
        
        if trail_active and price <= trail_stop:
            return price, f"trailing_stop(W={trail_width:.0%},A={trail_activate:.0%})", peak
        
        if price >= tp_price:
            return tp_price, "TP1", peak
    
    # Held to end of data — no exit triggered
    return daily_prices[-1], "time_stop", peak


def find_optimal_trail(
    entry_price: float,
    daily_prices: np.ndarray,
    hard_stop_pct: float = 0.15,
    tp1_pct: float = 0.20,
    trail_grid: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Grid search over (trail_width, trail_activate) to find optimal pair.
    
    Returns dict with optimal params, P&L, exit reason, and full grid results.
    """
    if trail_grid is None:
        # Dense grid over trailing parameters
        widths = np.arange(0.04, 0.20, 0.02)       # 4%, 6%, ..., 18%
        activates = np.arange(0.08, 0.22, 0.02)    # 8%, 10%, ..., 20%
        trail_grid = [(round(w, 2), round(a, 2)) for w in widths for a in activates]
    
    best_pnl_pct = -float('inf')
    best_params = None
    best_exit = None
    results = []
    
    for w, a in trail_grid:
        if a <= w:  # Activation must exceed width to avoid immediate trigger
            continue
        exit_price, reason, peak = simulate_trade_with_trail(
            entry_price, daily_prices, hard_stop_pct, tp1_pct, w, a
        )
        pnl_pct = (exit_price / entry_price - 1) * 100
        results.append({
            'width': w, 'activate': a, 'pnl_pct': round(pnl_pct, 2),
            'exit': reason, 'peak': round(float(peak), 2)
        })
        
        if pnl_pct > best_pnl_pct:
            best_pnl_pct = pnl_pct
            best_params = (w, a)
            best_exit = reason
    
    # Also compute baseline (no trail / TP1 only)
    baseline_exit, _, baseline_peak = simulate_trade_with_trail(
        entry_price, daily_prices, hard_stop_pct, tp1_pct, 0.99, 10.0
    )
    baseline_pnl = (baseline_exit / entry_price - 1) * 100
    
    return {
        'optimal_width': best_params[0],
        'optimal_activate': best_params[1],
        'optimal_pnl_pct': round(best_pnl_pct, 2),
        'optimal_exit': best_exit,
        'baseline_pnl_pct': round(baseline_pnl, 2),
        'baseline_exit': baseline_exit,
        'grid_results': results,
    }


# ═══════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_trade_features(
    ticker: str,
    entry_price: float,
    entry_date: str,
    daily_prices: np.ndarray,
) -> Dict:
    """
    Extract features for a trade to predict optimal trailing stop.
    """
    features = {
        'ticker': ticker,
        'entry_price': entry_price,
        'entry_date': entry_date,
    }
    
    # ATR-based volatility
    if len(daily_prices) >= 15:
        # Simple ATR-like: average daily % range
        pct_changes = np.abs(np.diff(daily_prices) / daily_prices[:-1])
        features['atr_pct'] = round(float(np.mean(pct_changes)), 4)
        features['vol_annualized'] = round(float(np.std(pct_changes) * np.sqrt(252)), 4)
    else:
        features['atr_pct'] = 0.02  # default
        features['vol_annualized'] = 0.30
    
    # Max drawdown during trade (proxy for gap risk)
    if len(daily_prices) > 5:
        running_max = np.maximum.accumulate(daily_prices)
        drawdowns = (daily_prices - running_max) / running_max
        features['max_intraday_dd'] = round(float(np.min(drawdowns)), 4)
    else:
        features['max_intraday_dd'] = -0.05
    
    # Price momentum before entry (last 20 days)
    if len(daily_prices) >= 21:
        pre_entry = daily_prices[-21:-1]  # 20 days before entry
        features['pre_momentum'] = round(float((pre_entry[-1] / pre_entry[0] - 1)), 4)
    else:
        features['pre_momentum'] = 0.0
    
    return features


def enrich_with_yfinance(ticker: str, entry_date: str) -> Dict:
    """Fetch current yfinance profile for market cap, sector, beta."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            'market_cap': info.get('marketCap', 0) or 0,
            'sector': info.get('sector', 'Unknown'),
            'beta': info.get('beta', 1.0) or 1.0,
            'price_to_book': info.get('priceToBook', 0) or 0,
        }
    except Exception:
        return {'market_cap': 0, 'sector': 'Unknown', 'beta': 1.0, 'price_to_book': 0}


# ═══════════════════════════════════════════════════════════════════
# Training Pipeline
# ═══════════════════════════════════════════════════════════════════

def learn_from_trades(trade_csv_path: str) -> pd.DataFrame:
    """
    Main training pipeline:
    1. Load trades from backtest CSV
    2. Fetch daily price data for each trade
    3. Grid-search optimal (W, A) per trade
    4. Extract features + optimal params
    5. Return training DataFrame
    """
    trades = []
    with open(trade_csv_path) as f:
        for row in csv.DictReader(f):
            trades.append(row)
    
    logger.info(f"Loaded {len(trades)} trades from {trade_csv_path}")
    
    training_data = []
    
    for i, trade in enumerate(trades):
        ticker = trade['ticker']
        entry_date = trade['entry_date']
        entry_price = float(trade['entry_price'])
        exit_date = trade['exit_date']
        actual_pnl = float(trade['return_pct'])
        exit_reason = trade['exit_reason']
        
        logger.info(f"\n[{i+1}/{len(trades)}] {ticker}: entry={entry_date} "
                    f"@ ${entry_price:.2f}, actual P&L={actual_pnl:+.1f}%, exit={exit_reason}")
        
        # Fetch daily price data
        try:
            end_date = pd.Timestamp(exit_date) + pd.Timedelta(days=5)
            start_date = pd.Timestamp(entry_date) - pd.Timedelta(days=30)
            daily = yf.download(ticker, start=str(start_date.date()),
                               end=str(end_date.date()), progress=False)
            if daily.empty or 'Close' not in daily.columns:
                logger.warning(f"  No price data for {ticker}, skipping")
                continue
            
            prices = daily['Close'].values.flatten()  # ensure 1D
            # Find entry index
            entry_target = pd.Timestamp(entry_date)
            daily.index = pd.to_datetime(daily.index)
            entry_idx = np.searchsorted(daily.index, entry_target)
            if entry_idx >= len(prices):
                entry_idx = len(prices) - 2
            trade_prices = prices[entry_idx:]  # from entry onward
        except Exception as e:
            logger.warning(f"  Price fetch failed: {e}, skipping")
            continue
        
        if len(trade_prices) < 2:
            logger.warning(f"  Not enough price data, skipping")
            continue
        
        # Find optimal trail
        opt = find_optimal_trail(entry_price, trade_prices, 0.15, 0.20)
        
        logger.info(f"  Optimal: W={opt['optimal_width']:.0%} A={opt['optimal_activate']:.0%} "
                    f"P&L={opt['optimal_pnl_pct']:+.1f}% exit={opt['optimal_exit']}")
        logger.info(f"  Baseline (no trail): P&L={opt['baseline_pnl_pct']:+.1f}%")
        
        if opt['optimal_pnl_pct'] > opt['baseline_pnl_pct'] + 0.5:
            logger.info(f"  📈 Trail ADDS {opt['optimal_pnl_pct'] - opt['baseline_pnl_pct']:+.1f}% vs no trail!")
        
        # Extract features
        features = extract_trade_features(ticker, entry_price, entry_date, trade_prices)
        yf_info = enrich_with_yfinance(ticker, entry_date)
        features.update(yf_info)
        features['optimal_width'] = opt['optimal_width']
        features['optimal_activate'] = opt['optimal_activate']
        features['optimal_pnl_pct'] = opt['optimal_pnl_pct']
        features['baseline_pnl_pct'] = opt['baseline_pnl_pct']
        features['actual_pnl_pct'] = actual_pnl
        features['exit_reason'] = exit_reason
        features['trade_days'] = len(trade_prices)
        
        training_data.append(features)
    
    df = pd.DataFrame(training_data)
    logger.info(f"\n{'='*60}")
    logger.info(f"Training data: {len(df)} trades with features")
    logger.info(f"Optimal W range: {df['optimal_width'].min():.0%} - {df['optimal_width'].max():.0%}")
    logger.info(f"Optimal A range: {df['optimal_activate'].min():.0%} - {df['optimal_activate'].max():.0%}")
    
    return df


# ═══════════════════════════════════════════════════════════════════
# Model: Random Forest Regression
# ═══════════════════════════════════════════════════════════════════

def train_models(df: pd.DataFrame) -> Tuple[object, object, Dict]:
    """
    Train Random Forest regressors for trailing_width and trailing_activate.
    Returns (width_model, activate_model, feature_importance).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    # Feature columns
    feature_cols = ['atr_pct', 'vol_annualized', 'max_intraday_dd',
                    'pre_momentum', 'entry_price', 'beta', 'trade_days']
    
    # Add log market cap
    df['log_market_cap'] = np.log1p(df['market_cap'].clip(lower=1))
    feature_cols.append('log_market_cap')
    
    # Drop rows with missing features
    train_df = df.dropna(subset=feature_cols + ['optimal_width', 'optimal_activate'])
    
    if len(train_df) < 5:
        logger.warning("Not enough training data (<5 trades), using fallback manual formula")
        return None, None, {}
    
    X = train_df[feature_cols].values
    y_w = train_df['optimal_width'].values
    y_a = train_df['optimal_activate'].values
    
    # Train width model
    rf_w = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, min_samples_leaf=2)
    rf_w.fit(X, y_w)
    cv_w = cross_val_score(rf_w, X, y_w, cv=min(3, len(train_df)), scoring='neg_mean_absolute_error')
    
    # Train activate model
    rf_a = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, min_samples_leaf=2)
    rf_a.fit(X, y_a)
    cv_a = cross_val_score(rf_a, X, y_a, cv=min(3, len(train_df)), scoring='neg_mean_absolute_error')
    
    # Feature importance
    importance = {
        'width': dict(zip(feature_cols, rf_w.feature_importances_)),
        'activate': dict(zip(feature_cols, rf_a.feature_importances_)),
    }
    
    logger.info(f"\nWidth model: CV MAE = {-cv_w.mean():.3f} (+/- {cv_w.std():.3f})")
    logger.info(f"Activate model: CV MAE = {-cv_a.mean():.3f} (+/- {cv_a.std():.3f})")
    logger.info(f"\nTop features for width:")
    for feat, imp in sorted(importance['width'].items(), key=lambda x: -x[1])[:5]:
        logger.info(f"  {feat}: {imp:.3f}")
    logger.info(f"\nTop features for activate:")
    for feat, imp in sorted(importance['activate'].items(), key=lambda x: -x[1])[:5]:
        logger.info(f"  {feat}: {imp:.3f}")
    
    return rf_w, rf_a, importance


# ═══════════════════════════════════════════════════════════════════
# Learned Predictor (replaces hand-crafted compute_trailing_stop)
# ═══════════════════════════════════════════════════════════════════

class LearnedTrailingStop:
    """
    ML-based trailing stop predictor. Trained on backtest optimal parameters.
    Falls back to hand-crafted formula if model not available.
    """
    
    def __init__(self):
        self.width_model = None
        self.activate_model = None
        self.feature_cols = ['atr_pct', 'vol_annualized', 'max_intraday_dd',
                             'pre_momentum', 'entry_price', 'beta', 'trade_days',
                             'log_market_cap']
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk."""
        import pickle
        w_path = MODEL_DIR / "trail_width_rf.pkl"
        a_path = MODEL_DIR / "trail_activate_rf.pkl"
        if w_path.exists() and a_path.exists():
            self.width_model = pickle.loads(w_path.read_bytes())
            self.activate_model = pickle.loads(a_path.read_bytes())
            logger.info("Loaded trained trail models from disk")
    
    def save_models(self, width_model, activate_model):
        """Save trained models to disk."""
        import pickle
        MODEL_DIR.mkdir(exist_ok=True)
        MODEL_DIR.joinpath("trail_width_rf.pkl").write_bytes(pickle.dumps(width_model))
        MODEL_DIR.joinpath("trail_activate_rf.pkl").write_bytes(pickle.dumps(activate_model))
        self.width_model = width_model
        self.activate_model = activate_model
        logger.info(f"Models saved to {MODEL_DIR}")
    
    def predict(self, features: Dict) -> Tuple[float, float]:
        """
        Predict optimal trailing stop (width, activate) for a trade.
        Falls back to hand-crafted formula if no model.
        """
        if self.width_model is None or self.activate_model is None:
            return self._fallback_predict(features)
        
        try:
            X = np.array([[
                features.get('atr_pct', 0.02),
                features.get('vol_annualized', 0.30),
                features.get('max_intraday_dd', -0.05),
                features.get('pre_momentum', 0.0),
                features.get('entry_price', 50),
                features.get('beta', 1.0),
                features.get('trade_days', 30),
                np.log1p(features.get('market_cap', 1e9)),
            ]])
            w = float(np.clip(self.width_model.predict(X)[0], 0.08, 0.18))
            a = float(np.clip(self.activate_model.predict(X)[0], 0.12, 0.20))
            return round(w, 3), round(a, 3)
        except Exception:
            return self._fallback_predict(features)
    
    def _fallback_predict(self, features: Dict) -> Tuple[float, float]:
        """Hand-crafted fallback formula (v3.2.1 base)."""
        trail = 0.12
        activate = 0.15
        
        if features.get('atr_pct', 0) > 0.05:
            trail += 0.03
        if features.get('market_cap', float('inf')) < 2e9:
            trail += 0.03
        if features.get('entry_price', 50) < 10:
            trail += 0.02
        
        trail = max(0.08, min(0.18, trail))
        activate = max(0.12, min(0.20, activate))
        return round(trail, 3), round(activate, 3)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VMAA Trail Learner")
    parser.add_argument('--trades', type=str, required=True,
                       help='Path to backtest trade log CSV')
    parser.add_argument('--train', action='store_true',
                       help='Train models on trade data')
    parser.add_argument('--predict', action='store_true',
                       help='Test prediction on a sample trade')
    parser.add_argument('--output', type=str, default='',
                       help='Save training data CSV')
    args = parser.parse_args()
    
    if args.train:
        df = learn_from_trades(args.trades)
        if args.output:
            df.to_csv(args.output, index=False)
            logger.info(f"Training data saved to {args.output}")
        
        width_model, activate_model, importance = train_models(df)
        if width_model:
            learner = LearnedTrailingStop()
            learner.save_models(width_model, activate_model)
            
            # Print learned insights
            print("\n" + "=" * 60)
            print("LEARNED TRAILING STOP MODEL")
            print("=" * 60)
            print(f"Samples: {len(df)} trades")
            print(f"\nFeature importance (width):")
            for feat, imp in sorted(importance['width'].items(), key=lambda x: -x[1])[:3]:
                print(f"  {feat}: {imp:.1%}")
            print(f"\nFeature importance (activate):")
            for feat, imp in sorted(importance['activate'].items(), key=lambda x: -x[1])[:3]:
                print(f"  {feat}: {imp:.1%}")
            print(f"\nModel saved to {MODEL_DIR}/")
    
    elif args.predict:
        learner = LearnedTrailingStop()
        # Sample prediction
        sample = {
            'atr_pct': 0.04, 'vol_annualized': 0.35,
            'max_intraday_dd': -0.08, 'pre_momentum': 0.05,
            'entry_price': 50, 'beta': 1.5, 'trade_days': 20,
            'market_cap': 1e9,
        }
        w, a = learner.predict(sample)
        print(f"Sample trade (small cap, beta=1.5, price=$50):")
        print(f"  Predicted: W={w:.0%} A={a:.0%}")
        
        sample2 = {**sample, 'market_cap': 3e12, 'atr_pct': 0.02, 'beta': 1.0}
        w2, a2 = learner.predict(sample2)
        print(f"Sample trade (large cap, beta=1.0, low vol):")
        print(f"  Predicted: W={w2:.0%} A={a2:.0%}")
    
    else:
        # Just show optimal analysis per trade
        df = learn_from_trades(args.trades)
        print("\n" + "=" * 80)
        print("PER-TRADE OPTIMAL TRAILING STOP ANALYSIS")
        print("=" * 80)
        for _, row in df.iterrows():
            delta = row['optimal_pnl_pct'] - row['actual_pnl_pct']
            marker = "🟢" if delta > 2 else ("🟡" if delta > 0 else "⚪")
            print(f"  {marker} {row['ticker']:6s} | "
                  f"Actual: {row['actual_pnl_pct']:+5.1f}% | "
                  f"Optimal: W={row['optimal_width']:.0%} A={row['optimal_activate']:.0%} "
                  f"→ {row['optimal_pnl_pct']:+5.1f}% (+{delta:+.1f}%) | "
                  f"Exit: {row['optimal_exit']}")


if __name__ == '__main__':
    main()
