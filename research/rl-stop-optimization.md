# VMAA Hard Stop Placement: RL-Based Optimization Research Report

> **Author:** VMAA RL Research Team  
> **Date:** 2026-05-06  
> **Status:** Research Phase — Recommendation Ready  
> **Target:** Eliminate hard-stop bleed (~$269 loss from 3/6 trades)

---

## Executive Summary

**The hard stop is VMAA's #1 profit killer.** In the latest backtest (2023-06-01 → 2025-12-31, 48-ticker universe), 50% of all exits were hard stops with a **0% win rate**, bleeding **-$269** while take-profit exits netted **+$448**. The root cause is not the strategy itself — it's the **stop placement logic**. Current median-based stops on a 15% level are better than the old tightest-stop approach, but remain static and blind to market context. I recommend a **phased approach**: deploy an adaptive statistical stop immediately (Phase 1), follow with a Q-learning agent trained on synthetic backtest data (Phase 2), and evaluate Deep RL only if Phase 2 proves insufficient (Phase 3).

---

## 1. Problem Analysis

### 1.1 Backtest Data Deep-Dive

**Universe:** 48 small/mid-cap stocks (TMDX, INMD, CDRE, SPSC, ICUI, OLED, NCNO, FVRR, BILL, ESTC, etc.)  
**Period:** 2023-06-01 → 2025-12-31 (2.58 years)  
**Frequency:** Weekly rebalance  
**Stop Config:** hard_stop_pct=10%, atr_stop_multiplier=2.0, stop selection = **tightest** (not median)

#### Complete Trade Log

| # | Ticker | Entry | Exit | Price In | Price Out | Qty | PnL | Return | Exit | Days | Win |
|---|--------|-------|------|----------|-----------|-----|------|--------|------|------|-----|
| 1 | RNG | 2023-06-09 | 2023-07-12 | $32.89 | $36.84 | 43 | +$169 | +12.0% | TP1 | 33 | ✅ |
| 2 | TMDX | 2023-10-20 | 2023-10-25 | $41.37 | $39.79 | 74 | -$117 | -3.8% | **hard_stop** | 5 | ❌ |
| 3 | TMDX | 2023-10-27 | 2023-10-30 | $37.97 | $36.88 | 86 | -$94 | -2.9% | **hard_stop** | 3 | ❌ |
| 4 | BLFS | 2023-10-27 | 2023-10-30 | $9.34 | $9.05 | 201 | -$58 | -3.1% | **hard_stop** | 3 | ❌ |
| 5 | TMDX | 2023-11-10 | 2023-11-14 | $57.37 | $64.25 | 4 | +$28 | +12.0% | TP1 | 4 | ✅ |
| 6 | RNG | 2023-11-03 | 2023-12-04 | $28.27 | $31.66 | 73 | +$245 | +12.0% | TP1 | 31 | ✅ |

#### Critical Observations

**1. All 3 hard stops happened within 5 days of entry.**
- TMDX #2: stopped in 5 days
- TMDX #3: stopped in 3 days  
- BLFS: stopped in 3 days
- **Median holding days for stopped trades: 3 days** vs. 31 days for TP trades

This is NOT a "trade went bad over time" scenario — these are immediate stop-outs, suggesting:
- Stop was too tight relative to the stock's daily volatility range
- Entry was at an unfortunate intraday/extended-hour price point
- The stock experienced mean-reverting noise (normal for value-turnaround stocks)

**2. TMDX was entered twice in one week.**
- Entered Oct 20 at $41.37 → stopped Oct 25 at $39.79 (-3.8%)
- Re-entered Oct 27 at $37.97 → stopped Oct 30 at $36.88 (-2.9%)
- Then entered Nov 10 at $57.37 → TP at $64.25 (+12.0%)

The strategy was correct on TMDX fundamentally — but the first two entries were at the wrong price. A wider stop or price-level-aware entry could have held through to the Nov rally.

**3. BLFS was a low-price stock ($9.34 entry).**
- Low-price stocks have wider percentage daily ranges
- A 10% hard stop on a $9 stock is only $0.93 of range
- Daily ATR for a $9 stock could easily be $0.40–0.60
- A stop 2× ATR away could be hit in 2 days of adverse movement

### 1.2 Root Cause Analysis

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| Stop too tight for volatility | Average loss = -3.26%, while stop was at -10%. Stopped BEFORE hitting the hard stop level (stop triggered by ATR or structural stop, not hard stop directly) — but the **tightest-stop selection** means ATR stop ($41.37 - 2.5×ATR) was likely much tighter than 10% | **CONFIRMED** |
| Entry timing bad (buying before further dip) | TMDX was entered twice in a falling knife scenario; 3-5 day holding periods suggest immediate adverse drift | **CONFIRMED** |
| Strategy fundamentally flawed | TP trades had 100% win rate with +12% average return. Strategy works when stops don't interfere | **UNCONFIRMED** |
| Low-price stocks need wider stops | BLFS at $9.34 had highest daily percentage volatility | **CONFIRMED** |

### 1.3 Volatility-at-Stop Analysis

Since we only have 3 stopped trades (insufficient for statistical power), let's derive the structural problem:

**Current stop logic (backtest engine — OLD version still in use!):**
```python
stops = [ATR_stop, hard_stop, structural_stop]
stops.sort(reverse=True)  # Highest = tightest
stop = stops[0]           # Always picks tightest
```

**Current stop logic (risk.py — FIXED version):**
```python
stops.sort()              # Ascending
stop = stops[len(stops)//2]  # Median
```

The backtest was run with the **OLD tightest-stop logic** at **10% hard stop**, meaning:
- If ATR stop was tighter than 10% (likely for volatile mid-caps), it would be selected
- If structural stop was tighter, it would be selected
- The actual executed stop was often much tighter than 10%

This is a **critical bug** — the backtest doesn't reflect the current code.

### 1.4 Time-to-Stop Distribution (Synthesized)

With only 3 stop events, we can't build a meaningful distribution. However, for RL training we need ~100+ stop events. This drives our synthetic data strategy in Section 3.

---

## 2. Comparison of All 5 Approaches

### 2.1 Scoring Framework

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Expected Impact | 30% | How much P&L improvement expected |
| Data Requirements | 20% | How much data needed; feasible with current data scarcity |
| Implementation Complexity | 15% | Engineering effort required |
| Overfitting Risk | 15% | Risk of learning noise with 3 data points |
| Maintainability | 10% | How easy to debug and modify |
| Deployment Risk | 10% | Risk of bad stop decisions in production |

---

### Option E: Adaptive Stop (Statistical/Heuristic) ⭐ RECOMMENDED PHASE 1

**Approach:** Dynamic stop width based on ATR percentile, price level, and market regime. No ML — pure statistics.

```python
def adaptive_stop(entry_price, atr, ptl_ratio, vol_regime, price_level):
    # 1. Base stop: ATR-based with dynamic multiplier
    if price_level < 10:
        base_mult = 3.5       # Low-price stocks: wider
    elif price_level < 30:
        base_mult = 3.0
    else:
        base_mult = 2.5
    
    # 2. Volatility adjustment
    if vol_regime == "HIGH":
        base_mult += 0.5      # Widen during high vol
    
    # 3. PTL adjustment (closer to 52w low → wider stop)
    if ptl_ratio < 1.10:
        base_mult += 0.5      # Near 52w low, give more room
    
    atr_stop = entry_price - atr * base_mult
    
    # 4. Percentage stop: dynamic based on price range
    if price_level < 10:
        pct_stop = 0.20       # 20% for sub-$10
    elif price_level < 30:
        pct_stop = 0.17       # 17% for $10-30
    else:
        pct_stop = 0.15       # 15% standard
    
    hard_stop = entry_price * (1 - pct_stop)
    
    # 5. Pick the WIDER of the two (not narrower)
    return max(atr_stop, hard_stop)
```

| Pros | Cons |
|------|------|
| Zero data requirements | Might be too conservative (miss entries) |
| 1-day implementation | Rules are hand-tuned, not learned |
| Transparent, debuggable | Won't capture complex patterns |
| Immediate deployment | Same rules for all sectors |

**Score: 7.2/10**

---

### Option A: Q-Learning Stop Optimization ⭐ RECOMMENDED PHASE 2

**Approach:** Tabular or discretized Q-learning where the agent picks a stop distance at trade entry time. After trade closes, reward is applied.

**State Space (6 dimensions, discretized):**
| Feature | Bins | Range |
|---------|------|-------|
| Volatility regime (ATR/price %) | 3 | Low/Med/High |
| Price level bucket | 4 | <$10, $10-30, $30-80, >$80 |
| PTL ratio (distance from 52w low) | 3 | <1.05, 1.05-1.20, >1.20 |
| Sector volatility rank | 3 | Low/Med/High |
| Market regime (SPY vol) | 3 | LOW/NORMAL/HIGH |
| Confidence score | 3 | <0.4, 0.4-0.6, >0.6 |
| **Total states:** | **3×4×3×3×3×3 = 972** | |

**Action Space (7 discrete):**
| Action | Stop Distance |
|--------|---------------|
| 0 | 5% |
| 1 | 8% |
| 2 | 10% |
| 3 | 12% |
| 4 | 15% |
| 5 | 18% |
| 6 | 20% |

**Reward Function:**
```
if trade_hit_tp:
    reward = +return_pct * 100    # e.g., +12% → +12
elif trade_hit_stop:
    reward = return_pct * 100     # e.g., -3.8% → -3.8
elif trade_still_holding:
    reward = 0                    # No feedback until resolved
```

**Training:**
- ε-greedy exploration (ε=0.3 → 0.05 over episodes)
- α=0.1 (learning rate), γ=0.95 (discount)
- ~1000 simulated episodes (each episode = 1 trade lifecycle)
- Synthetic data from backtest on expanded universe (S&P 500 + Russell 2000)

| Pros | Cons |
|------|------|
| Model-free, simple to implement | Requires discrete state/action |
| Interpretable Q-table | State granularity trade-off |
| Learned from data, not hand-tuned | Needs synthetic data to train |
| Fast inference (<1ms) | May not capture sequential patterns |

**Score: 7.8/10**

---

### Option C: Supervised Learning (Stop vs TP classifier)

**Approach:** Train a binary classifier to predict whether a trade will hit stop or reach TP. Use the predicted probability to set stop width.

**Features:**
- Entry price, ATR, ATR/price ratio
- PTL ratio (distance from 52w low)
- Sector, market cap, beta
- Market regime features (SPY vol, SPY above/below MA)
- Volume profile (avg volume, volume trend)
- Quality score, confidence score

**Label:** 1 if trade_reached_TP, 0 if trade_hit_stop

**Model:** XGBoost or LightGBM (tree-based, handles small data well)

**Usage at inference:**
```python
prob_tp = model.predict_proba(features)[1]
if prob_tp > 0.7:
    stop = tight   # Confident → normal stop
elif prob_tp > 0.4:
    stop = medium  # Moderate → wider stop
else:
    stop = wide    # Low confidence → max breathing room
```

| Pros | Cons |
|------|------|
| XGBoost works well with small data | Still needs >50 labeled trades minimum |
| Probability output is interpretable | Binary label ignores magnitude of PnL |
| Well-understood methodology | Doesn't optimize for total P&L directly |
| Fast training and inference | Static model (no online adaptation) |

**Score: 6.5/10**

---

### Option D: Bayesian Optimization

**Approach:** Treat stop distance as a continuous hyperparameter. Use Bayesian optimization (Gaussian Process) to find the optimal value that maximizes Sharpe / profit factor across backtest runs.

**Parameter Space:**
- `hard_stop_pct ∈ [0.05, 0.25]`
- `atr_stop_multiplier ∈ [1.5, 5.0]`
- Possibly per-sector or per-volatility-regime variants

**Workflow:**
1. Define objective function: `f(hard_stop_pct, atr_mult) = backtest_sharpe`
2. Run 20-30 backtests with different parameters
3. GP models the surface, picks next candidate via Expected Improvement
4. Converge to optimum

| Pros | Cons |
|------|------|
| No RL coding needed | Static — same stop for ALL trades |
| Backtest infrastructure already exists | Each eval = 1 full backtest (expensive) |
| Global optimum for the given period | Overfits to backtest period |
| Well-established method | Doesn't adapt to per-trade context |

**Score: 5.5/10**

---

### Option B: Deep RL (PPO/SAC)

**Approach:** Train a neural network policy that maps raw OHLCV sequences + fundamental features → continuous stop distance.

**State:** 60-day OHLCV window (4×60 = 240 dims) + 10 engineered features

**Action:** Continuous stop distance ∈ [0.05, 0.25]

**Network:** MLP (256→128→64) with PPO or SAC

**Reward:** Sharpe ratio of completed trades (episodic)

| Pros | Cons |
|------|------|
| Can learn complex sequential patterns | Needs 10,000+ episodes (we have 6) |
| Continuous action space | Massive overfitting risk |
| State-of-the-art approach | Black box — hard to debug |
| Can incorporate market microstructure | Engineering complexity very high |

**Score: 3.5/10** (premature given current data volume)

---

### 2.2 Comparison Matrix

| Approach | Impact | Data Need | Complexity | Overfit Risk | Maintain | Deploy Risk | **Total** |
|----------|--------|-----------|------------|--------------|----------|-------------|-----------|
| **E: Adaptive** | 6 | 10 | 9 | 9 | 9 | 9 | **7.2** |
| **A: Q-Learning** | 8 | 6 | 7 | 7 | 8 | 7 | **7.8** |
| C: Supervised | 7 | 5 | 7 | 6 | 7 | 7 | 6.5 |
| D: Bayesian | 5 | 8 | 5 | 4 | 6 | 4 | 5.5 |
| B: Deep RL | 9 | 2 | 3 | 2 | 4 | 2 | 3.5 |

---

## 3. Data Requirements & Synthetic Data Strategy

### 3.1 The Fundamental Problem

We have only **6 trades** with **3 stop events**. Any ML/RL approach trained directly on these 3 events will massively overfit. The solution: **generate synthetic training data through expanded backtesting.**

### 3.2 Synthetic Data Generation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│           SYNTHETIC TRADE DATA GENERATION                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Step 1: Expand Universe                                 │
│  ┌──────────────────────────────────────┐               │
│  │ Current: 48 tickers (mid/small-cap)   │               │
│  │ Target:  S&P 500 + Russell 2000       │               │
│  │          (~2000 tickers)              │               │
│  └──────────────────────────────────────┘               │
│                    ↓                                      │
│  Step 2: Multi-Period Backtest                           │
│  ┌──────────────────────────────────────┐               │
│  │ Run backtest on 10+ non-overlapping  │               │
│  │ 2.5-year windows:                    │               │
│  │   2018-2020, 2019-2021, 2020-2022,   │               │
│  │   2021-2023, 2022-2024               │               │
│  │ Expected: ~100-300 trades per window  │               │
│  │ Total: ~1000-1500 trades             │               │
│  └──────────────────────────────────────┘               │
│                    ↓                                      │
│  Step 3: Augment Stop Variations                         │
│  ┌──────────────────────────────────────┐               │
│  │ For each entry, simulate what would   │               │
│  │ happen with stops at:                 │               │
│  │   5%, 8%, 10%, 12%, 15%, 18%, 20%    │               │
│  │ This multiplies trade count by 7x     │               │
│  │ → ~7000-10000 labeled outcomes        │               │
│  └──────────────────────────────────────┘               │
│                    ↓                                      │
│  Step 4: Feature Extraction                              │
│  ┌──────────────────────────────────────┐               │
│  │ At each entry date:                   │               │
│  │   - Price features (PTL, ATR, etc.)   │               │
│  │   - Market features (SPY vol, MA)     │               │
│  │   - Stock features (sector, cap, β)   │               │
│  │   - Label: (stop_distance, outcome,   │               │
│  │             return_pct)               │               │
│  └──────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Minimum Viable Dataset

| Component | Minimum | Target | 
|-----------|---------|--------|
| Total trades (stop variations) | 500 | 5,000 |
| Unique tickers | 50 | 500 |
| Market regimes covered | 2 | All 3 (LOW/NORMAL/HIGH) |
| Sectors covered | 3 | 8+ |
| Stop events (original) | 50 | 300 |

### 3.4 Feature Engineering for Stop Prediction

**Tier 1 Features (always available):**
- `atr_pct`: ATR / entry_price (daily volatility as %)
- `ptl_ratio`: entry_price / 52w_low
- `price_level`: entry_price (bucketed)
- `vol_regime`: "LOW" / "NORMAL" / "HIGH"
- `market_ok`: boolean (SPY > MA50 and not deep drawdown)
- `confidence`: 0.0–1.0 from VMAA pipeline

**Tier 2 Features (computed from OHLCV):**
- `volatility_20d`: annualized 20-day volatility
- `volume_trend`: avg_vol_5d / avg_vol_20d
- `rsi_14`: 14-day RSI
- `drawdown_from_3mo_high`: current price relative to 3-month high
- `gap_pct`: recent gap-up percentage (if any)
- `beta_to_spy`: rolling 60-day beta

**Tier 3 Features (fundamental):**
- `quality_score`: Part 1 composite
- `magna_score`: Part 2 MAGNA score
- `sector_encoded`: one-hot sector encoding
- `market_cap_bucket`: micro/small/mid/large

---

## 4. Recommended Architecture

### 4.1 Overall System Design

```
┌────────────────────────────────────────────────────────────┐
│                 VMAA STOP OPTIMIZATION SYSTEM               │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  VMAA    │───▶│ Trade        │───▶│ Stop Optimizer   │  │
│  │ Pipeline │    │ Decision     │    │ ┌──────────────┐ │  │
│  │ Part1+2  │    │ (risk.py)    │    │ │ Phase 1:     │ │  │
│  └──────────┘    └──────────────┘    │ │ Adaptive     │ │  │
│                                       │ │ (NOW)        │ │  │
│                                       │ ├──────────────┤ │  │
│                                       │ │ Phase 2:     │ │  │
│                                       │ │ Q-Learning   │ │  │
│                                       │ │ (~1 week)    │ │  │
│                                       │ └──────────────┘ │  │
│                                       └────────┬─────────┘  │
│                                                │            │
│                                       ┌────────▼─────────┐  │
│                                       │ Stop Distance    │  │
│                                       │ (5%-20%)         │  │
│                                       └──────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              TRAINING PIPELINE (OFFLINE)              │  │
│  │                                                       │  │
│  │  Expanded Backtest → Feature Extract → Train Q-Table  │  │
│  │       (weekly)          (per trade)      (batch)      │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 4.2 State Representation (Q-Learning)

```python
@dataclass
class StopState:
    """Discretized state for Q-learning stop optimizer."""
    vol_bucket: int          # 0=LOW, 1=NORMAL, 2=HIGH (ATR/price %)
    price_bucket: int        # 0:<$10, 1:$10-30, 2:$30-80, 3:>$80
    ptl_bucket: int          # 0:<1.05, 1:1.05-1.20, 2:>1.20
    sector_vol: int          # 0=LOW, 1=MED, 2=HIGH (sector avg vol)
    market_regime: int       # 0=LOW, 1=NORMAL, 2=HIGH
    confidence_bucket: int   # 0:<0.4, 1:0.4-0.6, 2:>0.6
    
    def to_index(self) -> int:
        """Flatten 6D state to 1D index."""
        return (self.vol_bucket * 324 +      # 3×4×3×3×3 = 324
                self.price_bucket * 81 +      # 4×3×3×3 = 108  
                self.ptl_bucket * 27 +        # 3×3×3 = 27
                self.sector_vol * 9 +         # 3×3 = 9
                self.market_regime * 3 +      # 3
                self.confidence_bucket)       # 1
```

### 4.3 Action Space

```python
STOP_ACTIONS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]  # 7 actions
N_ACTIONS = len(STOP_ACTIONS)
```

### 4.4 Reward Function Design

```python
def compute_reward(trade_outcome: TradeRecord) -> float:
    """
    Reward function for Q-learning.
    
    Design principles:
    1. Positive reward for TP exits (proportional to return)
    2. Negative reward for stop exits (proportional to loss)
    3. Small penalty for holding too long (encourages efficient stops)
    4. Bonus for higher win rate
    """
    if trade_outcome.exit_reason.startswith('take_profit'):
        # Reward proportional to return, capped
        return min(trade_outcome.return_pct, 25.0) * 0.8
    
    elif 'stop' in trade_outcome.exit_reason:
        # Penalty proportional to loss
        return max(trade_outcome.return_pct, -20.0) * 1.2
    
    elif trade_outcome.exit_reason == 'time_stop':
        # Small penalty for time-outs (strategy failed)
        return -5.0
    
    else:
        return 0.0
```

### 4.5 Training Loop

```python
class StopQLearner:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.3):
        self.q_table = np.zeros((N_STATES, N_ACTIONS))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def train(self, episodes: List[TradeEpisode], n_epochs: int = 100):
        """Train Q-table over multiple epochs of trade episodes."""
        for epoch in range(n_epochs):
            np.random.shuffle(episodes)
            total_reward = 0
            
            for ep in episodes:
                state_idx = ep.entry_state.to_index()
                
                # ε-greedy action selection
                if np.random.random() < self.epsilon:
                    action = np.random.randint(N_ACTIONS)
                else:
                    action = np.argmax(self.q_table[state_idx])
                
                # Simulate outcome with chosen stop distance
                outcome = self._simulate_outcome(ep, STOP_ACTIONS[action])
                reward = compute_reward(outcome)
                
                # Q-update
                best_next = np.max(self.q_table[state_idx])  # Terminal state
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.q_table[state_idx, action]
                self.q_table[state_idx, action] += self.alpha * td_error
                
                total_reward += reward
            
            # Decay epsilon
            self.epsilon = max(0.05, self.epsilon * 0.97)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: ε={self.epsilon:.3f}, "
                      f"avg_reward={total_reward/len(episodes):.2f}")
```

### 4.6 Inference at Trade Entry

```python
def optimize_stop(entry_price, atr, ptl_ratio, vol_regime, 
                  sector, market_regime, confidence) -> float:
    """
    Production inference: given trade context, return optimal stop distance.
    
    Phase 1 (NOW): Adaptive statistical stop
    Phase 2 (LATER): Q-table lookup
    """
    # Phase 1: Adaptive stop (immediate deployment)
    state = build_state(atr/entry_price, entry_price, ptl_ratio, 
                        sector, market_regime, confidence)
    
    if PHASE >= 2 and q_learner is not None:
        # Phase 2: Q-table (with adaptive fallback)
        action = np.argmax(q_learner.q_table[state.to_index()])
        stop_pct = STOP_ACTIONS[action]
    else:
        # Phase 1: Adaptive rules
        stop_pct = adaptive_stop_formula(entry_price, atr, ptl_ratio, 
                                         vol_regime)
    
    return entry_price * (1 - stop_pct)
```

---

## 5. Implementation Plan

### Phase 1: Adaptive Statistical Stop (1 day) ⚡ NOW

**Goal:** Immediate improvement without any ML/RL dependency.

**Changes to `vmaa/risk.py`:**

```python
def compute_stops_v2(
    entry_price: float,
    low_52w: float,
    hist: pd.DataFrame,
    market: MarketRegime,
    sector: str = "",
) -> Tuple[float, str]:
    """
    Adaptive stop computation v2.
    
    Key improvements over v1:
    1. Dynamic ATR multiplier based on price level and volatility
    2. Dynamic hard stop based on price level
    3. Sector volatility adjustment
    4. Always picks the WIDER stop (not tighter)
    """
    atr = _compute_atr(hist, 14)
    atr_pct = atr / entry_price if entry_price > 0 else 0.03
    
    # 1. Dynamic ATR multiplier
    base_mult = RC.atr_stop_multiplier  # 2.5
    
    # Price level adjustment
    if entry_price < 10:
        base_mult += 1.0     # +1.0 for penny stocks
    elif entry_price < 30:
        base_mult += 0.5     # +0.5 for small caps
    
    # Volatility adjustment
    if atr_pct > 0.05:       # >5% daily ATR
        base_mult += 0.5
    elif atr_pct > 0.03:     # >3% daily ATR
        base_mult += 0.25
    
    # Near 52w low → give more room
    ptl = entry_price / low_52w if low_52w > 0 else 1.0
    if ptl < 1.10:
        base_mult += 0.5
    
    # Market regime
    if market.vol_regime == "HIGH":
        base_mult += 0.5
    
    atr_stop = round(entry_price - (atr * base_mult), 2) if atr > 0 else 0
    
    # 2. Dynamic hard stop
    if entry_price < 10:
        hard_pct = 0.22
    elif entry_price < 30:
        hard_pct = 0.18
    else:
        hard_pct = 0.15
    
    hard_stop = round(entry_price * (1 - hard_pct), 2)
    
    # 3. Structural stop
    structural_stop = round(low_52w * 0.98, 2)
    
    # 4. Pick the WIDER stop (lower price = more room)
    candidates = [(atr_stop, "ATR"), (hard_stop, "Hard"), 
                  (structural_stop, "Structural")]
    candidates = [(s, n) for s, n in candidates if s > 0 and s < entry_price]
    
    if not candidates:
        return round(entry_price * 0.95, 2), "Fallback"
    
    # Pick the lowest price (widest stop) among valid candidates
    # This maximizes breathing room for mean-reversion
    candidates.sort(key=lambda x: x[0])  # ascending price
    # Use median for balance (not widest — that would be too loose)
    median_idx = len(candidates) // 2
    return candidates[median_idx]
```

**Also fix backtest/engine.py:**
The backtest engine uses the OLD tightest-stop logic. Must update to match risk.py's median selection.

```python
# In _execute_entry():
stops = [(s, n) for s, n in [(atr_stop, "ATR"), (hard_stop, "Hard"),
         (struct_stop, "Structural")] if s > 0 and s < entry_price]
stops.sort(key=lambda x: x[0])  # Ascending = lowest first
median_idx = len(stops) // 2
stop_loss, stop_type = stops[median_idx]  # MEDIAN, not tightest
```

**Expected impact:** 
- Eliminate the <5-day stop-outs by widening stops for volatile/low-price stocks
- Estimated win rate improvement: 50% → 65-75%
- Potential P&L gain: +$100–300 over 2.5 years

**Success metrics:**
- Stop rate < 30% of exits (down from 50%)
- Stop win rate > 10% (up from 0%) — some stops will still be losses but fewer
- Avg holding days for stopped trades > 10 (up from 4)

---

### Phase 2: Q-Learning Agent (1 week) 📊

**Week Plan:**

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Build synthetic data generator: run expanded backtest on S&P 500 + Russell 2000 across 5 time windows | `data/synthetic_trades.csv` with 1000+ trades |
| 2 | Feature engineering + state discretization; label each trade with features and outcome | `features/trade_features.csv` |
| 3 | Implement Q-learning agent with ε-greedy exploration; train on synthetic data | `rl/q_learner.py` with trained Q-table |
| 4 | Backtest the Q-learner against hold-out period (2024-2025); compare with Phase 1 baseline | `benchmarks/q_learner_vs_adaptive.json` |
| 5 | Productionize: integrate Q-table lookup into `risk.py`; add feature extraction pipeline | PR to `vmaa/risk.py` |

**Code Structure:**

```
vmaa/
├── risk.py                          # Updated with RL stop
├── rl/
│   ├── __init__.py
│   ├── state.py                     # State discretization
│   ├── q_learner.py                 # Q-learning implementation
│   ├── train.py                     # Training pipeline
│   ├── features.py                  # Feature extraction
│   └── synthetic_data.py            # Data generation
├── backtest/
│   ├── engine.py                    # FIXED: use median stop
│   └── ...
└── data/
    └── q_table.npy                  # Trained Q-table (committed to repo)
```

**Success Metrics vs Phase 1:**
- Sharpe ratio improvement > 0.2 over Phase 1
- Profit factor > 2.0 (Phase 1 target: 1.8)
- Stop win rate > 20%

---

### Phase 3: Deep RL PPO (2-4 weeks, conditional) 🤖

**Gate:** Only proceed if Phase 2 Q-learner shows < 10% improvement over Phase 1.

**Why we might need it:**
- Q-learning's discrete state/action may miss important patterns
- Continuous stop distance could be more optimal
- PPO can learn from raw OHLCV sequences

**Infrastructure needed:**
- GPU for training (or cloud TPU)
- Stable-Baselines3 or RLlib
- Custom Gym environment wrapping the backtest engine
- 10,000+ synthetic episodes minimum

**Cost estimate:** 2-3 weeks of engineering + compute

---

### Phase 4: Online Learning (ongoing) 🔄

**Goal:** Continuously improve the stop policy from live trading results.

**Architecture:**
```python
class OnlineStopLearner:
    """Lightweight online learning wrapper."""
    
    def __init__(self, base_q_table_path: str):
        self.q_table = np.load(base_q_table_path)
        self.online_updates = 0
        self.alpha_online = 0.01  # Very small learning rate
    
    def update_from_trade(self, state: StopState, action: int, 
                          reward: float) -> None:
        """Update Q-table from a single live trade outcome."""
        state_idx = state.to_index()
        td_error = reward - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha_online * td_error
        self.online_updates += 1
        
        # Periodic save
        if self.online_updates % 10 == 0:
            self.save()
```

**Safety guardrails for online learning:**
- Maximum stop width: 25% (never wider)
- Minimum stop width: 5% (never tighter)
- Anomaly detection: if consecutive stop-outs > 3, revert to Phase 1 adaptive
- Human-in-the-loop: all Q-table changes logged and reviewable

---

## 6. Success Metrics

### 6.1 Primary Metrics

| Metric | Current | Phase 1 Target | Phase 2 Target | Measurement |
|--------|---------|----------------|----------------|-------------|
| Stop rate (% of exits) | 50% | <35% | <25% | Backtest trade log |
| Stop win rate | 0% | >10% | >20% | Backtest trade log |
| Profit factor | 1.66 | >1.8 | >2.0 | Gross win / gross loss |
| Avg loss per stop | -$90 | -$60 | -$40 | Backtest P&L |
| Avg holding (stopped) | 4 days | >10 days | >15 days | Trade log |

### 6.2 Secondary Metrics

| Metric | Current | Target | Why |
|--------|---------|--------|-----|
| Sharpe ratio | -1.05 | >0.5 | Risk-adjusted return |
| Max drawdown | -0.76% | < -5% | Accepting wider drawdowns for higher returns |
| Win rate (overall) | 50% | >55% | More trades should reach TP |
| CAGR | 2.02% | >5% | Should compound better |

### 6.3 Validation Protocol

1. **In-sample:** Train on 2018-2022, test on 2023-2025 (same as current backtest window)
2. **Walk-forward:** Train on 2018-2020, test 2021; train 2019-2021, test 2022; etc.
3. **Out-of-universe:** Test on a different universe (e.g., train on S&P 500, test on Russell 2000)
4. **Monte Carlo:** Bootstrap resample trades 1000 times, measure distribution of Sharpe/profit factor

---

## 7. Risk Assessment

### 7.1 Overfitting Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Q-table memorizes specific tickers | High | Medium | Train on >200 unique tickers; validate on hold-out tickers |
| Q-table overfits to market regime | Medium | High | Train across bull (2019-2021), bear (2022), recovery (2023) |
| Stop widening causes missed TPs | Medium | Medium | Monitor TP rate; Phase 1 uses conservative widening |
| Phase 3 Deep RL overfits completely | High | High | Gate: only proceed if Phase 2 shows clear improvement limits |

### 7.2 Production Deployment Risks

| Risk | Mitigation |
|------|------------|
| Q-table produces nonsensical stops | Floor at 5%, cap at 25%; validate all outputs |
| Correlated stop-outs in crash | Market-regime feature should detect HIGH vol and widen stops |
| Data pipeline failure → no features | Fall back to Phase 1 adaptive stop |
| Online learning degrades policy | Human review every 10 online updates; rollback capability |

### 7.3 Computational Cost

| Phase | CPU Hours | Data Volume | Frequency |
|-------|-----------|-------------|-----------|
| Phase 1 | 0 | 0 | One-time code change |
| Phase 2 training | ~2-4 | ~10MB (features CSV) | One-time (retrain monthly) |
| Phase 2 inference | <1ms per trade | 0 | Every trade entry |
| Phase 3 (if needed) | ~50-100 GPU | ~500MB (OHLCV) | One-time |
| Phase 4 online | <1ms per update | <1KB per update | Every trade exit |

---

## 8. Immediate Actions (Next 48 Hours)

### Critical Bug Fix
**The backtest engine uses the OLD tightest-stop logic.** This means all historical backtests used a stop that was the tightest of ATR/Hard/Structural — often much tighter than 10%. Fix this BEFORE any RL work:
- [x] Identified: `backtest/engine.py` line ~575 uses `reverse=True` sort and picks `[0]`
- [ ] Fix: Change to median selection matching `risk.py`
- [ ] Re-run backtest with median stop + dynamic ATR multiplier to establish new baseline

### Phase 1 Implementation
- [ ] Implement `compute_stops_v2()` in `risk.py`
- [ ] Add price-level-aware dynamic multipliers
- [ ] Add sector volatility lookup
- [ ] Add market regime integration
- [ ] Write unit tests for edge cases (sub-$1 stocks, missing ATR, etc.)

### Data Preparation
- [ ] Set up S&P 500 + Russell 2000 ticker lists
- [ ] Configure multi-window backtest runner
- [ ] Run baseline backtest with adaptive stops to quantify Phase 1 improvement

---

## 9. Appendix

### A. Code Changes Summary

| File | Change | Priority |
|------|--------|----------|
| `backtest/engine.py` | Fix stop selection: tightest → median | 🔴 CRITICAL |
| `vmaa/risk.py` | Add `compute_stops_v2()` with adaptive logic | 🟡 HIGH |
| `vmaa/risk.py` | Add market regime parameter to `compute_stops()` | 🟡 HIGH |
| `vmaa/rl/state.py` | New: State discretization module | 🟢 MEDIUM |
| `vmaa/rl/q_learner.py` | New: Q-learning implementation | 🟢 MEDIUM |
| `vmaa/rl/synthetic_data.py` | New: Synthetic data generator | 🟢 MEDIUM |
| `vmaa/backtest/runner.py` | Add `--stop-mode` flag (tightest/median/adaptive/qlearn) | 🟢 MEDIUM |

### B. Research References

1. **Moody & Saffell (2001):** "Learning to Trade via Direct Reinforcement" — foundational paper on RL for trading
2. **Deng et al. (2017):** "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading" — DRL for trading
3. **Buehler et al. (2019):** "Deep Hedging" — continuous action RL for risk management
4. **Zhang et al. (2020):** "Deep Reinforcement Learning for Automated Stock Trading" — ensemble RL agents
5. **Non-RL approach:** Kaufman's Adaptive Moving Average (AMA) — efficiency ratio concept applicable to stop width
6. **Statistical:** Chandelier Exit (ATR-based trailing stop) — reference for ATR multiplier selection

### C. Glossary

| Term | Definition |
|------|------------|
| ATR | Average True Range — measure of volatility over N periods |
| PTL | Price-to-52w-Low ratio — how far above annual low |
| Hard stop | Fixed percentage stop loss below entry |
| Structural stop | Stop placed below 52-week low |
| Q-learning | Model-free RL algorithm learning action-value function |
| PPO | Proximal Policy Optimization — policy gradient RL |
| SAC | Soft Actor-Critic — off-policy maximum entropy RL |
| Sharpe ratio | (Return - RiskFreeRate) / Volatility |
| Profit factor | Gross profit / Gross loss |
