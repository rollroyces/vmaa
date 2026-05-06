# VMAA Screening Threshold Optimisation Report
**Date:** 2026-05-05 | **Engineer:** Ironman 🦾 | **Context:** 2026 Bull Market

---

## Executive Summary

VMAA 兩階段篩選產出 24/91 Part 1 通過、24 MAGNA signals、11 entry-ready，但 **0 trades executed**。根源有二：(1) Part 1 thresholds 在 bull market 過緊，特別係 PTL 同 B/M；(2) Kelly position sizing 公式過度保守，confidence < 67% 即歸零。以下詳細分析 + 具體改善建議。

---

## 1. Part 1 Screening Breakdown

### 1.1 Per-Criterion Pass Rates (70-stock sample)

| Criterion | Threshold | Pass Rate | Median Value | Assessment |
|---|---|---|---|---|
| **PTL ≤ 1.30** | Price/52w-low ≤ 1.30x | **40%** (28/70) | 1.36x | 🔴 **Biggest killer** — bull market pushes prices well above 52w lows |
| **B/M ≥ 0.30** | Book/Market ≥ 0.30 | **41%** (29/70) | 0.24 | 🔴 **2nd killer** — tech/growth stocks are asset-light |
| **FCF/Y ≥ 3%** | FCF Yield ≥ 3% | **53%** (37/70) | 3.5% | 🟡 Median passes, lower quartile fails |
| **ROA ≥ 0%** | ROA ≥ 0% | **61%** (43/70) | — | 🟢 Most profitable stocks pass |
| **ALL 4 combined** | Hard pass on all | **16%** (11/70) | — | 🔴 Way too restrictive for 2026 |

> **Note:** Actual scan uses *weighted scoring* (min_quality_score = 0.40), not hard pass/fail on each criterion — hence 26.4% pass rate vs 16% hard-pass. But the scoring still disfavours stocks failing multiple criteria.

### 1.2 What's Killing Stocks (per-ticker detail)

**Tech/Growth killed by PTL + B/M:**
- `DDOG` (MCap $12B): PTL=1.62, B/M=0.27, ROA=-4.3%
- `MDB` (MCap $10B): PTL=1.41, B/M=0.09
- `ZS`, `NET`, `CRWD`, `OKTA`, `WDAY`, `HUBS`: All PTL > 1.30

**Crypto miners killed by PTL + negative FCF:**
- `CLSK`: PTL=1.69, FCF/Y=-9.1%, ROA=-12.6%
- `RIOT`: PTL=2.56, FCF/Y=-6.0%, ROA=-12.1%
- `IREN`: PTL=8.27, B/M=0.15, FCF/Y=-7.6%

**Energy/Materials killed by PTL + FCF:**
- `FCX` ($81B): PTL=1.61, B/M=0.25, FCF/Y=1.7% (also MCap > $10B!)
- `STLD` ($34B): PTL=1.95, B/M=0.26, FCF/Y=-0.1% (MCap > $10B!)
- `CRS`, `ATI`, `RS`, `CMC`: All PTL > 1.40

**Near-miss (marginal fails):**
- `RRC`: PTL=1.31 → **just 0.01 above threshold**
- `AR`: PTL=1.33 → **just 0.03 above**
- `UPST`: PTL=1.33, B/M=0.25 → close on both

---

## 2. Threshold Sensitivity Analysis

### 2.1 What-If Scenarios (68 stocks with valid data)

| Scenario | Pass Count | Pass Rate | Δ vs Current |
|---|---|---|---|
| **Current** (PTL≤1.3, B/M≥0.3, FCF/Y≥3%, ROA≥0%) | 8 | 12% | — |
| PTL 1.40 | 11 | 16% | +38% |
| PTL 1.50 | 12 | 18% | +50% |
| B/M 0.20 | 11 | 16% | +38% |
| B/M 0.15 | 12 | 18% | +50% |
| FCF/Y 2% | 9 | 13% | +13% |
| PTL 1.40 + B/M 0.20 | 14 | 21% | +75% |
| **PTL 1.40 + B/M 0.20 + FCF/Y 2%** | **16** | **24%** | **+100%** |
| PTL 1.50 + B/M 0.15 | 17 | 25% | +113% |
| PTL 1.50 + B/M 0.15 + FCF/Y 2% | 20 | 29% | +150% |

### 2.2 Recommendation: Tiered Thresholds by Market Regime

Rather than one-size-fits-all, VMAA should adjust thresholds based on market conditions:

| Parameter | Bear/Correction | Normal | **Bull (2026 Recommended)** |
|---|---|---|---|
| `max_ptl_ratio` | 1.20 | 1.30 | **1.50** |
| `min_bm_ratio` | 0.35 | 0.30 | **0.20** |
| `min_fcf_yield` | 0.04 | 0.03 | **0.02** |
| `min_quality_score` | 0.45 | 0.40 | **0.35** |

**Bull market rationale:** Prices are generally elevated → PTL should widen. Growth stocks dominate → B/M should relax. The quality floor drops slightly to let more candidates through to Part 2 (MAGNA filters them anyway).

Expected impact: **Pass rate ~24-29%** (up from 26% currently, but with a substantially larger, more diverse universe).

---

## 3. Ticker Universe Cleanup

### 3.1 Confirmed Delisted/Merged (REMOVE immediately)

```
NVRO, SMAR, AYX, X, CHK, MRO, NOVA, VERV
```

- `X` = U.S. Steel — acquired by Nippon Steel, delisted
- `CHK` = Chesapeake Energy — merged with Southwestern, ticker changed
- `MRO` = Marathon Oil — acquired by ConocoPhillips
- Others: delisted or reverse-split into oblivion

### 3.2 Stocks That Should Be Removed from Universe

| Ticker | Reason |
|---|---|
| `FCX` | MCap $81B → exceeds $10B cap |
| `SCCO` | MCap $142B → exceeds $10B cap |
| `STLD` | MCap $34B → exceeds $10B cap |
| `RS` | MCap $19B → exceeds $10B cap |
| `ATI` | MCap $21B → exceeds $10B cap |
| `CRS` | MCap $22B → exceeds $10B cap |
| `RKLB` | MCap $45B → exceeds $10B cap |
| `SOFI` | MCap $21B → exceeds $10B cap |
| `DKNG` | MCap $12B → exceeds $10B cap |

> These mega/mid-caps shouldn't be in the universe to begin with → they fail Part 1's `turnaround_max_cap = $10B` anyway. Removing them from the universe saves API calls.

### 3.3 Stocks to Add to Universe

**Quality mid-caps in current bull regime:**

| Sector | Suggested Tickers |
|---|---|
| Tech turnaround | `SMAR` replacement → `TEAM`, `ZS` (if PTL relaxed) |
| Healthcare | `TXG`, `VCYT`, `NARI` |
| Fintech | `MQ`, `FLYW` |
| Energy (new) | `AR`, `RRC`, `CTRA` (near 52w-lows, good FCF) |
| Industrial | `WOR` replacement → `GTLS`, `FSS` |

---

## 4. Position Sizing Root Cause: Why 0 Executed

### 4.1 The Kelly Problem

Current sizing formula in `risk.py`:

```python
win_prob = 0.50 * confidence          # ← Problem #1: halves confidence
payout_ratio = 2.0                     # ← Problem #2: assumes 2:1 R:R
kelly = (2.0 * win_prob - (1 - win_prob)) / 2.0
kelly = max(0, min(kelly, 0.25))
risk_capital = portfolio_value * kelly * 0.25  # ← Problem #3: quarter-Kelly on already conservative Kelly
```

**Math trace for confidence = 60%:**
- win_prob = 0.50 × 0.60 = 0.30
- Kelly = (2.0 × 0.30 − 0.70) / 2.0 = (0.60 − 0.70) / 2.0 = **−0.05 → 0**
- risk_capital = $0
- quantity = 0 → position_value = $price × 1 < $500 → **quantity → 0**

**Breakeven confidence for positive Kelly at 2:1 payout:**
- Need wp > 1/3 ≈ 33.3%
- wp = 0.50 × confidence → **confidence > 66.7%**

Only ONE candidate (TMDX at 68%) cleared this bar.

### 4.2 The MIN_POSITION Filter

Even when Kelly is barely positive:
- `RC.min_position_size = $500`
- Low-price stocks (INMD @ $14.45, CXM @ $5.43, FVRR @ $10.21) produce tiny position values
- Quantity = 1 → value < $500 → quantity → 0

### 4.3 Fix: Three Changes

**Change 1: Remove confidence halving**

```python
# Current:
win_prob = 0.50 * confidence
# Fix:
win_prob = 0.50 + (confidence - 0.50) * 0.5  # Center around 50%, scale up
# OR simply:
win_prob = max(0.45, confidence)  # Floor at 45% for filtered candidates
```

This ensures confidence ≥ 50% always produces positive Kelly.

**Change 2: Reduce payout_ratio to reflect actual stops**

```python
# Current (oversimplified):
payout_ratio = 2.0
# Fix (from actual TP levels):
payout_ratio = actual_reward / actual_risk  # Typically 1.5-3.0 from compute_take_profits
```

**Change 3: Reduce or remove kelly_fraction double-counting**

```python
# Current:
risk_capital = portfolio_value * kelly * RC.kelly_fraction  # 0.25 × already conservative Kelly
# Fix:
risk_capital = portfolio_value * max(kelly, 0.02)  # Min 2% risk capital per trade
```

### 4.4 Alternative: Replace Kelly with Fixed Fractional

For a strategy that already uses composite confidence scoring:

```python
# Simpler, more predictable:
base_risk_pct = 0.02   # 2% of portfolio at risk
conf_adj = confidence / 0.60  # Scale: 60% confidence = 1x, 80% = 1.33x
risk_capital = portfolio_value * base_risk_pct * min(conf_adj, 1.50)
```

This guarantees a minimum position for any candidate with confidence ≥ 35%.

---

## 5. Specific Code Changes

### 5.1 `config.py` — Bull Market Thresholds

```python
# Add market-regime-aware thresholds
@dataclass  
class MarketAdjustedConfig:
    """Adjust thresholds based on SPY vs 50MA position."""
    
    @staticmethod
    def get_config(market_regime: str) -> Part1Config:
        if market_regime == "BULL":
            return Part1Config(
                max_ptl_ratio=1.50,
                min_bm_ratio=0.20,
                min_fcf_yield=0.02,
                min_quality_score=0.35,
                # ... rest defaults
            )
        elif market_regime == "BEAR":
            return Part1Config(
                max_ptl_ratio=1.20,
                min_bm_ratio=0.35,
                min_fcf_yield=0.04,
                min_quality_score=0.45,
            )
        else:  # NORMAL
            return Part1Config()  # Current defaults
```

### 5.2 `risk.py` — Fixed Position Sizing

```python
def compute_position_size(...):
    # Replace Kelly with fixed fractional
    risk_per_share = max(entry_price - stop_loss, entry_price * 0.03)
    base_risk = portfolio_value * 0.015  # 1.5% base risk
    
    conf_mult = confidence / 0.55  # 55% confidence = 1x
    conf_mult = max(0.5, min(conf_mult, 2.0))
    
    risk_capital = base_risk * conf_mult * market.position_scalar
    quantity = int(risk_capital / risk_per_share)
    
    # Cap position at 20% of portfolio
    max_qty = int(portfolio_value * RC.max_position_pct / entry_price)
    quantity = min(quantity, max_qty, 5000)
    quantity = max(1, quantity)
    
    # Min check: allow as low as $200 in paper trading
    min_order = 200 if paper_trading else 500
    if quantity * entry_price < min_order:
        quantity = max(1, int(min_order / entry_price))
    
    return quantity, ...
```

### 5.3 Universe Cleanup Script

Add to `pipeline.py` or a separate `universe.py`:

```python
DELISTED_TICKERS = {'NVRO', 'SMAR', 'AYX', 'X', 'CHK', 'MRO', 'NOVA', 'VERV'}
MEGA_CAP_EXCEEDING_10B = {'FCX', 'SCCO', 'STLD', 'RS', 'ATI', 'CRS', 'RKLB', 'SOFI', 'DKNG'}
```

---

## 6. Impact Estimates

| Metric | Current | After Changes | Δ |
|---|---|---|---|
| Part 1 pass rate | 26% (24/91) | **30-35%** (27-32/91) | +15-35% |
| Usable universe (after cleanup) | 91 → ~77 valid | **~85** (adding new tickers) | +10% |
| Candidates reaching entry | 7-11 | **12-18** | +50-70% |
| Executed trades | 0 | **3-6** per scan | ∞ |
| Average position size | $0 | **$800-$3,000** | — |

---

## 7. Priority Actions (ordered)

1. 🔴 **CRITICAL:** Fix position sizing in `risk.py` — replace Kelly with fixed fractional or fix the win_prob formula. This is the #1 reason for 0 trades.

2. 🔴 **HIGH:** Remove 8 delisted tickers from universe → saves API calls + avoids errors.

3. 🟡 **MEDIUM:** Implement bull-market threshold relaxation (PTL→1.50, B/M→0.20). Unlocks 50% more candidates.

4. 🟡 **MEDIUM:** Remove 9 mega-cap stocks (>$10B) that auto-fail Part 1 → saves ~10 API calls per scan.

5. 🟢 **LOW:** Add 10-15 new tickers (energy, healthcare, fintech) to maintain universe diversity after cleanup.

6. 🟢 **LOW:** Implement market-regime-aware config switching in `config.py`.

---

## 8. Validation

To validate these changes, run:

```bash
# Test the fixed position sizing
cd vmaa && python3 -c "
from risk import compute_position_size
from models import MarketRegime
m = MarketRegime(spy_price=720, spy_ma50=700, above_ma50=True, 
                 volatility_20d=0.15, vol_regime='NORMAL', 
                 dd_from_3mo_high=-0.02, market_ok=True, position_scalar=0.80)
qty, pct, risk = compute_position_size('TEST', 50.0, 47.50, 100000, 0.55, m)
print(f'Qty: {qty}, Position: {pct}%, Risk: \${risk}')
# Should print non-zero quantity!
"

# Test updated thresholds
python3 pipeline.py --full-scan --dry-run --tickers ICUI,FIVN,YELP,SPSC,NCNO,FVRR,RNG,INMD,WOR,MNDY,DDOG,MDB,ZS,NET,CRWD,OKTA,TENB,CFLT,WDAY,HUBS,PATH,PCOR,UPST,SOFI,CHWY,CLSK,RIOT,RRC,AR,OVV,IONS,SRPT,TWST
```

---

*Report generated by VMAA Strategy Optimisation Subagent 🦾*
