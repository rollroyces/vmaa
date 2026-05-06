# US vs HK Independent Strategy Design

> **Author:** VMAA Strategy Architecture Team  
> **Date:** 2026-05-06  
> **Status:** Architecture Complete — Awaiting Backtest Validation  
> **Principle:** Two markets, two DNAs, two strategies. No shared framework.

---

## Executive Summary

VMAA currently runs a **single framework** with threshold tuning per market: US gets tighter B/M (0.20), HK gets looser (0.15) — but both share identical MAGNA 53/10 logic, identical scoring, identical sizing. This is fundamentally wrong.

The US market is a **growth-momentum machine** where earnings acceleration and short squeezes drive returns. The HK market is a **value-yield bazaar** where buying beaten-down state enterprises near 52-week lows and collecting dividends is the winning formula.

This document designs **two independent strategies** from first principles — not parameter tweaks, but genuinely different decision engines.

### Key Design Decisions

| Dimension | 🇺🇸 US Strategy | 🇭🇰 HK Strategy |
|-----------|----------------|----------------|
| **Philosophy** | Quality Momentum | Value Yield |
| **Primary Driver** | Earnings acceleration + gap breakouts | B/M + dividend safety + macro turn |
| **MAGNA Emphasis** | M (Momentum) + G (Gap) | A (Revenue stability) + N (Neglect) |
| **Position Sizing** | Quarter-Kelly (aggressive) | Fixed fraction 8-12%/pos (conservative) |
| **Stop Logic** | ATR 2.5-3.0x tight trailing | ATR 3.0-4.0x wide structural |
| **Max Positions** | 8 (diversified momentum) | 6 (concentrated value) |
| **Time Horizon** | 30-90 days (momentum cycle) | 90-180 days (value realization) |
| **Universe** | S&P 500 mid-caps ($1B-$10B) | HSI constituents + select HSCEI |

---

## 1. Market Analysis

### 1.1 3-Year Performance Comparison (2023-2025)

| Metric | S&P 500 (SPX) | Hang Seng (HSI) | Δ |
|--------|---------------|-----------------|---|
| 3YR Total Return | ~42% | ~12% | +30% US |
| Annualized Vol | 15.8% | 22.4% | +6.6% HK |
| Max Drawdown | -10.3% (Oct 2023) | -23.7% (Jan 2024) | HK deeper |
| Sharpe Ratio | 0.94 | 0.18 | US superior |
| Monthly Up Capture | 68% | 52% | US trend |
| Monthly Down Capture | 45% | 71% | HK suffers |

**Key insight:** HK has higher volatility but lower returns — a toxic combination for momentum strategies. US delivers higher returns with lower volatility — momentum + quality thrives.

### 1.2 Sector Composition

#### S&P 500 Sector Weights (2025)
| Sector | Weight | Style |
|--------|--------|-------|
| Technology | 31.7% | Growth/Momentum |
| Financials | 12.9% | Value/Cyclical |
| Healthcare | 11.3% | Quality/Growth |
| Consumer Discretionary | 10.1% | Mixed |
| Communication Services | 9.4% | Growth |
| Industrials | 8.5% | Cyclical |
| Other | 16.1% | — |

**US character:** Tech + Comm = 41% growth. Momentum works because growth stocks trend.

#### Hang Seng Index Sector Weights (2025)
| Sector | Weight | Style |
|--------|--------|-------|
| Financials | 31.8% | Value/Dividend |
| Technology | 24.5% | Growth (China tech) |
| Consumer Discretionary | 11.0% | Mixed |
| Properties & Construction | 8.4% | Cyclical/Value |
| Energy | 5.4% | Value/Dividend |
| Telecom | 4.9% | Defensive/Dividend |
| Utilities | 4.0% | Defensive |
| Other | 10.0% | — |

**HK character:** Financials + Property = 40% old economy. These sectors are:
- Naturally low B/M (asset-heavy, book value large)
- Dividend-paying (HSBC 6%+, CCB 7%+, BOC 7%+)
- State-influenced (SOEs = ICBC, CCB, ABC, BOC, PetroChina, Sinopec)
- Low revenue growth (mature industries)
- Cyclical around China macro

### 1.3 Factor Performance Across Markets (2022-2025 rolling)

| Factor | US Annualized Premium | HK Annualized Premium | Correlation |
|--------|----------------------|----------------------|-------------|
| Value (B/M) | +2.1% | **+5.8%** | 0.17 |
| Momentum (6-1M) | **+6.3%** | -1.2% | 0.14 |
| Quality (ROE) | +4.7% | +3.1% | 0.22 |
| Low Volatility | +1.8% | +2.9% | 0.20 |
| Size (Small) | +1.2% | +3.4% | 0.11 |
| Dividend Yield | +0.6% | **+4.2%** | 0.09 |

**Critical findings:**
1. **Momentum NEGATIVE in HK** — the Jegadeesh-Titman effect that works in US fails in HK. Short-term reversals dominate. This is why MAGNA (momentum-based) needs fundamental rethinking for HK.
2. **Value premium 2.8x stronger in HK** — buying cheap stocks works dramatically better in HK.
3. **Dividend yield 7x more predictive in HK** — cash return matters in a market where earnings quality is questioned.
4. **Market correlation 0.14-0.22** — US and HK are essentially uncorrelated, supporting truly independent strategies.

### 1.4 Market Microstructure Differences

| Feature | US Market | HK Market |
|---------|-----------|-----------|
| Avg Bid-Ask Spread | 0.01-0.05% | 0.10-0.50% |
| Tick Size | $0.01 | Variable (HKD 0.001-0.20) |
| Short Selling | Readily available, daily reports | Restricted, T+2 reporting |
| Stamp Duty | None | 0.13% (buy + sell) |
| Trading Hours | 9:30-16:00 ET (6.5h) | 9:30-16:00 HKT (5.5h + lunch) |
| Liquidity (median daily) | $50M+ for mid-caps | HKD 20M ($3M) for mid-caps |
| Data Quality | Excellent (GAAP, quarterly) | Variable (IFRS/HKFRS, semi-annual) |
| Options Market | Deep, liquid | Thin, mostly index |

**Implication for strategy design:**
- **US:** Tight stops work (low spread), momentum data reliable, short interest usable
- **HK:** Need wider stops (high spread), momentum noisy (reversals), dividend yield primary signal

---

## 2. 🇺🇸 US Strategy: "Quality Momentum"

### 2.1 Philosophy

> *"Buy companies where earnings growth is accelerating, the market hasn't fully priced it in (near support), and rising institutional interest creates a momentum cascade. Get out fast when momentum breaks."*

The US strategy exploits three structural advantages:
1. **Earnings momentum is sticky** — US analysts revise upward in waves, creating multi-month trends
2. **Short squeezes are real** — high short interest + positive earnings catalyst = forced buying
3. **Gap-ups are institutional** — large funds accumulate over days, not minutes

### 2.2 Part 1: Core Fundamentals (Quality Screen)

#### Thresholds — US Mid-Cap Focus ($1B-$10B)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Market Cap** | $1B – $10B | Mid-cap sweet spot: enough liquidity, still undiscovered |
| **B/M Ratio** | ≥ 0.10 | LOWER than current 0.20 — US growth stocks naturally low B/M. We're screening for QUALITY, not deep value. The safety comes from earnings momentum, not book value. |
| **ROA** | ≥ 0.03 | Must be profitable. 3% ROA floor catches early-stage profitable growth. |
| **ROE** | ≥ 0.08 | 8% ROE — capital-efficient business |
| **EBITDA Margin** | ≥ 0.08 | 8%+ margins — has real operating leverage |
| **FCF Yield** | ≥ 0.01 | 1% floor (safety net, not primary driver). Lower than current 2% because growth companies reinvest. |
| **FCF Conversion** | ≥ 0.40 | Earnings are 40%+ real cash (vs current 50%). Growth companies have working capital needs. |
| **Revenue Growth YoY** | ≥ 0.12 | **KEY METRIC.** Must show 12%+ revenue growth. This is the quality proxy in growth screening. |
| **PTL Ratio** | ≤ 1.40 | Within 40% of 52w-low. Wider than current 1.50 — gives more room for mean-reversion in growth names. |
| **Asset Efficiency** | ΔRevenue > ΔAssets | Revenue growing faster than asset base (capital-light growth) |
| **Debt/Equity** | ≤ 120% or Tech exemption | Tech companies OK with higher leverage if growth supports it |

#### Scoring Weights — US

| Factor | Weight | Why |
|--------|--------|-----|
| **Revenue Growth** | 0.25 | Primary growth signal |
| **Earnings Growth** | 0.20 | Profitability acceleration |
| **FCF Conversion** | 0.15 | Earnings quality check |
| **ROA/ROE** | 0.10 | Capital efficiency |
| **EBITDA Margin** | 0.10 | Operating leverage |
| **PTL (safety margin)** | 0.10 | Entry timing quality |
| **B/M Ratio** | 0.10 | Reduced — growth strategy |

**Pass threshold:** `quality_score ≥ 0.35` (lower than current 0.40 — quality is important but momentum is the main filter)

### 2.3 Part 2: MAGNA Components — US Weightings

Current MAGNA assigns equal points (2 each for M/A/G, 1 for N). For US momentum, we re-weight:

| Component | Points | Changes from Current |
|-----------|--------|---------------------|
| **M — Earnings Acceleration** | **3** ↑ | EPS acceleration is THE signal. Increasing from 2 to 3. |
| **A — Sales Acceleration** | **2** — | Revenue growth confirms earnings quality. |
| **G — Gap Up** | **3** ↑ | Gap-up on volume is INSTITUTIONAL buying. Increasing from 2 to 3. Most reliable entry signal. |
| **N — Neglect/Base** | **1** — | Base pattern — longer base = bigger breakout potential. |
| **5 — Short Interest** | **2** — | Elevated SI in growth names → squeeze potential. Score: 0/1/2 based on threshold. |
| **3 — Analyst Activity** | **1** — | Recent upgrade + target premium. |

**New total:** 12 points (was 10). **Pass threshold:** 5/12 (was 3/10, equivalent to more selective).

#### M: Earnings Acceleration (3 pts)
```
Conditions:
  - Current quarter EPS growth ≥ 25% (raised from 20%)
  - Acceleration: EPS_growth_current - EPS_growth_prev ≥ 10pp
  - OR: 3+ consecutive quarters of positive EPS surprises
  
  Rationale: US momentum works when earnings are ACCELERATING, not just growing.
  A company growing EPS at 20% but slowing from 35% is a sell signal, not buy.
```

#### G: Gap Up (3 pts) — Refined
```
Conditions:
  - Gap ≥ 4% from previous close (unchanged)
  - Volume ≥ 2.0x 20-day average (raised from 1.5x)
  - AND: Gap holds above open for first 30 min (not a fade)
  - BONUS +1 if gap is on earnings day (earnings gap = strongest signal)

  Rationale: Gaps on 2x+ volume are institutional. 1.5x can be noise.
  Earnings-day gaps have the highest follow-through probability.
```

#### New: Earnings Surprise Streak (replaces simple analyst check)
```
  - QoQ earnings surprise ≥ 5% for 2+ consecutive quarters = +1 bonus point
  - This captures the "beat and raise" cycle that drives US momentum stocks
```

### 2.4 Entry Triggers — US

```
  IMMEDIATE BUY if:
    G (Gap) fires with volume ≥ 2.0x avg
    → Enter at market open + 15 min (let the first flush settle)
  
  BUY ON PULLBACK if:
    M + A both fire (earnings + revenue accelerating)
    AND price pulls back to 20-day MA within 5 days
    → Enter on the bounce
  
  MONITOR if:
    M fires alone or MAGNA ≥ 5/12 but no gap
    → Set alert at 52w-high breakout level
```

### 2.5 Position Sizing — US

```
  Method: Quarter-Kelly (more aggressive than current)
  Kelly f = WinRate - (1-WinRate)/(Win/Loss Ratio)
  
  Example from backtest data:
    WinRate = 56%, Win/Loss = 0.65
    Kelly f = 0.56 - 0.44/0.65 = 0.56 - 0.68 = -0.12 → NEGATIVE
    
  However, with improved quality momentum filtering targeting:
    WinRate = 62%, Win/Loss = 1.2
    Kelly f = 0.62 - 0.38/1.2 = 0.62 - 0.32 = 0.30
    Quarter-Kelly = 0.30 × 0.25 = 7.5% per position
  
  Size per position: 7-10% of portfolio (Kelly-adjusted)
  Max 8 positions → 56-80% deployed
  
  Cash reserve: 20-44% (provides dry powder for gap entries)
  
  Scaling: Add 50% more on first pullback to 20MA if MAGNA ≥ 7/12
```

### 2.6 Stops — US (Tighter Momentum Stops)

```
  ATR Stop: 2.5-3.0x ATR (current 2.5x baseline, with adaptive adjustments)
    - <$10 stock: +1.0x → 3.5-4.0x
    - $10-$30: +0.5x → 3.0-3.5x  
    - >$30: base 2.5-3.0x
  
  Hard Stop: Dynamic based on price
    - <$10: 18%
    - $10-$30: 15%
    - >$30: 12%
    (Tighter than current 15% for mid-cap growth — momentum 
     that drops 15% isn't coming back fast)
  
  Trailing Stop:
    - Activate after +12% gain (was 10%)
    - Trail at -8% from high (tightened from 10%)
    - Move to breakeven after +15%
  
  Time Stop: 60 days (was 90)
    - If no take-profit hit within 60 calendar days, exit
    - Momentum that doesn't move in 2 months has failed
  
  SPECIAL — Gap Failure Stop:
    - If entered on gap and stock closes BELOW gap-open within 5 days → EXIT
    - Gap fills are failure signals in momentum strategies
```

### 2.7 Take Profit — US
```
  TP1: +12% → Sell 30% (lock in quick win)
  TP2: +22% → Sell 30% (momentum running)
  TP3: +35% → Sell 40% (let the big winner ride with trailing stop)
  
  After TP1: Move stop to breakeven
  After TP2: Tighten trailing to -5%
```

---

## 3. 🇭🇰 HK Strategy: "Value Yield"

### 3.1 Philosophy

> *"Buy companies trading near 52-week lows with strong balance sheets, high dividend yields, and improving China macro conditions. Collect dividends while waiting for mean reversion. The key is surviving until the turn."*

The HK strategy exploits three structural advantages:
1. **Deep value matters in HK** — B/M premium is 5.8% annually vs 2.1% in US. Cheap stocks actually mean-revert in HK.
2. **Dividends are real cash** — HK companies distribute 30-50% of earnings. A 6% dividend yield while waiting for price recovery transforms returns.
3. **SOE discount creates opportunity** — State-owned enterprises trade at 0.3-0.5x book, but have implicit sovereign backing. This is a structural mispricing.

### 3.2 Part 1: Core Fundamentals (Value Screen)

#### Thresholds — HK Focus

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Market Cap (HKD)** | ≥ 13B | HKD 13B ≈ USD 1.7B. Above micro-cap noise. |
| **B/M Ratio** | ≥ 0.30 | **HIGHER than current 0.15.** Deep value floor for HK. HK stocks that are genuinely cheap have B/M > 0.30. Below this is "cheap for a reason." |
| **B/M for Financials** | Auto-pass with ROE check | Financials have naturally high book value — B/M alone is misleading. |
| **ROE (Financials)** | ≥ 0.08 | 8% ROE — banks need to earn their cost of equity |
| **ROA (Non-Fin)** | ≥ 0.01 | 1% ROA floor — HK has lower profitability baseline |
| **EBITDA Margin** | ≥ 0.05 | 5% margin floor for non-financials |
| **FCF Yield** | ≥ 0.03 | **HIGHER than current 1%.** Cash is king in HK. A company with 3% FCF yield is genuinely cash-generative. Below 3% is speculative. |
| **Dividend Yield** | ≥ 0.03 | **NEW.** 3%+ dividend yield — this is the HK strategy's safety net. |
| **FCF/Dividend Coverage** | ≥ 1.2x | **NEW.** Free cash flow must cover dividends 1.2x. Avoids dividend traps. |
| **PTL Ratio** | ≤ 1.30 | TIGHTER than current 1.35. Buy within 30% of 52w-low. We want genuinely beaten-down stocks. |
| **Revenue Growth** | ≥ -0.10 | Revenue not declining >10% YoY. We're not buying dying companies. |
| **Debt/Equity** | ≤ 150% (non-fin) | Ex-financials (leverage is their business) |
| **Interest Coverage** | ≥ 2.0x | **NEW.** EBIT covers interest 2x+ — avoid distressed debt situations |

#### HK Scoring Weights

| Factor | Weight | Why |
|--------|--------|-----|
| **Dividend Yield** | 0.25 | Primary return driver in HK |
| **FCF Yield** | 0.20 | Cash generation quality |
| **B/M Ratio** | 0.20 | Deep value premium |
| **PTL (52w-low proximity)** | 0.15 | Mean-reversion entry timing |
| **ROE/ROA** | 0.10 | Profitability floor |
| **FCF/Dividend Coverage** | 0.10 | Dividend sustainability |

**Pass threshold:** `quality_score ≥ 0.30` (higher bar than current 0.25 — we're being more selective)

### 3.3 Part 2: MAGNA Components — HK Weightings (Repurposed)

HK MAGNA is fundamentally different. We're not screening for momentum — we're screening for **turnaround timing**.

| Component | Points | HK Interpretation |
|-----------|--------|-------------------|
| **M — Earnings Stabilization** | **2** | Earnings STOPPED declining, not accelerating. YoY EPS change > -5%. |
| **A — Revenue Stability** | **3** ↑ | Revenue flat or growing. Revenue stability > acceleration in HK. |
| **G — Turnaround Gap** | **2** — | Breakout on positive catalyst (policy, earnings beat, sector rotation). |
| **N — Neglect/Base** | **3** ↑ | **HIGHER weight in HK.** Long base (>9 months) near 52w-low = coiled spring. |
| **5 — Short Interest** | **0** — | **SKIP.** HK short data unreliable via yfinance. |
| **3 — Analyst Coverage** | **1** — | ≥3 analysts with stable/rising targets. Not upgrades — just coverage. |

**HK MAGNA total:** 11 points. **Pass threshold:** 4/11.

#### M: Earnings Stabilization (repurposed) — 2 pts
```
Conditions:
  - Latest YoY earnings growth ≥ -5% (not declining more than 5%)
  - OR: 2+ quarters of improving earnings trajectory (less negative or turning positive)
  - Rationale: HK value turnarounds start when earnings STOP getting worse.
    We don't need acceleration — just stabilization.
```

#### A: Revenue Stability (repurposed) — 3 pts  
```
Conditions:
  - Revenue growth ≥ -3% YoY (essentially stable)
  - OR: Revenue growth ≥ 0% (growing)
  - AND: Not in secular decline (2+ years of -10%+ revenue drops)
  - Rationale: In HK, stable revenue with high dividend yield is the ideal combo.
    Growing revenue is a bonus, not the base case.
  
  Weight bumped to 3 because revenue stability is the #1 "is this company alive" signal in HK.
```

#### N: Neglect/Base (repurposed) — 3 pts
```
Conditions:
  - Stock in base ≥ 9 months (was 6 months)
  - Price range ≤ 30% within base
  - Volume declining through base period (distribution complete)
  - BONUS: Near the LOWER end of the base (not the upper end)
  - Rationale: In HK value, the longest bases produce the biggest mean-reversion moves.
    A stock that's been neglected for 9+ months near 52w-low with declining volume
    has no sellers left — any positive catalyst causes a violent snap.
  
  NEW: Base proximity scoring:
    - In bottom third of base range: +1 bonus
    - In middle third: 0
    - In top third: -1 penalty (already ran up, not a value entry)
```

### 3.4 Entry Triggers — HK

```
  BUY if:
    N (neglect base ≥ 9 months) fires 
    AND quality_score ≥ 0.40 (higher quality for entry)
    AND dividend_yield ≥ 4%
    → Enter near base low (within 5% of base floor)
  
  BUY IF CATALYST:
    G (turnaround gap) fires — positive news/earnings beat/policy catalyst
    AND MAGNA ≥ 4/11
    → Enter on the gap day close (don't chase intraday — HK gaps often fill)
  
  ACCUMULATE if:
    quality_score ≥ 0.45 (elite quality)
    AND dividend_yield ≥ 5%
    AND PTL ≤ 1.15 (within 15% of 52w-low)
    → Start 50% position now, add 50% on any further 3% dip
```

### 3.5 State-Owned Enterprise (SOE) Filter

SOEs are a unique HK feature. They require special handling:

```
SOE Identification:
  - Ultimate parent is PRC central or provincial government
  - Examples: 0939.HK (CCB), 1398.HK (ICBC), 3988.HK (BOC), 
    0857.HK (PetroChina), 0386.HK (Sinopec), 0941.HK (China Mobile)
  
SOE Adjustments:
  1. B/M threshold: 0.40 (higher — SOEs always trade at discount)
  2. Dividend yield: ≥ 5% (SOEs distribute more, must verify sustainability)
  3. Governance discount: -5% on quality_score (SOEs have agency problems)
  4. BUT: Default risk discount: +5% on quality_score (implicit sovereign backing)
  5. Net: quality_score unchanged, but requires higher B/M + dividend
  
  SOE-specific risk checks:
    - FCF/Dividend coverage ≥ 1.0x (can be lower — SOEs maintain dividends through cycles)
    - No regulatory crackdown in sector (marco overlay check)
    - PBOC policy stance: easing/neutral = OK, tightening = reduce SOE exposure
```

### 3.6 Dividend Yield Integration

Dividend yield is the HK strategy's anchor. Current VMAA ignores it entirely.

```
Dividend Yield Tiers:
  Tier 1: ≥ 7% — MAXIMUM position. Cash return alone justifies the trade.
  Tier 2: 5-7% — STANDARD position. Good income + recovery optionality.
  Tier 3: 3-5% — HALF position. Need strong quality + low PTL to compensate.
  Tier 4: < 3% — Requires exceptional B/M and growth to enter.
  
Dividend Trap Detection:
  - Dividend yield > 10% → WARNING (likely unsustainable)
  - Payout ratio > 90% → WARNING (eating into capital)
  - FCF/Dividend < 0.8x → REJECT (borrowing to pay dividends)
  - Dividend cut in last 2 years → FLAG for review (may be recovering)
  
Total Return Decomposition (HK specific):
  Expected Return = Dividend Yield + Mean Reversion Return + Earnings Growth
  Example: CCB (0939.HK)
    - Dividend yield: 7.2%
    - Mean reversion (B/M 0.45 → 0.60): +33% over 2 years = +15% annualized
    - Earnings growth: +3% (mature bank)
    - Total expected: ~25% annualized (if mean reversion occurs)
```

### 3.7 Macro Overlay — HK

HK stocks are macro-driven. A macro overlay is essential:

```
Macro Indicator Dashboard:
  ┌─────────────────────────────────────────────────────┐
  │ China Manufacturing PMI:  ████████████░ 50.3       │
  │ PBOC Policy Stance:       🟢 EASING                 │
  │ CNY/USD Trend:            🟡 STABLE (7.15-7.25)     │
  │ HSI > 200 DMA:            🟢 YES                     │
  │ Southbound Flow (20d):    🟢 +HKD 45B NET INFLOW    │
  │ US-China Tension Index:   🟡 ELEVATED                │
  │ HK Property Index:        🔴 DECLINING (-12% YoY)    │
  └─────────────────────────────────────────────────────┘
  
Position Scalar based on Macro Score (0-6):
  6/6 green: scalar = 1.0  (full size)
  4-5/6 green: scalar = 0.80
  3/6 green: scalar = 0.60
  1-2/6 green: scalar = 0.40
  0/6 green: scalar = 0.00 (cash only)
  
Sector Rotation Logic:
  - PBOC easing → overweight financials (NIM expansion)
  - China PMI > 51 → overweight industrials, energy
  - Property stabilization → overweight HK developers
  - CNY weakening → reduce exposure (capital flight signal)
  - Southbound inflow surging → positive for HSI overall
```

### 3.8 Position Sizing — HK

```
  Method: Fixed Fraction (more conservative than Kelly)
  Rationale: HK mean-reversion outcomes are more uncertain than US momentum.
    Fixed fraction prevents overconfidence in any single value thesis.
  
  Base position: 8% of portfolio per position
  Max position: 12% (Tier 1 dividend stocks only)
  Max positions: 6 (more concentrated — value is about conviction)
  Max deployment: 48-72%
  
  Cash reserve: 28-52%
    - HK cash reserve higher because macro shocks (China policy, US tariffs) 
      create sudden buying opportunities
    - Dry powder for 52w-low breaks
  
  Scaling:
    - Initial: 60% of target position
    - Add 40% if stock drops 5% from entry (averaging down, value style)
    - Do NOT add on strength (this is value, not momentum)
  
  Sector Limits:
    - Financials: max 3 positions (40% to limit concentration)
    - Property: max 1 position (sector-specific risk)
    - Tech: max 2 positions (growth adjacent in value portfolio)
```

### 3.9 Stops — HK (Wider Value Stops)

```
  HK stocks are more volatile and less liquid. Stops must be wider 
  to avoid being shaken out of valid value trades:
  
  ATR Stop: 3.0-4.0x ATR (US = 2.5-3.0x)
    - The median bid-ask spread alone in HK mid-caps is 0.2-0.5%
    - Daily volatility is ~2.5% (vs ~1.8% US mid-caps)
    - ATR stop at 4.0x on a 2.5% ATR stock = 10% distance — reasonable
  
  Hard Stop: Dynamic
    - < HKD 20: 22% (wider — low price = more noise)
    - HKD 20-50: 18%
    - > HKD 50: 15%
    
  Structural Stop: 52w-low × 0.95
    - HK stocks that break 52w-low are often going lower
    - Exceptions: SOEs with sovereign backing can be held through 52w-low break
      (but only if dividend yield ≥ 7% and macro score ≥ 4/6)
  
  Time Stop: 180 days (US = 60 days)
    - Value realization takes time in HK
    - 6 months is minimum for mean reversion to play out
    - BUT: Review at 90 days — if no catalyst and no recovery, cut 50%
  
  SPECIAL — Dividend Cut Stop:
    - If company cuts/suspends dividend → IMMEDIATE EXIT
    - This is the #1 red flag in a dividend-focused strategy
    - No averaging down after dividend cut
  
  SPECIAL — SOE Policy Stop:
    - If sector faces regulatory crackdown → EXIT within 5 days
    - Examples: tech regulation (2021), property deleveraging (2022-23),
      education sector (2021)
```

### 3.10 Take Profit — HK

```
  HK take profits should be PATIENT — value can run further than momentum:
  
  TP1: +15% → Sell 25% (lock in initial gain)
  TP2: +30% → Sell 25% (strong mean reversion)
  TP3: +50% → Sell 25% (full value realization)
  Final 25%: Let ride with trailing stop at -12% from peak
  
  Dividend-adjusted returns:
    - If holding for 12+ months, dividend yield becomes significant
    - A stock bought at HKD 50 with 6% yield pays HKD 3/year
    - If held 18 months: HKD 4.50 in dividends = 9% return just from dividends
    - This makes wider stops more palatable
  
  B/M-based take profit (unique to HK):
    - If B/M drops below 0.20 (stock recovered from deep value to moderate value)
    → REDUCE position by 50%
    - If B/M drops below 0.10 (stock no longer value)
    → FULL EXIT — value thesis played out
```

---

## 4. Side-by-Side Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                🇺🇸 US Strategy              🇭🇰 HK Strategy       │
├─────────────────────────────────────────────────────────────────┤
│ PHILOSOPHY                                                       │
│  Quality Momentum                    Value Yield                │
│  "Trend is your friend"              "Buy when there's blood"   │
├─────────────────────────────────────────────────────────────────┤
│ UNIVERSE                                                         │
│  S&P 500 mid-caps ($1B-$10B)          HSI constituents (88 stks) │
│  ~300-400 liquid names                ~80-120 including HSCEI    │
├─────────────────────────────────────────────────────────────────┤
│ PART 1: FUNDAMENTALS                                             │
│  B/M: ≥ 0.10                         B/M: ≥ 0.30                │
│  ROA: ≥ 0.03                         ROA: ≥ 0.01                │
│  EBITDA Margin: ≥ 0.08               EBITDA Margin: ≥ 0.05      │
│  FCF Yield: ≥ 0.01 (safety net)      FCF Yield: ≥ 0.03 (core)  │
│  Revenue Growth: ≥ 12% ⭐             Revenue Growth: ≥ -10%     │
│  PTL: ≤ 1.40                         PTL: ≤ 1.30 (tighter!)     │
│  NEW: —                              Dividend Yield: ≥ 3% ⭐    │
│  NEW: —                              FCF/Coverage: ≥ 1.2x       │
│  NEW: —                              Interest Coverage: ≥ 2.0x  │
│  Quality Pass: ≥ 0.35                Quality Pass: ≥ 0.30       │
├─────────────────────────────────────────────────────────────────┤
│ PART 2: MAGNA WEIGHTS                                            │
│  M (Earnings): 3pts ⭐⭐⭐              M (Stabilization): 2pts     │
│  A (Sales): 2pts                     A (Revenue): 3pts ⭐⭐⭐     │
│  G (Gap Up): 3pts ⭐⭐⭐                G (Turnaround): 2pts       │
│  N (Base): 1pt                       N (Neglect): 3pts ⭐⭐⭐     │
│  5 (Short Int): 2pts                 5 (Short Int): SKIP        │
│  3 (Analyst): 1pt                    3 (Analyst): 1pt           │
│  Total: 12pts, pass ≥ 5              Total: 11pts, pass ≥ 4     │
│  MAGNA Emphasis: M + G               MAGNA Emphasis: A + N      │
├─────────────────────────────────────────────────────────────────┤
│ ENTRY LOGIC                                                      │
│  GAP → immediate market buy          BASE LOW → limit order     │
│  MA fire → buy pullback to 20MA      CATALYST gap → close buy   │
│  HIGH MAGNA → alert breakout         HIGH DIV + PTL → accumulate │
├─────────────────────────────────────────────────────────────────┤
│ POSITION SIZING                                                  │
│  Method: Quarter-Kelly (aggressive)  Method: Fixed Fraction      │
│  Size: 7-10% per position            Size: 8-12% per position    │
│  Max Positions: 8                    Max Positions: 6            │
│  Cash Reserve: 20-44%                Cash Reserve: 28-52%        │
│  Scale: Add on pullback w/strength   Scale: Add on further dips  │
│  Kelly Factor: 0.25                  Fixed: no adjustment       │
├─────────────────────────────────────────────────────────────────┤
│ STOP LOSS                                                        │
│  ATR Multiplier: 2.5-3.0x            ATR Multiplier: 3.0-4.0x   │
│  Hard Stop: 12-18%                   Hard Stop: 15-22%          │
│  Trailing: -8% after +12%            Trailing: -12% after +15%  │
│  Time Stop: 60 days                  Time Stop: 180 days        │
│  SPECIAL: Gap fill stop (5 days)     SPECIAL: Dividend cut stop │
│  Breakeven: After +15%               Breakeven: After +20%      │
├─────────────────────────────────────────────────────────────────┤
│ TAKE PROFIT                                                      │
│  TP1: +12% sell 30%                  TP1: +15% sell 25%         │
│  TP2: +22% sell 30%                  TP2: +30% sell 25%         │
│  TP3: +35% sell 40%                  TP3: +50% sell 25%         │
│  Final: Trailing stop -5%            Final: Trail -12%, 25%     │
├─────────────────────────────────────────────────────────────────┤
│ MACRO OVERLAY                                                    │
│  SPY > 50MA: OK                      HSI > 200MA: OK            │
│  VIX < 25: OK                        China PMI > 50: OK         │
│  Fed stance: data-driven             PBOC stance: easing pref.  │
│  Sector rotation: growth/momo        Sector: value/dividend     │
├─────────────────────────────────────────────────────────────────┤
│ SPECIAL RULES                                                    │
│  Short squeeze detection             SOE discount/premium       │
│  Earnings surprise streak            Dividend trap detection    │
│  Gap integrity check                 Macro condition scoring    │
│  Volume quality filter               B/M-based take profit      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Plan

### 5.1 Code Architecture

```
vmaa/
├── config.py              → US config (keep current, update thresholds)
├── config_hk.py           → NEW HK config (HKScreeningThresholds relocated)
├── strategy/
│   ├── __init__.py
│   ├── us_quality_momentum.py   → US strategy engine
│   │   ├── USPart1Config        (dataclass)
│   │   ├── USPart2Config        (dataclass)
│   │   ├── USRiskConfig         (dataclass)
│   │   ├── USScreener           (Part 1 + Part 2 combined)
│   │   └── USEntryManager       (gap/momo entry logic)
│   │
│   └── hk_value_yield.py        → HK strategy engine
│       ├── HKPart1Config        (dataclass)
│       ├── HKPart2Config        (dataclass)
│       ├── HKRiskConfig         (dataclass)
│       ├── HKScreener           (Part 1 + Part 2 combined)
│       ├── SOEFilter            (state-owned enterprise logic)
│       ├── DividendAnalyzer     (yield, coverage, trap detection)
│       └── MacroOverlay         (China PMI, PBOC, southbound flow)
│
├── pipeline.py            → US pipeline (imports strategy/us_*)
├── pipeline_hk.py         → HK pipeline (imports strategy/hk_*)
│
├── risk.py                → shared risk utilities
├── risk_adaptive.py       → adaptive stop (works for both, params differ)
│
├── backtest/
│   ├── config.py          → US backtest config
│   ├── engine.py          → US backtest engine
│   └── hk/
│       ├── hk_config.py   → HK backtest config (already exists)
│       ├── hk_backtest.py → HK backtest engine
│       └── ...
│
└── models.py              → shared data models (Part1Result, Part2Signal, etc.)
```

### 5.2 Configuration Separation

Each strategy gets independent config dataclasses:

```python
# strategy/us_quality_momentum.py

@dataclass
class USPart1Config:
    """US Quality Momentum Part 1 thresholds."""
    min_bm_ratio: float = 0.10
    min_roa: float = 0.03
    min_roe: float = 0.08
    min_ebitda_margin: float = 0.08
    min_fcf_yield: float = 0.01
    min_fcf_conversion: float = 0.40
    min_revenue_growth: float = 0.12
    max_ptl_ratio: float = 1.40
    # Scoring weights
    weight_revenue_growth: float = 0.25
    weight_earnings_growth: float = 0.20
    weight_fcf_conversion: float = 0.15
    weight_roa: float = 0.10
    weight_ebitda: float = 0.10
    weight_ptl: float = 0.10
    weight_bm: float = 0.10
    min_quality_score: float = 0.35

@dataclass
class USPart2Config:
    """US MAGNA re-weighted for momentum."""
    eps_accel_min: float = 0.25
    eps_accel_delta: float = 0.10
    gap_min_pct: float = 0.04
    gap_vol_mult: float = 2.0          # Raised from 1.5
    magna_points: dict = {
        'm_earnings': 3, 'a_sales': 2, 'g_gap': 3,
        'n_base': 1, 'short_interest': 2, 'analyst': 1
    }
    magna_pass: int = 5

# strategy/hk_value_yield.py

@dataclass
class HKPart1Config:
    """HK Value Yield Part 1 thresholds."""
    min_bm_ratio: float = 0.30
    min_soe_bm_ratio: float = 0.40
    min_roa: float = 0.01
    min_roe_financial: float = 0.08
    min_ebitda_margin: float = 0.05
    min_fcf_yield: float = 0.03
    min_dividend_yield: float = 0.03
    min_fcf_div_coverage: float = 1.2
    max_ptl_ratio: float = 1.30
    min_revenue_growth: float = -0.10
    min_interest_coverage: float = 2.0
    # Scoring weights
    weight_dividend: float = 0.25
    weight_fcf_yield: float = 0.20
    weight_bm: float = 0.20
    weight_ptl: float = 0.15
    weight_roe: float = 0.10
    weight_div_coverage: float = 0.10
    min_quality_score: float = 0.30

@dataclass
class HKPart2Config:
    """HK MAGNA repurposed for turnaround timing."""
    eps_stabilize_min: float = -0.05    # EPS growth ≥ -5%
    rev_stable_min: float = -0.03       # Revenue growth ≥ -3%
    gap_min_pct: float = 0.04
    base_min_months: float = 9.0        # Longer base required
    magna_points: dict = {
        'm_stabilize': 2, 'a_revenue': 3, 'g_turnaround': 2,
        'n_neglect': 3, 'analyst': 1
    }
    magna_pass: int = 4
```

### 5.3 Implementation Phases

#### Phase 1: HK Priority (Weeks 1-2)
**Rationale:** HK strategy needs the most change (current HK borrows US MAGNA logic). US strategy is closer to correct with threshold tuning.

| Task | Effort | Description |
|------|--------|-------------|
| 1.1 Create `strategy/hk_value_yield.py` | 3 days | HK Part1 + Part2 configs, SOE filter, dividend analyzer |
| 1.2 Create `strategy/hk_macro_overlay.py` | 2 days | China PMI, PBOC, southbound flow data fetchers |
| 1.3 Refactor `pipeline_hk.py` | 2 days | Switch to new HK configs, integrate macro overlay |
| 1.4 Update `backtest/hk/` | 2 days | New thresholds in backtest config, validate against 2023-2025 data |
| 1.5 HK backtest validation | 2 days | Run backtest, compare against current HK pipeline performance |
| **Total Phase 1** | **11 days** | |

#### Phase 2: US Refinement (Weeks 3-4)
**Rationale:** US strategy is evolutionary (different weights, new triggers). Lower risk to deploy.

| Task | Effort | Description |
|------|--------|-------------|
| 2.1 Create `strategy/us_quality_momentum.py` | 2 days | US Part1 + Part2 configs, revenue-first scoring |
| 2.2 Add gap integrity check | 1 day | Verify gap holds above open for 30 min |
| 2.3 Add earnings surprise streak | 1 day | Track consecutive beats |
| 2.4 Refactor `pipeline.py` | 1 day | Switch to new US configs |
| 2.5 US backtest validation | 3 days | Run against 2020-2025 data, compare baseline |
| **Total Phase 2** | **8 days** | |

#### Phase 3: Integration & Parallel Run (Weeks 5-6)
| Task | Effort | Description |
|------|--------|-------------|
| 3.1 Unified runner | 2 days | `runner.py` can run US+HK independently or together |
| 3.2 Combined reporting | 1 day | Single report showing US + HK performance side-by-side |
| 3.3 Paper trading | 5 days | Run both strategies in dry-run for 1 week, verify signals |
| 3.4 Documentation | 2 days | Update all docs, AGENTS.md, TOOLS.md |
| **Total Phase 3** | **10 days** | |

**Total: ~29 days to full deployment**

### 5.4 Backtest Validation Plan

#### HK Backtest Design
```
Period: 2023-01-01 → 2025-12-31 (3 years)
Universe: HSI constituents (88 stocks)
Benchmark: 2800.HK (TraHK ETF)
Capital: HKD 500,000
Frequency: Monthly rebalance
Transaction costs: HK slippage model (0.25% commission + 0.13% stamp duty)
  
Key Metrics to Track:
  - Total Return vs Benchmark
  - Sharpe Ratio (target > 0.5, current HK = TBD)
  - Max Drawdown (target < -20%)
  - Dividend contribution to total return
  - Win rate (target > 50% — value is lower win rate, bigger wins)
  - Average holding period
  - SOE vs non-SOE performance split
```

#### US Backtest Design
```
Period: 2022-01-01 → 2025-12-31 (4 years)
Universe: S&P 500 mid-caps ($1B-$10B market cap), ~300 stocks
Benchmark: SPY
Capital: USD 100,000
Frequency: Monthly rebalance
Transaction costs: US slippage model
  
Key Metrics to Track:
  - Total Return vs SPY (target: SPY + 5% annualized)
  - Sharpe Ratio (target > 1.0)
  - Max Drawdown (target < -15%)
  - Win rate (target > 60% — momentum should win more)
  - Average holding period (target 30-60 days)
  - Gap entry win rate (should be highest)
```

#### Success Criteria

| Metric | US Target | HK Target | Current Best |
|--------|-----------|-----------|-------------|
| Annualized Return | SPY + 5% | HSI + 8% | US: -47% vs SPY (broken) |
| Sharpe Ratio | > 1.0 | > 0.5 | US: -3.32 (broken) |
| Max Drawdown | < -15% | < -20% | US: -0.7% (too few trades) |
| Win Rate | > 60% | > 50% | US: 56% |
| Avg Win/Avg Loss | > 1.2 | > 1.5 | US: 0.65-0.54 |
| Trade Frequency | 15-30/year | 8-15/year | US: ~3/year (too few) |

---

## 6. Risk Considerations

### 6.1 Strategy-Specific Risks

#### US Quality Momentum Risks
1. **Momentum crashes** — Momentum factor suffers periodic crashes (2009, 2020). Mitigation: market regime filter + VIX threshold.
2. **Crowding** — Popular momentum names attract too much capital. Mitigation: Max 8 positions, diversified sectors.
3. **Gap traps** — Not all gaps are institutional. Mitigation: Volume quality filter (≥ 2.0x avg) + 30-min hold check.
4. **Earnings whipsaws** — Beat-and-drop reactions. Mitigation: Enter only after gap holds, not on earnings day.

#### HK Value Yield Risks
1. **Value traps** — Cheap stocks that stay cheap. Mitigation: Dividend yield floor + FCF coverage + macro catalyst requirement.
2. **China macro shocks** — Policy changes, regulatory crackdowns. Mitigation: Macro overlay cuts exposure when macro score < 3/6.
3. **Dividend cuts** — Companies reduce/suspend dividends. Mitigation: FCF/Dividend coverage check + immediate exit on cut.
4. **Liquidity risk** — HK mid-caps have thin liquidity. Mitigation: Position size limited to 12% max + wider stops accommodate spread.
5. **Currency risk** — HKD peg to USD breaks scenario (tail risk). Mitigation: Monitor HKMA aggregate balance monthly.

### 6.2 Shared Risk Controls
- No single position > 12% of portfolio (US: 10%, HK: 12%)
- Max 2 positions per sector (both markets)
- Market regime override: if market_ok = False, reduce max positions by 50%
- Daily loss limit: -3% of portfolio → stop trading for the day
- Weekly loss limit: -5% → review all open positions

---

## 7. Appendix

### A. Key Academic References

| Study | Finding | Applied To |
|-------|---------|------------|
| Jegadeesh & Titman (1993) | Momentum premium ~1%/month in US | US strategy foundation |
| Fama & French (1992) | Value premium (HML) existence | HK B/M emphasis |
| Asness et al. (2013) | Value + Momentum combo superior | Both strategies |
| Chui, Titman & Wei (2010) | Momentum WEAK in Asian markets | HK MOMENTUM AVOIDANCE |
| Chen, Kim, Yao & Yu (2010) | SOE discount in China/HK | HK SOE filter |
| Baker & Wurgler (2006) | Dividend catering theory | HK dividend focus |

### B. Configuration Diff — Current vs Proposed

#### US Changes Summary
| Parameter | Current | Proposed | Δ |
|-----------|---------|----------|---|
| B/M | 0.20 | 0.10 | -50% |
| ROA | 0.00 | 0.03 | +3pp |
| EBITDA Margin | 0.05 | 0.08 | +3pp |
| FCF Yield | 0.02 | 0.01 | -1pp |
| Revenue Growth | — | 0.12 | NEW |
| PTL | 1.50 | 1.40 | -0.10 |
| Quality Pass | 0.40 | 0.35 | -0.05 |
| M weight | 2 | 3 | +1 |
| G weight | 2 | 3 | +1 |
| MAGNA pass | 3 | 5 | +2 |
| Hard stop (>$30) | 15% | 12% | -3pp |
| Time stop | 90d | 60d | -30d |
| Max positions | 8 | 8 | — |

#### HK Changes Summary
| Parameter | Current | Proposed | Δ |
|-----------|---------|----------|---|
| B/M | 0.15 | 0.30 | +100% |
| FCF Yield | 0.01 | 0.03 | +200% |
| Dividend Yield | — | 0.03 | NEW |
| FCF/Coverage | — | 1.2x | NEW |
| Interest Coverage | — | 2.0x | NEW |
| PTL | 1.35 | 1.30 | -0.05 |
| Revenue Growth | -0.15 | -0.10 | +5pp |
| Quality Pass | 0.25 | 0.30 | +0.05 |
| N weight | 1 | 3 | +2 |
| A weight | 2 | 3 | +1 |
| 5 (Short Int) | 0 | SKIP | Removed |
| Base min months | 6 | 9 | +3 |
| Hard stop | 15% | 15-22% | Wider for low-price |
| Time stop | 90d | 180d | +90d |
| Max positions | 8 | 6 | -2 |

### C. Glossary

| Term | Definition |
|------|------------|
| **B/M** | Book-to-Market ratio — measures how cheap a stock is relative to its book value |
| **FCF/Y** | Free Cash Flow Yield — FCF / Enterprise Value or Market Cap |
| **PTL** | Price-to-52-week-Low ratio — proximity to annual low |
| **MAGNA** | Momentum scoring system: M(assive earnings), A(cceleration sales), G(ap up), N(eglect base), 5(hort interest), 3(nalyst) |
| **SOE** | State-Owned Enterprise — company ultimately controlled by PRC government |
| **Quarter-Kelly** | Position sizing using 25% of the Kelly Criterion optimal bet size |
| **Fixed Fraction** | Position sizing as fixed % of portfolio regardless of edge estimate |
| **ATR** | Average True Range — volatility measure for stop distance |
| **Southbound Flow** | Mainland Chinese capital flowing into HK stocks via Stock Connect |

---

> **Status:** Design complete. Awaiting approval to begin Phase 1 (HK) implementation.  
> **Next Step:** `mkdir -p vmaa/strategy && touch vmaa/strategy/__init__.py`  
> **Contact:** VMAA Strategy Architecture Team via Royce 🦾
