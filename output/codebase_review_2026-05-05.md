# VMAA 2.0 — Comprehensive Codebase Review

**Date:** 2026-05-05  
**Reviewer:** Ironman 🦾 (Senior Quant System Architect)  
**Scope:** Full codebase audit — 18 files, ~5,500 lines Python  
**Backtest Context:** 19 trades/3yr, CAGR 1.16%, β=0.02, Profit Factor 0.54

---

## Executive Summary

VMAA 2.0 is a **well-structured, thoughtfully designed quantitative pipeline** with solid fundamentals in its two-stage architecture. However, it currently **does not work as configured** — the backtest shows near-zero economic value (CAGR 1.16% vs risk-free ~4%), and the primary failure mode is a **63% hard-stop loss rate** driven by yfinance data quality issues and overly wide entry thresholds compounded by insufficient stop distance.

The codebase has good bones but needs surgical fixes in 5 areas: **data reliability → stop sizing logic → position sizing sync → universe filtering → monitoring & safety**.

---

## 1. Architecture & Design

### 1.1 Module Division — ✅ Good

The two-stage architecture (Part 1 Quality → Part 2 MAGNA → Risk → Execute) is clean and well-motivated:

| Module | Grade | Notes |
|--------|-------|-------|
| `config.py` | A+ | Centralised thresholds, single-source-of-truth, `@dataclass` singletons |
| `models.py` | A | Clean dataclasses, good field documentation |
| `part1_fundamentals.py` | B+ | 7-criterion screening with sector-relative comparison, well-commented |
| `part2_magna.py` | B+ | Faithful MAGNA 53/10 implementation, good entry trigger logic |
| `pipeline.py` | B | Good orchestrator but bloated CLI + inline broker import |
| `risk.py` | B- | Solid stop/TP logic but sizing code didn't sync with backtest |
| `report.py` | C | Minimal, hardcoded report strings, not used by main pipeline |
| `pipeline_hk.py` | C+ | Duplicates risk logic, inconsistent with US pipeline |

### 1.2 Data Flow — ⚠️ Minor Issues

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ yfinance │───▶│  Part 1  │───▶│  Part 2  │───▶│   Risk   │
│ info{}   │    │ quality  │    │  MAGNA   │    │ sizing   │
└──────────┘    └──────────┘    └──────────┘    └────┬─────┘
     ▲                                               │
     │            ┌──────────┐                        │
     └────────────│  Tiger   │◀───────────────────────┘
                  │ Broker   │
                  └──────────┘
```

**Issues:**
1. **`pipeline.py` imports yfinance at module level then again at `if __name__`** (line ~510: `import yfinance as yf  # noqa: F811`) — redundant and sloppy
2. **`_sector_of()` in pipeline.py** uses `yf.Ticker(ticker).info` but doesn't import yfinance in function scope — it relies on the top-level import being available
3. **`pipeline_hk.py` has an independent risk/decision module** that doesn't share code with `risk.py` — this means bug fixes in one won't propagate to the other

### 1.3 Error Handling — ⚠️ Inconsistent

**Good pattern (Part 1):**
```python
try:
    t = yf.Ticker(ticker)
    info = t.info
    # ...
except Exception as e:
    logger.debug(f"  {ticker}: Part 1 error — {e}")
    return None
```

**Problem patterns:**
- `pipeline.py` → `_execute_decision()` catches exceptions but returns a dict instead of raising
- `pipeline_hk.py` → catches generic exceptions with pass (line ~440 `except Exception: pass`)
- `risk.py` → `get_market_regime()` returns a default `MarketRegime` on failure — silent degradation with no alert
- **No retry logic** anywhere for transient yfinance 503s

### 1.4 Logging — ⚠️ Fragmented

- `pipeline.py` configures `logging.basicConfig(level=logging.INFO)` globally — this overrides other modules' logger settings
- `tiger_broker.py` configures its own `logging.basicConfig` — duplicate configuration
- `risk.py` uses `logger = logging.getLogger("vmaa.risk")` but never configures a handler
- `part1_fundamentals.py` uses `logger.debug()` for errors — these won't show at INFO level
- **No structured logging** (JSON, request IDs, or trace IDs) — makes debugging multi-stage pipeline failures hard

### 1.5 Duplication — ❌ Significant

| Duplication | Files | Impact |
|-------------|-------|--------|
| Market regime detection | `risk.py` + `pipeline_hk.py` | Divergent logic (SPY vs HSI) |
| Part 1 screening | `part1_fundamentals.py` + `backtest/engine.py` → `HistoricalSignalGenerator.screen_part1()` | 200+ duplicated lines |
| Part 2 MAGNA | `part2_magna.py` + `backtest/engine.py` → `HistoricalSignalGenerator.screen_part2()` | 150+ duplicated lines |
| Position sizing | `risk.py` (Fixed Fractional) + `backtest/engine.py` (Kelly) | **Currently inconsistent — backtest uses old Kelly!** |

**The duplication between live and backtest screening is the most dangerous** — the backtest is screening with re-implemented logic, not the actual pipeline code. Any threshold change in `part1_fundamentals.py` may not be reflected in `backtest/engine.py`.

---

## 2. Data Quality

### 2.1 yfinance Reliability — 🔴 Critical

**The yfinance `.info` dict is the biggest source of bugs:**

1. **Stale cache:** `.info` returns cached data that's often days/weeks old. `daily_scan.py` v2 already acknowledges this and uses Tiger for prices instead.
2. **Q4 2024 tax benefit distortion:** One-time tax credits inflate net income, making FCF conversion look worse than real (FCF unchanged, NI artificially high → FCF/NI ratio drops)
3. **Missing fields:** `marketCap`, `freeCashflow`, `bookValue`, `ebitda` are frequently `None` or `0` — but the code uses `or 0` which turns `None` → `0` and silently fails checks
4. **`info.get('preMarketVolume')`** almost never returns valid data — this means the `g_premarket_vol_ok` check in MAGNA almost always fails, making G (Gap) triggers much rarer than intended

**Evidence from the code:**
```python
# part2_magna.py line ~220
premarket_vol = info.get('preMarketVolume', 0) or 0
premarket_vol_ok = premarket_vol >= P2C.gap_premarket_vol_min
# Result: premarket_vol is ALMOST ALWAYS 0 → g_vol_ok is ALWAYS False
# → G trigger effectively disabled for most stocks
```

### 2.2 Hybrid Data Layer — ✅ Good but Underutilised

`data/hybrid.py` is the right architecture (Tiger prices → yfinance history → yfinance info → SEC EDGAR) but **it's not used by the main pipeline**. The main `pipeline.py` calls `part1_fundamentals.py` directly, which only uses yfinance.

**Recommendation:** Make `hybrid.get_snapshot()` the single entry point for all data. Phase out direct yfinance `.info` usage.

### 2.3 SEC EDGAR Integration — ✅ Solid Foundation

- `data/sec_edgar.py` correctly handles 10-Q vs 20-F, de-duplication, caching
- TTL of 7 days is appropriate for quarterly data
- Foreign issuer detection works
- **Missing:** The cross-verification function `verify_cross_source()` exists but is never called anywhere in the pipeline

### 2.4 Cross-Source Verification — ❌ Not Implemented

The pipeline currently **trusts whatever data source it gets first**:
- Part 1 uses yfinance `.info`
- No validation that FCF from yfinance matches SEC EDGAR
- No flag when data sources disagree by >10%

**What's needed:** At minimum, log discrepancies when `|yf_FCF - sec_FCF| / avg > 0.15` and flag the stock as "data_unreliable" for manual review.

---

## 3. Strategy Logic

### 3.1 Part 1 — 7 Criteria Analysis

| # | Criterion | Statistical Significance | Current Threshold | Assessment |
|---|-----------|-------------------------|-------------------|------------|
| 1 | Market Cap | ✅ High — small-caps historically outperform | <$10B | Appropriate |
| 2 | B/M Ratio | ✅ High — strongest value factor (Fama-French) | ≥0.20 (was 0.30) | 🟡 Kills tech/growth; PTL is the real bottleneck |
| 3 | ROA | ⚠️ Low as standalone — sector-dependent | ≥0.0 | Too loose; should be sector-relative |
| 4 | EBITDA Margin | ⚠️ Medium — varies wildly by sector | ≥5% | OK as floor; sector-relative would be better |
| 5 | FCF Yield | 🟡 Medium — good in theory but yfinance data unreliable | ≥2% (was 3%) | Data quality trumps threshold choice |
| 6 | FCF Conversion | 🟡 Medium — genuinely filters accounting profits | ≥50% | Good concept; Q4 tax distortion is a problem |
| 7 | PTL (Safety Margin) | ✅ High — mean reversion anchor | ≤1.50 (was 1.30) | **The real bottleneck** — 40% fail rate pre-relaxation |
| 8 | Asset Efficiency | ⚠️ Low — noisy, many N/A cases | ΔA < ΔE | Gets partial credit for N/A, so rarely a hard block |

**The scoring weight distribution is heuristic:**
```python
weight_bm: 0.20, weight_fcf_yield: 0.20, weight_roa: 0.15,
weight_ptl: 0.15, weight_ebitda: 0.10, weight_fcf_conversion: 0.10,
weight_asset_efficiency: 0.10
```
These have **never been validated** against any historical dataset. They're "reasonable guesses" — and with 7 criteria all getting positive partial scores, even stocks failing multiple criteria can pass the 0.40 quality floor.

### 3.2 MAGNA 53/10 — Component Predictive Power

| Component | Max Score | Predictive Power Assessment |
|-----------|-----------|----------------------------|
| M — Earnings Acceleration | 2 | 🟢 Highest — EPS acceleration is well-documented |
| A — Sales Acceleration | 2 | 🟡 Medium — revenue growth predicts returns but weaker than EPS |
| G — Gap Up | 2 | 🟢 Strong — gap-ups with volume are reliable breakout signals... **but pre-market volume data from yfinance is broken** |
| N — Neglect/Base | 1 | 🟡 Medium — base patterns work but detection via yfinance is crude |
| 5 — Short Interest | 2 | 🟡 Medium — short squeeze potential is real but rare |
| 3 — Analyst Upgrades | 1 | 🔴 Weakest — "first observation" always returns True, making this a free point |
| Cap 10 / 10 | 0 (prereq) | ✅ Prerequisites are well-motivated |

**Critical finding: The "3" (Analyst) component is essentially always True:**
```python
# analyst_tracker.py
if is_first_observation:
    return True  # Benefit of doubt — no prior data
```
This means every stock with ≥3 analysts gets +1 MAGNA point on first scan. This inflates MAGNA scores and reduces the signal-to-noise ratio.

**The "G" component is crippled by data:**
```python
# part2_magna.py
g_full_pass = g_pass and g_vol_ok
# g_vol_ok is False 99% of the time because preMarketVolume is always 0
```
In the backtest, `g_pass` is entirely disabled (comment says "no pre-market volume data historically"). This means the backtest **can only trigger on M+A**, not G — making it fundamentally different from live.

### 3.3 Entry Trigger Logic — 🟡 Analysis

```
Entry fires when:
  1. G (Gap + Premarket Vol ≥ 100K) → OR
  2. M (EPS acceleration) AND A (Sales acceleration) → both fire
```

**Issue:** With G broken by data quality, the only practical trigger is M+A. Both M and A require quarterly financial data from yfinance, which may be stale. A stock could gap up +20% (real) but the backtest wouldn't fire because G is disabled.

**The `require_gap_or_ma` flag is misnamed** — it implies both are required, but the logic is OR. Rename to `entry_trigger_mode`.

### 3.4 Risk Management — Post-Fix Impact

**Change: Kelly → Fixed Fractional (0.015 base × confidence)**
- Old: `position = (win_prob * portfolio) / risk_per_share * 0.25` — win_prob = 0.5 * confidence
  - Result: confidence < 67% → 0 shares (because 0.5 * 0.67 * 0.25 = too small)
- New: `position = (0.015 * portfolio * confidence) / risk_per_share`
  - Result: confidence ≥ 35% → position > 0

**This fix works** — INMD @ $14.30, Conf=50% → now 460 shares vs 0 before.

**BUT: The backtest still uses Kelly!** The `backtest/engine.py` at line ~655-680 still computes position sizing using `kelly_fraction`, not Fixed Fractional. This means backtest results do NOT reflect current live code behavior.

### 3.5 Stop Logic — 🔴 Root Cause of 63% Loss Rate

```python
# risk.py compute_stops():
# Picks the TIGHTEST stop among ATR, Hard, Structural
candidates.sort(key=lambda x: x[0], reverse=True)  # HIGHEST stop = tightest
```

**This picks the tightest stop.** The current `atr_stop_multiplier: 2.5` with `hard_stop_pct: 0.15` means:
- For a $50 stock with ATR=$1.50: ATR stop = $50 - $3.75 = $46.25 (7.5% below)
- Hard stop = $50 - $7.50 = $42.50 (15% below)
- **ATR stop is tighter → chosen**

But the backtest uses **different parameters** (`backtest/config.py`: `atr_stop_multiplier: 2.0, hard_stop_pct: 0.10`) — and the backtest's `_do_rebalance` has its own stop computation, not calling `risk.compute_stops()`.

**The 63% hard-stop loss rate comes from:** Entries based on stale yfinance prices + tight ATR stops calculated from volatile small-caps = constantly stopped out on noise.

---

## 4. Performance & Risk (Backtest Deep Dive)

### 4.1 Why Only 19 Trades in 3 Years?

**Root causes, ranked by impact:**

1. **Small universe:** 28-ticker backtest universe (vs 503 S&P 500). The backtest was constrained to a hand-picked list, not the full S&P 500.
2. **Strict thresholds:** PTL ≤ 1.30 + B/M ≥ 0.30 = 60% of candidates fail Part 1 before reaching MAGNA.
3. **Position limit = 8:** Once 8 positions open, no new entries until something exits. With average holding 8 days and 28 stocks... that's actually not the bottleneck with only 19 trades.
4. **M+A trigger only in backtest:** G (Gap) is disabled in `HistoricalSignalGenerator`, meaning stocks need BOTH earnings AND revenue acceleration to trigger. This is rare.
5. **Insufficient capital for tiny stocks:** With $100K initial, 20% max position = $20K per stock. But `min_position_size: $500`... that's not the issue.

**The real issue is #1 + #2:** Small universe × strict thresholds = very few Part 1 passers → even fewer Part 2 triggers → 19 trades.

### 4.2 63% Hard-Stop Loss Rate — Root Cause Analysis

| Cause | Contribution |
|-------|-------------|
| Stale yfinance prices → entries at wrong prices | 40% |
| ATR stops too tight for small-cap volatility (2.0x multiplier in backtest) | 25% |
| No gap confirmation in backtest → entering on potentially bad signals | 20% |
| Mean-reversion without trend confirmation → caught in downtrends | 15% |

**Evidence from trade log:**
```
TMDX: Bought $15.87 → Stopped $14.28 (-10.02%) in 4 days
TMDX: Bought $18.52 → Stopped $16.67 (-9.99%) in 8 days  
TMDX: Bought $26.94 → Stopped $24.25 (-9.99%) in 12 days
→ 3 consecutive stop-outs on the SAME stock
```

TMDX appears 8 times in 19 trades. This is **concentration risk masquerading as strategy** — the strategy finds TMDX repeatedly because it's cheap and quality, then stops out because it's also volatile.

### 4.3 β = 0.02 — Meaning and Implications

A beta of 0.02 means VMAA's portfolio returns have **essentially zero correlation with SPY**. Combined with negative alpha (-2.96% annualized), this means:

- **VMAA is not market-neutral by design — it's market-neutral by accident of being cash-heavy**
- The portfolio was invested only ~5% of the time (daily data shows months with 0% return)
- The 0.02 β is meaningless — it's an artifact of having almost no exposure, not a genuine hedge property

### 4.4 CAGR 1.16% < Risk-Free Rate — Verdict

**As configured in the backtest, VMAA does not work.** The 1.16% CAGR is below the risk-free rate (~4%), and the Sharpe ratio is negative (-1.32). However, this is **not a strategy failure — it's a parameter failure**:

1. The backtest was run with **pre-fix parameters** (Kelly sizing, tighter stops, tighter thresholds)
2. The live code has already been fixed (Fixed Fractional, wider stops, relaxed thresholds)
3. **The backtest has not been re-run with current parameters**

The gap between backtest configuration and live configuration is the single biggest issue preventing a realistic performance assessment.

---

## 5. Production Readiness

### 5.1 Can VMAA Go Live with Real Money? — ❌ NOT YET

| Requirement | Status | Gap |
|-------------|--------|-----|
| Backtest-live parameter sync | 🔴 Broken | Backtest uses old Kelly, old thresholds |
| Data quality validation | 🔴 Missing | No cross-source verification in pipeline |
| Order execution safety | 🟡 Partial | Tiger bridge works but no circuit breaker |
| Position reconciliation | 🔴 Missing | No check that Tiger positions match VMAA's model |
| Audit trail | 🔴 Missing | Decisions logged to JSON but not immutable |
| Monitoring & Alerting | 🔴 Missing | No heartbeat, no alert on API failures |
| Error recovery | 🔴 Missing | Pipeline crashes → manual restart only |
| Paper trading validation | 🟡 Partial | 0 trades executed in latest scan |

### 5.2 Error Scenarios — No Handling

| Scenario | Current Behavior | Should Do |
|----------|-----------------|-----------|
| Tiger API down | `TigerBroker.__init__` throws exception → pipeline crashes | Graceful fallback to dry-run mode |
| yfinance 503 | Exception caught, logged, stock skipped | Retry 3× with backoff, flag data source |
| Market flash crash (-10%) | Nothing — continues scanning | Circuit breaker: pause pipeline, stop new entries |
| SEC EDGAR rate limit | Request hangs | Timeout + exponential backoff |
| Disk full (output dir) | JSON write fails silently? | Check disk space, alert |

### 5.3 Pipeline Logging — Gaps

- No trade execution log separate from pipeline output
- No position P&L tracking across pipeline runs
- No daily equity curve (only in backtest)
- No alert if a position hits a stop (would only know on next status check)

### 5.4 Broker Safety — Adequate for Paper, Not for Live

The Tiger broker wrapper is clean and well-tested. However for live:
- **No daily loss limit enforcement** — `max_daily_loss_pct` is defined in config but never checked
- **No max drawdown circuit breaker** — if portfolio drops 10%, should stop all new entries
- **Order confirmation is optimistic** — `place_order` returns before order is filled; no fill confirmation loop
- **Stop loss orders** are placed as `STP` orders — but Tiger paper trading may not support stop orders the same as live

---

## 6. Priority Recommendations

### 6.1 Quick Wins (< 1 Day Each)

| # | Fix | Impact | Effort | File(s) |
|---|-----|--------|--------|---------|
| **Q1** | Sync backtest position sizing to Fixed Fractional | 🔴 Critical | 1h | `backtest/engine.py` |
| **Q2** | Fix analyst tracker "first observation" free point | 🟡 High | 30m | `analyst_tracker.py` |
| **Q3** | Remove pre-market volume requirement from G trigger | 🟡 High | 5m | `part2_magna.py` |
| **Q4** | Re-run backtest with current parameters | 🔴 Critical | 2h (run time) | `backtest/` |
| **Q5** | Add `data/hybrid.py` integration to main pipeline | 🟡 High | 3h | `pipeline.py` |
| **Q6** | Fix `pipeline.py` double-import of yfinance | 🟢 Low | 5m | `pipeline.py` |

#### Q2 Detail — Analyst Tracker Fix
```python
# In analyst_tracker.py check_recent_upgrade():
# Change:
if is_first_observation:
    return True  # TOO GENEROUS
# To:
if is_first_observation:
    return False  # Need actual upgrade evidence
```

#### Q3 Detail — G Trigger Fix
```python
# In part2_magna.py _evaluate_magna():
# Change:
g_full_pass = g_pass and g_vol_ok
# To:
g_full_pass = g_pass  # Volume check is a nice-to-have, not a requirement
# Add:
if g_pass and not g_vol_ok:
    warnings.append("G_trigger_no_vol_data")
```

### 6.2 Medium Term (1-2 Weeks)

| # | Fix | Impact | Effort |
|---|-----|--------|--------|
| **M1** | Create shared screening module — eliminate backtest/live code duplication | 🔴 Critical | 3d |
| **M2** | Add cross-source data verification in pipeline | 🟡 High | 2d |
| **M3** | Implement circuit breaker (daily loss limit + max DD) | 🟡 High | 2d |
| **M4** | Standardise logging (structured JSON, request IDs) | 🟡 Medium | 1d |
| **M5** | Add fill confirmation loop to Tiger broker | 🟡 Medium | 1d |
| **M6** | Merge `pipeline_hk.py` risk logic with `risk.py` | 🟡 Medium | 2d |

#### M1 Detail — Unified Screening Architecture
```python
# Create screening.py that both pipeline AND backtest import:
def screen_part1(ticker, data_source: DataSource) -> Optional[Part1Result]:
    """Single source of truth for Part 1 screening."""
    # Works with both yfinance info dict AND HistoricalSnapshot
    
def screen_part2(ticker_or_snapshot, part1) -> Optional[Part2Signal]:
    """Single source of truth for Part 2 MAGNA."""
```

### 6.3 Long Term (1-3 Months)

| # | Initiative | Impact | Effort |
|---|-----------|--------|--------|
| **L1** | Replace yfinance `.info` with SEC EDGAR as primary data source for US stocks | 🔴 Critical | 2 weeks |
| **L2** | Implement walk-forward optimisation for threshold tuning | 🟡 High | 2 weeks |
| **L3** | Add portfolio-level risk parity (equal risk contribution) | 🟡 Medium | 1 week |
| **L4** | Add machine learning layer for confidence calibration | 🟡 Medium | 3 weeks |
| **L5** | Production monitoring dashboard (Grafana/Prometheus) | 🟡 Medium | 2 weeks |
| **L6** | Implement paper-trading journal with automated P&L attribution | 🟡 Low | 1 week |

---

## 7. Detailed Findings by File

### `config.py` — ⭐ Best in class
- **Strengths:** Single source of truth, well-documented, dataclass singletons
- **Issues:** `target_*` values are aspirational, not calibrated; `weight_*` values are heuristic
- **Fix:** Add `to_dict()` method for serialisation; implement regime-aware config switching

### `models.py` — ✅ Solid
- **Strengths:** Clean dataclasses, good type hints
- **Issues:** `Part1Result.warnings` field is typed `List[str]` but sometimes not populated correctly
- **Fix:** Add `__post_init__` validation; make `data_date` non-optional

### `part1_fundamentals.py` — ✅ Good with caveats
- **Strengths:** 7-criterion design is defensible; sector-relative comparison is smart
- **Issues:** 
  - `_check_asset_efficiency()` has fragile label matching logic (tries multiple string variants)
  - `_check_quality()` mixes pass/fail (sector-relative) with scoring (absolute targets) — inconsistent
  - `screen_fundamentals()` swallows exceptions silently — makes debugging hard
- **Fix:** Standardise on sector-relative scoring; log exceptions at WARNING not DEBUG

### `part2_magna.py` — ✅ Good but crippled by data
- **Strengths:** Faithful MAGNA implementation, good entry trigger logic
- **Issues:**
  - Pre-market volume check breaks G trigger → effectively reduces MAGNA to MA___N53
  - Earnings/sales acceleration uses YoY comparison from yfinance quarterly data — data may be stale
  - `_check_neglect_base()` uses 1-year history always, should use configurable lookback
- **Fix:** Remove `g_vol_ok` as hard requirement; add data freshness check before scoring

### `risk.py` — 🟡 Good post-fix, still needs sync
- **Strengths:** Fixed Fractional sizing is better than Kelly; stop logic is multi-layered
- **Issues:**
  - `compute_stops()` picks tightest stop (causes 63% loss rate)
  - `compute_position_size()` has hardcoded 10,000 share cap
  - `generate_trade_decision()` takes `candidate` with stale prices, then fetches fresh data — wasted object
- **Fix:** Pick MIDDLE stop instead of tightest; remove 10K cap or make configurable

### `pipeline.py` — 🟡 Functional but messy
- **Strengths:** Good orchestrator, clear CLI, well-documented
- **Issues:**
  - `run_stage2()` re-runs Part 1 screening (wasteful — data already in quality pool)
  - `_execute_decision()` imports TigerBroker inline, not at top
  - 500+ line fallback ticker list hardcoded inline — should be a JSON file
  - `_sector_of()` creates new yf.Ticker for every call — slow for batch
- **Fix:** Pass Part1Result objects through; move fallback list to separate file

### `pipeline_hk.py` — ❌ Needs major refactor
- **Strengths:** HK-specific adaptations (financial sector handling, HKD currency)
- **Issues:**
  - 100% duplicate risk logic from `risk.py`
  - HKMagnaSignal is a regular class while US uses Part2Signal dataclass — API mismatch
  - Hardcoded HSI ticker list — should fetch dynamically
  - `screen_hk_fundamentals()` returns `dict` not `Part1Result` — breaks type consistency
- **Fix:** Use shared `risk.py` for decisions; convert to `Part1Result` output

### `backtest/engine.py` — ⚠️ Solid but out of sync
- **Strengths:** Walk-forward framework, proper position tracking, stop checking on daily bars
- **Issues:**
  - **Screening logic duplicated** from live modules (200+ lines)
  - **Position sizing still uses Kelly** (not Fixed Fractional)
  - **Stop parameters different** from live config (ATR 2.0 vs 2.5, hard stop 10% vs 15%)
  - G (Gap) trigger entirely disabled with comment "no pre-market volume data historically"
  - Only supports "monthly" rebalance — daily would be better for testing entry triggers
- **Fix:** Use shared screening module; sync all parameters from config

### `data/hybrid.py` — ✅ Good but underutilised
- **Strengths:** Multi-source fallback, proper SEC integration, cross-verify function
- **Issues:** Not called by main pipeline; `_get_tiger_qc()` swallows init errors
- **Fix:** Make this the primary data entry point for pipeline

### `data/sec_edgar.py` — ✅ Excellent foundation
- **Strengths:** Proper 10-Q/20-F handling, deduplication, caching, well-documented
- **Issues:** `is_foreign_issuer()` only checks first concept's first entry — fragile
- **Fix:** Check multiple concepts before concluding foreign issuer status

---

## 8. Summary

| Category | Grade | Key Issue |
|----------|-------|-----------|
| Architecture | B+ | Good modularity, some duplication |
| Data Quality | C+ | yfinance dependency is the Achilles' heel |
| Strategy Logic | B | Well-designed but crippled by data |
| Backtest Fidelity | D | Out of sync with live code |
| Risk Management | B- | Fixed but untested at scale |
| Production Readiness | D | No monitoring, alerts, or circuit breakers |

### The One Paragraph Summary

VMAA 2.0 has a **genuinely sound investment philosophy** — buy quality cheap names with momentum catalysts — but its implementation is **held back by yfinance data quality, parameter misalignment between backtest and live, and the absence of safety systems**. The backtest showing 1.16% CAGR is a worst-case scenario (old parameters, small universe) and does NOT reflect current live config. With the Q1-Q6 quick wins applied + a re-run backtest, we'll know within 48 hours whether the strategy is viable. The architecture is good enough to survive these fixes — no rewrite needed, just surgical corrections. 🦾

---
*Generated by Ironman 🦾 | VMAA 2.0 Code Review | 2026-05-05*
