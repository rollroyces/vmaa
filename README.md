# VMAA v3 — WIDE_STOP Strategy 🦾

**Multi-market quantitative trading framework — US + Hong Kong.**

VMAA is a three-stage value + momentum + sentiment framework that screens for quality at value prices, triggers entries on MAGNA momentum acceleration, filters through multi-source sentiment analysis, and manages risk with backtest-optimized WIDE_STOP parameters.

WIDE_STOP is the winner of 8 backtest experiments: full TP exit, wide stops for mean-reversion, no time limits. Built for paper trading on Tiger Trade; live execution is opt-in.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    VMAA v3 — WIDE_STOP Pipeline                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Layer (yfinance 📈 + SEC 🏛️ + Tiger 🐅 + Tushare 🇨🇳)     │
│         │                                                        │
│         ▼                                                        │
│  Stage 1: Part 1 — Quality Screening (7 criteria)                │
│         │  → Quality Pool (~26% pass US, ~75% pass HK)           │
│         ▼                                                        │
│  Stage 2: Part 2 — MAGNA 53/10 Momentum                         │
│         │  → Entry-ready Candidates                              │
│         ▼                                                        │
│  Stage 3: Part 3 — Sentiment Analysis (5 sources)                │
│         │  → Filtered Buy Signals                                │
│    ┌────┼────┐                                                   │
│    ▼    ▼    ▼                                                   │
│ 📐    🎯    💰   ← Optional Engines                              │
│ Tech  Chip  Earnings                                             │
│    └────┼────┘                                                   │
│         ▼                                                        │
│  Risk Management (Quarter-Kelly + WIDE_STOP)                     │
│         │                                                        │
│         ▼                                                        │
│  Trade Decision (BUY/HOLD/AVOID)                                 │
│         │                                                        │
│         ▼                                                        │
│  Tiger Trade Execution 🐅                                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## WIDE_STOP Strategy 🎯

Winner of 8 backtest experiments (2026-05-07 tuning). The core insight: **partial fills destroy returns.**

| Parameter | Old | **WIDE_STOP** | Why |
|-----------|-----|:------------:|-----|
| TP1 Exit | 30% at +12% | **100% at +15%** | Partial fills bled the remaining 70% |
| Hard Stop | 15% | **25%** | Gives room for value mean-reversion |
| Time Stop | 120 days | **Disabled** | Let trades fully play out |
| Kelly Fraction | 0.25 | **0.15** | More conservative sizing |
| Max Position | 20% | **18%** | Better diversification |
| Max Concurrent | None | **5** | Prevents over-concentration |

### Backtest Results (30 Liquid US Stocks, 2022-2024)

| Period | SPY | VMAA WIDE_STOP | Max Drawdown | Profit Factor |
|--------|:---:|:-------------:|:------------:|:-------------:|
| 2022 Bear | -18.6% | **-9.7%** | -10.7% | — |
| 2023-24 Bull | +58.8% | +35.8% | -7.5% | 3.14 |
| **2022-24 Full** | +28.7% | **+15.5%** | -10.9% | 1.82 |

VMAA outperforms SPY in bear markets (-9.7% vs -18.6%) while capturing ~60% of bull market upside with significantly lower drawdown.

---

## Quick Start

```bash
# US daily scan (cron: 21:00 HKT weekdays)
python3 pipeline.py --full-scan

# US scan with sentiment (recommended)
python3 pipeline.py --full-scan --sentiment

# HK live scan
python3 pipeline_hk.py --full-scan

# HK with sentiment
python3 pipeline_hk.py --full-scan --sentiment

# Sentiment analysis for specific tickers
python3 -c "from part3_sentiment import batch_sentiment; print(batch_sentiment(['AAPL','MSFT']))"

# Backtest with WIDE_STOP config
python3 backtest/runner.py --tickers INMD,TMDX,CDRE --start 2022-01-01

# Backtest HK
python3 backtest/hk/hk_runner.py --full-scan

# Single engine analysis
python3 engine/demo.py --chip AAPL
python3 engine/demo.py --technical AAPL,MSFT
python3 engine/demo.py --screen AAPL,MSFT
```

---

## Core Pipeline

### Stage 1: Part 1 — Core Financial Fundamentals (Quality)

**7 criteria** that eliminate value traps and ensure genuine cash-generation ability:

| # | Criterion | Threshold | Purpose |
|---|-----------|-----------|---------|
| 1 | Market Cap | < $10B (turnaround) / < $250M (deep value) | Avoid mega-caps |
| 2 | Quality | B/M ≥ 0.3, ROA ≥ 0%, EBITDA margin ≥ 5% | Genuine value |
| 3 | FCF Yield | ≥ 3% (target 8%) | Cash generation |
| 4 | Safety Margin (PTL) | ≤ 1.30x (52-week low proximity) | Entry near support |
| 5 | Asset Efficiency | ΔAssets < ΔEarnings | Capital discipline |
| 6 | Interest Sensitivity | Flag high D/E, high beta, IR sectors | Macro awareness |
| 7 | FCF/NI Conversion | ≥ 50% (weight: 20% of score) | Earnings authenticity |

### Stage 2: Part 2 — MAGNA 53/10 (Momentum)

| Component | Signal | Weight | Entry Trigger |
|-----------|--------|:------:|---------------|
| **M**assive Earnings Accel | EPS growth ↑ ≥ 20% + accel | 2 pts | M+A = Entry |
| **A**cceleration of Sales | Revenue ↑ ≥ 10% + accel | 2 pts | M+A = Entry |
| **G**ap Up | > 4% gap + volume ≥ 1.5x avg | 2 pts | G = Entry |
| **N**eglect/Base | Sideways ≥ 3mo, ≤ 30% range, declining vol | 1 pt | — |
| **5** Short Interest | Ratio ≥ 3 (1pt) / ≥ 5 (2pts) | 0-2 pts | — |
| **3** Analyst Target | ≥ 3 analysts, target ≥ 15% above | 1 pt | — |
| Cap **10** | Market Cap < $10B | Prerequisite | — |
| IPO **10** | Listed ≤ 10 years | Prerequisite | — |

**Graduated Growth Scoring**: Partial credit for near-miss thresholds — produces 3× more signals than binary pass/fail.

### Stage 3: Part 3 — Sentiment Analysis 🆕

**5-source multi-dimensional sentiment scoring** (weighted composite):

| Source | Weight | Data | Purpose |
|--------|:------:|------|---------|
| Analyst Consensus | 25% | yfinance recommendations + targets | Professional outlook |
| News Sentiment | 30% | VADER NLP on headlines | Real-time market narrative |
| Social Buzz | 20% | Reddit mentions + trend | Retail sentiment |
| Technical Sentiment | 15% | Price-momentum indicators | Market psychology |
| Insider/Institutional | 10% | Ownership flow | Smart money tracking |

**Composite Score**: -1.0 (max bearish) → +1.0 (max bullish)

**Key Signals:**
- 🟢 **CONTRARIAN_BUY**: Sentiment < -0.25 + strong fundamentals → Value opportunity (boosted entry)
- 🟡 **CROWDED_TRADE**: Sentiment > 0.65 → Caution flag
- 🔵 **SENTIMENT_DIVERGENCE**: Price ↓ but sentiment ↑ → Accumulation detected
- 🔴 **BEARISH_REJECT**: Sentiment < -0.40 + weak fundamentals → Entry rejected

**Modes:**
- **Historical backtest**: Drawdown-aware analyst scoring, no look-ahead bias
- **Live**: Real-time news headlines via yfinance + VADER NLP
- Integrated into both `pipeline.py` (US) and `pipeline_hk.py` (HK)

---

## Risk Management — WIDE_STOP

### Position Sizing (Quarter-Kelly)
- Kelly fraction: **0.15** (conservative)
- Max position: **18%** of portfolio
- Max concurrent positions: **5**
- Max per sector: **2**
- Cash reserve: 15%

### Stops
- **Hard stop**: 25% (wide — allows mean-reversion to work)
- **Trailing stop**: 12%, activates after +18% gain
- **ATR multiplier**: 3.0× (wider breathing room)
- **Time stop**: Disabled — no artificial exit deadline
- **Market regime**: Stops widen +0.5× ATR in high volatility

### Take Profit
- **TP1**: +15% — **SELL 100%** (full exit, no partial fills)
- **TP2**: +25% (secondary reference)
- **TP3**: +40% (tertiary reference)

> ⚠️ **The Lesson**: Selling 30% at +12% locked tiny wins while the remaining 70% bled out. Full exit at first meaningful target was the single biggest improvement in 8 backtest experiments.

---

## Markets

### 🇺🇸 US Market
- **Strategy**: Quality Value + MAGNA Momentum + Sentiment
- **Universe**: S&P 500 (503 stocks)
- **Pass rate**: ~26% quality → ~5 MAGNA → ~2 entry after sentiment
- **Data**: yfinance + SEC EDGAR
- **Broker**: Tiger Trade (paper + live)

### 🇭🇰 HK Market
- **Strategy**: Value Yield (HK-adapted)
- **Universe**: 90 HSI constituents
- **Pass rate**: ~75% quality, ~12 entry-ready
- **Data**: yfinance (.HK suffix) + Tushare supplementary
- **Currency**: HKD
- **Broker**: Tiger Trade (pending)

---

## Engines (Optional)

| Engine | Lines | Purpose | CLI |
|--------|-------|---------|-----|
| **Selection** | 3,941 | Multi-factor screening, condition combos, dynamic pools, auto rotation | `--screen` |
| **Risk** | 3,740 | VaR (6 models), Volatility (5 methods), Exposure (7 dimensions), Sizing (4 methods) | `--risk` |
| **Monitor** | 4,002 | Price alerts, conditional orders, anomaly detection, push notifications (Telegram) | `--monitor` |
| **Technical** | 4,134 | 25+ indicators (MA, MACD, RSI, KDJ, Bollinger, Ichimoku), custom formulas, signal aggregation | `--technical` |
| **Chip** | 2,332 | Volume Profile, Value Area, POC, cost distribution, money flow, S/R detection | `--chip` |
| **Earnings** | 2,757 | Consensus, surprise history, rating changes, earnings calendar | `--earnings` |

**Total: 20,906 engine lines** | **32K+ total codebase including Part 3**

---

## Changelog

### WIDE_STOP (2026-05-07)
| # | Change | Impact |
|---|--------|--------|
| 1 | **TP1 full exit** — Sell 100% at +15% (was 30% at +12%) | #1 fix: partial fills destroyed returns |
| 2 | **Hard stop 25%** (was 15%) | Allows value mean-reversion to work |
| 3 | **Time stop disabled** | Trades play out fully, no artificial deadline |
| 4 | **Quarter-Kelly 0.15** (was 0.25) | More conservative sizing |
| 5 | **Part 3 Sentiment** — 5-source multi-dimensional analysis | Filters bearish traps, boosts contrarian buys |
| 6 | **Max 5 concurrent** + 18% per position | Prevents over-concentration |
| 7 | **Graduated MAGNA scoring** | 3× more signals than binary pass/fail |
| 8 | **Volatility-based bear stops** | 50% wider stops in turbulent markets |

### Structural Fixes (2026-05-05)
| # | Fix | Impact |
|---|-----|--------|
| 1 | **Analyst tracker** — no false positives on first observation | MAGNA score integrity |
| 2 | **G trigger** — real volume check replacing broken preMarketVolume | G signal now works |
| 3 | **Sector comparison** — 10% premium (not just ">= median") | Better quality selection |
| 4 | **Config weights** — FCF/NI 10%→20% | Aligned with requirements |
| 5 | **Stop selection** — median instead of tightest | 33% fewer hard stops |
| 6 | **Backtest engine** — uses live modules, no duplicate logic | Config changes propagate |

---

## Dependencies

```
Python 3.10+
numpy, pandas, yfinance, vaderSentiment
tigeropen (Tiger Trade SDK)
requests (SEC EDGAR API)
tushare (HK supplementary data)
```

---

## Configuration

All thresholds in `config.py`:
- `Part1Config` — Quality screening params
- `Part2Config` — MAGNA scoring params
- `RiskConfig` — WIDE_STOP strategy params
- `PipelineConfig` — Operational settings

Sentiment config in `part3_sentiment.py` (`SENT_CONFIG` dict).

Engine configs in:
- `engine/config.py` — Global engine settings
- `engine/risk/config/` — Risk engine YAML
- `engine/earnings/config.json` — Earnings engine
- `engine/chip/config.json` — Chip engine

---

## License

Private — rollroyces/vmaa
