# VMAA v3 — Value Mean-reversion Algorithmic Advisor

**Multi-market quantitative trading framework — US + Hong Kong.**

VMAA is a two-stage value + momentum framework that scans markets through a disciplined quality filter, triggers entries on momentum acceleration, manages risk with adaptive stops, and extends with 6 optional analysis engines (technical, chip, earnings, risk, selection, monitoring).

Built for paper trading on Tiger Trade; live execution is opt-in.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     VMAA v3 Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Data Layer (Tiger 🐅 + SEC 🏛️ + yfinance 📈)              │
│         │                                                   │
│         ▼                                                   │
│  Stage 1: Market Regime (SPY, VIX, MA50)                    │
│         │                                                   │
│         ▼                                                   │
│  Stage 2: Part 1 — Quality Screening (7 criteria)           │
│         │  → Quality Pool                                   │
│         ▼                                                   │
│  Stage 3: Part 2 — MAGNA 53/10 Momentum                     │
│         │  → Entry-ready Candidates                         │
│         │                                                   │
│    ┌────┼────┐                                              │
│    ▼    ▼    ▼                                              │
│ 📐    🎯    💰   ← Optional Engines                         │
│ Tech  Chip  Earnings                                        │
│    └────┼────┘                                              │
│         ▼                                                   │
│  Composite Score (weighted)                                 │
│         │                                                   │
│         ▼                                                   │
│  Risk Assessment (VaR + Adaptive Stop)                      │
│         │                                                   │
│         ▼                                                   │
│  Trade Decision (BUY/HOLD/MONITOR)                          │
│         │                                                   │
│         ▼                                                   │
│  Monitor Alerts (price/anomaly/notifications)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# US daily scan (cron: 21:00 HKT weekdays)
python3 scripts/daily_scan.py

# Full v3 pipeline with all engines
python3 engine/demo.py --full

# Backtest US
python3 backtest/runner.py --tickers INMD,TMDX,CDRE --start 2023-01-01

# HK live scan
python3 pipeline_hk.py --full-scan

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
| 2 | B/M, ROA, EBITDA | ≥ sector median × **1.1** (10% premium) | Quality vs peers |
| 3 | FCF Yield | ≥ 2% | Cash generation |
| 4 | Safety Margin (PTL) | ≤ 1.50x (52-week low proximity) | Entry near support |
| 5 | Asset Efficiency | ΔAssets < ΔEarnings | Capital discipline |
| 6 | Interest Sensitivity | Flag high D/E, high beta, IR sectors | Macro awareness |
| 7 | FCF/NI Conversion | ≥ 50% (weight: **20%** of score) | Earnings authenticity |

### Stage 2: Part 2 — MAGNA 53/10 (Momentum)

| Component | Signal | Weight | Entry Trigger |
|-----------|--------|--------|---------------|
| **M**assive Earnings Accel | EPS growth ↑ ≥ 20% + accel | 2 pts | M+A = Entry |
| **A**cceleration of Sales | Revenue ↑ ≥ 10% + accel | 2 pts | M+A = Entry |
| **G**ap Up | > 4% gap + **volume ≥ 1.5x avg** | 2 pts | G = Entry |
| **N**eglect/Base | Sideways ≥ 6 months, ≤ 30% range | 1 pt | — |
| **5** Short Interest | Ratio ≥ 3 (1pt) / ≥ 5 (2pts) | 0-2 pts | — |
| **3** Analyst Target | ≥ 3 analysts, target ≥ 15% above | 1 pt | — |
| Cap **10** | Market Cap < $10B | Prerequisite | — |
| IPO **10** | Listed ≤ 10 years | Prerequisite | — |

### Risk Management

**Adaptive Stop (Phase 1 — 2026-05-06):**
- Dynamic ATR multiplier based on price level (+1x for < $10 stocks)
- Dynamic hard stop (15% base, 18% for < $30, 22% for < $10)
- Market regime adjustment (HIGH vol → +0.5x ATR)
- Near 52w-low → +0.5x ATR (room for bounce)
- **Median stop selection** (not tightest — backtest showed 33% fewer stops)

**Position Sizing:**
- Fixed Fractional: 1.5% of portfolio × confidence scalar
- Max position: 20% of portfolio
- Market regime scalar (0.5-1.0x)

---

## Markets

### 🇺🇸 US Market
- **Strategy**: Quality Momentum
- **Universe**: 85 mid-cap stocks (< $10B, liquid)
- **Pass rate**: ~26% (22/85 quality)
- **Cron**: Daily @ 21:00 HKT (13:00 UTC)
- **Broker**: Tiger Trade paper/live

### 🇭🇰 HK Market
- **Strategy**: Value Yield
- **Universe**: 90 HSI constituents
- **Pass rate**: ~75% (68/90 quality) — looser thresholds, financial bypass
- **Data**: yfinance (.HK suffix)
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

**Total: 20,906 engine lines** | **32K+ total codebase**

---

## Backtest

### US Backtest
```bash
python3 backtest/runner.py --tickers <list> --mode monthly_rebalance
python3 backtest/runner.py --mode weekly_rebalance  # More signals
```

**Latest results (adaptive stop, 48 stocks, 4yr 2022-2025):**
| Metric | Tightest Stop | Median Stop | **Adaptive Stop** |
|--------|:-------------:|:-----------:|:-----------------:|
| Hard Stops | 6 (67%) | 4 (44%) | **2 (22%)** 🔥 |
| Win Rate | 33% | 56% | 44% |
| Profit Factor | 0.38 | 0.81 | **1.13** 🎯 |
| Net P&L | -$473 | -$114 | **+$24** ✅ |

### HK Backtest
```bash
python3 backtest/hk/hk_runner.py --full-scan
python3 backtest/hk/hk_runner.py --tickers 0700.HK,0388.HK
```

---

## Structural Fixes (2026-05-05)

| # | Fix | Impact |
|---|-----|--------|
| 1 | **Analyst tracker** — no false positives on first observation | MAGNA score integrity |
| 2 | **G trigger** — replaced broken preMarketVolume with real volume check | G signal now works |
| 3 | **Sector comparison** — 10% premium (not just ">= median") | Better quality selection |
| 4 | **Config weights** — FCF/NI 10%→20% per Royce spec | Aligned with requirements |
| 5 | **Stop selection** — median instead of tightest | 33% fewer hard stops |
| 6 | **Backtest engine** — uses live modules, no duplicate logic | Config changes propagate |

---

## Dependencies

```
Python 3.10+
numpy, pandas, yfinance
tigeropen (Tiger Trade SDK)
requests (SEC EDGAR API)
```

---

## Configuration

All thresholds in `config.py`:
- `Part1Config` — Quality screening params
- `Part2Config` — MAGNA scoring params
- `RiskConfig` — Stop/sizing params

Engine configs in:
- `engine/config.py` — Global engine settings
- `engine/risk/config/` — Risk engine YAML
- `engine/earnings/config.json` — Earnings engine
- `engine/chip/config.json` — Chip engine

---

## License

Private — rollroyces/vmaa
