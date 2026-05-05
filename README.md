# VMAA 2.0 — Value Mean-reversion Algorithmic Advisor

**Two-stage quantitative pipeline — Quality fundamentals → MAGNA momentum → Risk-managed execution.**

VMAA scans the S&P 500 (or any universe) through a disciplined two-stage filter, then generates risk-calibrated trade decisions with position sizing, stops, and confidence scoring. Built for paper trading on Tiger Trade; live execution is opt-in.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  VMAA 2.0 Pipeline                    │
├──────────────┬──────────────┬──────────────┬─────────┤
│   Stage 1    │   Stage 2    │    Risk      │ Execute │
│  Quality     │  Momentum    │  Management  │         │
│  (Part 1)    │  (Part 2)    │              │         │
├──────────────┼──────────────┼──────────────┼─────────┤
│ 7 criteria   │ MAGNA 53/10  │ Kelly sizing │ Tiger   │
│ Fundamentals │ Entry signal │ Stops + Conf │ Broker  │
│ → 14/503     │ → 2-5 entry  │ → Decisions  │ → Paper │
│   quality    │   ready      │   4 trades   │   /Live │
└──────────────┴──────────────┴──────────────┴─────────┘
```

### Stage 1: Core Financial Fundamentals (Part 1)
Ensures the company has genuine cash-generation ability — eliminates value traps.

| # | Criterion | Threshold | Purpose |
|---|-----------|-----------|---------|
| 1 | **Market Cap** | <$10B (turnaround) or <$250M (deep value) | Avoid mega-cap stagnation |
| 2 | **Quality** | B/M ≥0.3, ROA ≥0%, EBITDA margin ≥5% | Proven profitability |
| 3 | **FCF Yield** | ≥3% (target 8%) | Strong cash generation |
| 4 | **Safety Margin** | PTL ≤1.30x (near 52-week low) | Buy low, not chasing |
| 5 | **Asset Efficiency** | ΔAssets < ΔEarnings | Capital-efficient growth |
| 6 | **IR Sensitivity** | High D/E, beta, or IR-sensitive sector | Rate-risk awareness |
| 7 | **Earnings Authenticity** | FCF/NI ≥50% | Real earnings, not accounting |

**Output:** Quality Pool (typically 10–20 stocks from S&P 500) with composite quality scores.

### Stage 2: MAGNA 53/10 (Part 2)
Captures momentum and breakout signals on quality-verified stocks.

| Component | Description | Score |
|-----------|-------------|-------|
| **M** — Massive Earnings Acceleration | EPS growth ↑ ≥20% YoY acceleration | 2 pts |
| **A** — Acceleration of Sales | Revenue growth ↑ ≥10% YoY acceleration | 2 pts |
| **G** — Gap Up | >4% gap + pre-market vol >100K | ⚡ **Entry** |
| **N** — Neglect / Base | ≥3mo sideways, ≤30% range, declining vol | 1 pt |
| **5** — Short Interest | SI ratio ≥5 (2pts), ≥3 (1pt) | 1–2 pts |
| **3** — Analyst Coverage | ≥3 analysts, target ≥15% above current | 1 pt |
| **Cap 10** | Market cap <$10B | Required |
| **10** — IPO recency | IPO within 10 years | Required |

**Entry Triggers:**
- **G (Gap Up)** fires alone → immediate entry signal
- **M + A** both fire → entry signal (dual fundamental acceleration)
- Otherwise → MONITOR (in quality pool, waiting for trigger)

### Stage 3: Risk Management
- **Market regime detection**: SPY vs MA50, 20-day volatility, drawdown from high
- **Position sizing**: Quarter-Kelly with market-regime scalar (0.5x–1.0x)
- **Stop management**: ATR-based trailing stops with structural floors
- **Confidence scoring**: Weighted composite of quality + MAGNA + market fit
- **Sector limits**: Max 2 positions per sector, correlation checks

### Stage 4: Execution
- Tiger Trade OpenAPI integration (paper + live)
- Order types: MKT, LMT, STP
- Auto-stop placement on fill
- Portfolio & buying-power awareness

---

## Installation

```bash
# Clone
git clone https://github.com/rollroyces/vmaa.git
cd vmaa

# Python 3.11+ recommended
pip install -r requirements.txt
```

**Requirements:** `yfinance`, `numpy`, `pandas`, `tigeropen` (for broker execution only)

---

## Usage

### Quick Start

```bash
# Full dry-run scan (S&P 500)
python3 pipeline.py --full-scan

# Only Stage 1 — build quality pool
python3 pipeline.py --scan-part1

# Only Stage 2 — check quality pool for MAGNA signals
python3 pipeline.py --scan-part2

# Scan specific tickers
python3 pipeline.py --tickers AAPL,MSFT,GOOGL

# Russell 2000 universe
python3 pipeline.py --full-scan --source russell2000
```

### Live Trading

```bash
# LIVE mode — executes orders on Tiger Trade
python3 pipeline.py --full-scan --live

# Dry run first (default, safe)
python3 pipeline.py --full-scan --dry-run
```

### Portfolio Management

```bash
# Show current portfolio + risk dashboard
python3 pipeline.py --status
```

---

## Output

All results saved to `output/`:

```
output/
├── pipeline_result.json    # Full pipeline output (JSON)
├── quality_pool.json        # Stage 1 quality pool
├── magna_signals.json       # Stage 2 MAGNA signals
└── trade_decisions.json     # Final trade decisions
```

Generate a Telegram-friendly report:

```bash
python3 report.py output/pipeline_result.json
```

---

## Configuration

All thresholds centralized in `config.py`:

```python
# config.py — Part1Config
min_fcf_yield: float = 0.03          # Minimum 3% FCF yield
max_ptl_ratio: float = 1.30          # Max 30% above 52w-low
min_fcf_conversion: float = 0.50     # FCF/NI >= 50%

# config.py — Part2Config
gap_min_pct: float = 0.04            # 4% gap threshold
gap_min_premarket_vol: int = 100000  # Minimum pre-market volume
magna_entry_threshold: int = 5       # Minimum MAGNA score for entry

# config.py — RiskConfig
max_positions: int = 8              # Max concurrent positions
max_position_pct: float = 0.125     # 12.5% per position (1/8 Kelly)
max_sector_exposure: float = 0.25   # 25% max per sector
```

---

## Broker Setup (Tiger Trade)

1. Place `tiger_openapi_config.properties` and `tiger_openapi_token.properties` in `broker/`
2. Test connection:

```bash
python3 broker/test_connection.py
```

3. Dry-run before live — always.

---

## Module Map

| File | Role |
|------|------|
| `pipeline.py` | Two-stage orchestrator + CLI |
| `config.py` | All thresholds (Part1, Part2, Risk, Pipeline) |
| `models.py` | Shared dataclasses (Part1Result, Part2Signal, TradeDecision) |
| `part1_fundamentals.py` | Core Financial Fundamentals screener (7 criteria) |
| `part2_magna.py` | MAGNA 53/10 momentum screener |
| `risk.py` | Risk management, position sizing, stops, confidence |
| `analyst_tracker.py` | Analyst revision tracking |
| `report.py` | Telegram/Markdown report generator |
| `broker/tiger_broker.py` | Tiger Trade execution layer |
| `broker/vmaa_tiger_bridge.py` | VMAA ↔ Tiger signal bridge |

---

## Example Output (May 5, 2026)

```
📊 S&P 500 Full Scan — 503 stocks in 298s
💰 SPY $718.01 | MA50 ✓ | Vol NORMAL | Scalar 0.8x

Stage 1: 14/503 passed quality (2.8%)
Stage 2: 5 MAGNA signals, 2 entry-ready

⚡ BUY  IT   536sh @ $149.19  Stop $137.34  MAGNA 7/10  Q=74%
⚡ BUY  RVTY 906sh @ $88.27   Stop $80.79   MAGNA 6/10  Q=74%
📊 HOLD ARE        @ $41.14   MAGNA 3/10 — waiting for trigger

Executed: 2 | Skipped: 1 | Candidates: 5
```

---

## Disclaimer

**For educational and research purposes only.** This is not financial advice. Past performance does not guarantee future results. Paper trading results do not account for slippage, liquidity constraints, or market impact. Always consult a qualified financial advisor before making investment decisions.

---

## License

MIT © rollroyces
