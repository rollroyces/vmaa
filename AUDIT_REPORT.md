# VMAA 2.0 代码审计报告
**日期:** 2026-05-07  
**审计范围:** 完整的 VMAA Python 代码库（12 个核心文件）  
**审计员:** 量化交易代码审计员

---

## 🔴 关键问题 — 影响交易决策/P&L 的错误

### 1. 回测前瞻偏差：当前基本面数据泄露
**文件:** `backtest/engine.py`，第 115-170 行  
**函数:** `HistoricalSignalGenerator.cache_yfinance()` 和 `_get_cached_info()`

回测引擎预取**当前** yfinance `.info` 数据（`cache_yfinance()` 第 115 行），并将其作为历史基本面数据的代理。`_get_cached_info()`（第 147 行）以 `cached = self._yf_cache.get(ticker, {})` 开头，仅覆盖价格相关字段。**所有基本面比率**——ROA、ROE、账面价值、EBITDA、FCF、债务/股权——都来自当前数据。

**影响:** 回测使用今天的基本面数据模拟 2020 年的交易决策。这意味着回测中的公司拥有与今天相同的盈利、现金流和估值比率。这对 2020 年来说是完全前瞻性的信息，回测结果不具有真实性。

**修复:** 从 `backtest/data.py` 的 `_build_snapshot()` 中获取已存在的历史财务报表数据。`HistoricalSnapshot` 数据类**已包含**从 `balance_sheet`、`cashflow` 和 `quarterly_financials` 提取的字段，并在日期截断之前。使用 `snapshot` 数据来构建信息字典，而不是当前的 yfinance 信息。将 `_get_cached_info()` 替换或重新架构为仅从 `HistoricalSnapshot` 提取。

---

### 2. 回测：市值推断有误
**文件:** `backtest/data.py`，约第 258 行  
**函数:** `_build_snapshot()`

```python
current_mcap = info.get('marketCap', 0) or 0
current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0) or 0
if current_mcap > 0 and current_price > 0:
    market_cap = current_mcap * (float(row['Close']) / current_price)
```

这通过价格比率缩放当前的市值，**假设流通股数量不变**。对于进行过稀释/回购的公司，这会产生显著误差。例如，一家自 2020 年以来回购了 30% 股票的公司，其 2020 年的市值将是今天的 70%。

**修复:** 优先使用 `shares * historical_price`。如果快照中有 `shares_outstanding`，直接使用它。如果不行，使用 yfinance 的 `impliedSharesOutstanding` 字段。记录此限制并在回测输出中添加警告横幅。

---

### 3. backtest/data.py 中债务/股权计算的维度不匹配
**文件:** `backtest/data.py`，约第 326 行  
**函数:** `_build_snapshot()`

```python
debt_to_equity = info.get('debtToEquity', 0) or 0
if not debt_to_equity and book_value > 0 and total_debt > 0:
    debt_to_equity = (total_debt / book_value) * 100
```

这里 `book_value` 是**每股**账面价值（来自 `info.get('bookValue', 0)`），而 `total_debt` 是原始美元金额。这是**维度不匹配**——将数十亿美元的总债务除以每股 50 美元的账面价值，将得到一个数量级错误的结果。

**修复:** 与 `total_debt` 处于同一数量级的 `total_equity` 值：从资产负债表中提取 `Total Stockholder Equity`，并用它来除 `total_debt`。

---

### 4. 收益认证逻辑有缺陷（可能偏差 FCF/NI > 50%）
**文件:** `part1_fundamentals.py`，约第 290-314 行  
**函数:** `_check_earnings_authenticity()`

```python
fcf_ni = fcf / ni if ni > 0 else (fcf / abs(ni) if fcf > 0 else 0)
```

当净收入为负且 FCF 为正时，这会产生一个**正**比率（例如，NI = -$100M，FCF = $50M → fcf_ni = 0.5 → **通过**）。一家亏损公司绝不应获得高的"真实收益"分数。由于昂贵的 SBC 或营运资本变化，FCF 可以暂时为正，但这并不能使收益真实。

**修复:** 当净收入为负时：如果 FCF 为正，仍然返还失败。将其改为 `fcf_ni = fcf / max(ni, 1.0)` 并使其失败，或者对于赤字为负的 NI 明确返还 `passed = False`。

---

### 5. 自适应止损 PTL 条件不可达
**文件:** `risk_adaptive.py`，约第 83-87 行  
**函数:** `compute_stops_adaptive()`

```python
if ptl < 1.10:
    base_mult += 0.5
elif ptl < 1.05:
    base_mult += 1.0  # 在接近 52 周低点时给予最大空间
```

`ptl < 1.10` 会捕获所有 `<1.05` 的情况，然后才到达 `elif ptl < 1.05`。`+1.0` 的"接近 52 周低点"逻辑永远不会触发。

**修复:** 翻转顺序：
```python
if ptl < 1.05:
    base_mult += 1.0
elif ptl < 1.10:
    base_mult += 0.5
```

---

### 6. 回测 DRY_RUN 跳过安全检查
**文件:** `pipeline.py`，`_execute_decision()`，第 321-328 行

```python
if dry_run:
    return {'executed': True, 'reason': 'dry_run'}
```

此提前返回**绕过**了所有验证——成本检查、现金储备检查、持仓数量限制。这意味着模拟运行会报告执行了实际会被拒绝的交易。

**修复:** 移除提前返回。在函数末尾附近，设置 `return {'executed': True, 'reason': 'dry_run'}`，但保留所有安全检查。

---

### 7. 回测入场价格是收盘价——重新平衡时的存活偏差
**文件:** `backtest/engine.py`，`_execute_entry()`，约第 625 行

```python
entry_price = p1.current_price  # 这是快照的收盘价
```

在每日/每周/每月的重新平衡中，入场发生在条形图的**结束时**，此时所有信息已知。这赋予回测一种微小的前瞻性优势——如果在同一天以收盘价入场，你无法捕捉到日内波动。

**修复:** 使用下一根 K 线的开盘价（`next_day_open`），或应用明确的滑点模型来近似 t+1 的入场。

---

## 🟡 高优先级 — 显著改进

### 8. VADER 不适用于金融新闻
**文件:** `part3_sentiment.py`，第 200-250 行  
**严重性:** 高

VADER（Valence Aware Dictionary and sEntiment Reasoner）是在社交媒体/电影评论上训练的，而不是金融文本。像"股票因强劲财报飙升 15%，公司下调指引"这样的标题，VADER 会给出正分，因为"飙升"、"强劲"和"增长"占主导。金融新闻中的负面内容通常隐藏在技术语言中（"指引下调"、"保证金压缩"、"去库存"），而 VADER 对此视而不见。备用方案 `_simple_sentiment` 仅使用 38 个单词，更为糟糕。

**修复:**
- 添加 `finnhub`/`polygon.io` 情绪端点的集成（如果订阅）
- 作为最小可行修复：在 `_simple_sentiment` 中添加金融专用词典（`finbert` 或 `FinVADER`）
- 记录这是一个已知限制

---

### 9. 情绪平均化掩盖了极端信号
**文件:** `part3_sentiment.py`，`_news_sentiment()`

```python
avg_score = np.mean(scores) if scores else 0.0
```

一篇极其负面的文章和 9 篇中性的文章 → 情绪"略微看跌"。价值均值回归策略需要一个**真实的**情绪层来检测恐慌。求平均会消除极端情况。一个 `min(scores)` 或偏度指标将更有效地捕捉恐慌/贪婪。

**修复:** 添加一个补充指标：`extreme_negative_ratio` = 得分 < -0.6 的比例。如果 `extreme_ratio > 0.30` → 触发 `NEWS_EXTREME_NEGATIVE` 信号。这已经存在（`news_negative_pct > 0.60`），但使用 VADER 的 -0.05 阈值而不是复合的 -0.6 导致了稀释。

---

### 10. 港股的 MAGNA 不一致——使用当前信息而非季度数据
**文件:** `pipeline_hk.py`，`screen_hk_magna()`

```python
earnings_growth = float(info.get('earningsGrowth', 0) or 0)
rev_growth = float(info.get('revenueGrowth', 0) or 0)
```

美股管道（`part2_magna.py`）使用来自 `t.quarterly_financials` 的实际季度增长，并计算 QoQ 加速度。港股版本使用来自 yfinance 信息的预计算增长指标（trailing vs. forward，不透明）。这从根本上就不一致——港股管道测量的是不同的东西。

**修复:** 构建设票的季度财务数据（如美股版本），并对港股 .HK 适配器使用相同的 `_check_earnings_accel` 逻辑。

---

### 11. 回测中的"无重新筛选"模式无效
**文件:** `backtest/engine.py`，`_do_rebalance()`

```python
if self.config.re_screen_fundamentals:
    p1 = self.signal_gen.screen_part1(snapshot)
else:
    p1 = None
if p1 is None:
    continue
```

当 `re_screen_fundamentals=False` 时，引擎总是跳过（p1=None → continue）。`quality_pool_path` 配置从未被读取或使用。这使得重新筛选切换毫无意义——您**必须**在回测中启用重新筛选，或者获得零笔交易。

**修复:** 实现从预先计算的质量池中加载，或移除死代码。

---

### 12. 回测中的入场信号可能无限排队
**文件:** `backtest/engine.py`，`_do_rebalance()`

`candidates` 列表是在每个重新平衡日期重建的。但每次都会重新运行完整的第 1 部分和第 2 部分筛选。第 1 部分通过但第 2 部分未通过的股票会无限期地留在质量池中，并且在每个周期中被重新计算，而没有跟踪信号降级或停滞的逻辑。

**修复:** 添加一个"自上次第 2 部分信号以来的天数"计数器。仅对近期有活跃信号的股票运行第 2 部分。这可以提高性能（50-100 个股票的第 1 部分 + 第 2 部分在一个完整的标普 500 中是昂贵的）。

---

### 13. 头寸规模：min() 逻辑可以严格限制
**文件:** `risk.py`，`compute_position_size()`

```python
adj_shares = int(raw_shares * market.position_scalar)
quantity = min(raw_shares, alloc_shares, adj_shares)
```

由于 `adj_shares = raw_shares * scalar` 且 `scalar ≤ 1.0`，`adj_shares ≤ raw_shares` 总是成立。`min()` 总是选择 `adj_shares` 或 `alloc_shares`，使 `raw_shares` 变得多余。在高波动市场中，`scalar=0.5`，头寸规模减半两次（通过标量然后通过分配上限）——这可能过于保守。

**修复:** 澄清逻辑：
```python
# 1. 根据固定分数风险规模
risk_shares = int(risk_capital / risk_per_share)
# 2. 上限分配
alloc_shares = int(max_alloc / entry_price)
# 3. 市场调整
market_shares = int(risk_shares * market.position_scalar)
# 4. 取中间值（既不是最激进也不是最保守）
quantity = int(np.median([risk_shares, alloc_shares, market_shares]))
```

---

### 14. 报告中的字段名称不匹配
**文件:** `report.py`，`generate_report()`

```python
f"Candidates: {pipeline.get('candidates_found', 0)} | "
f"Decisions: {pipeline.get('decisions_made', 0)} | "
```

实际管道输出使用：
- `candidates`（计数，不是 "candidates_found"）
- `decisions`（计数，不是 "decisions_made"）

这些将始终显示为 0。

**修复:** 更正为 `pipeline.get('candidates', 0)` 和 `pipeline.get('decisions', 0)`。

---

### 15. 回测中的第 2 部分差距仅找到第一个结果
**文件:** `part2_magna.py`，`_check_gap_up()`

```python
if gap >= P2C.gap_min_pct:
    gap_detected = True
    gap_pct = max(gap_pct, gap)
    gap_day_volume = int(recent['Volume'].iloc[i])
    # ...
    break  # 经过第一个差距后中断
```

这会找到第一个差距，但可能不是最大的那个。如果第 i-2 天有一个 4.5% 的差距，第 i-1 天有一个 8% 的差距，它将捕获 4.5% 的差距并中断。`max(gap_pct, gap)` 对此进行了部分缓解，但成交量计算使用的是第一个差距的成交量，而不是最大差距的成交量。

**修复:** 不要中断。遍历所有差距，记录最佳（差距*成交量）组合。

---

## 🔵 中等 — 显著改进

### 16. 全局可变单例配置
**文件:** `config.py`，底部

```python
P1C = Part1Config()
P2C = Part2Config()
RC = RiskConfig()
PC = PipelineConfig()
```

这些是模块级单例。如果任何代码修改了 `P1C.min_bm_ratio = 0.5`，它会影响所有模块。对于测试或调优来说是危险的。

**修复:** 使配置通过依赖注入可注入，或使用冻结数据类。或者像 `part3_sentiment.py` 的 `SENT_CONFIG` 那样添加一个配置获取函数。

---

### 17. `_compute_atr` 重复
**文件:** `risk.py` 第 320 行和 `risk_adaptive.py` 第 130 行

两个文件中相同的 ATR 实现。`backtest/engine.py` 中还有一个内联版本。

**修复:** 移动到一个共享的 `utils.py` 或 `indicators.py` 模块。

---

### 18. 第 2 部分中的 magna_score 类型不匹配
**文件:** `models.py`，`Part2Signal` 数据类

```python
magna_score: int = 0  # 0–10 综合评分
```

但第 2 部分使用**分级评分**与半分数（0.5, 1.5）。代码通过 `int(sales_score)` 截断它们。因此，EPS 增长 12% → `eps_score = 1.0`（被截断），丢失了区分粒度。

**修复:** 在 `magna_score: float` 和 `trigger_signals` 中使用浮点数，而不仅仅是整数点。

---

### 19. `batch_sentiment()` 中没有重试逻辑
**文件:** `part3_sentiment.py`，`batch_sentiment()`

```python
except Exception as e:
    logger.warning(f"Sentiment failed for {ticker}: {e}")
    results[ticker] = SentimentResult(ticker=ticker)
```

一个失败的网络调用 → 为该股票分配一个空的 `SentimentResult`（composite_score=0.0，NEUTRAL）。这是一种无声的数据丢失。该股票仍然在管道中传递，情绪没有影响。

**修复:** 添加 1 次重试。或者将 `sources_available=0` 标记为需要特别关注的股票。

---

### 20. 价格获取回退逻辑不一致
**文件:** 多个文件

`_get_price()` 存在于 `part1_fundamentals.py` 中。在 `part2_magna.py`、`pipeline_hk.py` 和 `part3_sentiment.py` 中内联了相同的逻辑。每个版本的回退顺序略有不同。

**修复:** 单个 `get_price(info)` 工具函数。

---

### 21. 港股管道中的硬编码 HKD 头寸规模
**文件:** `pipeline_hk.py`，`compute_hk_trade_decision()`

```python
position_size = min(int(80000 * scalar / price), 5000)
```

80,000 港币的硬上限是硬编码的。对于像腾讯（~350 港币）这样的股票，这提供约 228 股，而对于像汇丰银行（~65 港币）这样的股票，这提供约 1,230 股。应该可配置。

**修复:** 添加到配置中，作为 `HK_RC.max_position_size_hkd`。

---

### 22. 回测未清持仓值缺乏日内数据
**文件:** `backtest/engine.py`，`_check_exits()`

```python
bar_high = price
bar_low = price
# ...
if not bar.empty:
    bar_high = float(bar['High'].iloc[0])
    bar_low = float(bar['Low'].iloc[0])
```

在非交易日，这会回退到收盘价。止损可能**在**周五收盘价和周一开盘价之间被触发，而高频路径假设日内检查会捕获它。对于每日条形图回测来说，这是可以接受的，但值得记录。

---

### 23. `compute_confidence()` 中缺少市场制度权重
**文件:** `backtest/engine.py`，`_compute_confidence()`

回测版本的置信度计算使用固定的 25% 市场权重（第 10-15 点），而实时版本（`risk.py` 中的 `compute_confidence()`）使用可变的制度重量。这两种实现随着时间的推移会偏离。

**修复:** 将两个文件中的置信度评分重构为单个共享函数。

---

### 24. 0 个分析师的非空分析师计数
**文件:** `part3_sentiment.py`，`_analyst_sentiment()`

```python
if analyst_count < 3:
    score *= (analyst_count / 3)
```

当 `analyst_count = 0` 时，`score *= 0/3 = 0`。这没问题。但对于 `analyst_count = 1`，`score *= 0.33`，接近 0。衰减过于激进——在有 1-2 个分析师的小盘股中，分析师情绪几乎完全被抑制。

**修复:** 使用 `min(1.0, analyst_count / 3)` 或平方根缩放：`score *= sqrt(analyst_count / 3)`。

---

## 🔵 低优先级 — 锦上添花

### 25. 投资回报率目标为零时除以零
**文件:** `part1_fundamentals.py`，`_check_quality()`，约第 155 行

```python
roa_score = min(result['roa'] / P1C.target_roa, 1.0) if P1C.target_roa > 0 else 0.5
```

当 `target_roa = 0` 时，如果该股票通过了 ROA 检查（任何 >= 0 的值都通过），它会盲目地分配 0.5 分。如果配置被改为排除 ROA 权重，这就会显得不协调。

### 26. 魔术数字：资产负债表中"最近 4 个季度"
**文件:** `part1_fundamentals.py`，`_check_asset_efficiency()`

```python
assets_latest = float(total_assets.iloc[0])
assets_prev = float(total_assets.iloc[1])
```

假设 `.iloc[0]` = 最新季度，`.iloc[1]` = 上一季度。对于 yfinance 来说，这是正确的，但在注释中记录这一点对于可维护性来说很重要。

### 27. `pipeline.py` 中已弃用的 `_sector_of()` 函数调用 yfinance
**文件:** `pipeline.py`，`_sector_of()`

```python
info = yf.Ticker(ticker).info
```

这为每个股票触发新的 yfinance 调用，仅用于检索行业。缓存或使用来自 `Part1Result` 的现有数据，其中已经包含行业信息。

### 28. 港股管道中的 `np_safe()` 嵌套为闭包
**文件:** `pipeline_hk.py`，`run_hk_pipeline()`

`np_safe()` 是在函数内部定义的，作为一种方法但未复用。将其移动到一个共享的序列化工具中。

### 29. 回测中同一函数的两个不同 `_compute_confidence()` 版本
在 `risk.py` 中有一个版本，在 `backtest/engine.py` 中有一个几乎相同的副本。它们现在*已经*分歧（回测版本添加了情感，但权重不同）。这在将来会导致混淆。

### 30. 第 2 部分中第 1 部分数据的已弃用硬刷新
**文件:** `pipeline.py`，`run_stage2()`

```python
# 重新运行第 1 部分以获得完整的 Part1Result 对象
for i, ticker in enumerate(quality_pool_tickers):
    p1 = screen_fundamentals(ticker)
```

每个股票都会重新调用 yfinance。应该使用来自第 1 阶段保存的 `Part1Result` 对象，而不是再次获取它们。如果自第 1 阶段以来数据发生了变化，这还会创建一个数据新鲜度问题。

---

## 总结统计

| 类别 | 数量 |
|--------|-------|
| 🔴 关键 | 7 |
| 🟡 高 | 8 |
| 🔵 中等 | 9 |
| 🔵 低 | 6 |
| **总计** | **30** |

## 最重要的操作项（按影响排序）

1. **修复前瞻偏差** (#1, #2) — 回测结果不可靠
2. **修复债务/股权维度不匹配** (#3) — 错误筛选了基本面
3. **修复收益认证逻辑** (#4) — 可能让亏损公司通过
4. **修复自适应止损中不可达的条件** (#5) — 功能失效
5. **修复 DRY_RUN 安全检查** (#6) — 虚假报告
6. **修复入场价格偏差** (#7) — 轻微但真实的回测偏倚
7. **修复报告字段名称** (#14) — 报告显示为零
8. **修复第 2 部分差距检测** (#15) — 可能会错过更强的差距
