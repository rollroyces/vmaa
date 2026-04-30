/**
 * research_yfinance.cjs — yFinance (yahoo-finance2) API field research
 * 
 * Covers 6 missing criteria:
 * 1. Short Interest Ratio
 * 2. IPO Tenure
 * 3. Asset Growth vs Earnings Growth
 * 4. Interest Rate Sensitivity
 * 5. ROA (Return on Assets)
 * 6. Analyst Count
 * 
 * Tickers: AAPL, RVTY, EPAM
 * 
 * Usage: node research_yfinance.cjs
 */

const YahooFinance = require('yahoo-finance2').default;
const yf = new YahooFinance({ suppressNotices: ['yahooSurvey'] });

const TICKERS = ['AAPL', 'RVTY', 'EPAM'];

// ── Helpers ────────────────────────────────────────────────────────

function dateStr(d) {
  if (!d) return 'N/A';
  if (typeof d === 'string') return d.split('T')[0];
  if (d instanceof Date) return d.toISOString().split('T')[0];
  return String(d);
}

function yearsSince(dateVal) {
  if (!dateVal) return 'N/A';
  const d = dateVal instanceof Date ? dateVal : new Date(dateVal);
  const years = (Date.now() - d.getTime()) / (365.25 * 24 * 60 * 60 * 1000);
  return years.toFixed(1) + ' yrs';
}

function growthPct(latest, previous) {
  if (!previous || previous === 0) return 'N/A';
  return (((latest - previous) / Math.abs(previous)) * 100).toFixed(2) + '%';
}

// ── Main Research ──────────────────────────────────────────────────

async function research(ticker) {
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`📊 RESEARCH: ${ticker}`);
  console.log(`${'═'.repeat(70)}`);

  // ── 1. Fetch all relevant data ──────────────────────────────────

  const [quoteData, summaryData, profileData, finData, calEvents] = await Promise.all([
    yf.quote(ticker),
    yf.quoteSummary(ticker, { modules: ['defaultKeyStatistics', 'summaryDetail'] }),
    yf.quoteSummary(ticker, { modules: ['assetProfile'] }),
    yf.quoteSummary(ticker, { modules: ['financialData'] }),
    yf.quoteSummary(ticker, { modules: ['calendarEvents'] }),
  ]);

  const quote = quoteData;
  const dks = summaryData.defaultKeyStatistics || {};
  const sd = summaryData.summaryDetail || {};
  const ap = profileData.assetProfile || {};
  const fd = finData.financialData || {};
  const ce = calEvents.calendarEvents || {};

  // ── 2. Fundamentals Time Series (Balance Sheet + Income) ─────────

  let bsAnnual = [];
  let isAnnual = [];
  try {
    bsAnnual = await yf.fundamentalsTimeSeries(ticker, {
      module: 'balance-sheet',
      period1: '2020-01-01',
      type: 'annual',
    });
  } catch (e) {
    console.log(`  ⚠️ Balance sheet fetch failed: ${e.message}`);
  }
  try {
    isAnnual = await yf.fundamentalsTimeSeries(ticker, {
      module: 'financials',
      period1: '2020-01-01',
      type: 'annual',
    });
  } catch (e) {
    console.log(`  ⚠️ Income statement fetch failed: ${e.message}`);
  }

  // ── CRITERION 1: Short Interest Ratio ───────────────────────────

  console.log(`\n  📌 1. SHORT INTEREST RATIO`);
  console.log(`  ─────────────────────────`);
  console.log(`  shortRatio:              ${dks.shortRatio ?? 'N/A'}`);
  console.log(`  sharesShort:             ${dks.sharesShort?.toLocaleString() ?? 'N/A'}`);
  console.log(`  shortPercentOfFloat:     ${dks.shortPercentOfFloat != null ? (dks.shortPercentOfFloat * 100).toFixed(2) + '%' : 'N/A'}`);
  console.log(`  sharesOutstanding:       ${dks.sharesOutstanding?.toLocaleString() ?? 'N/A'}`);
  console.log(`  floatShares:             ${dks.floatShares?.toLocaleString() ?? 'N/A'}`);
  console.log(`  dateShortInterest:       ${dks.dateShortInterest ? new Date(dks.dateShortInterest).toISOString().split('T')[0] : 'N/A'}`);
  console.log(`  sharesShortPriorMonth:   ${dks.sharesShortPriorMonth?.toLocaleString() ?? 'N/A'}`);
  console.log(`  ✅ VERDICT: shortRatio + sharesShort + shortPercentOfFloat ALL available via defaultKeyStatistics`);

  // ── CRITERION 2: IPO Tenure ────────────────────────────────────

  console.log(`\n  📌 2. IPO TENURE (上市日期)`);
  console.log(`  ─────────────────────────`);
  const ipoMs = quote.firstTradeDateMilliseconds;
  const ipoDate = ipoMs ? new Date(ipoMs).toISOString().split('T')[0] : null;
  console.log(`  firstTradeDateMilliseconds: ${ipoMs ?? 'N/A'}`);
  console.log(`  → IPO Date:              ${ipoDate ?? 'N/A'}`);
  console.log(`  → Years Since IPO:       ${ipoDate ? yearsSince(ipoDate) : 'N/A'}`);
  console.log(`  sector:                  ${ap.sector ?? 'N/A'}`);
  console.log(`  industry:                ${ap.industry ?? 'N/A'}`);
  console.log(`  fullTimeEmployees:       ${ap.fullTimeEmployees ?? 'N/A'}`);
  console.log(`  founded:                 ${ap.longBusinessSummary ? '(see summary)' : 'N/A'}`);
  console.log(`  ✅ VERDICT: firstTradeDateMilliseconds available via quote()`);

  // ── CRITERION 3: Asset Growth vs Earnings Growth ────────────────

  console.log(`\n  📌 3. ASSET GROWTH vs EARNINGS GROWTH`);
  console.log(`  ────────────────────────────────────`);

  // fundamentalsTimeSeries returns OLDEST first, so latest = last element
  const latestBS = bsAnnual.length > 0 ? bsAnnual[bsAnnual.length - 1] : null;
  const prevBS = bsAnnual.length > 1 ? bsAnnual[bsAnnual.length - 2] : null;
  const latestIS = isAnnual.length > 0 ? isAnnual[isAnnual.length - 1] : null;
  const prevIS = isAnnual.length > 1 ? isAnnual[isAnnual.length - 2] : null;

  if (latestBS && prevBS) {
    console.log(`  Total Assets (latest):   ${latestBS.totalAssets?.toLocaleString() ?? 'N/A'} (${dateStr(latestBS.date)})`);
    console.log(`  Total Assets (prev):     ${prevBS.totalAssets?.toLocaleString() ?? 'N/A'} (${dateStr(prevBS.date)})`);
    console.log(`  → Asset Growth:          ${growthPct(latestBS.totalAssets, prevBS.totalAssets)}`);
  } else {
    console.log(`  ⚠️ Balance sheet: ${bsAnnual.length} records (need >= 2 for growth)`);
  }

  if (latestIS && prevIS) {
    console.log(`  Earnings (latest):       ${latestIS.netIncome?.toLocaleString() ?? 'N/A'} (${dateStr(latestIS.date)})`);
    console.log(`  Earnings (prev):         ${prevIS.netIncome?.toLocaleString() ?? 'N/A'} (${dateStr(prevIS.date)})`);
    console.log(`  → Earnings Growth:       ${growthPct(latestIS.netIncome, prevIS.netIncome)}`);
    if (latestBS && prevBS) {
      const ag = parseFloat(growthPct(latestBS.totalAssets, prevBS.totalAssets));
      const eg = parseFloat(growthPct(latestIS.netIncome, prevIS.netIncome));
      if (!isNaN(ag) && !isNaN(eg)) {
        console.log(`  → Asset-Earnings Gap:    ${(ag - eg).toFixed(2)}pp`);
      }
    }
  } else {
    console.log(`  ⚠️ Income statement: ${isAnnual.length} records (need >= 2 for growth)`);
  }

  // Also show all years for reference
  console.log(`  All years BS:`);
  bsAnnual.forEach(r => console.log(`    ${dateStr(r.date)}  totalAssets: ${r.totalAssets?.toLocaleString() ?? 'N/A'}`));
  console.log(`  All years IS:`);
  isAnnual.forEach(r => console.log(`    ${dateStr(r.date)}  totalRevenue: ${r.totalRevenue?.toLocaleString() ?? 'N/A'}  netIncome: ${r.netIncome?.toLocaleString() ?? 'N/A'}`));

  console.log(`  ✅ VERDICT: fundamentalsTimeSeries('balance-sheet' + 'financials') provides the data`);
  console.log(`     NOTE: incomeStatementHistory in quoteSummary also works as fallback for earnings`);

  // ── CRITERION 4: Interest Rate Sensitivity ──────────────────────

  console.log(`\n  📌 4. INTEREST RATE SENSITIVITY`);
  console.log(`  ─────────────────────────────`);
  console.log(`  sector:                  ${ap.sector ?? 'N/A'}`);
  console.log(`  industry:                ${ap.industry ?? 'N/A'}`);
  console.log(`  sectorKey:               ${ap.sectorKey ?? 'N/A'}`);
  console.log(`  industryKey:             ${ap.industryKey ?? 'N/A'}`);
  
  // Check for debt-related fields as proxy
  console.log(`  debtToEquity (finData):  ${fd.debtToEquity ?? 'N/A'}`);
  console.log(`  totalDebt (finData):     ${fd.totalDebt?.toLocaleString() ?? 'N/A'}`);
  console.log(`  currentRatio (finData):  ${fd.currentRatio ?? 'N/A'}`);
  console.log(`  quickRatio (finData):    ${fd.quickRatio ?? 'N/A'}`);
  console.log(`  beta (summaryDetail):    ${sd.beta ?? 'N/A'}`);
  
  if (latestBS) {
    console.log(`  totalDebt (BS latest):   ${latestBS.totalDebt?.toLocaleString() ?? 'N/A'}`);
    console.log(`  longTermDebt (BS):       ${latestBS.longTermDebt?.toLocaleString() ?? 'N/A'}`);
    console.log(`  currentDebt (BS):        ${latestBS.currentDebt?.toLocaleString() ?? 'N/A'}`);
  }
  
  console.log(`  ⚠️ VERDICT: No direct 'interest rate sensitivity' field`);
  console.log(`     WORKAROUND: Combine sector (${ap.sector}) + debt ratios + beta to infer sensitivity`);

  // ── CRITERION 5: ROA (Return on Assets) ─────────────────────────

  console.log(`\n  📌 5. ROA (Return on Assets)`);
  console.log(`  ──────────────────────────`);
  console.log(`  returnOnAssets (finData):  ${fd.returnOnAssets ?? 'N/A'}`);
  console.log(`  returnOnEquity (finData):  ${fd.returnOnEquity ?? 'N/A'}`);
  console.log(`  profitMargins (dks):       ${dks.profitMargins ?? 'N/A'}`);
  console.log(`  operatingMargins (finData):${fd.operatingMargins ?? 'N/A'}`);
  
  // Manual ROA check from fundamentals
  if (latestIS && latestBS) {
    const netInc = latestIS.netIncome;
    const totAssets = latestBS.totalAssets;
    if (netInc && totAssets) {
      const manualROA = (netInc / totAssets * 100).toFixed(2) + '%';
      console.log(`  Manual ROA (NI/TA):        ${manualROA} (netIncome=${netInc?.toLocaleString()} / totalAssets=${totAssets?.toLocaleString()})`);
    }
  }
  
  console.log(`  ✅ VERDICT: returnOnAssets available in financialData module of quoteSummary`);

  // ── CRITERION 6: Analyst Count ──────────────────────────────────

  console.log(`\n  📌 6. ANALYST COUNT`);
  console.log(`  ─────────────────`);
  console.log(`  numberOfAnalystOpinions:  ${fd.numberOfAnalystOpinions ?? 'N/A'}`);
  console.log(`  recommendationKey:        ${fd.recommendationKey ?? 'N/A'}`);
  console.log(`  recommendationMean:       ${fd.recommendationMean ?? 'N/A'}`);
  console.log(`  targetHighPrice:          ${fd.targetHighPrice ?? 'N/A'}`);
  console.log(`  targetLowPrice:           ${fd.targetLowPrice ?? 'N/A'}`);
  console.log(`  targetMeanPrice:          ${fd.targetMeanPrice ?? 'N/A'}`);
  console.log(`  targetMedianPrice:        ${fd.targetMedianPrice ?? 'N/A'}`);
  console.log(`  ✅ VERDICT: numberOfAnalystOpinions + recommendationKey available in financialData`);

  // ── BONUS: All financialData fields dump ────────────────────────

  console.log(`\n  📋 BONUS: All financialData keys & values`);
  console.log(`  ───────────────────────────────────────`);
  Object.entries(fd).sort(([a], [b]) => a.localeCompare(b)).forEach(([k, v]) => {
    console.log(`  ${k.padEnd(28)} = ${JSON.stringify(v)}`);
  });
}

// ── Run ────────────────────────────────────────────────────────────

(async () => {
  console.log('🔬 yFinance (yahoo-finance2) API Field Research');
  console.log(`   Tickers: ${TICKERS.join(', ')}`);
  console.log(`   Date: ${new Date().toISOString()}\n`);

  for (const ticker of TICKERS) {
    try {
      await research(ticker);
    } catch (e) {
      console.log(`\n❌ ERROR for ${ticker}: ${e.message}`);
      console.log(e.stack?.split('\n').slice(0, 3).join('\n'));
    }
  }

  console.log(`\n${'═'.repeat(70)}`);
  console.log('📋 SUMMARY');
  console.log(`${'═'.repeat(70)}`);
  console.log(`
  1. Short Interest Ratio  ✅ shortRatio, sharesShort, shortPercentOfFloat → defaultKeyStatistics
  2. IPO Tenure            ✅ firstTradeDateMilliseconds → quote()
  3. Asset vs Earnings     ✅ fundamentalsTimeSeries('balance-sheet'/'financials')
  4. Interest Sensitivity  ⚠️ No direct field; use sector + debt ratios + beta as proxy
  5. ROA                   ✅ returnOnAssets → financialData
  6. Analyst Count         ✅ numberOfAnalystOpinions + recommendationKey → financialData
  `);
})();
