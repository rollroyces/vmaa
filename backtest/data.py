#!/usr/bin/env python3
"""
Historical Data Loader
=======================
Loads and caches historical price and fundamental data for backtesting.

Uses yfinance for:
  - Daily OHLCV price data
  - Quarterly financial statements (balance sheet, income, cash flow)
  - Historical shares outstanding (for market cap reconstruction)

Data Caching:
  - Price data → pickle cache keyed by ticker + date range
  - Financials → pickle cache keyed by ticker
  - Auto-refresh if cache is stale (>7 days)

Key Challenge: yfinance's `.info` always returns the CURRENT snapshot.
For backtesting, we reconstruct historical fundamentals using:
  - quarterly_financials / balance_sheet dated index
  - Price * sharesOutstanding for historical market cap
  - Financial ratios computed from dated statement data
"""
from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.config import BacktestConfig, BTC

logger = logging.getLogger("vmaa.backtest.data")


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HistoricalSnapshot:
    """A point-in-time snapshot of all available data for a stock."""
    ticker: str
    date: str  # YYYY-MM-DD

    # Price data (as of this date)
    close: float
    open: float
    high: float
    low: float
    volume: int
    low_52w: float
    high_52w: float

    # Fundamental data (from most recent quarter available on this date)
    market_cap: float
    shares_outstanding: float
    book_value: float
    roa: Optional[float] = None
    roe: Optional[float] = None
    ebitda: Optional[float] = None
    total_revenue: Optional[float] = None
    net_income: Optional[float] = None
    free_cashflow: Optional[float] = None
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    debt_to_equity: Optional[float] = None
    beta: Optional[float] = None
    sector: str = ""
    industry: str = ""
    short_name: str = ""
    short_ratio: Optional[float] = None
    short_pct_float: Optional[float] = None
    analyst_count: int = 0
    analyst_target: Optional[float] = None

    # Earnings data (quarterly)
    eps_quarterly: List[float] = field(default_factory=list)
    revenue_quarterly: List[float] = field(default_factory=list)

    # Previous period fundamentals
    net_income_prev: Optional[float] = None
    total_assets_prev: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════
# Historical Data Loader
# ═══════════════════════════════════════════════════════════════════

class HistoricalDataLoader:
    """
    Fetch and cache historical data for multiple tickers.

    Usage:
        loader = HistoricalDataLoader(config)
        loader.fetch_all(tickers, start_date, end_date)
        snapshot = loader.get_snapshot("AAPL", "2023-06-15")
        prices = loader.get_price_history("AAPL")
    """

    def __init__(self, config: BacktestConfig = BTC):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._fin_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._info_cache: Dict[str, dict] = {}
        self._snapshot_cache: Dict[str, Dict[str, HistoricalSnapshot]] = {}

    # ── Public API ──

    def fetch_all(self, tickers: List[str],
                  start_date: str, end_date: str) -> None:
        """
        Fetch all data for all tickers.
        Call once before running backtest.
        """
        total = len(tickers)
        logger.info(f"Loading historical data for {total} tickers "
                    f"({start_date} → {end_date})...")

        # Extend start_date by 1 year for 52-week low/high and financial lag
        extended_start = (pd.Timestamp(start_date) -
                          pd.DateOffset(days=400)).strftime('%Y-%m-%d')

        for i, ticker in enumerate(tickers):
            if (i + 1) % 50 == 0:
                logger.info(f"  Data fetch: {i+1}/{total}")
            try:
                self._fetch_ticker(ticker, extended_start, end_date)
            except Exception as e:
                logger.debug(f"  {ticker}: Data fetch failed — {e}")
            time.sleep(0.05)  # Rate limit

        logger.info(f"Data loading complete: {len(self._price_cache)} tickers")

    def get_price_history(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get full daily price history for a ticker."""
        return self._price_cache.get(ticker)

    def get_snapshot(self, ticker: str, date_str: str) -> Optional[HistoricalSnapshot]:
        """
        Get a point-in-time snapshot for a ticker on a given date.
        Builds snapshot from cached price/financial data if not already cached.
        """
        if ticker not in self._snapshot_cache:
            self._snapshot_cache[ticker] = {}

        if date_str in self._snapshot_cache[ticker]:
            return self._snapshot_cache[ticker][date_str]

        snapshot = self._build_snapshot(ticker, date_str)
        if snapshot:
            self._snapshot_cache[ticker][date_str] = snapshot
        return snapshot

    def get_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Get cached financial statements for a ticker."""
        return self._fin_cache.get(ticker, {})

    def get_info(self, ticker: str) -> dict:
        """Get cached info dict (current snapshot, not historical)."""
        return self._info_cache.get(ticker, {})

    # ── Private: Data Fetching ──

    def _fetch_ticker(self, ticker: str, start: str, end: str) -> None:
        """Fetch all data for a single ticker."""
        t = yf.Ticker(ticker)

        # Price history
        price_cache_path = self._cache_path(f"{ticker}_price_{start}_{end}")
        hist = self._load_cache(price_cache_path)
        if hist is None:
            hist = t.history(start=start, end=end)
            if not hist.empty:
                # Normalize timezone: strip tz-aware to tz-naive for consistent comparison
                if hasattr(hist.index, 'tz') and hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)
                self._save_cache(price_cache_path, hist)
            else:
                return
        if hist.empty:
            return
        # Also normalize cached data that might have tz
        if hasattr(hist.index, 'tz') and hist.index.tz is not None:
            hist = hist.copy()
            hist.index = hist.index.tz_localize(None)
        self._price_cache[ticker] = hist

        # Financial statements
        fin_cache_path = self._cache_path(f"{ticker}_financials")
        fin = self._load_cache(fin_cache_path)
        if fin is None:
            fin = {}
            try:
                bs = t.balance_sheet
                if bs is not None and not bs.empty:
                    fin['balance_sheet'] = bs
            except Exception:
                pass
            try:
                cf = t.cashflow
                if cf is not None and not cf.empty:
                    fin['cashflow'] = cf
            except Exception:
                pass
            try:
                qf = t.quarterly_financials
                if qf is not None and not qf.empty:
                    fin['quarterly_financials'] = qf
            except Exception:
                pass
            try:
                af = t.financials
                if af is not None and not af.empty:
                    fin['annual_financials'] = af
            except Exception:
                pass
            if fin:
                self._save_cache(fin_cache_path, fin)
        self._fin_cache[ticker] = fin

        # Info (current snapshot — used as fallback for static fields)
        self._info_cache[ticker] = t.info

    # ── Private: Snapshot Construction ──

    def _build_snapshot(self, ticker: str,
                        date_str: str) -> Optional[HistoricalSnapshot]:
        """
        Build a point-in-time fundamental snapshot.

        Hybrid approach: price data from historical bars, fundamentals from
        financial statements (balance sheet, cashflow, quarterly) with
        yfinance info as fallback.

        Known limitation: yfinance only provides ~2 years of quarterly data.
        For deeper backtests, current fundamental ratios serve as a proxy.
        """
        hist = self._price_cache.get(ticker)
        if hist is None or hist.empty:
            return None

        target_date = pd.Timestamp(date_str)

        # Find the trading day at or before target_date
        available = hist[hist.index <= target_date]
        if available.empty:
            return None
        row = available.iloc[-1]
        actual_date = available.index[-1].strftime('%Y-%m-%d')

        # 52-week highs/lows using 1 year of data before target_date
        window = available.tail(min(len(available), 252))
        low_52w = float(window['Low'].min())
        high_52w = float(window['High'].max())

        # ── Price-based metrics (exact from historical data) ──
        info = self._info_cache.get(ticker, {})
        # Shares for per-share calculations
        shares = info.get('sharesOutstanding', 0) or info.get('impliedSharesOutstanding', 0) or 0
        # Market cap: shares × historical price.
        # NOTE: Uses *current* shares outstanding from yfinance info — does
        # NOT account for historical buybacks, splits, or dilution.  Paid
        # data (Bloomberg / FactSet) would be needed for true historical
        # share counts.  yfinance does not expose time-series shares.
        if shares > 0:
            market_cap = shares * float(row['Close'])
        else:
            # Ultimate fallback: scale current market cap by price ratio
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0) or 0
            current_mcap = info.get('marketCap', 0) or 0
            if current_mcap > 0 and current_price > 0:
                market_cap = current_mcap * (float(row['Close']) / current_price)
            else:
                market_cap = 0

        # ── Fundamentals: extract from financial statements ──
        bs = self._fin_cache.get(ticker, {}).get('balance_sheet')
        cf_stmt = self._fin_cache.get(ticker, {}).get('cashflow')
        qf = self._fin_cache.get(ticker, {}).get('quarterly_financials')

        # Total Assets (from balance sheet, most recent column)
        total_assets = self._extract_from_stmt(bs, ['Total Assets', 'TotalAssets'])
        if not total_assets:
            total_assets = info.get('totalAssets', 0) or 0

        # Book value per share
        bv_per_share = info.get('bookValue', 0) or 0
        if not bv_per_share:
            total_equity = self._extract_from_stmt(bs, [
                'Total Stockholder Equity', 'StockholdersEquity',
                'TotalEquityGrossMinorityInterest', 'Common Stock Equity'])
            if total_equity and shares > 0:
                bv_per_share = total_equity / shares
        book_value = bv_per_share

        # Total Debt (in dollars, for D/E calculation)
        total_debt = self._extract_from_stmt(bs, [
            'Total Debt', 'TotalDebt', 'LongTermDebt', 'Long Term Debt'])
        if not total_debt:
            total_debt = info.get('totalDebt', 0) or 0

        # FCF (from cashflow)
        free_cashflow = self._extract_from_stmt(cf_stmt, [
            'Free Cash Flow', 'FreeCashFlow'])
        if not free_cashflow:
            free_cashflow = info.get('freeCashflow', 0) or 0

        # Key financials from quarterly or info
        net_income = self._extract_from_stmt(qf, [
            'Net Income', 'NetIncome', 'Net Income Common Stockholders'])
        if not net_income:
            net_income = info.get('netIncomeToCommon', 0) or info.get('netIncome', 0) or 0

        total_revenue = self._extract_from_stmt(qf, [
            'Total Revenue', 'TotalRevenue', 'Revenue', 'Operating Revenue'])
        if not total_revenue:
            total_revenue = info.get('totalRevenue', 0) or 0

        # EBITDA: from quarterly or compute from margin
        ebitda = self._extract_from_stmt(qf, ['EBITDA', 'Ebitda', 'Normalized EBITDA'])
        if not ebitda:
            ebitda_margin = info.get('ebitdaMargins', 0) or 0
            if ebitda_margin > 0 and total_revenue > 0:
                ebitda = total_revenue * ebitda_margin
            else:
                # Fallback: estimate from operating income
                ebitda = info.get('ebitda', 0) or 0

        # ROA: from info or compute
        roa = info.get('returnOnAssets')
        if roa is None and total_assets > 0 and net_income:
            roa = net_income / total_assets
        roe = info.get('returnOnEquity')
        if roe is None and book_value > 0 and net_income:
            roe = net_income / book_value

        # D/E: from info or compute.
        # total_debt is in raw dollars; must divide by *total* equity
        # (not per-share book value) to produce a meaningful ratio.
        debt_to_equity = info.get('debtToEquity', 0) or 0
        if not debt_to_equity:
            total_equity_de = self._extract_from_stmt(bs, [
                'Total Stockholder Equity', 'StockholdersEquity',
                'TotalEquityGrossMinorityInterest',
                'Common Stock Equity',
                'Stockholders Equity',
                'Total Equity Gross Minority Interest',
            ])
            if not total_equity_de:
                total_equity_de = info.get('totalStockholderEquity', 0) or 0
            if total_equity_de and total_equity_de > 0 and total_debt > 0:
                debt_to_equity = (total_debt / total_equity_de) * 100

        # Prior period (use same as current for want of better data)
        ni_prev = net_income
        ta_prev = total_assets
        # Try to get prior year from financial statements
        if qf is not None and not qf.empty and len(qf.columns) >= 5:
            # Column 4 is ~4 quarters ago
            col4 = qf.columns[min(4, len(qf.columns)-1)]
            ni_prev_val = self._get_value_at(qf, col4,
                ['Net Income', 'NetIncome', 'Net Income Common Stockholders'])
            if ni_prev_val:
                ni_prev = ni_prev_val
        if bs is not None and not bs.empty and len(bs.columns) >= 2:
            col1 = bs.columns[min(1, len(bs.columns)-1)]
            ta_prev_val = self._get_value_at(bs, col1, ['Total Assets', 'TotalAssets'])
            if ta_prev_val:
                ta_prev = ta_prev_val

        snapshot = HistoricalSnapshot(
            ticker=ticker,
            date=actual_date,
            close=round(float(row['Close']), 2),
            open=round(float(row['Open']), 2),
            high=round(float(row['High']), 2),
            low=round(float(row['Low']), 2),
            volume=int(row['Volume']),
            low_52w=round(low_52w, 2),
            high_52w=round(high_52w, 2),
            market_cap=market_cap,
            shares_outstanding=shares,
            book_value=book_value,
            roa=roa,
            roe=roe,
            ebitda=ebitda,
            total_revenue=total_revenue,
            net_income=net_income,
            free_cashflow=free_cashflow,
            total_assets=total_assets,
            total_debt=total_debt,
            debt_to_equity=debt_to_equity,
            beta=info.get('beta'),
            sector=info.get('sector', ''),
            industry=info.get('industry', ''),
            short_name=info.get('shortName', ticker),
            short_ratio=info.get('shortRatio'),
            short_pct_float=info.get('shortPercentOfFloat'),
            analyst_count=info.get('numberOfAnalystOpinions', 0) or 0,
            analyst_target=info.get('targetMeanPrice'),
            net_income_prev=ni_prev,
            total_assets_prev=ta_prev,
        )

        # Quarterly EPS and revenue series from actual historical data
        if qf is not None and not qf.empty:
            try:
                ni_series = self._get_fin_row(qf, [
                    'Net Income', 'NetIncome', 'Net Income Common Stockholders'])
                rev_series = self._get_fin_row(qf, [
                    'Total Revenue', 'TotalRevenue', 'Revenue', 'Operating Revenue'])
                if ni_series is not None:
                    vals = [float(x) for x in ni_series.values[:8] if not np.isnan(x)]
                    if vals:
                        snapshot.eps_quarterly = vals
                if rev_series is not None:
                    vals = [float(x) for x in rev_series.values[:8] if not np.isnan(x)]
                    if vals:
                        snapshot.revenue_quarterly = vals
            except Exception:
                pass

        return snapshot

    @staticmethod
    def _extract_from_stmt(stmt: Optional[pd.DataFrame],
                           labels: List[str]) -> Optional[float]:
        """Extract a value from the most recent column of a financial statement."""
        if stmt is None or stmt.empty:
            return None
        for label in labels:
            if label in stmt.index:
                val = stmt.loc[label, stmt.columns[0]]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if pd.notna(val):
                    return float(val)
        return None

    def _get_historical_financials(self, ticker: str,
                                    cutoff: pd.Timestamp) -> Dict[str, Any]:
        """
        Extract point-in-time fundamental data from quarterly financials.
        Uses the most recent quarter ending before the cutoff date.
        Falls back to current info for static fields.
        """
        result: Dict[str, Any] = {}
        bs = self._fin_cache.get(ticker, {}).get('balance_sheet')
        cf = self._fin_cache.get(ticker, {}).get('cashflow')
        qf = self._fin_cache.get(ticker, {}).get('quarterly_financials')
        info = self._info_cache.get(ticker, {})

        # Find most recent quarter before cutoff
        available_cols = None
        if qf is not None and not qf.empty:
            try:
                before = qf.loc[:, qf.columns <= cutoff]
                if not before.empty:
                    available_cols = before.columns[-1]
            except Exception:
                pass

        # Extract key metrics
        if available_cols is not None:
            col = available_cols
            result['net_income'] = self._get_value_at(qf, col,
                ['Net Income', 'NetIncome', 'netIncome'])
            result['total_revenue'] = self._get_value_at(qf, col,
                ['Total Revenue', 'TotalRevenue', 'Revenue'])
            result['ebitda'] = self._get_value_at(qf, col, ['EBITDA', 'Ebitda'])

            # Previous period
            prev_cols = qf.columns[qf.columns < col]
            if len(prev_cols) >= 1:
                prev_col = prev_cols[-1]
                result['net_income_prev'] = self._get_value_at(qf, prev_col,
                    ['Net Income', 'NetIncome', 'netIncome'])
            if len(prev_cols) >= 4:
                result['total_assets_prev'] = self._get_total_assets_at(bs, prev_cols[-4])

        # Balance sheet
        if bs is not None and not bs.empty:
            try:
                bs_before = bs.loc[:, bs.columns <= cutoff]
                if not bs_before.empty:
                    bs_col = bs_before.columns[-1]
                    result['book_value'] = self._get_value_at(bs, bs_col,
                        ['Total Stockholder Equity', 'StockholdersEquity',
                         'TotalEquityGrossMinorityInterest'])
                    result['total_assets'] = self._get_total_assets_at(bs, bs_col)
                    result['total_debt'] = self._get_value_at(bs, bs_col,
                        ['Total Debt', 'TotalDebt', 'LongTermDebt',
                         'Long Term Debt'])
                    # D/E ratio
                    equity = result.get('book_value', 0) or 0
                    debt = result.get('total_debt', 0) or 0
                    if equity > 0:
                        result['debt_to_equity'] = (debt / equity) * 100
            except Exception:
                pass

        # ROA/ROE from income vs balance sheet
        ni = result.get('net_income', 0) or 0
        ta = result.get('total_assets', 0) or 0
        bv = result.get('book_value', 0) or 0
        if ta > 0:
            result['roa'] = ni / ta
        if bv > 0:
            result['roe'] = ni / bv

        # FCF from cash flow
        if cf is not None and not cf.empty:
            try:
                cf_before = cf.loc[:, cf.columns <= cutoff]
                if not cf_before.empty:
                    cf_col = cf_before.columns[-1]
                    result['free_cashflow'] = self._get_value_at(cf, cf_col,
                        ['Free Cash Flow', 'FreeCashFlow', 'freeCashflow'])
            except Exception:
                pass

        # Fallback to current info for any missing fields
        if 'book_value' not in result or not result.get('book_value'):
            result['book_value'] = info.get('bookValue', 0)
        if 'roa' not in result or not result.get('roa'):
            result['roa'] = info.get('returnOnAssets')
        if 'roe' not in result or not result.get('roe'):
            result['roe'] = info.get('returnOnEquity')
        if 'ebitda' not in result or not result.get('ebitda'):
            result['ebitda'] = info.get('ebitda', 0)

        return result

    def _get_shares_outstanding(self, ticker: str,
                                 cutoff: pd.Timestamp,
                                 info: dict) -> float:
        """Get shares outstanding as of a historical date."""
        # yfinance doesn't provide historical shares outstanding directly
        # Fallback to current shares outstanding from info
        shares = info.get('sharesOutstanding', 0)
        if shares:
            return float(shares)
        # Try to infer from market cap / price
        return 0.0

    def _get_total_assets_at(self, bs: pd.DataFrame,
                              col) -> Optional[float]:
        """Get total assets from balance sheet at a specific column."""
        if bs is None:
            return None
        return self._get_value_at(bs, col,
            ['Total Assets', 'TotalAssets', 'totalAssets'])

    def _get_value_at(self, df: pd.DataFrame, col,
                       labels: List[str]) -> Optional[float]:
        """Get a financial metric value at a specific column, trying multiple labels."""
        for label in labels:
            if label in df.index:
                val = df.loc[label, col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if pd.notna(val):
                    return float(val)
        # Case-insensitive fallback
        for idx in df.index:
            for label in labels:
                if label.lower() in str(idx).lower():
                    val = df.loc[idx, col]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    if pd.notna(val):
                        return float(val)
        return None

    @staticmethod
    def _get_fin_row(df: pd.DataFrame, labels: List[str]) -> Optional[pd.Series]:
        """Find a row in a financial DataFrame by trying multiple labels."""
        for label in labels:
            if label in df.index:
                return df.loc[label]
        for idx in df.index:
            for label in labels:
                if label.lower() in str(idx).lower():
                    return df.loc[idx]
        return None

    # ── Cache Helpers ──

    def _cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = key.replace('/', '_').replace('\\', '_').replace(' ', '_')
        return self.cache_dir / f"{safe_key}.pkl"

    def _load_cache(self, path: Path) -> Optional[Any]:
        """Load from pickle cache if exists and fresh (<7 days)."""
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > 7 * 86400:  # 7 days
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def _save_cache(self, path: Path, data: Any) -> None:
        """Save data to pickle cache."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._price_cache.clear()
        self._fin_cache.clear()
        self._info_cache.clear()
        self._snapshot_cache.clear()
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
