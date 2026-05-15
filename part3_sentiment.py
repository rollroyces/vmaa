#!/usr/bin/env python3
"""
VMAA 2.0 — Part 3: Sentiment Analysis Layer
============================================
Multi-source sentiment analysis for better entry/exit decisions.

Philosophy (Value Mean-Reversion):
  Extreme negative sentiment + strong fundamentals = Contrarian BUY opportunity
  Extreme positive sentiment + weak fundamentals = Potential SELL/AVOID

Sentiment Sources:
  1. Analyst Consensus (yfinance)          — weight: 25%
  2. News Sentiment (VADER on headlines)   — weight: 30%
  3. Social Buzz (Reddit mentions)         — weight: 20%
  4. Price-Momentum Sentiment (technical)  — weight: 15%
  5. Insider/Institutional Flow            — weight: 10%

Composite Score: -1.0 (max bearish) to +1.0 (max bullish)

Key Signals:
  - CONTRARIAN_BUY: Sentiment < -0.25 AND fundamentals strong → VALUE OPPORTUNITY
  - CROWDED_TRADE: Sentiment > 0.65 → CAUTION
  - SENTIMENT_DIVERGENCE: Price ↓ but sentiment ↑ → ACCUMULATION

Usage:
  from part3_sentiment import analyze_sentiment, batch_sentiment
  result = analyze_sentiment("AAPL")
  results = batch_sentiment(["AAPL", "MSFT", "GOOGL"])
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("vmaa.sentiment")


# ═══════════════════════════════════════════════════════════════════
# Dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SentimentResult:
    """Multi-source sentiment analysis result for a single ticker."""
    ticker: str

    analyst_score: float = 0.0
    news_score: float = 0.0
    social_score: float = 0.0
    technical_score: float = 0.0
    insider_score: float = 0.0

    composite_score: float = 0.0
    sentiment_label: str = "NEUTRAL"

    analyst_consensus: str = ""
    analyst_count: int = 0
    analyst_target_mean: float = 0.0
    analyst_target_upside_pct: float = 0.0

    news_headline_count: int = 0
    news_positive_pct: float = 0.0
    news_negative_pct: float = 0.0
    news_neutral_pct: float = 0.0
    news_summary: str = ""

    social_mentions: int = 0
    social_trend: str = ""
    social_subreddits: List[str] = field(default_factory=list)

    signals: List[str] = field(default_factory=list)
    sources_available: int = 0
    data_quality: str = "ok"
    news_extreme_neg_ratio: float = 0.0
    news_extreme_pos_ratio: float = 0.0
    data_date: str = ""


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SentimentConfig:
    weight_analyst: float = 0.25
    weight_news: float = 0.30
    weight_social: float = 0.20
    weight_technical: float = 0.15
    weight_insider: float = 0.10

    contrarian_buy_threshold: float = -0.25
    crowded_trade_threshold: float = 0.65
    divergence_threshold: float = 0.30

    bearish_threshold: float = -0.35
    slightly_bearish_threshold: float = -0.15
    slightly_bullish_threshold: float = 0.15
    bullish_threshold: float = 0.35

    news_lookback_days: int = 7
    min_news_articles: int = 3
    social_lookback_days: int = 3
    min_social_mentions: int = 5

    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    reddit_enabled: bool = True
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "VMAA-Sentiment/2.0"


SENT_CONFIG = SentimentConfig()

# Lazy-load VADER
_vader = None


def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed. pip install vaderSentiment")
            _vader = False
    return _vader


# ═══════════════════════════════════════════════════════════════════
# 1. Analyst Sentiment (yfinance)
# ═══════════════════════════════════════════════════════════════════

def _analyst_sentiment(ticker: str) -> Dict[str, Any]:
    info = {}
    # Primary: yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        pass
    
    # Fallback: YahooDirect (bypasses yfinance rate-limit)
    if not info or not info.get('recommendationKey'):
        try:
            from data.yahoo_direct import YahooDirect
            yd = YahooDirect(delay=0.08)
            yd_info = yd.get_info(ticker)
            if yd_info:
                info = {**yd_info, **info}  # merge, yfinance values take priority
        except Exception:
            pass
    
    try:
        recommendation = info.get('recommendationKey', '').lower()
        analyst_count = info.get('numberOfAnalystOpinions', 0) or 0
        target_mean = info.get('targetMeanPrice', 0) or 0
        current = info.get('currentPrice', info.get('regularMarketPrice', 0)) or 0
        rec_map = {'strong_buy': 1.0, 'buy': 0.6, 'overweight': 0.5, 'hold': 0.0,
                   'neutral': 0.0, 'underweight': -0.5, 'sell': -0.6, 'strong_sell': -1.0}
        rec_score = rec_map.get(recommendation, 0.0)
        upside = 0.0
        if target_mean > 0 and current > 0:
            upside = (target_mean - current) / current
        upside_adj = np.clip(upside / 0.20 * 0.30, -0.30, 0.30)
        score = rec_score * 0.70 + upside_adj
        if analyst_count < 3:
            score *= min(1.0, (analyst_count / 3) ** 0.5)
        score = round(np.clip(score, -1.0, 1.0), 3)
        return {'score': score, 'consensus': recommendation, 'count': analyst_count,
                'target_mean': target_mean, 'upside_pct': round(upside, 4) if upside != 0 else 0}
    except Exception as e:
        logger.debug(f"Analyst sentiment failed for {ticker}: {e}")
        return {'score': 0.0, 'consensus': '', 'count': 0, 'target_mean': 0, 'upside_pct': 0}


# ═══════════════════════════════════════════════════════════════════
# 2. News Sentiment (VADER)
# ═══════════════════════════════════════════════════════════════════

def _news_sentiment(ticker: str) -> Dict[str, Any]:
    headlines = []
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news
        if news:
            for item in news[:20]:
                title = item.get('title', '')
                if not title:
                    content = item.get('content', {})
                    title = content.get('title', '') if isinstance(content, dict) else ''
                if not title:
                    if isinstance(content, dict):
                        title = content.get('summary', '')
                if title:
                    headlines.append(title)
    except Exception as e:
        logger.debug(f"yfinance news failed for {ticker}: {e}")

    # Finnhub supplement (if key available)
    finnhub_key = _get_finnhub_key()
    if finnhub_key and len(headlines) < 10:
        try:
            import requests
            now = datetime.now()
            from_date = (now - timedelta(days=SENT_CONFIG.news_lookback_days)).strftime('%Y-%m-%d')
            to_date = now.strftime('%Y-%m-%d')
            resp = requests.get(
                f'https://finnhub.io/api/v1/company-news',
                params={'symbol': ticker, 'from': from_date, 'to': to_date, 'token': finnhub_key},
                timeout=10)
            if resp.status_code == 200:
                for item in resp.json()[:20]:
                    title = item.get('headline', '')
                    if title:
                        headlines.append(title)
        except Exception:
            pass

    if not headlines:
        return {'score': 0.0, 'count': 0, 'positive_pct': 0, 'negative_pct': 0,
                'neutral_pct': 0, 'summary': ''}

    # VADER is a general-purpose sentiment tool. Financial news sentiment should
    # ideally use FinBERT or a financial-domain model. VADER scores are used as a
    # supplementary signal only.
    analyzer = _get_vader()
    if not analyzer:
        scores = [_simple_sentiment(h) for h in headlines]
        simple_scores = scores  # same scores when VADER unavailable
    else:
        scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        simple_scores = [_simple_sentiment(h) for h in headlines]

    # Detect VADER vs financial-dictionary conflicts
    conflict_count = 0
    for v, s_score in zip(scores, simple_scores):
        if (v > 0.05 and s_score < -0.05) or (v < -0.05 and s_score > 0.05):
            conflict_count += 1

    avg_score = np.mean(scores) if scores else 0.0
    positive = sum(1 for s in scores if s > 0.05)
    negative = sum(1 for s in scores if s < -0.05)
    neutral = len(scores) - positive - negative
    total = max(len(scores), 1)

    # Extreme-ratio detection — a plain mean masks panic/euphoria clusters
    extreme_neg_ratio = round(sum(1 for s in scores if s < -0.5) / total, 3)
    extreme_pos_ratio = round(sum(1 for s in scores if s > 0.5) / total, 3)

    summary = _extract_theme(headlines)

    return {'score': round(np.clip(avg_score, -1.0, 1.0), 3), 'count': len(headlines),
            'positive_pct': round(positive / total, 3), 'negative_pct': round(negative / total, 3),
            'neutral_pct': round(neutral / total, 3), 'summary': summary,
            'extreme_neg_ratio': extreme_neg_ratio, 'extreme_pos_ratio': extreme_pos_ratio,
            'conflict_count': conflict_count}


def _simple_sentiment(text: str) -> float:
    # Financial-domain sentiment dictionary (~90 terms).
    # VADER is trained on social media / movie reviews and can miss financial nuance.
    # This fallback catches domain-specific terms VADER may misinterpret.
    positive_words = {
        'beat', 'raise', 'upgrade', 'surge', 'jump', 'rally', 'growth',
        'profit', 'record', 'strong', 'bullish', 'outperform', 'buy',
        'positive', 'gain', 'higher', 'boost', 'breakthrough', 'expansion',
        # Financial-domain bullish terms
        'beat estimates', 'raised guidance', 'buyback', 'dividend increase',
        'fda approval', 'contract win', 'partnership', 'market share gain',
        'record revenue', 'profit beat', 'accelerating growth', 'margin expansion',
        'cost reduction', 'share repurchase', 'special dividend', 'spin-off',
        'strategic review', 'activist investor', 'price target raised',
        'earnings surprise', 'guidance raised', 'positive pre-announcement',
        'revenue beat', 'subscriber growth', 'expanding margins',
    }
    negative_words = {
        'miss', 'cut', 'downgrade', 'plunge', 'drop', 'crash', 'decline',
        'loss', 'weak', 'bearish', 'underperform', 'sell', 'negative',
        'risk', 'concern', 'warning', 'layoff', 'restructuring', 'debt',
        # Financial-domain bearish terms
        'cuts guidance', 'margin compression', 'inventory destocking',
        'impairment', 'write-down', 'delisting', 'bankruptcy',
        'default', 'liquidity crunch', 'cash burn', 'debt restructuring',
        'missed estimates', 'revenue decline', 'profit warning',
        'sec investigation', 'lawsuit', 'regulatory fine', 'going concern',
        'covenant breach', 'credit downgrade', 'share dilution',
        'supply chain disruption', 'customer loss', 'patent expiration',
        'pipeline failure', 'clinical hold', 'recall', 'accounting irregularity',
        'suspension', 'force majeure', 'asset sale', 'filing delay',
    }
    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    total = pos_count + neg_count
    return (pos_count - neg_count) / total if total > 0 else 0.0


def _extract_theme(headlines: List[str]) -> str:
    keywords = ['earnings', 'revenue', 'growth', 'profit', 'loss', 'guidance',
                'upgrade', 'downgrade', 'acquisition', 'merger', 'ipo', 'layoff',
                'expansion', 'contract', 'deal', 'partnership', 'regulatory',
                'lawsuit', 'dividend', 'buyback', 'restructuring', 'product',
                'launch', 'approval', 'trial', 'data', 'report', 'forecast',
                'outlook', 'target', 'estimate', 'beat', 'miss']
    all_text = ' '.join(headlines).lower()
    counts = {kw: all_text.count(kw) for kw in keywords if kw in all_text}
    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return ", ".join(f"{k}({v})" for k, v in top)


def _get_finnhub_key() -> Optional[str]:
    """
    Retrieve Finnhub API key from environment variable or OpenClaw config.
    Tries multiple potential config paths for robustness.
    """
    import os
    key = os.environ.get('FINNHUB_KEY', '')
    if key:
        return key

    # Try multiple possible config locations
    config_paths = [
        os.path.expanduser('~/.openclaw/openclaw.json'),
        os.path.expanduser('~/openclaw.json'),
    ]
    try:
        import json
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                key = cfg.get('skills', {}).get('entries', {}).get('finnhub', {}).get('env', {}).get('FINNHUB_KEY', '')
                if key:
                    return key
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════════════════════════════════
# 3. Social Sentiment
# ═══════════════════════════════════════════════════════════════════

def _social_sentiment(ticker: str) -> Dict[str, Any]:
    if not SENT_CONFIG.reddit_enabled:
        return {'score': 0.0, 'mentions': 0, 'trend': 'stable', 'subreddits': []}
    reddit_data = _reddit_api(ticker)
    if reddit_data:
        return reddit_data
    return _social_fallback(ticker)


def _reddit_api(ticker: str) -> Optional[Dict[str, Any]]:
    import os
    client_id = SENT_CONFIG.reddit_client_id or os.environ.get('REDDIT_CLIENT_ID', '')
    client_secret = SENT_CONFIG.reddit_client_secret or os.environ.get('REDDIT_CLIENT_SECRET', '')
    if not client_id or not client_secret:
        return None
    try:
        import praw
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                             user_agent=SENT_CONFIG.reddit_user_agent)
        mentions = []
        for sub_name in ['wallstreetbets', 'stocks', 'investing', 'stockmarket']:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.search(ticker, sort='new', time_filter='week', limit=10):
                    text = f"{post.title} {post.selftext}"[:500]
                    mentions.append({'text': text, 'score': post.score,
                                     'num_comments': post.num_comments, 'subreddit': sub_name})
            except Exception:
                continue
        if not mentions:
            return None
        analyzer = _get_vader()
        sentiment_scores = []
        for m in mentions:
            s = analyzer.polarity_scores(m['text'])['compound'] if analyzer else 0.0
            weight = np.log1p(m['score'] + m['num_comments'])
            sentiment_scores.append((s, weight))
        if sentiment_scores:
            weighted = sum(s * w for s, w in sentiment_scores) / sum(w for _, w in sentiment_scores)
            score = round(np.clip(weighted, -1.0, 1.0), 3)
        else:
            score = 0.0
        subreddits_found = list(set(m['subreddit'] for m in mentions))
        trend = _detect_trend(mentions)
        return {'score': score, 'mentions': len(mentions), 'trend': trend,
                'subreddits': subreddits_found}
    except ImportError:
        logger.debug("praw not installed")
        return None
    except Exception as e:
        logger.debug(f"Reddit API error: {e}")
        return None


def _social_fallback(ticker: str) -> Dict[str, Any]:
    info = {}
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception:
        pass
    
    if not info:
        try:
            from data.yahoo_direct import YahooDirect
            yd = YahooDirect(delay=0.08)
            info = yd.get_info(ticker) or {}
        except Exception:
            pass
    
    try:
        held_inst = info.get('heldPercentInstitutions', 0) or 0
        short_pct = info.get('shortPercentOfFloat', 0) or 0
        inst_score = 0.3 if held_inst > 0.7 else (0.1 if held_inst > 0.4 else (-0.2 if held_inst < 0.2 else 0.0))
        short_score = -0.3 if short_pct > 0.20 else (-0.1 if short_pct > 0.10 else (0.1 if short_pct < 0.03 else 0.0))
        score = round(np.clip(inst_score * 0.6 + short_score * 0.4, -1.0, 1.0), 3)
        return {'score': score, 'mentions': 0, 'trend': 'stable', 'subreddits': []}
    except Exception:
        return {'score': 0.0, 'mentions': 0, 'trend': 'stable', 'subreddits': []}


def _detect_trend(mentions: List[Dict]) -> str:
    if len(mentions) < 3:
        return 'stable'
    mid = len(mentions) // 2
    avg_first = np.mean([m['score'] for m in mentions[:mid]]) if mentions[:mid] else 0
    avg_second = np.mean([m['score'] for m in mentions[mid:]]) if mentions[mid:] else 0
    if avg_second > avg_first * 1.5:
        return 'rising'
    elif avg_second < avg_first * 0.5:
        return 'falling'
    return 'stable'


# ═══════════════════════════════════════════════════════════════════
# 4. Technical Sentiment
# ═══════════════════════════════════════════════════════════════════

def _technical_sentiment(ticker: str) -> Dict[str, Any]:
    hist = None
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo")
    except Exception:
        pass
    
    # Fallback: YahooDirect
    if hist is None or len(hist) < 20:
        try:
            from data.yahoo_direct import YahooDirect
            yd = YahooDirect(delay=0.08)
            hist = yd.get_history(ticker, period="3mo")
        except Exception:
            pass
    
    try:
        if hist is None or len(hist) < 20:
            return {'score': 0.0}
        close = hist['Close']
        volume = hist['Volume']
        rsi = _compute_rsi(close, 14)
        rsi_score = -0.5 if rsi < 30 else (-0.3 if rsi < 40 else (0.5 if rsi > 70 else (0.3 if rsi > 60 else 0.0)))
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        current = close.iloc[-1]
        ma_score = 0.3 if current > ma50 else (-0.3 if current < ma50 * 0.9 else -0.1)
        vol_score = 0.0
        if len(volume) >= 50:
            vt = float(volume.tail(10).mean()) / float(volume.tail(50).mean()) if float(volume.tail(50).mean()) > 0 else 1.0
            vol_score = 0.15 if vt > 1.3 else (-0.15 if vt < 0.7 else 0.0)
        macd_score = 0.0
        if len(close) >= 26:
            macd_l = close.ewm(span=12).mean() - close.ewm(span=26).mean()
            sig = macd_l.ewm(span=9).mean()
            macd_score = 0.2 if float(macd_l.iloc[-1]) > float(sig.iloc[-1]) else -0.2
        score = round(np.clip(rsi_score * 0.30 + ma_score * 0.30 + vol_score * 0.20 + macd_score * 0.20, -1.0, 1.0), 3)
        return {'score': score}
    except Exception as e:
        logger.debug(f"Technical sentiment failed: {e}")
        return {'score': 0.0}


# ═══════════════════════════════════════════════════════════════════
# 5. Insider Sentiment
# ═══════════════════════════════════════════════════════════════════

def _insider_sentiment(ticker: str) -> Dict[str, Any]:
    info = {}
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception:
        pass
    if not info:
        try:
            from data.yahoo_direct import YahooDirect
            yd = YahooDirect(delay=0.08)
            info = yd.get_info(ticker) or {}
        except Exception:
            pass
    try:
        insider_pct = info.get('heldPercentInsiders', 0) or 0
        inst_pct = info.get('heldPercentInstitutions', 0) or 0
        iscore = 0.3 if insider_pct > 0.15 else (0.1 if insider_pct > 0.05 else (-0.1 if insider_pct < 0.01 else 0.0))
        sscore = 0.25 if inst_pct > 0.80 else (0.10 if inst_pct > 0.50 else (-0.2 if inst_pct < 0.20 else 0.0))
        score = round(np.clip(iscore * 0.40 + sscore * 0.60, -1.0, 1.0), 3)
        return {'score': score}
    except Exception:
        return {'score': 0.0}


def _compute_rsi(series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0


# ═══════════════════════════════════════════════════════════════════
# Composite Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_sentiment(ticker: str) -> SentimentResult:
    cfg = SENT_CONFIG
    result = SentimentResult(ticker=ticker)
    available = 0
    total_weight = 0.0

    a = _analyst_sentiment(ticker)
    result.analyst_score = a['score']
    result.analyst_consensus = a['consensus']
    result.analyst_count = a['count']
    result.analyst_target_mean = a['target_mean']
    result.analyst_target_upside_pct = a['upside_pct']
    if a['count'] > 0:
        available += 1
        total_weight += cfg.weight_analyst

    n = _news_sentiment(ticker)
    result.news_score = n['score']
    result.news_headline_count = n['count']
    result.news_positive_pct = n['positive_pct']
    result.news_negative_pct = n['negative_pct']
    result.news_neutral_pct = n['neutral_pct']
    result.news_summary = n['summary']
    result.news_extreme_neg_ratio = n.get('extreme_neg_ratio', 0.0)
    result.news_extreme_pos_ratio = n.get('extreme_pos_ratio', 0.0)
    news_conflict = (
        n.get('conflict_count', 0) >= n.get('count', 0) * 0.3
        and n.get('count', 0) >= 3
    )
    if n['count'] >= cfg.min_news_articles:
        available += 1
        total_weight += cfg.weight_news
    elif n['count'] > 0:
        total_weight += cfg.weight_news * (n['count'] / cfg.min_news_articles)

    s = _social_sentiment(ticker)
    result.social_score = s['score']
    result.social_mentions = s['mentions']
    result.social_trend = s['trend']
    result.social_subreddits = s['subreddits']
    if s['mentions'] >= cfg.min_social_mentions or s['score'] != 0.0:
        available += 1
        total_weight += cfg.weight_social

    t = _technical_sentiment(ticker)
    result.technical_score = t['score']
    available += 1
    total_weight += cfg.weight_technical

    i = _insider_sentiment(ticker)
    result.insider_score = i['score']
    if i['score'] != 0.0:
        available += 1
        total_weight += cfg.weight_insider

    result.sources_available = available

    if total_weight > 0:
        news_wf = 0.5 if news_conflict else 1.0  # halve news weight on VADER/dict conflict
        composite = (
            result.analyst_score * (cfg.weight_analyst / total_weight) +
            result.news_score * (cfg.weight_news / total_weight) * news_wf +
            result.social_score * (cfg.weight_social / total_weight) +
            result.technical_score * (cfg.weight_technical / total_weight) +
            result.insider_score * (cfg.weight_insider / total_weight)
        )
    else:
        composite = 0.0

    result.composite_score = round(np.clip(composite, -1.0, 1.0), 3)

    if result.composite_score <= cfg.bearish_threshold:
        result.sentiment_label = "BEARISH"
    elif result.composite_score <= cfg.slightly_bearish_threshold:
        result.sentiment_label = "SLIGHTLY_BEARISH"
    elif result.composite_score >= cfg.bullish_threshold:
        result.sentiment_label = "BULLISH"
    elif result.composite_score >= cfg.slightly_bullish_threshold:
        result.sentiment_label = "SLIGHTLY_BULLISH"
    else:
        result.sentiment_label = "NEUTRAL"

    signals = []
    if result.composite_score <= cfg.contrarian_buy_threshold:
        signals.append("CONTRARIAN_BUY")
    if result.composite_score >= cfg.crowded_trade_threshold:
        signals.append("CROWDED_TRADE")
    tech_vs_comp = abs(result.technical_score - result.composite_score)
    if tech_vs_comp >= cfg.divergence_threshold:
        if result.technical_score < -0.2 and result.composite_score > 0.2:
            signals.append("SENTIMENT_DIVERGENCE_POSITIVE")
        elif result.technical_score > 0.2 and result.composite_score < -0.2:
            signals.append("SENTIMENT_DIVERGENCE_NEGATIVE")
    if result.news_extreme_neg_ratio > 0.30:
        signals.append("NEWS_EXTREME_NEGATIVE")
    if result.news_extreme_pos_ratio > 0.30:
        signals.append("NEWS_EXTREME_POSITIVE")
    if news_conflict:
        signals.append("SENTIMENT_CONFLICT")
    if result.analyst_target_upside_pct > 0.25 and result.analyst_count >= 3:
        signals.append("HIGH_ANALYST_UPSIDE")
    if result.social_trend == 'rising' and result.social_mentions >= 10:
        signals.append("SOCIAL_BUZZ_RISING")
    result.signals = signals
    result.data_date = datetime.now().isoformat()
    return result


def batch_sentiment(tickers: List[str], delay: float = 0.15,
                    progress_every: int = 20) -> Dict[str, SentimentResult]:
    results = {}
    for i, ticker in enumerate(tickers):
        if (i + 1) % progress_every == 0:
            logger.info(f"  Sentiment: {i+1}/{len(tickers)}")
        for attempt in range(2):
            try:
                result = analyze_sentiment(ticker)
                results[ticker] = result
                break
            except Exception as e:
                if attempt == 0:
                    time.sleep(2)
                else:
                    logger.warning(f"Sentiment failed for {ticker} after retry: {e}")
                    result = SentimentResult(ticker=ticker, sources_available=0)
                    result.data_quality = "missing"
                    result.signals.append("SENTIMENT_DATA_MISSING")
                    results[ticker] = result
        time.sleep(delay)
    return results


def sentiment_confidence_adjustment(
    sentiment: SentimentResult,
    base_confidence: float,
) -> Tuple[float, List[str]]:
    adjustment = 0.0
    notes = []
    signals = sentiment.signals
    if "CONTRARIAN_BUY" in signals:
        adjustment += 0.10
        notes.append("Contrarian(+0.10)")
    if "CROWDED_TRADE" in signals:
        adjustment -= 0.10
        notes.append("Crowded(-0.10)")
    if "NEWS_EXTREME_NEGATIVE" in signals:
        adjustment += 0.05
        notes.append("NewsOverreaction?(+0.05)")
    if "NEWS_EXTREME_POSITIVE" in signals:
        adjustment -= 0.05
        notes.append("NewsEuphoria?(-0.05)")
    if "HIGH_ANALYST_UPSIDE" in signals:
        adjustment += 0.05
        notes.append("AnalystUpside(+0.05)")
    if "SOCIAL_BUZZ_RISING" in signals:
        adjustment += 0.03
        notes.append("Buzz(+0.03)")
    if "SENTIMENT_DIVERGENCE_POSITIVE" in signals:
        adjustment += 0.08
        notes.append("Divergence(+0.08)")
    if "SENTIMENT_DIVERGENCE_NEGATIVE" in signals:
        adjustment -= 0.08
        notes.append("Divergence(-0.08)")
    adjustment += sentiment.composite_score * 0.05
    adjusted = round(np.clip(base_confidence + adjustment, 0.0, 1.0), 3)
    return adjusted, notes


def should_exit_on_sentiment(
    ticker: str, entry_sentiment: Optional[SentimentResult],
    current_sentiment: SentimentResult) -> Tuple[bool, str]:
    if entry_sentiment and entry_sentiment.composite_score > 0.0:
        drop = entry_sentiment.composite_score - current_sentiment.composite_score
        if drop > 0.50:
            return True, f"Sentiment crash: {entry_sentiment.composite_score:+.2f} → {current_sentiment.composite_score:+.2f}"
    if (current_sentiment.analyst_score < -0.5 and current_sentiment.news_score < -0.5 and
        current_sentiment.composite_score < -0.6):
        return True, "All sources extremely bearish"
    return False, ""


# ═══════════════════════════════════════════════════════════════════
# Historical-Mode Sentiment (for Backtesting)
# ═══════════════════════════════════════════════════════════════════

def analyze_sentiment_historical(
    ticker: str,
    price_history: Any = None,
    yf_info: Optional[Dict] = None,
) -> SentimentResult:
    result = SentimentResult(ticker=ticker)
    available = 0

    tech = _technical_sentiment_from_hist(price_history)
    result.technical_score = tech['score']
    available += 1

    if yf_info:
        a = _analyst_from_info(yf_info, ticker)
    else:
        a = _analyst_sentiment(ticker)
    result.analyst_score = a['score']
    result.analyst_consensus = a['consensus']
    result.analyst_count = a['count']
    result.analyst_target_mean = a['target_mean']
    result.analyst_target_upside_pct = a['upside_pct']
    if a['count'] > 0:
        available += 1

    if yf_info:
        i = _insider_from_info(yf_info)
    else:
        i = _insider_sentiment(ticker)
    result.insider_score = i['score']
    if i['score'] != 0.0:
        available += 1

    result.sources_available = available
    result.news_score = 0.0
    result.social_score = 0.0

    # Analyst discount for deep drawdowns
    analyst_weight = 0.20 if a['count'] > 0 else 0
    dd = abs(min(result.technical_score, 0))
    if dd > 0.20:
        analyst_weight *= 0.50
    if dd > 0.35:
        analyst_weight *= 0.25
    if dd > 0.50:
        analyst_weight = 0.00

    total_weight = 0.70 + analyst_weight + (0.10 if i['score'] != 0.0 else 0)
    if total_weight > 0:
        composite = (result.technical_score * (0.70 / total_weight) +
                     result.analyst_score * (analyst_weight / total_weight if analyst_weight > 0 else 0) +
                     result.insider_score * (0.10 / total_weight if i['score'] != 0.0 else 0))
    else:
        composite = result.technical_score

    result.composite_score = round(np.clip(composite, -1.0, 1.0), 3)

    cfg = SENT_CONFIG
    if result.composite_score <= cfg.bearish_threshold:
        result.sentiment_label = "BEARISH"
    elif result.composite_score <= cfg.slightly_bearish_threshold:
        result.sentiment_label = "SLIGHTLY_BEARISH"
    elif result.composite_score >= cfg.bullish_threshold:
        result.sentiment_label = "BULLISH"
    elif result.composite_score >= cfg.slightly_bullish_threshold:
        result.sentiment_label = "SLIGHTLY_BULLISH"
    else:
        result.sentiment_label = "NEUTRAL"

    signals = []
    if result.composite_score <= -0.25:
        signals.append("CONTRARIAN_BUY")
    if result.composite_score >= 0.65:
        signals.append("CROWDED_TRADE")
    if result.analyst_target_upside_pct > 0.25 and result.analyst_count >= 3:
        signals.append("HIGH_ANALYST_UPSIDE")
    if result.technical_score > 0.15 and result.analyst_score < 0:
        signals.append("TECHNICAL_IMPROVING")
    result.signals = signals
    result.data_date = datetime.now().isoformat()
    return result


def _technical_sentiment_from_hist(hist: Any) -> Dict[str, Any]:
    if hist is None or (hasattr(hist, 'empty') and hist.empty) or len(hist) < 20:
        return {'score': 0.0}
    try:
        close = hist['Close'] if 'Close' in hist else hist['close']
        volume = hist['Volume'] if 'Volume' in hist else hist.get('volume', None)
        high = hist['High'] if 'High' in hist else hist.get('high', close)
        if len(close) < 20:
            return {'score': 0.0}
        current = float(close.iloc[-1])
        high_1y = float(high.tail(252).max()) if len(high) >= 252 else float(high.max())
        dd_from_high = (current - high_1y) / high_1y if high_1y > 0 else 0
        dd_score = 0.0
        if dd_from_high < -0.50:
            dd_score = -0.85
        elif dd_from_high < -0.30:
            dd_score = -0.55
        elif dd_from_high < -0.20:
            dd_score = -0.30
        elif dd_from_high < -0.10:
            dd_score = -0.10
        elif dd_from_high > 0.05:
            dd_score = 0.20

        rsi = _compute_rsi(close, 14)
        rsi_score = -0.50 if rsi < 25 else (-0.25 if rsi < 35 else (0.40 if rsi > 75 else (0.20 if rsi > 65 else 0.0)))

        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        ma_score = 0.25 if current > ma50 else (-0.25 if current < ma50 * 0.9 else -0.10)

        vol_score = 0.0
        if volume is not None and len(volume) >= 50:
            vt = float(volume.tail(10).mean()) / float(volume.tail(50).mean()) if float(volume.tail(50).mean()) > 0 else 1.0
            vol_score = 0.12 if vt > 1.3 else (-0.12 if vt < 0.7 else 0.0)

        macd_score = 0.0
        if len(close) >= 26:
            macd_l = close.ewm(span=12).mean() - close.ewm(span=26).mean()
            sig = macd_l.ewm(span=9).mean()
            macd_score = 0.15 if float(macd_l.iloc[-1]) > float(sig.iloc[-1]) else -0.15

        score = round(np.clip(
            dd_score * 0.50 + rsi_score * 0.20 + ma_score * 0.15 +
            vol_score * 0.08 + macd_score * 0.07, -1.0, 1.0), 3)
        return {'score': score}
    except Exception:
        return {'score': 0.0}


def _analyst_from_info(info: Dict, ticker: str = "") -> Dict[str, Any]:
    recommendation = info.get('recommendationKey', '')
    analyst_count = info.get('numberOfAnalystOpinions', 0) or 0
    target_mean = info.get('targetMeanPrice', 0) or 0
    current = info.get('currentPrice', info.get('regularMarketPrice', 0)) or 0
    rec_map = {'strong_buy': 1.0, 'buy': 0.6, 'overweight': 0.5, 'hold': 0.0,
               'neutral': 0.0, 'underweight': -0.5, 'sell': -0.6, 'strong_sell': -1.0}
    rec_score = rec_map.get(str(recommendation).lower(), 0.0)
    upside_adj = np.clip(((target_mean - current) / current) / 0.20 * 0.30, -0.30, 0.30) if target_mean > 0 and current > 0 else 0.0
    score = rec_score * 0.70 + upside_adj
    # Square-root decay: gentler penalty for small analyst coverage
    if analyst_count < 3 and analyst_count > 0:
        score *= min(1.0, (analyst_count / 3) ** 0.5)
    upside = ((target_mean - current) / current) if target_mean > 0 and current > 0 else 0.0
    return {'score': round(np.clip(score, -1.0, 1.0), 3), 'consensus': str(recommendation),
            'count': analyst_count, 'target_mean': target_mean,
            'upside_pct': round(upside, 4) if upside != 0 else 0}


def _insider_from_info(info: Dict) -> Dict[str, Any]:
    insider_pct = info.get('heldPercentInsiders', 0) or 0
    inst_pct = info.get('heldPercentInstitutions', 0) or 0
    iscore = 0.3 if insider_pct > 0.15 else (0.1 if insider_pct > 0.05 else (-0.1 if insider_pct < 0.01 else 0.0))
    sscore = 0.25 if inst_pct > 0.80 else (0.10 if inst_pct > 0.50 else (-0.2 if inst_pct < 0.20 else 0.0))
    return {'score': round(np.clip(iscore * 0.40 + sscore * 0.60, -1.0, 1.0), 3)}


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ['AAPL', 'TSLA', 'GME']
    for tkr in tickers:
        result = analyze_sentiment(tkr)
        print(f"\n{'='*60}")
        print(f"  {tkr} — Sentiment: {result.sentiment_label} ({result.composite_score:+.3f})")
        print(f"  Analyst: {result.analyst_consensus} ({result.analyst_score:+.3f}) "
              f"[{result.analyst_count} analysts, +{result.analyst_target_upside_pct:.1%} upside]")
        print(f"  News:    {result.news_score:+.3f} ({result.news_headline_count} headlines, "
              f"+{result.news_positive_pct:.0%}/-{result.news_negative_pct:.0%})")
        print(f"  Social:  {result.social_score:+.3f} ({result.social_mentions} mentions, {result.social_trend})")
        print(f"  Tech:    {result.technical_score:+.3f} | Insider: {result.insider_score:+.3f}")
        print(f"  Sources: {result.sources_available}/5")
        if result.signals:
            print(f"  Signals: {', '.join(result.signals)}")
