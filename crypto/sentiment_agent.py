"""
Sentiment Agent
===============
Reads multiple free data sources and produces a unified sentiment signal
for each trading symbol. Runs every 15 minutes (cached between cycles).

Sources:
  1. CryptoCompare News API (free, no key needed for basic)
  2. CoinGecko trending + market data (free)
  3. Yahoo Finance RSS headlines (free)
  4. Alternative.me Fear & Greed Index (free)
  5. Binance Funding Rates + Open Interest (already have client)

Output per symbol:
  {
    'direction':        'BULLISH' | 'BEARISH' | 'NEUTRAL',
    'strength':         0.0–1.0,
    'sources':          ['fear_greed', 'news', 'funding'],
    'fear_greed_value': 45,
    'fear_greed_label': 'Fear',
    'funding_rate':     0.0012,
    'funding_signal':   'BULLISH' | 'BEARISH' | 'NEUTRAL',
    'news_score':       0.3,   # -1 to +1
    'oi_change_pct':    2.1,   # open interest % change
    'raw':              {...}   # full raw data for logging
  }
"""

from __future__ import annotations

import time
import json
import logging
import requests
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS  = 900     # 15 minutes — refresh sentiment every 15 min
REQUEST_TIMEOUT    = 8       # seconds per HTTP request
MAX_NEWS_ARTICLES  = 20      # articles to score per fetch
NEWS_LOOKBACK_HOURS = 6      # only use news from last 6 hours

# Funding rate thresholds — positive = longs paying shorts (market bullish/crowded)
FUNDING_BULLISH_THRESHOLD  =  0.0008   # >0.08% per 8h = bullish crowd, potential short
FUNDING_BEARISH_THRESHOLD  = -0.0008   # <-0.08% per 8h = bearish crowd, potential long

# Keyword scoring for news headlines — simple but effective
BULLISH_WORDS = [
    'surge', 'rally', 'bull', 'pump', 'breakout', 'record', 'high',
    'adoption', 'approved', 'institutional', 'etf', 'buy', 'bullish',
    'upgrade', 'partnership', 'launch', 'milestone', 'recovery', 'bounce',
    'support', 'accumulate', 'long', 'moon', 'all-time', 'growth',
]
BEARISH_WORDS = [
    'crash', 'dump', 'bear', 'sell', 'hack', 'scam', 'fraud', 'ban',
    'regulation', 'crackdown', 'fear', 'panic', 'collapse', 'liquidat',
    'plunge', 'drop', 'fall', 'decline', 'warning', 'risk', 'concern',
    'investigation', 'lawsuit', 'fine', 'restrict', 'short', 'bearish',
    'correction', 'breakdown', 'resistance', 'loss', 'contagion',
]

SYMBOL_KEYWORDS = {
    'BTCUSDT': ['bitcoin', 'btc', 'crypto', 'digital asset', 'blockchain'],
    'ETHUSDT': ['ethereum', 'eth', 'ether', 'defi', 'smart contract', 'layer 2'],
    'SOLUSDT': ['solana', 'sol', 'solana network'],
}


class SentimentAgent:
    """
    Fetches and caches multi-source sentiment. Call get_signal(symbol)
    to get the unified sentiment dict for a symbol.
    """

    def __init__(self, binance_client=None):
        self._client   = binance_client   # optional — for funding/OI data
        self._cache    = {}               # symbol -> (signal, timestamp)
        self._global_cache = {}           # global data (fear/greed, news)
        self._global_ts    = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_signal(self, symbol: str) -> dict:
        """
        Returns unified sentiment signal for symbol.
        Uses cache if data is less than CACHE_TTL_SECONDS old.
        """
        now = time.time()

        # Check symbol cache
        if symbol in self._cache:
            cached_signal, cached_ts = self._cache[symbol]
            if now - cached_ts < CACHE_TTL_SECONDS:
                return cached_signal

        # Refresh global data if stale
        if self._global_ts is None or (now - self._global_ts) > CACHE_TTL_SECONDS:
            self._refresh_global()
            self._global_ts = now

        # Build per-symbol signal
        signal = self._build_signal(symbol)
        self._cache[symbol] = (signal, now)

        log.info(f"[SENTIMENT][{symbol[:3]}] {signal['direction']} "
                 f"(strength={signal['strength']:.0%}) | "
                 f"F&G={signal['fear_greed_label']}({signal['fear_greed_value']}) | "
                 f"funding={signal['funding_rate']:+.4f} | "
                 f"news_score={signal['news_score']:+.2f}")

        return signal

    def invalidate(self, symbol: str = None):
        """Force refresh on next call."""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()
            self._global_ts = None

    # ── Internal refresh ─────────────────────────────────────────────────────

    def _refresh_global(self):
        """Fetch all global data sources (news, fear/greed)."""
        log.info("[SENTIMENT] Refreshing global sentiment data...")

        self._global_cache['fear_greed']      = self._fetch_fear_greed()
        self._global_cache['cryptocompare']   = self._fetch_cryptocompare_news()
        self._global_cache['coingecko_news']  = self._fetch_coingecko_trending()
        self._global_cache['yahoo_headlines'] = self._fetch_yahoo_crypto_headlines()

        total_articles = (
            len(self._global_cache['cryptocompare'])
            + len(self._global_cache['yahoo_headlines'])
        )
        log.info(f"[SENTIMENT] Loaded {total_articles} articles | "
                 f"F&G: {self._global_cache['fear_greed'].get('label', 'N/A')} "
                 f"({self._global_cache['fear_greed'].get('value', 'N/A')})")

    def _build_signal(self, symbol: str) -> dict:
        """Combine all sources into one signal for a symbol."""
        scores  = []
        sources = []

        # ── 1. Fear & Greed ──────────────────────────────────────────────────
        fg = self._global_cache.get('fear_greed', {})
        fg_value = fg.get('value', 50)
        fg_label = fg.get('label', 'Neutral')
        fg_score  = (fg_value - 50) / 50.0    # -1 to +1
        scores.append(('fear_greed', fg_score, 0.30))  # 30% weight
        sources.append('fear_greed')

        # ── 2. News scoring ───────────────────────────────────────────────────
        all_articles = (
            self._global_cache.get('cryptocompare', [])
            + self._global_cache.get('yahoo_headlines', [])
        )
        news_score = self._score_news(all_articles, symbol)
        scores.append(('news', news_score, 0.40))   # 40% weight
        sources.append('news')

        # ── 3. CoinGecko trending ─────────────────────────────────────────────
        cg_trending = self._global_cache.get('coingecko_news', {})
        cg_score = self._score_coingecko(cg_trending, symbol)
        if cg_score is not None:
            scores.append(('coingecko', cg_score, 0.10))
            sources.append('coingecko')

        # ── 4. Funding rate (Binance) ─────────────────────────────────────────
        funding_rate = self._fetch_funding_rate(symbol)
        funding_signal = 'NEUTRAL'
        funding_score  = 0.0
        if funding_rate is not None:
            # High positive funding = longs crowded = bearish signal (squeeze risk)
            # High negative funding = shorts crowded = bullish signal
            if funding_rate > FUNDING_BULLISH_THRESHOLD:
                funding_signal = 'BEARISH'   # crowded longs
                funding_score  = -min(funding_rate / 0.003, 1.0)
            elif funding_rate < -FUNDING_BULLISH_THRESHOLD:
                funding_signal = 'BULLISH'   # crowded shorts
                funding_score  = min(abs(funding_rate) / 0.003, 1.0)
            scores.append(('funding', funding_score, 0.20))
            sources.append('funding')
        else:
            funding_rate = 0.0

        # ── 5. Open Interest change ───────────────────────────────────────────
        oi_change_pct = self._fetch_oi_change(symbol)

        # ── Weighted aggregate ────────────────────────────────────────────────
        total_weight = sum(w for _, _, w in scores)
        if total_weight == 0:
            agg_score = 0.0
        else:
            agg_score = sum(s * w for _, s, w in scores) / total_weight

        # Map to direction + strength
        if agg_score > 0.10:
            direction = 'BULLISH'
            strength  = min(agg_score * 1.5, 1.0)
        elif agg_score < -0.10:
            direction = 'BEARISH'
            strength  = min(abs(agg_score) * 1.5, 1.0)
        else:
            direction = 'NEUTRAL'
            strength  = 0.0

        return {
            'direction':        direction,
            'strength':         round(strength, 3),
            'sources':          sources,
            'fear_greed_value': fg_value,
            'fear_greed_label': fg_label,
            'funding_rate':     round(funding_rate, 6),
            'funding_signal':   funding_signal,
            'news_score':       round(news_score, 3),
            'oi_change_pct':    round(oi_change_pct or 0.0, 2),
            'agg_score':        round(agg_score, 3),
            'raw': {
                'fear_greed': fg,
                'funding_rate': funding_rate,
                'oi_change_pct': oi_change_pct,
            }
        }

    # ── Data fetchers ─────────────────────────────────────────────────────────

    def _fetch_fear_greed(self) -> dict:
        """Alternative.me Crypto Fear & Greed Index — free, no key."""
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=REQUEST_TIMEOUT
            )
            data = resp.json()['data'][0]
            return {
                'value': int(data['value']),
                'label': data['value_classification'],
                'timestamp': data['timestamp']
            }
        except Exception as e:
            log.warning(f"[SENTIMENT] Fear & Greed fetch failed: {e}")
            return {'value': 50, 'label': 'Neutral', 'timestamp': None}

    def _fetch_cryptocompare_news(self) -> list:
        """CryptoCompare News API — free tier, no key for basic use."""
        try:
            resp = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest",
                timeout=REQUEST_TIMEOUT
            )
            payload = resp.json()
            if not isinstance(payload, dict):
                log.warning(f"[SENTIMENT] CryptoCompare unexpected response type: {type(payload)}")
                return []
            raw = payload.get('Data', [])
            if not isinstance(raw, list):
                log.warning(f"[SENTIMENT] CryptoCompare 'Data' field is not a list "
                            f"(got {type(raw).__name__}) — API may have returned an error: "
                            f"{payload.get('Message', 'no message')}")
                return []
            articles = raw[:MAX_NEWS_ARTICLES]
            cutoff = time.time() - (NEWS_LOOKBACK_HOURS * 3600)
            return [
                {
                    'title': a.get('title', ''),
                    'body': a.get('body', '')[:200],
                    'published': a.get('published_on', 0),
                    'source': 'cryptocompare'
                }
                for a in articles
                if isinstance(a, dict) and a.get('published_on', 0) > cutoff
            ]
        except Exception as e:
            log.warning(f"[SENTIMENT] CryptoCompare news fetch failed: {e}")
            return []

    def _fetch_coingecko_trending(self) -> dict:
        """CoinGecko trending coins + market dominance — free."""
        try:
            trending_resp = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=REQUEST_TIMEOUT
            )
            trending = trending_resp.json()

            market_resp = requests.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=REQUEST_TIMEOUT
            )
            market = market_resp.json().get('data', {})

            return {
                'trending_coins': [
                    c['item']['symbol'].lower()
                    for c in trending.get('coins', [])
                ],
                'market_cap_change_24h': market.get('market_cap_change_percentage_24h_usd', 0),
                'btc_dominance': market.get('market_cap_percentage', {}).get('btc', 50),
            }
        except Exception as e:
            log.warning(f"[SENTIMENT] CoinGecko fetch failed: {e}")
            return {}

    def _fetch_yahoo_crypto_headlines(self) -> list:
        """Yahoo Finance RSS — crypto headlines, no key needed."""
        results = []
        feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=ETH-USD&region=US&lang=en-US",
        ]
        cutoff = time.time() - (NEWS_LOOKBACK_HOURS * 3600)

        for url in feeds:
            try:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                                    headers={'User-Agent': 'Mozilla/5.0'})
                text = resp.text

                # Simple XML parse — avoid lxml dependency
                import re
                titles = re.findall(r'<title><!\[CDATA\[(.+?)\]\]></title>', text)
                for title in titles[:8]:
                    results.append({
                        'title': title,
                        'body': '',
                        'published': time.time(),  # approximate
                        'source': 'yahoo'
                    })
            except Exception as e:
                log.warning(f"[SENTIMENT] Yahoo RSS fetch failed ({url}): {e}")

        return results

    def _fetch_funding_rate(self, symbol: str) -> float | None:
        """Fetch current funding rate from Binance."""
        if self._client is None:
            try:
                resp = requests.get(
                    f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}",
                    timeout=REQUEST_TIMEOUT
                )
                data = resp.json()
                return float(data.get('lastFundingRate', 0))
            except Exception as e:
                log.warning(f"[SENTIMENT] Funding rate fetch failed ({symbol}): {e}")
                return None
        try:
            data = self._client.futures_mark_price(symbol=symbol)
            return float(data[0]['lastFundingRate']) if isinstance(data, list) else float(data.get('lastFundingRate', 0))
        except Exception as e:
            log.warning(f"[SENTIMENT] Funding rate (client) failed ({symbol}): {e}")
            return None

    def _fetch_oi_change(self, symbol: str) -> float | None:
        """Fetch open interest change % over last 4 hours."""
        try:
            resp = requests.get(
                f"https://fapi.binance.com/futures/data/openInterestHist"
                f"?symbol={symbol}&period=1h&limit=5",
                timeout=REQUEST_TIMEOUT
            )
            data = resp.json()
            if isinstance(data, list) and len(data) >= 2:
                latest = float(data[-1]['sumOpenInterest'])
                earlier = float(data[0]['sumOpenInterest'])
                if earlier > 0:
                    return ((latest - earlier) / earlier) * 100
        except Exception as e:
            log.warning(f"[SENTIMENT] OI fetch failed ({symbol}): {e}")
        return None

    # ── Scoring helpers ───────────────────────────────────────────────────────

    def _score_news(self, articles: list, symbol: str) -> float:
        """Score articles for a symbol. Returns -1 to +1."""
        keywords = SYMBOL_KEYWORDS.get(symbol, ['crypto', 'bitcoin'])
        scores   = []

        for art in articles:
            text = (art.get('title', '') + ' ' + art.get('body', '')).lower()

            # Only count if relevant to this symbol
            if not any(kw in text for kw in keywords + ['crypto', 'market']):
                continue

            bull = sum(1 for w in BULLISH_WORDS if w in text)
            bear = sum(1 for w in BEARISH_WORDS if w in text)
            total = bull + bear
            if total == 0:
                continue

            score = (bull - bear) / total  # -1 to +1
            scores.append(score)

        if not scores:
            return 0.0

        # Trim extremes, average
        scores.sort()
        trimmed = scores[len(scores)//5 : -max(1, len(scores)//5)] or scores
        return round(sum(trimmed) / len(trimmed), 3)

    def _score_coingecko(self, data: dict, symbol: str) -> float | None:
        """Score CoinGecko data for a symbol."""
        if not data:
            return None

        score = 0.0
        sym_map = {'BTCUSDT': 'btc', 'ETHUSDT': 'eth', 'SOLUSDT': 'sol'}
        coin_id = sym_map.get(symbol, '')

        # Trending = mild bullish
        if coin_id and coin_id in data.get('trending_coins', []):
            score += 0.3

        # Market cap change
        mc_change = data.get('market_cap_change_24h', 0)
        score += max(-1.0, min(1.0, mc_change / 10))  # normalize ±10% → ±1

        return max(-1.0, min(1.0, score))
