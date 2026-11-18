"""Simple content-based ranker. No ML libraries needed.
score = sector match + ticker match + recency of the headline (slot 0 = newest)."""
from collections import Counter

# ticker -> (sector, market cap tier)
STOCKS = {
    "AAPL": ("Technology", "large"), "MSFT": ("Technology", "large"), "NVDA": ("Technology", "large"),
    "AMD": ("Technology", "large"), "ROKU": ("Technology", "mid"),
    "LLY": ("Healthcare", "large"), "JNJ": ("Healthcare", "large"), "PFE": ("Healthcare", "large"),
    "GILD": ("Healthcare", "large"), "MRNA": ("Healthcare", "mid"),
    "JPM": ("Financials", "large"), "BAC": ("Financials", "large"), "GS": ("Financials", "large"),
    "SCHW": ("Financials", "large"), "ALLY": ("Financials", "mid"),
    "XOM": ("Energy", "large"), "CVX": ("Energy", "large"), "COP": ("Energy", "large"),
    "OXY": ("Energy", "large"), "APA": ("Energy", "mid"),
    "AMZN": ("Consumer", "large"), "WMT": ("Consumer", "large"), "NKE": ("Consumer", "large"),
    "SBUX": ("Consumer", "large"), "ETSY": ("Consumer", "mid"),
    "GE": ("Industrials", "large"), "CAT": ("Industrials", "large"), "UPS": ("Industrials", "large"),
    "DAL": ("Industrials", "large"), "AAL": ("Industrials", "mid"),
    "NEE": ("Utilities", "large"), "DUK": ("Utilities", "large"), "SO": ("Utilities", "large"),
    "AEP": ("Utilities", "large"), "PNW": ("Utilities", "mid"),
    "LIN": ("Materials", "large"), "SHW": ("Materials", "large"), "FCX": ("Materials", "large"),
    "NEM": ("Materials", "large"), "CLF": ("Materials", "mid"),
}

# 10 placeholder headline templates per ticker (slot 0 = newest)
TEMPLATES = [
    "{t} schedules annual shareholder meeting", "{t} beats quarterly earnings estimates",
    "Analysts raise price target on {t}", "{t} announces new share buyback",
    "{t} CEO discusses outlook for next year", "{t} expands operations in new markets",
    "{t} faces regulatory review", "{t} signs multi-year supply agreement",
    "{t} declares quarterly dividend", "{t} unveils cost-cutting plan",
]


def rank(watchlist, recent_items, top_k=10):
    watchlist = [t.upper() for t in watchlist if t.upper() in STOCKS]
    recent = [r for r in recent_items if r.split(":")[0] in STOCKS]
    read = set(recent)

    # what do we know about the user's interest?
    if recent:
        mode, basis = "session", [r.split(":")[0] for r in recent]
    elif watchlist:
        mode, basis = "watchlist", watchlist
    else:
        mode, basis = "generic", []
    sectors = Counter(STOCKS[t][0] for t in basis)
    tickers = Counter(basis)
    tiers = Counter(STOCKS[t][1] for t in basis)
    n = max(len(basis), 1)
    source_word = "recent reading" if mode == "session" else "watchlist"

    items = []
    for ticker, (sector, tier) in STOCKS.items():
        for slot in range(10):
            item_id = f"{ticker}:{slot}"
            if item_id in read:
                continue
            s_part = sectors[sector] / n
            t_part = tickers[ticker] / n
            c_part = 0.2 * tiers[tier] / n
            recency = (10 - slot) / 10 * 0.15
            content = 0.5 * s_part + 0.5 * t_part + c_part + recency
            score = content
            parts = {"long_term_collaborative": 0.0, "long_term_content": round(content if mode == "watchlist" else 0, 4),
                     "session_collaborative": 0.0, "session_content": round(content if mode == "session" else 0, 4)}
            reasons = []
            if t_part: reasons.append(f"You follow {ticker} in your {source_word}")
            if s_part: reasons.append(f"Fits the {sector} stories in your {source_word}")
            if c_part: reasons.append(f"{tier.capitalize()}-cap, similar to your {source_word}")
            if not reasons: reasons.append("Popular market-wide story" if slot < 3 else "Recent market story")
            items.append({
                "item_id": item_id, "item_type": "news", "ticker": ticker, "sector": sector,
                "market_cap_tier": tier, "headline": TEMPLATES[slot].format(t=ticker),
                "source": "sample", "url": None, "score": round(min(score, 1.0), 4),
                "explanation": {
                    "reasons": reasons,
                    "features": {"sector": round(s_part, 3), "ticker": round(t_part, 3),
                                 "market_cap": round(c_part, 3), "recency": round(recency, 3)},
                    "parts": parts, "raw_score": round(content, 4),
                },
            })

    items.sort(key=lambda i: (-i["score"], i["ticker"], i["item_id"]))
    out, per_ticker = [], Counter()
    for it in items:                       # at most 2 stories per ticker
        if per_ticker[it["ticker"]] < 2:
            per_ticker[it["ticker"]] += 1
            out.append(it)
        if len(out) == top_k:
            break
    return {"mode": "simple", "personalisation": mode, "catalogue_mode": "sample", "recommendations": out}
