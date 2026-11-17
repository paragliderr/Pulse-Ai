"""
The live catalogue: for every ticker, its 10 latest headlines plus the two numbers the model
reads from them (news sentiment and price momentum).

Modes (never crashes, always falls back):
  sample : deterministic placeholder headlines, features taken from the training data.
           Used when PULSE_LIVE_DATA is not "1", keys are missing, or a request fails.
  live   : NewsAPI headlines (needs NEWSAPI_KEY) + yfinance 5-day price move (optional).

Each item carries a `source` ("live" or "sample") so the UI can label it honestly.
NOTE: the live paths could not be exercised inside the development sandbox (no API access);
test them once with your keys:  PULSE_LIVE_DATA=1 python -m app.data.catalogue
"""
import os
import time
import threading
import numpy as np

from app.universe import UNIVERSE, SECTORS, TIERS, STORIES_PER_TICKER, N_ITEMS, item_id

POS = set("beat beats surge surges soar soars rally rallies gain gains rise rises jump jumps record strong "
          "upgrade upgrades raises raised growth profit profits approval approves approved expands boost "
          "outperform climbs optimistic".split())
NEG = set("miss misses fall falls drop drops plunge plunges slump cut cuts downgrade downgrades lawsuit "
          "probe recall loss losses warning warns weak decline declines layoffs fraud investigation fine "
          "bankruptcy concern concerns pressure".split())

SAMPLE_POS = ["{t}: analysts raise outlook after strong quarter", "{t} shares climb on upbeat guidance",
              "{t} announces expansion plan, investors respond well", "{t} posts results ahead of expectations",
              "Brokers upgrade {t} on improving demand"]
SAMPLE_NEG = ["{t} under pressure as demand outlook softens", "{t} dips after cautious management commentary",
              "Regulators scrutinise {t} over recent practices", "{t} warns of higher costs next quarter",
              "Analysts trim estimates for {t}"]
SAMPLE_NEU = ["{t} to present at upcoming industry conference", "{t} schedules annual shareholder meeting",
              "{t} names new head of investor relations", "What to watch in {t} ahead of earnings",
              "{t} files routine quarterly report"]


def lexicon_sentiment(text: str) -> float:
    words = [w.strip(".,:;!?\"'()").lower() for w in text.split()]
    pos = sum(w in POS for w in words)
    neg = sum(w in NEG for w in words)
    if pos + neg == 0:
        return 0.0
    return float(np.clip(0.8 * (pos - neg) / (pos + neg), -1, 1))


class Catalogue:
    def __init__(self, items, feat, mode):
        self.items, self.feat, self.mode = items, feat, mode
        self.built_at = time.time()

    def __len__(self):
        return len(self.items)


def build_sample(default_feat: np.ndarray) -> Catalogue:
    """default_feat: the (400, 13) feature matrix the model was trained on."""
    items = []
    for t, (sym, _name, s, c) in enumerate(UNIVERSE):
        for k in range(STORIES_PER_TICKER):
            i = t * STORIES_PER_TICKER + k
            sent, mom = float(default_feat[i, 11]), float(default_feat[i, 12])
            pool = SAMPLE_POS if sent > 0.2 else SAMPLE_NEG if sent < -0.2 else SAMPLE_NEU
            items.append(dict(item_id=item_id(sym, k), ticker=sym, slot=k, sector=SECTORS[s],
                              market_cap_tier=TIERS[c], headline=pool[(t + k) % len(pool)].format(t=sym),
                              url=None, source="sample", sentiment=sent, momentum=mom))
    return Catalogue(items, np.asarray(default_feat, dtype=np.float32).copy(), "sample")


def _fetch_headlines(name: str, key: str):
    import requests
    r = requests.get("https://newsapi.org/v2/everything",
                     params=dict(q=f'"{name}"', searchIn="title", language="en", sortBy="publishedAt",
                                 pageSize=STORIES_PER_TICKER, apiKey=key), timeout=10)
    r.raise_for_status()
    return [(a["title"], a.get("url")) for a in r.json().get("articles", []) if a.get("title")]


def _momentum(symbols):
    try:
        import yfinance as yf
        data = yf.download(symbols, period="1mo", progress=False, auto_adjust=True)["Close"]
        out = {}
        for s in symbols:
            c = data[s].dropna()
            if len(c) >= 6:
                out[s] = float(np.clip((c.iloc[-1] / c.iloc[-6] - 1) / 0.06, -1, 1))
        return out
    except Exception as e:                                    # offline, rate limited, not installed...
        print(f"[catalogue] momentum unavailable: {e}")
        return {}


def build_live(default_feat: np.ndarray) -> Catalogue:
    key = os.getenv("NEWSAPI_KEY", "")
    cat = build_sample(default_feat)
    if not key:
        print("[catalogue] NEWSAPI_KEY missing, using sample headlines")
        return cat
    mom = _momentum([u[0] for u in UNIVERSE])
    live_count = 0
    for t, (sym, name, _s, _c) in enumerate(UNIVERSE):
        try:
            heads = _fetch_headlines(name, key)
        except Exception as e:
            print(f"[catalogue] {sym}: news fetch failed ({e}); keeping sample items")
            heads = []
        for k, (title, url) in enumerate(heads[:STORIES_PER_TICKER]):
            it = cat.items[t * STORIES_PER_TICKER + k]
            it.update(headline=title, url=url, source="live", sentiment=lexicon_sentiment(title))
            live_count += 1
        if sym in mom:
            for k in range(STORIES_PER_TICKER):
                cat.items[t * STORIES_PER_TICKER + k]["momentum"] = mom[sym]
    cat.feat[:, 11] = [it["sentiment"] for it in cat.items]
    cat.feat[:, 12] = [it["momentum"] for it in cat.items]
    cat.mode = "live" if live_count else "sample"
    print(f"[catalogue] live headlines: {live_count}/{N_ITEMS}")
    return cat


_lock = threading.Lock()
_cache = {"cat": None}


def get_catalogue(default_feat: np.ndarray) -> Catalogue:
    """Cached catalogue. Live data refreshes every CATALOGUE_TTL_MIN minutes (default 720, which
    keeps a free NewsAPI key under its daily request limit)."""
    live = os.getenv("PULSE_LIVE_DATA") == "1"
    ttl = float(os.getenv("CATALOGUE_TTL_MIN", "720")) * 60
    with _lock:
        cat = _cache["cat"]
        expired = cat is not None and live and time.time() - cat.built_at > ttl
        if cat is None or expired:
            cat = build_live(default_feat) if live else build_sample(default_feat)
            _cache["cat"] = cat
        return cat


if __name__ == "__main__":
    f = np.zeros((N_ITEMS, 13), dtype=np.float32)
    c = build_live(f)
    print(c.mode, len(c))
    for it in c.items[:3]:
        print(it["item_id"], it["source"], it["headline"], round(it["sentiment"], 2), round(it["momentum"], 2))
