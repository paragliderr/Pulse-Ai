"""
Ranking entry point used by POST /recommend.

mode "model": the exported Pulse-Hybrid ranks the live catalogue (40 tickers x 10 latest headlines).
mode "stub" : torch or the checkpoint is missing, so plausible random scores are returned and the
              response says so. The app stays runnable with only requirements-lite.txt installed.

How a request becomes a "session" (what the model reads):
  1. recent_items  - the user's last clicks, e.g. ["NVDA:2", "AMD:0"]          -> personalisation "session"
  2. watchlist     - newest headline of each followed ticker                    -> "watchlist"
  3. nothing       - a market-wide seed: one leading name per sector, scores averaged -> "generic"
"""
import os
import random
from typing import List, Dict, Any, Optional

from app.config import TICKER_UNIVERSE
from app.explainability.explainer import explain_stub
from app.universe import parse_item_id, SYMBOL_INDEX, STORIES_PER_TICKER
from app.serving.registry import get_service

GENERIC_SEEDS = ["AAPL", "LLY", "JPM", "XOM", "AMZN", "GE", "NEE", "LIN"]   # one leader per sector
MAX_SESSION = 8


def build_session(watchlist: List[str], recent_items: Optional[List[str]]):
    """Returns (sessions, how, seen). Several sessions are averaged (used for the generic feed)."""
    ids = [parse_item_id(x) for x in (recent_items or [])]
    ids = [i for i in ids if i is not None]
    if ids:
        ids = ids[-MAX_SESSION:]
        return [ids], "session", ids
    wl = [SYMBOL_INDEX[s.upper()] * STORIES_PER_TICKER for s in (watchlist or []) if s.upper() in SYMBOL_INDEX]
    if wl:
        return [wl[:MAX_SESSION]], "watchlist", wl[:MAX_SESSION]
    seeds = [SYMBOL_INDEX[s] * STORIES_PER_TICKER for s in GENERIC_SEEDS]
    return [[i] for i in seeds], "generic", seeds


def _stub(watchlist: List[str], top_k: int):
    candidates = watchlist if watchlist else TICKER_UNIVERSE
    rows = [{"item_id": t, "item_type": "ticker", "score": round(random.uniform(0.4, 0.99), 3),
             "explanation": explain_stub(t)} for t in candidates]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"mode": "stub", "personalisation": "none", "catalogue_mode": "none", "items": rows[:top_k]}


def rank_for_user(user_id: str, watchlist: List[str], top_k: int = 20,
                  recent_items: Optional[List[str]] = None) -> Dict[str, Any]:
    svc = get_service()
    if svc is None:
        return _stub(watchlist, top_k)

    from app.data.catalogue import get_catalogue
    from app.serving.pulse_service import reasons_for

    cat = get_catalogue(svc.default_feat)
    sessions, how, seen = build_session(watchlist, recent_items)
    recs = svc.recommend(sessions, cat.feat, top_k=top_k, exclude=seen,
                         tier_lambda=float(os.getenv("PULSE_TIER_LAMBDA", "0.25")))
    items = []
    for r in recs:
        it = cat.items[r["index"]]
        items.append({
            "item_id": it["item_id"], "item_type": "news", "ticker": it["ticker"], "sector": it["sector"],
            "market_cap_tier": it["market_cap_tier"], "headline": it["headline"], "source": it["source"],
            "url": it["url"], "score": round(r["relevance"], 4),
            "explanation": {
                "reasons": reasons_for(r["effects"], it, how),
                "features": {k: round(v, 3) for k, v in r["effects"].items()},
                "parts": {k: round(v, 3) for k, v in r["parts"].items()},
                "raw_score": round(r["raw"], 4),
            },
        })
    return {"mode": "model", "personalisation": how, "catalogue_mode": cat.mode, "items": items}
