"""
Feature fusion + ranking layer.

Combines LightGCN embeddings (long-term), SR-GNN session embeddings
(short-term), and financial features (sentiment, sector, momentum) into a
single score per candidate item, then attaches a SHAP/LIME explanation.

STATUS: `rank_for_user` below is currently a working STUB — it returns
plausible-looking scores over the watchlist (or a small default universe)
so the backend and frontend can integrate against a stable contract while
the LightGCN + SR-GNN branches are trained (see Milestone 2 report, Section
7 — Next Steps). Replace the body of `rank_for_user` once both branches
and the fusion layer are trained; keep the function signature and response
shape the same so nothing downstream breaks.
"""

import random
from typing import List, Dict, Any

from app.config import TICKER_UNIVERSE
from app.explainability.explainer import explain_stub


def rank_for_user(user_id: str, watchlist: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
    candidates = watchlist if watchlist else TICKER_UNIVERSE
    candidates = candidates[: max(top_k, len(candidates))]

    results = []
    for ticker in candidates:
        score = round(random.uniform(0.4, 0.99), 3)
        results.append({
            "item_id": ticker,
            "item_type": "ticker",
            "score": score,
            "explanation": explain_stub(ticker),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# TODO once both branches are trained:
# def rank_for_user(user_id, watchlist, top_k=20):
#     user_emb, item_embs = lightgcn_model(norm_adj)          # long-term signal
#     session_emb = srgnn_model(session_items, adj_in, adj_out)  # short-term signal
#     fused = fusion_layer(user_emb, session_emb, sentiment_features, momentum_features)
#     scores = ranking_head(fused, item_embs)
#     explanations = shap_explainer(scores, features)
#     return top_k results sorted by score, each with its explanation
