# Pulse AI — Recommendation Engine

Python + FastAPI + PyTorch. This is the core ML service: LightGCN + SR-GNN, feature fusion,
explainability. Owned by the ML lead — the other two services treat this as a black box behind
one HTTP endpoint.

## Structure

```
app/
  main.py                    FastAPI app, mounts routers
  config.py                  API keys, ticker universe, model hyperparameters
  api/
    recommend.py              POST /recommend — the contract the backend calls
  data/
    ingestion/
      price.py                 yfinance — OHLCV history
      news.py                   NewsAPI — headlines per ticker
      fundamentals.py           Alpha Vantage — sector/market cap/P-E
    preprocessing/
      text.py                   cleaning, dedup, sentiment scoring
      price.py                  technical indicators, normalization, gap-filling
      session.py                session segmentation + per-session transition graphs
  graph/
    build_graph.py              bipartite user-item graph + normalized adjacency (LightGCN input)
  models/
    lightgcn.py                 long-term collaborative-filtering branch
    srgnn.py                    short-term session-based branch
    fusion.py                   combines both branches + financial features -> ranked scores
  explainability/
    explainer.py                SHAP/LIME attribution per recommendation
tests/
  test_recommend.py            validates the API contract end-to-end
notebooks/                     scratch space for experimentation (not imported by the app)
```

## Local dev (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000/docs for interactive API docs
pytest                          # run the contract tests
```

## Current status

`models/fusion.py` — `rank_for_user()` — is a **working stub**: it returns realistic-shaped
scores and explanations over the watchlist (or a default ticker universe for cold-start users)
without running real models yet. This exists so the backend and frontend can integrate against
a stable contract immediately. The ingestion and preprocessing modules underneath it are real
and functional, not stubs.

## Build order (matches Milestone 2 report, Section 7 — Next Steps)

1. Populate real interaction data (simulate personas + click logs, or start logging real usage
   via the backend's `Interaction` table).
2. Build the interaction graph (`graph/build_graph.py`) from that data and train `LightGCN`
   (add a `train_lightgcn.py` script — BPR loss, see the paper).
3. Build session graphs (`data/preprocessing/session.py`) and train `SRGNN` (add a
   `train_srgnn.py` script — cross-entropy over next-item prediction).
4. Replace the stub body of `rank_for_user()` in `fusion.py` with the real pipeline (the
   commented-out sketch at the bottom of that file shows the intended shape) — keep the
   function signature and response shape unchanged so nothing downstream breaks.
5. Replace `explainability/explainer.py`'s stub with a real `shap.Explainer` call against the
   trained fusion model.
6. Add the fairness/privacy/drift-detection checks described in the Milestone 2 report once the
   core pipeline is producing real recommendations to audit.

## Notes

- `NEWSAPI_KEY` and `ALPHA_VANTAGE_KEY` must be set in the root `.env` for ingestion to work —
  see `.env.example`. Alpha Vantage's free tier is rate-limited (~5 calls/min); `fetch_fundamentals_bulk`
  already paces requests accordingly.
- `text.py`'s sentiment model is a general-purpose DistilBERT model as a baseline; swapping in a
  finance-tuned model (e.g. FinBERT) is a natural upgrade once the pipeline works end-to-end.
- Keep `/recommend`'s request/response shape stable — it's the contract the backend and frontend
  are both built against. If it needs to change, update `backend/src/services/recommendation.service.js`
  and `frontend/src/components/FeedCard.js` in the same change.
