# Pulse AI - model side (how to run everything)

All commands run inside `recommendation-engine/` with the venv active:
`source .venv/bin/activate` and `pip install -r requirements-ml.txt` (once).

## 1. Serve the trained model (what the app uses)
```bash
cd experiments && python export_model.py      # about 1 minute, writes app/model_store/pulse_model.pt (0.4 MB)
cd .. && uvicorn app.main:app --port 8000     # GET /health shows "mode": "model"
pytest -q tests                               # 5 tests, all should pass
```
A copy of `pulse_model.pt` is included, so you can skip the export and just start the API.
Without torch or the checkpoint the API still starts and answers in `"mode": "stub"`.

`POST /recommend`  body: `{"user_id": "u1", "watchlist": ["XOM"], "recent_items": ["NEE:0","DUK:1"], "top_k": 10}`
- `recent_items` (the user's latest clicks, format `TICKER:slot`) takes priority -> personalisation `session`
- else `watchlist` -> `watchlist`; else a market-wide feed -> `generic`
`GET /model/info` states what the model was trained on (simulated data).

## 2. Run the remaining model experiments
```bash
cd experiments
python extras.py --quick     # about 1 minute, checks the setup
python extras.py             # full run, about 6-8 minutes; writes results/extras.json
python extras.py --only drift adversarial     # pick experiments
```
| experiment | what it answers |
|---|---|
| explain | do the built-in explanations agree with exact Shapley values and a LIME-style surrogate? |
| drift | how much does a regime shift hurt, can a monitor catch it, does fine-tuning recover it (and forget)? |
| adversarial | can manipulated headline tone or injected clicks push a target stock into feeds? does bounding help? |
| privacy | can an observer infer a user's persona from the top-10 list? what does noise cost? |
| compress | pruning, low-rank, int8: accuracy, size, latency |
| personas | do behavioural clusters recover the investor types? |

## 3. Live data (optional, needs your keys in the root .env)
`PULSE_LIVE_DATA=1 NEWSAPI_KEY=... uvicorn app.main:app` fills the catalogue with real headlines
(sentiment = a simple word-list score; momentum from yfinance if installed).
This path could not be tested in the development sandbox: run `python -m app.data.catalogue` once to check it.

## Honest limits
- The model is trained on SIMULATED sessions. App users are served through the session + content path;
  the long-term (LightGCN) user vector only becomes useful after retraining on real click logs.
- Sector and market-cap labels in `app/universe.py` are approximate; edit them, then re-run `export_model.py`.
- Sentiment for live headlines is a word-list score, not a validated model.
