# Pulse AI

News feed that re-ranks itself around what you read, and explains why.

```
browser -> frontend :3000 -> backend :4000 -> recommendation-engine :8000
```

## Run it

You need **Node 18+** and **Python 3.9+**. No Docker, no database server.

```bash
./scripts/setup.sh      # once
./scripts/start.sh      # starts all three; Ctrl+C stops them
```

Open http://localhost:3000

(Windows: run the three commands inside `scripts/*.sh` by hand, one terminal each, or use WSL.)

## Use it

1. Click **Sign up** (any email, password 8+ characters), or **Try the demo** for a feed without an account.
2. **Watchlist**: add tickers like `XOM`, `NEE`, `AAPL`. The feed starts from these.
3. **Feed**: click **Read** on a story and the feed re-ranks toward it. **Why this?** shows the score breakdown. Filter by sector with the chips.

Headlines are placeholders tagged `sample`.

## Check it works

```bash
curl localhost:8000/health    # engine
curl localhost:4000/health    # backend
cd recommendation-engine && .venv/bin/pip install pytest httpx && .venv/bin/python -m pytest
```

## Notes

- Users, watchlists and clicks are saved in `backend/data.json` (delete it to reset).
- Settings (optional env vars): `JWT_SECRET`, `ENGINE_URL` (backend), `BACKEND_INTERNAL_URL` (frontend).
- API: `GET /api/feed/demo?recent=NEE:0&watchlist=XOM`, `GET /api/feed`, `GET /api/status`, `POST /api/interactions`,
  `GET|POST /api/watchlist`, `DELETE /api/watchlist/:ticker`, `POST /api/auth/signup|login`.
