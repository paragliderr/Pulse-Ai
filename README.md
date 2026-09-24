# Pulse AI

Personalized, explainable news & market recommendation engine — a module of Finance OS.

## Architecture

Three services, orchestrated with Docker Compose:

```
frontend/                Next.js + React + Tailwind — dashboard UI
backend/                 Node.js + Express + Prisma — auth, users, watchlists, API gateway
recommendation-engine/   Python + FastAPI + PyTorch — LightGCN + SR-GNN, explainability (owned by ML lead)
```

The frontend never talks to the recommendation engine directly — it goes through the backend,
which handles auth and forwards ranking requests internally. This keeps the ML service free to
change its internals without breaking the frontend contract.

```
[frontend] --HTTP--> [backend] --internal HTTP--> [recommendation-engine]
                          |
                     [postgres] [redis]
```

## Who owns what

- **Recommendation engine** (`recommendation-engine/`) — data ingestion, preprocessing, LightGCN +
  SR-GNN models, fusion/ranking, SHAP/LIME explainability. This is the core ML work.
- **Backend** (`backend/`) — auth, user/watchlist CRUD, database schema, calling the recommendation
  engine and shaping its output for the frontend. Pick this up if you're doing API/infra work.
- **Frontend** (`frontend/`) — dashboard, feed UI, watchlist management, explanation panel display.
  Pick this up if you're doing UI work.

## Quick start

1. Copy `.env.example` to `.env` and fill in API keys (NewsAPI, Alpha Vantage).
2. `docker-compose up --build`
3. Frontend: http://localhost:3000
4. Backend API: http://localhost:4000
5. Recommendation engine (internal, for debugging): http://localhost:8000/docs (FastAPI auto-docs)

Each service also has its own README with more detail and can be run standalone outside Docker
for faster iteration — see `frontend/README.md`, `backend/README.md`, `recommendation-engine/README.md`.

## Troubleshooting

**"Couldn't load feed: Network Error" in the browser** — this means the browser's request
never got any HTTP response at all (not even an error one). Check, in order:

1. **Is the backend actually running and printing "Pulse AI backend listening on :4000"?**
   If you're running services natively (not via `docker-compose up`) and launched from inside
   `backend/` with `cd backend && npm run dev`, make sure you're on the latest code — earlier
   versions of `dotenv.config()` only looked in the current directory and silently missed the
   root `.env`. Fixed now: it resolves `.env` from the repo root regardless of where you run it
   from. Restart `npm run dev` after pulling this fix.
2. **Are you running frontend and backend separately (not Docker)?** The frontend now proxies
   `/api/*` to the backend server-side (see `next.config.js`'s `rewrites`), controlled by
   `BACKEND_INTERNAL_URL` in `.env` (defaults to `http://localhost:4000`). If your backend runs
   on a different port or host, set `BACKEND_INTERNAL_URL` accordingly and restart the frontend
   dev server (Next only reads env vars at server start).
3. **Cloud/sandboxed dev environments** (Codespaces-style forwarded ports, remote containers,
   etc.): this is the most common cause. If the frontend tries to call the backend directly at
   a hardcoded `localhost:PORT`, "localhost" from the browser's perspective often does not mean
   what you think it means in these environments. The same-origin proxy above sidesteps this
   entirely — the browser only ever talks to the frontend's own origin.
4. **Recommendation engine not running is NOT what causes this specific error** — the backend
   catches that and returns a normal (non-2xx) JSON error, or for `/api/feed/demo`, silently
   falls back to static demo data. If you see a proper error message (not "Network Error"),
   that's a different, more specific problem — check the backend's terminal output for details.

**Want to see the UI working before wiring up login or the recommendation engine at all?**
`GET /api/feed/demo` needs no auth and always returns 200 (real recommendations if the engine
is reachable, static fallback data if not). The homepage already calls this automatically when
there's no login token in `localStorage`, so `docker-compose up` (or running frontend + backend
alone) is enough to see a working feed immediately — the recommendation engine is optional for
frontend/backend work to proceed.

## Data flow (high level)

1. Recommendation engine ingests price (yfinance), news (NewsAPI), and fundamentals (Alpha Vantage)
   data on a schedule, and preprocesses it (sentiment scoring, technical indicators, graph/session
   construction).
2. Backend stores users, watchlists, and interaction events in Postgres, and caches hot data in Redis.
3. When a user opens their feed, backend calls the recommendation engine's `/recommend` endpoint
   with the user's ID and watchlist, gets back a ranked list with explanations, and returns it to
   the frontend.
