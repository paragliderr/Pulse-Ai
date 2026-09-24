# Pulse AI — Backend

Node.js + Express + Prisma (PostgreSQL) + Redis. Handles auth, users, watchlists, interaction
logging, and proxies feed requests to the recommendation engine.

## Structure

```
src/
  index.js              app entry point
  routes/                route definitions (thin)
  controllers/            request handling + response shaping
  services/                business logic (DB queries, calling the recommendation engine)
  middleware/              auth.middleware.js — JWT verification
  config/                  prisma.js — Prisma client singleton
prisma/
  schema.prisma           DB schema: User, WatchlistItem, Session, Interaction, FeedCache
```

## Local dev (without Docker)

```bash
npm install
npx prisma migrate dev --name init   # creates tables from schema.prisma
npm run dev                          # nodemon, http://localhost:4000
```

## What's stubbed vs. real

- Auth (signup/login) and watchlist CRUD are functional against Prisma.
- `GET /api/feed` calls the recommendation engine at `RECOMMENDATION_ENGINE_URL` — this will
  return an error until that service is running and its `/recommend` endpoint is implemented.
- `FeedCache` gives a simple 15-minute cache so the frontend isn't waiting on the ML service
  every page load — tune `CACHE_TTL_MINUTES` in `feed.controller.js` as needed.

## Extending this

Good first tasks for whoever picks this up: pagination on `/api/feed`, a `/api/user/persona`
endpoint to let users set their investor persona explicitly, rate limiting on auth routes,
and swapping the naive "one open session" logic in `interaction.service.js` for a proper
time-gap session boundary (matches the session-segmentation approach in the ML engine).
