# Pulse AI — Frontend

Next.js (App Router) + React + Tailwind CSS.

## Structure

```
src/
  app/            pages (App Router) — page.js is the home feed, layout.js is the shell
  components/     UI components (FeedCard.js renders one recommendation + its explanation)
  lib/api.js      axios client, auto-attaches JWT from localStorage
  styles/         Tailwind entry point
```

## Local dev (without Docker)

```bash
npm install
npm run dev   # http://localhost:3000
```

Expects the backend running at `NEXT_PUBLIC_API_URL` (default `http://localhost:4000`).

## What's stubbed vs. real

- `page.js` fetches `/api/feed` and renders whatever comes back — functional once the backend
  and recommendation engine are both running.
- No auth screens yet (login/signup pages, token storage flow) — the API client already expects
  a `pulse_token` in localStorage, so wiring up login/signup pages is a clean first task.
- `FeedCard` expects `item.explanation` as a flat `{feature: weight}` object from the SHAP/LIME
  output — adjust if the recommendation engine's response shape changes.

## Good first tasks

Login/signup pages, a watchlist management page (add/remove tickers, calls `/api/watchlist`),
loading/error states polish, and a toggle between "general feed" and "personalized feed" views
(the backend returns whichever the engine decides based on watchlist size, but a manual toggle
is a nice UX addition per the project's before/after personalization demo).
