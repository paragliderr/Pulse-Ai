#!/usr/bin/env bash
# Starts engine :8000, backend :4000, frontend :3000. Ctrl+C stops all.
cd "$(dirname "$0")/.."
trap 'kill 0' EXIT INT TERM
(cd recommendation-engine && .venv/bin/python -m uvicorn app.main:app --port 8000) &
(cd backend && npm run dev) &
(cd frontend && npm run dev) &
wait
