#!/usr/bin/env bash
# Starts recommendation engine (:8000), backend (:4000) and frontend (:3000) together.
# Ctrl+C stops all three.
cd "$(dirname "$0")/.."

if [ ! -d recommendation-engine/.venv ] || [ ! -d backend/node_modules ] || [ ! -d frontend/node_modules ]; then
  echo "Dependencies are missing (a fresh unzip/clone or a clean removes them)."
  echo "Run:  ./scripts/setup.sh"
  exit 1
fi

trap 'kill 0' EXIT INT TERM
(cd recommendation-engine && . .venv/bin/activate && uvicorn app.main:app --port 8000 --reload) &
(cd backend && npm run dev) &
(cd frontend && npm run dev) &
wait