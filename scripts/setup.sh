#!/usr/bin/env bash
# One-time install of all three services.
set -e
cd "$(dirname "$0")/.."
echo "==> engine";   (cd recommendation-engine && python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt)
echo "==> backend";  (cd backend && npm install)
echo "==> frontend"; (cd frontend && npm install)
echo "Done. Run: ./scripts/start.sh"
