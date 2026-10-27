#!/usr/bin/env bash
# One-command (re)install of everything. Safe to run again any time.
#   ./scripts/setup.sh        -> lite: runs the whole app, no ML libraries
#   ./scripts/setup.sh ml     -> also installs torch etc. for the experiments
set -e
cd "$(dirname "$0")/.."
MODE="${1:-lite}"

echo "==> .env"
[ -f .env ] || { cp .env.example .env; echo "    created .env from .env.example (fill in your keys later)"; }

echo "==> Python virtual environment (recommendation-engine/.venv)"
cd recommendation-engine
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements-lite.txt
if [ "$MODE" = "ml" ]; then pip install -r requirements-ml.txt; fi
deactivate
cd ..

echo "==> Backend (Node)"
(cd backend && npm install && npx prisma generate)

echo "==> Frontend (Node)"
(cd frontend && npm install)

echo
echo "Done. Start everything with:  ./scripts/start.sh"