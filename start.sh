#!/usr/bin/env bash
# Launch the stack in Docker (default) or against local processes.
set -euo pipefail

MODE="${1:-docker}"

if [[ ! -f .env ]]; then
  echo "No .env found. Copying .env.example — add your API keys before continuing."
  cp .env.example .env
  exit 1
fi

case "$MODE" in
  docker)
    exec docker compose up --build
    ;;

  local)
    command -v redis-server >/dev/null || { echo "redis-server not on PATH"; exit 1; }
    redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes

    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    BACKEND=$!
    trap 'kill $BACKEND 2>/dev/null || true' EXIT

    # Backend loads ~2GB of models before it answers; wait rather than racing it
    until curl -sf http://127.0.0.1:8000/health >/dev/null; do sleep 2; done
    echo "Backend ready on :8000"

    API_URL=http://127.0.0.1:8000 streamlit run app/frontend.py --server.port=8501
    ;;

  *)
    echo "Usage: ./start.sh [docker|local]"
    exit 1
    ;;
esac
