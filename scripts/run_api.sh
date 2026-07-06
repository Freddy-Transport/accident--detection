#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/traffic_accident_rnd}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
mkdir -p logs/service
exec .venv/bin/uvicorn traffic_accident_rnd.api:app --host "${HOST}" --port "${PORT}" --log-level info
