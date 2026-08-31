#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/traffic_accident_rnd
DEMO="$ROOT/0831可视化"
echo "The label-assisted demo has been retired. Running the annotation-free full-video benchmark."
exec "$DEMO/scripts/run_unknown_benchmark.sh" "$@"
