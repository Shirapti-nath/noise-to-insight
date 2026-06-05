#!/usr/bin/env bash
# Run full pipeline on demo/fixture data (heuristic mode, no API key required).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
INPUT="${1:-tests/fixtures/segment_orders.csv}"
RUN_ID="${2:-demo}"

"$PYTHON" -m src.cli --input "$INPUT" --run-id "$RUN_ID" --no-llm
echo "Artifacts: data/artifacts/${RUN_ID}/"
