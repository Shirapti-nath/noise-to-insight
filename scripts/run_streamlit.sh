#!/usr/bin/env bash
# Launch Streamlit from project root so `src` imports resolve.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [ -d .venv ]; then
  exec .venv/bin/streamlit run app/streamlit_app.py "$@"
fi
exec streamlit run app/streamlit_app.py "$@"
