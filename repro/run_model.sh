#!/usr/bin/env bash
# Portable entry point; e.g. CEDDAR_RUNS=/work/runs/experiment DATA_DIR=/data ...
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -m sbgm.cli.main_app "$@"
