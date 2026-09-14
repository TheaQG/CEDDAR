#!/usr/bin/env bash
# Deliberate reduced training experiment, not a smoke test; requires real data.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CEDDAR_RUNS="${CEDDAR_RUNS:-$REPO_DIR/../CEDDAR_runs/repro/02_reduced_run}"
exec bash "$REPO_DIR/repro/run_model.sh" \
  --config_path "$REPO_DIR/repro/02_reduced_run/reduced_run_config.yaml" \
  --mode full_pipeline --device "${DEVICE:-cpu}" "$@"
