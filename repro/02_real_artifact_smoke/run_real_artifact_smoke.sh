#!/usr/bin/env bash
# Use the checked smoke path: explicit checkpoint, no training or synthetic fallback.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${SMOKE_CONFIG:-$REPO_DIR/repro/02_real_artifact_smoke/real_artifact_smoke_config.yaml}"
: "${DATA_DIR:?Set DATA_DIR to the real Data_DiffMod directory}"
: "${PUBLISHED_CHECKPOINT:?Set PUBLISHED_CHECKPOINT to the archived checkpoint file}"
: "${STATS_LOAD_DIR:?Set STATS_LOAD_DIR to the matching training statistics root}"
export CEDDAR_RUNS="${CEDDAR_RUNS:-$REPO_DIR/../CEDDAR_runs/repro/02_real_artifact_smoke}"
exec bash "$REPO_DIR/repro/01_small_test/run_small_test.sh" \
  --require-real --config "$CFG" --data-root "$DATA_DIR" \
  --checkpoint "$PUBLISHED_CHECKPOINT" --stats-root "$STATS_LOAD_DIR" \
  --device "${DEVICE:-cpu}" "$@"
