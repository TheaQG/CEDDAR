#!/bin/bash
set -e

echo "----------------------------------------"
echo " Running CEDDAR 02_reduced_run (local) "
echo "----------------------------------------"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Export experiment date if not set, for config interpolation
export EXP_DATE="${EXP_DATE:-$(date -u +%Y%m%dT%H%M%SZ)}"

# Where to write outputs (override if you want)
export CEDDAR_RUNS="${CEDDAR_RUNS:-$HOME/ceddar_runs}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/../Data_DiffMod_small}"

export CKPT_DIR="$CEDDAR_RUNS/repro/02_reduced_run/outputs/checkpoints"
export SAMPLE_DIR="$CEDDAR_RUNS/repro/02_reduced_run/outputs/samples"
export EVAL_DIR="$SAMPLE_DIR/evaluation"
export LOG_DIR="$CEDDAR_RUNS/repro/02_reduced_run/outputs/logs"

export STATS_LOAD_DIR="$PROJECT_ROOT/repro/assets/stats/statistics_run/stats"

mkdir -p "$CKPT_DIR" "$SAMPLE_DIR" "$EVAL_DIR" "$LOG_DIR"

# Sanity checks
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: DATA_DIR not found: $DATA_DIR"
  echo "Place Data_DiffMod_small next to the repo, or export DATA_DIR=/path/to/Data_DiffMod_small"
  exit 1
fi
if [ ! -d "$STATS_LOAD_DIR" ]; then
  echo "ERROR: STATS_LOAD_DIR not found: $STATS_LOAD_DIR"
  exit 1
fi

# Decide device:
# - user can force with DEVICE=cpu or DEVICE=cuda
# - otherwise try cuda, fall back to cpu
if [ -z "${DEVICE:-}" ]; then
  if python - <<'PY' 2>/dev/null | grep -q "True"; then
import torch
print(torch.cuda.is_available())
PY
    export DEVICE="cuda"
  else
    export DEVICE="cpu"
  fi
fi

echo "Project root: $PROJECT_ROOT"
echo "Data dir: $DATA_DIR"
echo "Runs dir: $CEDDAR_RUNS"
echo "Device: $DEVICE"

# DEVICE is still exported for config
export DEVICE

python -m sbgm.cli.main_app \
  --config "$PROJECT_ROOT/repro/02_reduced_run/reduced_run_config.yaml"

echo "----------------------------------------"
echo " Reduced run completed successfully. "
echo "Outputs in: $CEDDAR_RUNS/repro/02_reduced_run/outputs"
echo "----------------------------------------"