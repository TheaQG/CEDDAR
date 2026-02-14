#!/bin/bash
set -e

echo "----------------------------------------"
echo " Running CEDDAR 01_small_test (CPU) "
echo "----------------------------------------"

# -----------------------------
# Resolve project root
# -----------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# -----------------------------
# Define paths (portable)
# -----------------------------
# Where to write outputs (can be overridden by user)
export CEDDAR_RUNS="${CEDDAR_RUNS:-$HOME/ceddar_runs}"

# Where the small dataset lives (expected to be a sibling of the CEDDAR repo by default)
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/../Data_DiffMod_small}"

# Output dirs (always outside the repo to avoid cluttering and ensure reproducibility)
export CKPT_DIR="$CEDDAR_RUNS/repro/01_small_test/outputs/checkpoints"
export SAMPLE_DIR="$CEDDAR_RUNS/repro/01_small_test/outputs/samples"
export EVAL_DIR="$SAMPLE_DIR/evaluation"
export LOG_DIR="$CEDDAR_RUNS/repro/01_small_test/outputs/logs"

# Path to precomputed statistics for conditioning (expected to be inside the repo)
export STATS_LOAD_DIR="$PROJECT_ROOT/repro/assets/stats/statistics_run/stats"

# -----------------------------
# Sanity checks
# -----------------------------
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found at $DATA_DIR"
    echo "Expected by default at: $PROJECT_ROOT/../Data_DiffMod_small"
    echo "Fix by either:"
    echo "  a) Placing Data_DiffMod_small next to the CEDDAR repo, or"
    echo "  b) exporting DATA_DIR=/path/to/Data_DiffMod_small before running this script."
    exit 1
fi

if [ ! -d "$STATS_LOAD_DIR" ]; then
    echo "ERROR: Statistics directory not found at $STATS_LOAD_DIR"
    echo "This repo should include stats JSON files at repro/assets/stats/statistics_run/stats"
    echo "Fix by ensuring you have the latest version of the repo with the stats files included."
    exit 1
fi

# Create output directories
mkdir -p "$CKPT_DIR"
mkdir -p "$SAMPLE_DIR"
mkdir -p "$EVAL_DIR"
mkdir -p "$LOG_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Data dir: $DATA_DIR"
echo "Statistics dir: $STATS_LOAD_DIR"

# -----------------------------
# Run training + generation + evaluation
# -----------------------------
python -m sbgm.cli.main_app \
    --config "$PROJECT_ROOT/repro/01_small_test/small_test_config.yaml"


echo "----------------------------------------"
echo " Evaluation summary (from prob_summary.csv) "
echo "----------------------------------------"

# Find the newest prob_summary.csv produced by the run (written under samples/evaluation/<MODEL_NAME>/...)
SUMMARY_CSV="$(find "$SAMPLE_DIR/evaluation" -type f -name "prob_summary.csv" -print0 2>/dev/null \
  | xargs -0 ls -t 2>/dev/null | head -n 1)"

if [ -z "$SUMMARY_CSV" ]; then
  echo "WARNING: Could not find prob_summary.csv under $SAMPLE_DIR/evaluation"
  echo "Look for it under: .../prcp/probabilistic/tables/prob_summary.csv"
else
  echo "Found: $SUMMARY_CSV"
  SUMMARY_CSV="$SUMMARY_CSV" python - <<'PY'
import pandas as pd, os
csv_path = os.environ["SUMMARY_CSV"]
df = pd.read_csv(csv_path)
keep = ["CRPS_ensemble", "PMM_MAE"]
df = df[df["metric"].isin(keep)]
for _, r in df.iterrows():
    print(f'{r["metric"]:22s} mean={r["mean"]:8.3f}  std={r.get("std", float("nan")):8.3f}  N={r.get("N","")}')
PY
fi


echo "----------------------------------------"
echo " Small test completed successfully. "
echo "----------------------------------------"