#!/usr/bin/env bash
# Run full legacy sigma* experiment using the existing ATMO environment.
#   legacy:    global schedule, fixed legacy initial amplitude
#
# Launch from tcsh with: bash repro/run_sigma_legacy.sh [--prepare-only]
# Add --prepare-only to inspect resolved configs without running inference.
# Every invocation needs a fresh output directory; existing runs are preserved.
set -euo pipefail

# Locate the source tree and use Python from the active environment.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHON="${PYTHON:-python}"

# Input artifacts: environment variables can override these ATMO defaults.
DEFAULT_CHECKPOINT_DIR=/home/theaqg/CEDDAR_runs/paper1_original/checkpoints_from_lumi
DEFAULT_CHECKPOINT_NAME=B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5
DEFAULT_CHECKPOINT_NAME+=__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56.pth.tar

export DATA_DIR="${DATA_DIR:-/home/theaqg/CEDDAR_migration/Data/Data_DiffMod}"
export STATS_LOAD_DIR="${STATS_LOAD_DIR:-$REPO_DIR/repro/assets/stats/statistics_run/stats}"
export PUBLISHED_CHECKPOINT="${PUBLISHED_CHECKPOINT:-$DEFAULT_CHECKPOINT_DIR/$DEFAULT_CHECKPOINT_NAME}"

# All runs execute sequentially with the same CPU thread settings.
export DEVICE=cpu
export OMP_NUM_THREADS="${CPU_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export EXP_DATE="$(date -u +%Y%m%dT%H%M%SZ)"

# Shared experiment settings. Ramp parameters are inherited from SIGMA_CONFIG:
# F_final_test_eval.yaml specifies ramp_start_frac=0.60, ramp_end_frac=0.85.
SIGMA_SEED="${SIGMA_SEED:-504}"
MAX_DATES=1000 #"${MAX_DATES:-24}"
ENSEMBLE_SIZE=32 #"${ENSEMBLE_SIZE:-8}"
SIGMA_CONFIG="${SIGMA_CONFIG:-$REPO_DIR/sbgm/config/component_study/F_final_test_eval.yaml}"
SIGMA_STAR_GRID="${SIGMA_STAR_GRID:-0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25}"

# A timestamped parent separates this run from all earlier results.
DEFAULT_OUTPUT_ROOT=/home/theaqg/CEDDAR_runs/paper1_revision
LEGACY_NAME="sigma_init_d${MAX_DATES}_m${ENSEMBLE_SIZE}_s${SIGMA_SEED}_${EXP_DATE}"
LEGACY_DIR="${LEGACY_DIR:-$DEFAULT_OUTPUT_ROOT/$LEGACY_NAME}"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --prepare-only ) ]]; then
    echo 'Usage: bash repro/run_sigma_legacy.sh [--prepare-only]' >&2
    exit 2
fi

# Validate inputs and reject an existing RUN or any source-tree output.
LEGACY_DIR="$(
"$PYTHON" -c '
import os
import sys
from pathlib import Path

from sbgm.runtime import external_output

for key in ("DATA_DIR", "STATS_LOAD_DIR"):
    if not Path(os.environ[key]).is_dir():
        raise SystemExit(f"Missing directory: {key}")

if not Path(os.environ["PUBLISHED_CHECKPOINT"]).is_file():
    raise SystemExit("Missing checkpoint")

root = external_output(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
print(root)
' "$LEGACY_DIR"
)"

LEGACY="$LEGACY_DIR/legacy"
RUN_DIR="$LEGACY"

# Keep one log per stage. pipefail stops the script if the command fails,
# even when tee successfully writes its output to the log.
run_logged() {
    local label="$1"
    shift
    "$@" 2>&1 | tee "$LEGACY_DIR/$label.log"
}
trap 'echo "Stopped on error. Keep partial results for inspection: $LEGACY_DIR" >&2' ERR

# Prepare all configurations first. Only scaling mode and initialization differ;
# generation and evaluation subsequently read each saved resolved_config.yaml.
COMMON=(
    --config "$SIGMA_CONFIG"
    --noise-mode paired
    --seed "$SIGMA_SEED"
    --sigma-star-grid $SIGMA_STAR_GRID
    --steps 56
    --ensemble-size "$ENSEMBLE_SIZE"
    --max-dates "$MAX_DATES"
    --split valid
)

run_logged prepare_legacy \
    bash "$REPO_DIR/repro/run_sigma_star.sh" prepare \
    --run-dir "$LEGACY" "${COMMON[@]}" \
    --sigma-star-mode global --initial-state legacy_sigma_max


run_logged config_check \
    "$PYTHON" -m repro.check_sigma_initialization "$RUN_DIR" --config-only

if [[ "${1:-}" == --prepare-only ]]; then
    echo "Prepared and checked: $RUN_DIR (no inference)"
    exit 0
fi

# Generate separately, then verify checkpoint identity, paired draws and exact
# sigma*=1 sample agreement across all three runs before evaluating anything.
run_logged "generate_legacy" \
    bash "$REPO_DIR/repro/run_sigma_star.sh" generate --run-dir "$RUN_DIR"

run_logged generation_check \
    "$PYTHON" -m repro.check_sigma_initialization "$RUN_DIR" --generation-only

# Evaluate each run independently and export three distinctly named CSVs.
# sources.json records their original paths and hashes to avoid mixing tables.
run_logged "evaluate_legacy" \
    bash "$REPO_DIR/repro/run_sigma_star.sh" evaluate --run-dir "$RUN_DIR"

run_logged result_check \
    "$PYTHON" -m repro.check_sigma_initialization "$RUN_DIR" \
    --output "$LEGACY_DIR/run"

echo "Finished. Named tables and their source hashes: $LEGACY_DIR/run"
