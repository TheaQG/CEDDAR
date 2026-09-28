#!/usr/bin/env bash

# =============================================================================
# Full legacy sigma* experiment
#
# Sampling setup:
#   - sigma_star_mode = global
#   - initial_state   = legacy_sigma_max
#
# This is the main dense sigma* sweep intended for the manuscript analysis.
#
# Launch from tcsh with:
#   bash repro/run_sigma_legacy.sh
# or inspect the resolved configuration without running inference:
#   bash repro/run_sigma_legacy.sh --prepare-only
#
# Every invocation creates a fresh timestamped output directory.
# =============================================================================

set -euo pipefail


# -----------------------------------------------------------------------------
# Repository and Python environment
# -----------------------------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHON="${PYTHON:-python}"


# -----------------------------------------------------------------------------
# Input artifacts. Environment variables may override these defaults.
# -----------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIR=/home/theaqg/CEDDAR_runs/paper1_original/checkpoints_from_lumi

DEFAULT_CHECKPOINT_NAME=B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5
DEFAULT_CHECKPOINT_NAME+=__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56.pth.tar

export DATA_DIR="${
    DATA_DIR:-/home/theaqg/CEDDAR_migration/Data/Data_DiffMod
}"

export STATS_LOAD_DIR="${
    STATS_LOAD_DIR:-$REPO_DIR/repro/assets/stats/statistics_run/stats
}"

export PUBLISHED_CHECKPOINT="${
    PUBLISHED_CHECKPOINT:-$DEFAULT_CHECKPOINT_DIR/$DEFAULT_CHECKPOINT_NAME
}"


# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------

export DEVICE="${DEVICE:-cpu}"
export OMP_NUM_THREADS="${CPU_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

EXP_DATE="$(date -u +%Y%m%dT%H%M%SZ)"


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------

SIGMA_SEED="${SIGMA_SEED:-504}"

MAX_DATES="${MAX_DATES:-1000}"

ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-32}"

SIGMA_CONFIG="${
    SIGMA_CONFIG:-$REPO_DIR/sbgm/config/component_study/F_final_test_eval.yaml
}"

# Main dense manuscript grid.
SIGMA_STAR_GRID="${
    SIGMA_STAR_GRID:-0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25
}"


# -----------------------------------------------------------------------------
# Output directory
#
# Parent name identifies this as the main legacy sweep.
# Internal "legacy" directory identifies the sampling mechanism.
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT=/home/theaqg/CEDDAR_runs/paper1_revision

EXPERIMENT_NAME="sigma_legacy_main_d${MAX_DATES}_m${ENSEMBLE_SIZE}_seed${SIGMA_SEED}_${EXP_DATE}"

LEGACY_MAIN_DIR="${
    LEGACY_MAIN_DIR:-$DEFAULT_OUTPUT_ROOT/$EXPERIMENT_NAME
}"


# -----------------------------------------------------------------------------
# CLI validation
# -----------------------------------------------------------------------------

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--prepare-only" ) ]]; then
    echo "Usage: bash repro/run_sigma_legacy.sh [--prepare-only]" >&2
    exit 2
fi


# -----------------------------------------------------------------------------
# Validate inputs and create a fresh external output directory
# -----------------------------------------------------------------------------

LEGACY_MAIN_DIR="$(
"$PYTHON" -c '
import os
import sys
from pathlib import Path

from sbgm.runtime import external_output


for key in ("DATA_DIR", "STATS_LOAD_DIR"):
    path = Path(os.environ[key])

    if not path.is_dir():
        raise SystemExit(
            f"Missing directory: {key}={path}"
        )


checkpoint = Path(
    os.environ["PUBLISHED_CHECKPOINT"]
)

if not checkpoint.is_file():
    raise SystemExit(
        f"Missing checkpoint: {checkpoint}"
    )


root = external_output(
    sys.argv[1]
)

root.mkdir(
    parents=True,
    exist_ok=False,
)

print(root)
' "$LEGACY_MAIN_DIR"
)"


RUN_DIR="$LEGACY_MAIN_DIR/legacy"


# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------

run_logged() {
    local label="$1"
    shift

    "$@" 2>&1 | tee "$LEGACY_MAIN_DIR/$label.log"
}


trap '
echo "Stopped on error. Partial results retained at:" >&2
echo "$LEGACY_MAIN_DIR" >&2
' ERR


# -----------------------------------------------------------------------------
# Shared sigma* preparation arguments
# -----------------------------------------------------------------------------

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


# =============================================================================
# 1. Prepare
# =============================================================================

echo
echo "============================================================"
echo "Preparing main legacy sigma* sweep"
echo "Grid: $SIGMA_STAR_GRID"
echo "Dates: $MAX_DATES"
echo "Members: $ENSEMBLE_SIZE"
echo "Seed: $SIGMA_SEED"
echo "Output: $RUN_DIR"
echo "============================================================"
echo


run_logged prepare_legacy \
    bash "$REPO_DIR/repro/run_sigma_star.sh" prepare \
    --run-dir "$RUN_DIR" \
    "${COMMON[@]}" \
    --sigma-star-mode global \
    --initial-state legacy_sigma_max


if [[ "${1:-}" == "--prepare-only" ]]; then
    echo
    echo "Prepared main legacy sigma* sweep:"
    echo "$RUN_DIR"
    echo
    echo "No inference performed."
    exit 0
fi


# =============================================================================
# 2. Generate
# =============================================================================

echo
echo "============================================================"
echo "Generating main legacy sigma* sweep"
echo "============================================================"
echo


run_logged generate_legacy \
    bash "$REPO_DIR/repro/run_sigma_star.sh" generate \
    --run-dir "$RUN_DIR"


# =============================================================================
# 3. Evaluate
# =============================================================================

echo
echo "============================================================"
echo "Evaluating main legacy sigma* sweep"
echo "============================================================"
echo


run_logged evaluate_legacy \
    bash "$REPO_DIR/repro/run_sigma_star.sh" evaluate \
    --run-dir "$RUN_DIR"


# =============================================================================
# Finished
# =============================================================================

echo
echo "============================================================"
echo "Finished main legacy sigma* sweep"
echo
echo "Run directory:"
echo "$RUN_DIR"
echo "============================================================"