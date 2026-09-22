#!/usr/bin/env bash
# Run from the activated ATMO environment, preferably inside tmux:
#   bash repro/run_sigma_ramp_comparison.sh [--prepare-only]
# Six ramps, each with 24 validation dates, 8 members and paired noise.
# Edit RAMPS below to add cases. Existing output directories are never reused.
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --prepare-only ) ]]; then
    echo "Usage: bash repro/run_sigma_ramp_comparison.sh [--prepare-only]" >&2
    exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHON="${PYTHON:-python}"
export DEVICE=cpu
export OMP_NUM_THREADS="${CPU_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export EXP_DATE="$(date -u +%Y%m%dT%H%M%SZ)"

# Same artifacts as the earlier initialization comparison; overrides are optional.
CHECKPOINT_NAME=B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5
CHECKPOINT_NAME+=__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56.pth.tar
export DATA_DIR="${DATA_DIR:-/home/theaqg/CEDDAR_migration/Data/Data_DiffMod}"
export STATS_LOAD_DIR="${STATS_LOAD_DIR:-$REPO_DIR/repro/assets/stats/statistics_run/stats}"
export PUBLISHED_CHECKPOINT="${PUBLISHED_CHECKPOINT:-/home/theaqg/CEDDAR_runs/paper1_original/checkpoints_from_lumi/$CHECKPOINT_NAME}"
SIGMA_CONFIG="${SIGMA_CONFIG:-$REPO_DIR/sbgm/config/component_study/F_final_test_eval.yaml}"
SIGMA_SEED="${SIGMA_SEED:-504}"
COMPARISON_DIR="${COMPARISON_DIR:-/home/theaqg/CEDDAR_runs/paper1_revision/sigma_ramps_d24_m8_s${SIGMA_SEED}_${EXP_DATE}}"

# First four: shift a ramp of width 0.25. Last two: change width at start 0.30.
# In the default 56-step sampler, these all begin after active churn ends.
RAMPS=("0.15 0.40" "0.30 0.55" "0.45 0.70" "0.60 0.85"
       "0.30 0.40" "0.30 0.85")

# Reuse the existing path guard and prepare command's artifact/config validation.
COMPARISON_DIR="$("$PYTHON" -c '
import sys
from sbgm.runtime import external_output
root = external_output(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
print(root)
' "$COMPARISON_DIR")"
echo "Comparison: $COMPARISON_DIR"

run_logged() {
    local log="$1"
    shift
    "$@" 2>&1 | tee "$COMPARISON_DIR/$log.log"
}
trap 'echo "Stopped on error; partial outputs remain in $COMPARISON_DIR" >&2' ERR

# Prepare every run before spending time on inference. Only ramp bounds differ.
RUN_DIRS=()
for ramp in "${RAMPS[@]}"; do
    read -r start end <<< "$ramp"
    name="ramp_${start}_${end}"
    config="$COMPARISON_DIR/$name.yaml"
    run="$COMPARISON_DIR/$name"
    RUN_DIRS+=("$run")

    # Write a per-ramp YAML outside the repository, retaining config interpolation.
    # Clear optional sigma thresholds so they cannot override the fractions.
    "$PYTHON" - "$SIGMA_CONFIG" "$config" "$start" "$end" <<'PY'
import sys
import yaml
from pathlib import Path
source, target, start, end = sys.argv[1:]
cfg = yaml.safe_load(Path(source).read_text())
cfg['full_gen_eval'].setdefault('sigma_control', {}).update(
    ramp_start_frac=float(start), ramp_end_frac=float(end),
    ramp_start_sigma=None, ramp_end_sigma=None)
Path(target).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

    run_logged "prepare_$name" bash "$REPO_DIR/repro/run_sigma_star.sh" prepare \
        --run-dir "$run" --config "$config" --split valid \
        --max-dates 24 --ensemble-size 8 --steps 56 --seed "$SIGMA_SEED" \
        --noise-mode paired --sigma-star-mode late_ramp --initial-state schedule \
        --sigma-star-grid 0.95 1.00 1.05
done

if [[ "${1:-}" == --prepare-only ]]; then
    echo "Prepared all six ramps; no inference. Configs: $COMPARISON_DIR"
    exit 0
fi

# Sequential CPU runs; evaluation also produces the existing sigma* plots.
for run in "${RUN_DIRS[@]}"; do
    run_logged "generate_${run##*/}" bash "$REPO_DIR/repro/run_sigma_star.sh" generate --run-dir "$run"
    run_logged "evaluate_${run##*/}" bash "$REPO_DIR/repro/run_sigma_star.sh" evaluate --run-dir "$run"
done

echo "Finished: $COMPARISON_DIR"
