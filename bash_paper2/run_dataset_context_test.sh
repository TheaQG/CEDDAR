#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------
# Paper 2 – Dataset Context Test Runner
# -------------------------------------------------

echo "Running Paper 2 dataset context test..."

# --- Project root (adjust if needed)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# --- Base directories
export DATA_DIR="/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/Data/Data_DiffMod_small"
export RUN_ROOT="$PROJECT_ROOT/runs/paper2_test"

export CKPT_DIR="$RUN_ROOT/checkpoints"
export SAMPLE_DIR="$RUN_ROOT/samples"
export EVAL_DIR="$RUN_ROOT/evaluation"
export LOG_DIR="$RUN_ROOT/logs"
export STATS_LOAD_DIR="$PROJECT_ROOT/repro/assets/stats/statistics_run/stats"

# --- Other required variables
export EXP_DATE=$(date +"%Y%m%d")
export SLURM_CPUS_PER_TASK=0 # Force single-process loading for quick testing

# --- Create folders
mkdir -p "$CKPT_DIR" "$SAMPLE_DIR" "$EVAL_DIR" "$LOG_DIR"

echo "Environment configured:"
echo "DATA_DIR=$DATA_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "STATS_LOAD_DIR=$STATS_LOAD_DIR"

# --- Activate environment (adjust to your setup)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate ceddar

# --- Run dataset context test
cd "$PROJECT_ROOT"
python -m sbgm.data.dataset_context_test --config sbgm/config/paper2/P0.yaml

echo "Done."