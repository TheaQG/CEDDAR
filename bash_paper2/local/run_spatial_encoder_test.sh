#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------
# Paper 2 – Spatial Encoder Test Runner
# -------------------------------------------------

echo "Running Paper 2 spatial encoder test..."

# --- Project root (adjust if needed)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

# --- Base directories
export DATA_DIR="$HOME/Data/Data_DiffMod_small"
export RUN_ROOT="$PROJECT_ROOT/runs/paper2_test"

export CKPT_DIR="$RUN_ROOT/checkpoints"
export SAMPLE_DIR="$RUN_ROOT/samples"
export EVAL_DIR="$RUN_ROOT/evaluation"
export LOG_DIR="$RUN_ROOT/logs"
export STATS_LOAD_DIR="$PROJECT_ROOT/repro/assets/stats/statistics_run/stats"

# Provide default config for the python test (can still override via --config)
export SPATIAL_ENCODER_CONFIG="${SPATIAL_ENCODER_CONFIG:-sbgm/config/paper2/P0.yaml}"

# --- Other required variables
export EXP_DATE=$(date +"%Y%m%d")
export SLURM_CPUS_PER_TASK=0 # Force single-process loading for quick testing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# --- Create folders
mkdir -p "$CKPT_DIR" "$SAMPLE_DIR" "$EVAL_DIR" "$LOG_DIR"

echo "Environment configured:"
echo "DATA_DIR=$DATA_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "STATS_LOAD_DIR=$STATS_LOAD_DIR"
echo "SPATIAL_ENCODER_CONFIG=$SPATIAL_ENCODER_CONFIG"

# --- Run spatial encoder integration test
cd "$PROJECT_ROOT"
python -m sbgm.module_tests.spatial_encoder_test --smoke-train --config "$SPATIAL_ENCODER_CONFIG"

echo "Done."
