#!/usr/bin/env bash
set -euo pipefail

echo "Running Paper 2 P0 quick eval (CPU) ..."

# --- Project root (this file lives in bash_paper2/local/)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

# --- Config
export CONFIG_PATH="${CONFIG_PATH:-sbgm/config/paper2/P0_eval_quick.yaml}"

# --- Data + run dirs
export DATA_DIR="${DATA_DIR:-$HOME/Data/Data_DiffMod_small}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/paper2_quick_cpu}"

export CKPT_DIR="$RUN_ROOT/checkpoints"
export SAMPLE_DIR="$RUN_ROOT/samples"
export EVAL_DIR="$RUN_ROOT/evaluation"
export LOG_DIR="$RUN_ROOT/logs"
export STATS_LOAD_DIR="${STATS_LOAD_DIR:-$PROJECT_ROOT/repro/assets/stats/statistics_run/stats}"

# --- Repro stamp
export EXP_DATE=$(date +"%Y%m%d")

# --- Force CPU + stable single-process loading
export CUDA_VISIBLE_DEVICES=""
export SLURM_CPUS_PER_TASK=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$CKPT_DIR" "$SAMPLE_DIR" "$EVAL_DIR" "$LOG_DIR"

echo "Environment configured:"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "DATA_DIR=$DATA_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "CKPT_DIR=$CKPT_DIR"
echo "SAMPLE_DIR=$SAMPLE_DIR"
echo "EVAL_DIR=$EVAL_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "STATS_LOAD_DIR=$STATS_LOAD_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$PROJECT_ROOT"

# -------------------------------------------------
# 1) TRAIN (quick)
# -------------------------------------------------
# echo ""
# echo "=== TRAIN ==="

# IMPORTANT: Replace this command with your actual pipeline entrypoint.
# Common patterns:
#   python -m sbgm.cli.main_app --mode train --config_path "$CONFIG_PATH"
#   python main_app.py --train --config_path "$CONFIG_PATH"
#   python -m sbgm.cli.training_main --config_path "$CONFIG_PATH"
#
# Below we try a few plausible ones; keep the one that works and delete the rest.

# set +e

# python -m sbgm.cli.main_app --mode train --config_path "$CONFIG_PATH"
# RC=$?

# set -e

# if [ $RC -ne 0 ]; then
#   echo ""
#   echo "[ERROR] Could not find a working training entrypoint."
#   echo "Fix the TRAIN command in this script to match your repo's CLI."
#   exit 1
# fi

# echo ""
# echo "Training finished."

# -------------------------------------------------
# 2) GENERATE (quick)
# -------------------------------------------------
# echo ""
# echo "=== GENERATE ==="

# set +e

# python -m sbgm.cli.main_app --mode generate --config_path "$CONFIG_PATH"
# RC=$?

# set -e

# if [ $RC -ne 0 ]; then
#   echo ""
#   echo "[ERROR] Could not find a working generation entrypoint."
#   echo "Fix the GENERATE command in this script to match your repo's CLI."
#   exit 1
# fi

# echo ""
# echo "Generation finished."

# echo ""
# echo "Done. Run root: $RUN_ROOT"

# -------------------------------------------------
# 3) EVALUATE (quick)
# -------------------------------------------------

echo ""
echo "=== EVALUATE ==="

set +e

python -m sbgm.cli.main_app --mode evaluate --config_path "$CONFIG_PATH"
RC=$?

set -e

if [ $RC -ne 0 ]; then
  echo ""
  echo "[ERROR] Could not find a working evaluation entrypoint."
  echo "Fix the EVALUATE command in this script to match your repo's CLI."
  exit 1
fi

echo ""
echo "Evaluation finished."

echo ""
echo "Done. Run root: $RUN_ROOT"