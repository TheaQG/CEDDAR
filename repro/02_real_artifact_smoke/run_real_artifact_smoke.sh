#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="$REPO_DIR/repro/02_real_artifact_smoke/real_artifact_smoke_config.yaml"

: "${DATA_DIR:?Set DATA_DIR to the real Data_DiffMod directory}"
: "${PUBLISHED_CHECKPOINT:?Set PUBLISHED_CHECKPOINT to the final Paper I checkpoint}"

DATA_DIR="$(realpath "$DATA_DIR")"
PUBLISHED_CHECKPOINT="$(realpath "$PUBLISHED_CHECKPOINT")"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi

if [[ ! -f "$PUBLISHED_CHECKPOINT" ]]; then
  echo "PUBLISHED_CHECKPOINT does not exist: $PUBLISHED_CHECKPOINT" >&2
  exit 1
fi

export DATA_DIR
export CKPT_DIR="$(dirname "$PUBLISHED_CHECKPOINT")"
export CHECKPOINT_NAME="$(basename "$PUBLISHED_CHECKPOINT")"

export CEDDAR_RUNS="${CEDDAR_RUNS:-$REPO_DIR/../CEDDAR_runs/repro/02_real_artifact_smoke}"
export SAMPLE_DIR="${SAMPLE_DIR:-$CEDDAR_RUNS/samples}"
export EVAL_DIR="${EVAL_DIR:-$CEDDAR_RUNS/evaluation}"
export LOG_DIR="${LOG_DIR:-$CEDDAR_RUNS/logs}"
export STATS_LOAD_DIR="${STATS_LOAD_DIR:-$REPO_DIR/repro/assets/stats/statistics_run/stats}"
export EXP_DATE="${EXP_DATE:-$(date -u +%Y%m%dT%H%M%SZ)}"
export DEVICE="${DEVICE:-cpu}"

mkdir -p "$CEDDAR_RUNS" "$SAMPLE_DIR" "$EVAL_DIR" "$LOG_DIR"

echo "------------------------------------------"
echo " CEDDAR real-artifact smoke test"
echo "------------------------------------------"
echo "Repository        : $REPO_DIR"
echo "Data              : $DATA_DIR"
echo "Checkpoint        : $PUBLISHED_CHECKPOINT"
echo "Samples           : $SAMPLE_DIR"
echo "Evaluation        : $EVAL_DIR"
echo "Logs              : $LOG_DIR"
echo "Device            : $DEVICE"

echo
echo "Git state:"
git -C "$REPO_DIR" rev-parse HEAD
if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
  echo "[WARNING] Repository has uncommitted changes."
else
  echo "Working tree clean."
fi

echo
echo "Starting bounded real-model generation..."

bash "$REPO_DIR/repro/run_model.sh" \
  --config_path "$CFG" \
  --mode generation \
  --device "$DEVICE" \
  "$@"


echo
echo "[PASS] Real-artifact smoke command completed."
echo "Artifacts: $CEDDAR_RUNS"
