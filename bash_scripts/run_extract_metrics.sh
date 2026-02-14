#!/bin/bash
#SBATCH --job-name=extract_metrics
#SBATCH --output=logs/slurm_extract_%x_%j.log
#SBATCH --error=logs/slurm_extract_%x_%j.err
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00


set -eo pipefail

# --- User/site configuration (override when submitting) ---
# Example:
#   sbatch --account=project_xxxxx \
#     --export=ALL,ROOT_DIR=/scratch/project_xxxxx/$USER/Code/CEDDAR,DATA_DIR=/scratch/project_xxxxx/$USER/Data/Data_DiffMod_small \
#     <script>.sh
ACCOUNT="${SLURM_JOB_ACCOUNT:-${ACCOUNT:-project_xxxxx}}"
USER_BASE="/scratch/${ACCOUNT}/${USER}"
ROOT_DIR="${ROOT_DIR:-${USER_BASE}/Code/CEDDAR}"
CEDDAR_RUNS="${CEDDAR_RUNS:-${USER_BASE}/runs/CEDDAR}"
DATA_BASE="${DATA_BASE:-${USER_BASE}/Data}"
DATA_DIR="${DATA_DIR:-${DATA_BASE}/Data_DiffMod_small}"
CONTAINER="${CONTAINER:-${USER_BASE}/containers/ceddar.sif}"
export ACCOUNT USER_BASE ROOT_DIR CEDDAR_RUNS DATA_BASE DATA_DIR CONTAINER

set -eo pipefail

module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
module load lumi-tools || true

CONTAINER=${USER_BASE}/images/my_torch_container_with_plotting.sif

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR="$SCRATCH/$USER"
ROOT_DIR="$USER_DIR/Code/CEDDAR"

EVAL_ROOT="$ROOT_DIR/models_and_samples/generated_samples/evaluation"
OUT_DIR="$ROOT_DIR/models_and_samples/generated_samples/evaluation/metrics_tables"
mkdir -p logs "$OUT_DIR"

# Optional: set this to a model folder name to extract just one
MODEL_NAME="${1:-}"
YEAR="${2:-2017}"

# where you placed the script in the repo
PY_SCRIPT="$ROOT_DIR/sbgm/evaluate/tools/extract_eval_metrics.py"

echo "[INFO] ROOT_DIR  = $ROOT_DIR"
echo "[INFO] EVAL_ROOT = $EVAL_ROOT"
echo "[INFO] OUT_DIR   = $OUT_DIR"
echo "[INFO] MODEL     = ${MODEL_NAME:-<scan all>}"
echo "[INFO] YEAR      = ${YEAR}"

srun singularity exec "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${ROOT_DIR}:${PYTHONPATH:-}'
  python '$PY_SCRIPT' \
    --eval_root '$EVAL_ROOT' \
    ${MODEL_NAME:+--model \"$MODEL_NAME\"} \
    --out_csv '$OUT_DIR/edm_ablation_metrics.csv' \
    --ignore baselines ablations1 ablations2 ablations3 \
    --year ${YEAR} \
    --strict 
"
