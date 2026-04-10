#!/bin/bash
#SBATCH --job-name=paper2_spatial_enc_test
#SBATCH --output=logs/slurm_%x_%j.log
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -eo pipefail

# -----------------------------------------------------------------------------
# Paper2: Spatial encoder test on LUMI (inside container)
#
# Example submit:
# sbatch --account=project_465002493 \
#   --export=ALL,ROOT_DIR=/scratch/project_465002493/$USER/Code/CEDDAR,DATA_DIR=/scratch/project_465002493/$USER/Data/Data_DiffMod_small \
#   bash_paper2/lumi/run_spatial_encoder_test.sh
#
# Optional overrides:
#   CONFIG_REL=sbgm/config/paper2/P0.yaml
#   RUN_NAME=paper2_test
#   CONTAINER=.../containers/ceddar.sif
# -----------------------------------------------------------------------------

# --- User/site configuration (override when submitting) ---
ACCOUNT="${SLURM_JOB_ACCOUNT:-${ACCOUNT:-project_465002493}}"
USER_BASE="/scratch/${ACCOUNT}/${USER}"
CONT_BASE="/scratch/${ACCOUNT}/containers/images"
ROOT_DIR="${ROOT_DIR:-${USER_BASE}/Code/CEDDAR}"
CEDDAR_RUNS="${CEDDAR_RUNS:-${USER_BASE}/runs/CEDDAR}"
DATA_BASE="${DATA_BASE:-${USER_BASE}/Data}"
DATA_DIR="${DATA_DIR:-${DATA_BASE}/Data_DiffMod_small}"

# Container path (override at submit-time)
CONTAINER="${CONTAINER:-${CONT_BASE}/my_torch_container_with_plotting.sif}"

# Config (relative to ROOT_DIR by default)
CONFIG_REL="${CONFIG_REL:-sbgm/config/paper2/P0.yaml}"

# Run naming / paths
RUN_NAME="${RUN_NAME:-paper2_test}"
RUN_ROOT="${RUN_ROOT:-${CEDDAR_RUNS}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
STATS_LOAD_DIR="${STATS_LOAD_DIR:-${ROOT_DIR}/repro/assets/stats/statistics_run/stats}"

export ACCOUNT USER_BASE ROOT_DIR CEDDAR_RUNS DATA_BASE DATA_DIR CONTAINER
export RUN_ROOT LOG_DIR STATS_LOAD_DIR

# Now it’s safe to enable -u
set -u

# --- Modules (match your other LUMI scripts) ---
module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
module load lumi-tools || true

# --- Create logs dirs ---
mkdir -p logs
mkdir -p "${LOG_DIR}"

# --- Threading caps inside container (keep modest for Zarr IO) ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# --- MIOpen workaround: per-job DB (safe even if unused) ---
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
MIOPEN_DB_DIR="${SCRATCH}/${USER}/miopen_db_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_DB_DIR}"
export MIOPEN_USER_DB_PATH="${MIOPEN_DB_DIR}/userdb.sql"
export MIOPEN_SYSTEM_DB_PATH="${MIOPEN_DB_DIR}/systemdb.sql"

# --- Derived paths ---
CFG="${ROOT_DIR}/${CONFIG_REL}"

# --- Export PYTHONPATH ---
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "[INFO] ROOT_DIR        = ${ROOT_DIR}"
echo "[INFO] DATA_DIR        = ${DATA_DIR}"
echo "[INFO] RUN_ROOT        = ${RUN_ROOT}"
echo "[INFO] LOG_DIR         = ${LOG_DIR}"
echo "[INFO] STATS_LOAD_DIR  = ${STATS_LOAD_DIR}"
echo "[INFO] CONTAINER       = ${CONTAINER}"
echo "[INFO] CONFIG          = ${CFG}"

# --- Make sure the test can discover config via env if needed ---
export SPATIAL_ENCODER_CONFIG="${CFG}"

echo "[INFO] Running spatial encoder test (unit + integration + smoke train) inside container..."

srun singularity exec "${CONTAINER}" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${PYTHONPATH}'
  export DATA_DIR='${DATA_DIR}'
  export RUN_ROOT='${RUN_ROOT}'
  export LOG_DIR='${LOG_DIR}'
  export STATS_LOAD_DIR='${STATS_LOAD_DIR}'
  export SPATIAL_ENCODER_CONFIG='${SPATIAL_ENCODER_CONFIG}'
  cd '${ROOT_DIR}'

  python -m sbgm.module_tests.spatial_encoder_test \
    --config '${CFG}' \
    --device cuda \
    --smoke-train \
    --epochs 2 \
    --max-train-batches 5 \
    --max-val-batches 2
"

echo "[INFO] Done."