#!/bin/bash
#SBATCH --job-name=CEDDAR_reduced
#SBATCH --account=project_465002493
#SBATCH --output=logs/slurm_CEDDAR_reduced_%j.log
#SBATCH --error=logs/slurm_CEDDAR_reduced_%j.err
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:59:00

# Fail fast but set -u only after handling env defaults
set -eo pipefail

echo "----------------------------------------"
echo " Running CEDDAR 02_reduced_run (LUMI) "
echo "----------------------------------------"

# --- Modules ---
module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
module load lumi-tools || true

# --- Container ---
# User can override when submitting: CONTAINER=/path/to/image.sif sbatch repro/02_reduced_run/run_reduced_lumi.sh
CONTAINER="${CONTAINER:-/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif}"

# --- Paths (defaults; can be overridden at submit time) ---
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR="$SCRATCH/$USER"
ROOT_DIR="${ROOT_DIR:-$USER_DIR/Code/CEDDAR}"
DATA_DIR="${DATA_DIR:-$USER_DIR/Data/Data_DiffMod_small}"
CEDDAR_RUNS="${CEDDAR_RUNS:-$USER_DIR/runs/CEDDAR}"

# --- Experiment date for config interpolation ---
EXP_DATE="${EXP_DATE:-$(date -u +%Y%m%dT%H%M%SZ)}"

# --- Output dirs (outside of repo) ---
CKPT_DIR="$CEDDAR_RUNS/repro/02_reduced_run/outputs/checkpoints"
SAMPLE_DIR="$CEDDAR_RUNS/repro/02_reduced_run/outputs/samples"
EVAL_DIR="$SAMPLE_DIR/evaluation"
LOG_DIR="$CEDDAR_RUNS/repro/02_reduced_run/outputs/logs"

# --- Stats JSONs committed in repo ---
STATS_LOAD_DIR="$ROOT_DIR/repro/assets/stats/statistics_run/stats"

# Enable -u after safely handling defaults
set -u

# Create log dir for slurm outputs if it doesn't exist
mkdir -p logs

# Create output dirs
mkdir -p "$CKPT_DIR" "$SAMPLE_DIR" "$EVAL_DIR" "$LOG_DIR"

# --- Export env vars form YAML ---
export ROOT_DIR DATA_DIR CEDDAR_RUNS CKPT_DIR SAMPLE_DIR EVAL_DIR LOG_DIR EXP_DATE STATS_LOAD_DIR
export DEVICE="cuda"

# Ensure repo is in PYTHONPATH (importable across machines)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Threading caps inside container
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- MIOpen workaround: per-job DB ---
MIOPEN_DB_DIR="$SCRATCH/$USER/miopen_db_${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_DB_DIR"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB_DIR/userdb.sql"
export MIOPEN_SYSTEM_DB_PATH="$MIOPEN_DB_DIR/systemdb.sql"

CFG="$ROOT_DIR/repro/02_reduced_run/reduced_run_config.yaml"

echo "[INFO] ROOT_DIR      = $ROOT_DIR"
echo "[INFO] DATA_DIR      = $DATA_DIR"
echo "[INFO] CEDDAR_RUNS   = $CEDDAR_RUNS"
echo "[INFO] CKPT_DIR      = $CKPT_DIR"
echo "[INFO] SAMPLE_DIR    = $SAMPLE_DIR"
echo "[INFO] EVAL_DIR      = $EVAL_DIR"
echo "[INFO] STATS_LOAD_DIR= $STATS_LOAD_DIR"
echo "[INFO] EXP_DATE      = $EXP_DATE"
echo "[INFO] DEVICE        = $DEVICE"
echo "[INFO] CONTAINER     = $CONTAINER"

# Run inside the container
srun singularity exec "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${PYTHONPATH}'
  python -m sbgm.cli.main_app --config '${CFG}'
"

echo "Done. Outputs in: $CEDDAR_RUNS/repro/02_reduced_run/outputs"