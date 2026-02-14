#!/bin/bash
###############################################################################
# run_optuna_sweep_lumi.sh – one-trial-per-GPU Optuna sweep on LUMI
###############################################################################

#SBATCH --job-name=sbgm_optuna_sweep
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard-g
#SBATCH --array=0-2%2           # change range/concurrency as needed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/optuna_%A_%a.out
#SBATCH --error=logs/optuna_%A_%a.err


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

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# ── Paths ────────────────────────────────────────────────────────────────────
CONTAINER=${USER_BASE}/images/my_torch_container_with_plotting.sif
OVERLAY=${USER_BASE}/overlays/hpo_overlay.img

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR=$SCRATCH/$USER
HOST_CODE=$USER_DIR/Code/CEDDAR

OPTUNA_DB_DIR=$SCRATCH/optuna_db
STUDY_NAME=sbgm_optuna_v1
STORAGE="sqlite:///$OPTUNA_DB_DIR/$STUDY_NAME.db"
mkdir -p "$OPTUNA_DB_DIR" logs

# Data & output dirs for container
export DATA_DIR=$USER_DIR/Data/Data_DiffMod
export SAMPLE_DIR="$HOST_CODE/models_and_samples/generated_samples"
export CKPT_DIR="$HOST_CODE/models_and_samples/trained_models"

echo "[INFO] Job $SLURM_JOB_ID | Task $SLURM_ARRAY_TASK_ID"
echo "[INFO] Optuna DB : $STORAGE"

# ── Launch one trial ─────────────────────────────────────────────────────────
srun singularity exec \
     --cleanenv \
     --overlay "$OVERLAY":ro \
     --bind "$HOST_CODE:/workspace" \
     --env STORAGE="$STORAGE" \
     "$CONTAINER" \
     bash -eu <<'INNER'
# ---------------------- inside container + overlay --------------------------
# 1) Point Python to overlay’s Optuna (no micromamba run → no lock)
HPO_SITE="\$HOME/micromamba/envs/hpo/lib/python3.10/site-packages"
export PYTHONPATH="/workspace:\$HPO_SITE:\${PYTHONPATH:-}"
export MAMBA_NO_LOCK=1            # just in case anything touches micromamba

# 2) Run one Optuna trial (Torch comes from container, Optuna from overlay)
python -m sbgm.sweep.run_optuna \
       --n-trials 1 \
       --study-name sbgm_optuna_v1 \
       --storage    "\$STORAGE" \
       --enable-medium \
       --epochs 3
INNER
