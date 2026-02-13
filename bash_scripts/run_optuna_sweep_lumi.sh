#!/bin/bash
###############################################################################
# run_optuna_sweep_lumi.sh – one-trial-per-GPU Optuna sweep on LUMI
###############################################################################

#SBATCH --job-name=sbgm_optuna_sweep
#SBATCH --account=project_465002493
#SBATCH --partition=standard-g
#SBATCH --array=0-2%2           # change range/concurrency as needed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/optuna_%A_%a.out
#SBATCH --error=logs/optuna_%A_%a.err

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# ── Paths ────────────────────────────────────────────────────────────────────
CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif
OVERLAY=/scratch/project_465002493/containers/overlays/hpo_overlay.img

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