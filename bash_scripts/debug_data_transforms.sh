#!/bin/bash
#SBATCH --job-name=test_transforms
#SBATCH --output=logs/slurm_test_data_%x_%j.log
#SBATCH --error=logs/slurm_test_data_%x_%j.err
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1          # 1 GPU is enough
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00


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

# -----------------------
# User-adjustable
# -----------------------
PROJECT="${ACCOUNT}"
CONTAINER="/scratch/${PROJECT}/containers/images/my_torch_container_with_plotting.sif"
REPO_NAME="SBGM_SD"

# -----------------------
# Modules
# -----------------------
module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
module load lumi-tools || true

# -----------------------
# Paths
# -----------------------
SCRATCH="/scratch/${PROJECT}"
USER_DIR="${SCRATCH}/${USER}"
ROOT_DIR="${USER_DIR}/Code/${REPO_NAME}"
DATA_DIR="${USER_DIR}/Data/Data_DiffMod"
CONFIG_DIR="${ROOT_DIR}/sbgm/config"

CONFIG_FILE="test_data_transforms.yaml"

export ROOT_DIR DATA_DIR CONFIG_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p logs

echo "[INFO] ROOT_DIR = ${ROOT_DIR}"
echo "[INFO] DATA_DIR = ${DATA_DIR}"
echo "[INFO] CONFIG   = ${CONFIG_DIR}/${CONFIG_FILE}"

srun singularity exec "${CONTAINER}" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${PYTHONPATH}'
  python ${ROOT_DIR}/sbgm/data/test_data_transforms.py \
      --cfg ${CONFIG_DIR}/${CONFIG_FILE}
"
