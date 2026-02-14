#!/bin/bash
#SBATCH --job-name=container_build
#SBATCH --output=cotainr.out
#SBATCH --error=cotainr.err
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0:45:00



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

# Setup software environment
module use /appl/local/training/modules/AI-20240529/
module load LUMI cotainr

# Point to the container
CONTAINER=/scratch/project_xxxxxxxxx/containers/images/my_torch_container_with_pandas.sif

# Point to the environment yml file
ENV_YML=/scratch/project_xxxxxxxxx/containers/build_files/torch_lumi_w_pandas.yml

srun cotainr build my_torch_container_with_pandas.sif --system=lumi-g --conda-env=/scratch/project_xxxxxxxxx/containers/build_files/torch_lumi_w_pandas.yml #cotainr build $CONTAINER --system=lumi-g --conda-env=$ENV_YML --accept-licenses
