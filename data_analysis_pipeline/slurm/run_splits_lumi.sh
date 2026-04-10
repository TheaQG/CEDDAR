#!/bin/bash
#SBATCH --job-name=split_data
#SBATCH --output=logs/split_data%j.log
#SBATCH --error=logs/split_data%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00



set -eo pipefail

# --- User/site configuration (override when submitting) ---
# Example:
#   sbatch --account=project_xxxxx \
#     --export=ALL,ROOT_DIR=/scratch/project_xxxxx/$USER/Code/CEDDAR,DATA_DIR=/scratch/project_xxxxx/$USER/Data/Data_DiffMod_small \
#     <script>.sh
ACCOUNT="${SLURM_JOB_ACCOUNT:-${ACCOUNT:-project_465002493}}"
USER_BASE="/scratch/${ACCOUNT}/${USER}"
CONT_BASE="/scratch/${ACCOUNT}/containers/images"
ROOT_DIR="${ROOT_DIR:-${USER_BASE}/Code/CEDDAR}"
CEDDAR_RUNS="${CEDDAR_RUNS:-${USER_BASE}/runs/CEDDAR}"
DATA_BASE="${DATA_BASE:-${USER_BASE}/Data}"
DATA_DIR="${DATA_DIR:-${DATA_BASE}/Data_DiffMod_small}"
export ACCOUNT USER_BASE ROOT_DIR CEDDAR_RUNS DATA_BASE DATA_DIR CONT_BASE

# === Environment setup ===
module purge 
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# === Point to the container ===
CONTAINER=${CONT_BASE}/my_torch_container_with_plotting.sif

# === Define paths ===
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR=$SCRATCH/$USER
export ROOT_DIR="$USER_DIR/Code/CEDDAR"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# === Data and config directories === 
export DATA_DIR=$USER_DIR/Data/Data_DiffMod # Data_DiffMod_small
export CONFIG_DIR="$ROOT_DIR/data_analysis_pipeline/configs/split_config.yaml"

# === Optional: create logs directory if it doesn't exist ===
mkdir -p logs
echo "starting run"

echo "Container: $CONTAINER"
echo "Root Directory: $ROOT_DIR"
echo "Data Directory: $DATA_DIR"
echo "Config Directory: $CONFIG_DIR"
# === Launch the training ===
srun singularity exec $CONTAINER \
    python -m data_analysis_pipeline.cli.main_data_app --mode "create_splits" --config $CONFIG_DIR

echo "finished run"