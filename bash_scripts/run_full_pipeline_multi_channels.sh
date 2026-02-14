#!/bin/bash
#SBATCH --job-name=allChannels
#SBATCH --output=logs/allChannels_%j.log
#SBATCH --error=logs/allChannels_%j.err
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56 # 7 * 8 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00



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

# === Environment setup ===
module purge 
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# === Point to the container ===
CONTAINER=${USER_BASE}/images/my_torch_container_with_plotting.sif

# === Define paths ===
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR=$SCRATCH/$USER
export ROOT_DIR="$USER_DIR/Code/CEDDAR"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# === Data and output directories === 
export DATA_DIR=$USER_DIR/Data/Data_DiffMod # Data_DiffMod_small # 
export SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
export CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"
export CONFIG_DIR="$ROOT_DIR/sbgm/config"
export STATS_LOAD_DIR="$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/stats" # load from stats run
export EVAL_DIR="$ROOT_DIR/evaluate_sbgm/test"
# === Define date in format DD_MM_YYYY for logging purposes ===
# Export for use in the Python script
export EXP_DATE=$(date +%d_%m_%Y)


# === Optional: create logs directory if it doesn't exist ===
mkdir -p logs

# === Optional: Log the directories to verify ===
echo "[INFO] Date of experiment = $EXP_DATE"
echo "[INFO] ROOT_DIR      = $ROOT_DIR"
echo "[INFO] DATA_DIR      = $DATA_DIR"
echo "[INFO] SAMPLE_DIR    = $SAMPLE_DIR"
echo "[INFO] CKPT_DIR      = $CKPT_DIR"

# === Launch the training ===
echo "[INFO] Launching the full training-generation-evaluation pipeline..."
srun singularity exec $CONTAINER \
    python -m sbgm.cli.main_app --mode full_pipeline --config $CONFIG_DIR/full_run_all_data_config.yaml