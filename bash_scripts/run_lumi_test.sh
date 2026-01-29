#!/bin/bash
#SBATCH --job-name=cfg_full
#SBATCH --output=logs/cfg_full_%j.log
#SBATCH --error=logs/cfg_full_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56 # 7 * 8 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=12:00:00


# === Environment setup ===
module purge 
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# === Point to the container ===
CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif

# === Define paths ===
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR=$SCRATCH/$USER
export ROOT_DIR="$USER_DIR/Code/SBGM_SD"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# === Data and output directories === 
export DATA_DIR=$USER_DIR/Data/Data_DiffMod # Data_DiffMod_small
export SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
export CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"
export STATS_LOAD_DIR="$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/stats" # load from stats run

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
srun singularity exec $CONTAINER \
    python -m sbgm.cli.main_app train 
