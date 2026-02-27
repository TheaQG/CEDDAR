#!/bin/bash
#SBATCH --job-name=stats_data_full
#SBATCH --output=logs/stats_data_full%j.log
#SBATCH --error=logs/stats_data_full%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00


# === Environment setup ===
module purge 
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# === Point to the container ===
CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif

# === Define paths ===
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR=$SCRATCH/$USER
export ROOT_DIR="$USER_DIR/Code/CEDDAR"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# === Data and config directories === 
export DATA_DIR=$USER_DIR/Data/Data_DiffMod # Data_DiffMod_small # 
export STATS_SAVE_DIR="$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/stats"
export DATA_FIGS_DIR="$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/figs"
export CONFIG_DIR="$ROOT_DIR/data_analysis_pipeline/configs/stats_config.yaml"

# === Optional: create logs directory if it doesn't exist ===
mkdir -p logs
echo "starting run"
echo "Container: $CONTAINER"
echo "Root Directory: $ROOT_DIR"
echo "Data Directory: $DATA_DIR"
echo "Config Directory: $CONFIG_DIR"
# === Launch the training ===
srun singularity exec $CONTAINER \
    python -m data_analysis_pipeline.cli.main_data_app --mode "run_statistics" --config $CONFIG_DIR

echo "finished run"
