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
