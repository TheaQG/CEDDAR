#!/bin/bash
#SBATCH --job-name=era5_quicklook
#SBATCH --output=logs/era5_quicklook_%j.log
#SBATCH --error=logs/era5_quicklook_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR="$SCRATCH/$USER"
export ROOT_DIR="$USER_DIR/Code/CEDDAR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export RAW_BASE="/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/raw"
export OUT_DIR="/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/quicklooks_$(date +%Y%m%dT%H%M%S)"
SCRIPT_DIR="/scratch/project_465002493/quistgaa/Code/CEDDAR/era5_download_pipeline/pipeline"

mkdir -p logs
mkdir -p "$OUT_DIR"

echo "starting run"
echo "Container: $CONTAINER"
echo "ROOT_DIR: $ROOT_DIR"
echo "RAW_BASE: $RAW_BASE"
echo "OUT_DIR:  $OUT_DIR"

srun singularity exec "$CONTAINER" \
  python "$SCRIPT_DIR/quicklook_era5_raw.py" \
    --raw_base "$RAW_BASE" \
    --out_dir "$OUT_DIR"

echo "[DONE] Outputs in: $OUT_DIR"