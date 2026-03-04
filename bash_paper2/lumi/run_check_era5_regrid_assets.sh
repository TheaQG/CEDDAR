#!/bin/bash
#SBATCH --job-name=check_regrid_assets
#SBATCH --output=logs/check_regrid_assets_%j.log
#SBATCH --error=logs/check_regrid_assets_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

set -euo pipefail

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_pandas.sif

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR="$SCRATCH/$USER"
export ROOT_DIR="$USER_DIR/Code/CEDDAR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---- Set these to your actual paths (from your config) ----
GRID_FILE="/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/grid/mygrid_danra_small"
WEIGHTS_FILE="/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/weights/ERA5_to_DANRA_bil_weights_new.nc"

# Pick ONE raw ERA5 file (or a dir/glob); script will pick first .nc if you pass a directory
ERA5_SAMPLE="/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/raw/tp"   # or "/path/to/file.nc" or "/path/**/*.nc"

ZARR_SAMPLE="/scratch/project_465002493/quistgaa/Data/Data_DiffMod/data_ERA5/size_589x789/prcp_589x789/zarr_files/train.zarr"
ZARR_HINT="tp_589x789"   # or prcp key if that’s what is inside the zarr
ZARR_N=3

# Optional: expected DANRA bbox to test coverage (replace with your real bounds if you want)
# BBOX="54.0,59.0,7.0,16.0"   # latmin,latmax,lonmin,lonmax
BBOX=""

mkdir -p logs

echo "[INFO] Container: $CONTAINER"
echo "[INFO] GRID_FILE: $GRID_FILE"
echo "[INFO] WEIGHTS_FILE: $WEIGHTS_FILE"
echo "[INFO] ERA5_SAMPLE: $ERA5_SAMPLE"
echo "[INFO] ZARR_SAMPLE: $ZARR_SAMPLE"
ARGS=( "$ROOT_DIR/era5_download_pipeline/pipeline/check_era5_regrid_assets.py"
       --grid_file "$GRID_FILE"
       --weights_file "$WEIGHTS_FILE"
       --era5_sample "$ERA5_SAMPLE"
)

ARGS+=( --zarr_sample "$ZARR_SAMPLE" --zarr_hint "$ZARR_HINT" --zarr_n "$ZARR_N" )

if [[ -n "$BBOX" ]]; then
  ARGS+=( --bbox "$BBOX" )
fi

srun singularity exec "$CONTAINER" python "${ARGS[@]}"