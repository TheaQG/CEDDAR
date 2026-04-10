#!/bin/bash
#SBATCH --job-name=zarr_to_zip
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=era5_logs/zarr_to_zip_%j.out
#SBATCH --error=era5_logs/zarr_to_zip_%j.err



# ============================================================
# Convert directory-backed Zarr stores to ZIP-backed Zarr stores
# using convert_zarr_to_zip.py on a compute node.
#
# Usage examples:
#   sbatch bash_paper2/lumi/zarr_to_zip.sh
#
#   sbatch --export=ALL,ZARR_INPUT=/scratch/.../train.zarr \
#          bash_paper2/lumi/zarr_to_zip.sh
#
#   sbatch --export=ALL,ZARR_INPUT=/scratch/.../train.zarr,ZARR_OUTPUT=/scratch/.../train.zarr.zip \
#          bash_paper2/lumi/zarr_to_zip.sh
# ============================================================

# --- Use Singularity container (has zarr installed) ---
module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits


ACCOUNT="project_465002493"
USER_BASE="/scratch/${ACCOUNT}/${USER}"
CONT_BASE="/scratch/${ACCOUNT}/containers"
ROOT_DIR_DEFAULT="${USER_BASE}/Code/CEDDAR"
export ROOT_DIR="${ROOT_DIR:-$ROOT_DIR_DEFAULT}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
CONTAINER="${CONT_BASE}/images/my_torch_container_with_plotting.sif"
OVERLAY="${CONT_BASE}/overlays/my_overlay.img"
PY_SCRIPT="${ROOT_DIR}/sbgm/data/convert_zarr_to_zip.py"

# Default small test dataset path. Override with --export=ALL,ZARR_INPUT=...
ZARR_INPUT_DEFAULT="${USER_BASE}/Data/Data_DiffMod_small/data_ERA5/size_589x789/temp_589x789/zarr_files/valid.zarr"
ZARR_INPUT="${ZARR_INPUT:-$ZARR_INPUT_DEFAULT}"

# If output is not provided, append .zip to input path
ZARR_OUTPUT="${ZARR_OUTPUT:-${ZARR_INPUT}.zip}"

mkdir -p era5_logs

echo "[INFO] ROOT_DIR      : ${ROOT_DIR}"
echo "[INFO] PY_SCRIPT     : ${PY_SCRIPT}"
echo "[INFO] ZARR_INPUT    : ${ZARR_INPUT}"
echo "[INFO] ZARR_OUTPUT   : ${ZARR_OUTPUT}"
echo "[INFO] Host          : $(hostname)"
echo "[INFO] Start         : $(date)"

srun singularity exec $CONTAINER \
    python "$PY_SCRIPT" "$ZARR_INPUT" "$ZARR_OUTPUT"
