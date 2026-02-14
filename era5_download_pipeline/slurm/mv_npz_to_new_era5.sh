#!/usr/bin/env bash
#SBATCH --job-name=mv_npz
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=era5_logs/mv_npz_%j.out
#SBATCH --error=era5_logs/mv_npz_%j.err


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

SRC=${USER_BASE}/Data/Data_ERA5_tmp/npz/
DEST=${USER_BASE}/Data/Data_DiffMod/data_ERA5/size_589x789/new_ERA5/

mkdir -p "$DEST"
rsync -avh --info=progress2 --remove-source-files "$SRC" "$DEST"
find "$SRC" -type d -empty -delete
echo "Done."
