#!/usr/bin/env bash
#SBATCH --job-name=flatten_era5
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=era5_logs/flatten_%j.out
#SBATCH --error=era5_logs/flatten_%j.err



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

BASE="${USER_BASE}/Data/Data_DiffMod/data_ERA5/size_589x789/new_ERA5"
OVERWRITE="${OVERWRITE:-0}" # To overwrite existing files, set to 1

shopt -s nullglob # Enable nullglob (glob patterns that match no files will expand to a null string)

for VARDIR in "$BASE"/*_589x789; do # Iterate over each variable directory
    [[ -d "$VARDIR" ]] || continue # Skip non-directories
    echo "Flattening $VARDIR" 
    mkdir -p "$VARDIR/all" # Create the 'all' directory if it doesn't exist


    for YDIR in "$VARDIR"/[12][0-9][0-9][0-9]; do # Iterate over each year directory
        echo "Processing $YDIR"
        [[ -d "$YDIR" ]] || continue # Skip non-directories
        if [ "$OVERWRITE" -eq 1 ]; then # If overwrite is enabled
            echo "Overwriting files in $YDIR"
            mv "$YDIR"/*.npz "$VARDIR/all"/ 2>/dev/null || true # Move files to 'all' directory (2> /dev/null means suppress errors)
        else
            echo "Copying files from $YDIR"
            mv -n "$YDIR"/*.npz "$VARDIR/all"/ 2>/dev/null || true # Move files to 'all' directory (2> /dev/null means suppress errors, and -n means do not overwrite existing files)
        fi
    done

    find "$VARDIR" -mindepth 1 -maxdepth 1 -type d -regex '.*/[12][0-9]{3}' -empty -delete
done

echo "Flattening completed"
