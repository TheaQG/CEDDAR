#!/usr/bin/env bash
#SBATCH --job-name=mv_npz
#SBATCH --account=<set your account here>
#SBATCH --partition=standard
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=era5_logs/mv_npz_%j.out
#SBATCH --error=era5_logs/mv_npz_%j.err

set -euo pipefail

SRC=/scratch/<your_project>/<your_user>/Data/Data_ERA5_tmp/npz
DEST=/scratch/<your_project>/<your_user>/Data/Data_DiffMod/data_ERA5/size_589x789/new_ERA5

mkdir -p "$DEST"
mkdir -p era5_logs

echo "[INFO] SRC:  $SRC"
echo "[INFO] DEST: $DEST"
echo "[INFO] START: $(date)"


RSYNC_OPTS=(
    -avh
    --info=progress2
    --ignore-existing
    --remove-source-files
    --partial
    --partial-dir=.rsync-partial
)

if [[ ! -d "$SRC" ]]; then
    echo "[ERROR] Source directory does not exist: $SRC"
    exit 1
fi

shopt -s nullglob

# Expected structure is roughly:
#   $SRC/<var>/<year>/*.npz
# We sync one <var>/<year>/ subtree at a time.
for var_dir in "$SRC"/*; do
    [[ -d "$var_dir" ]] || continue
    var_name=$(basename "$var_dir")

    echo "[INFO] ------------------------------------------------------------"
    echo "[INFO] Variable: $var_name"

    year_dirs=("$var_dir"/*)

    # If there are no year subdirectories, fall back to syncing the variable dir.
    if [[ ${#year_dirs[@]} -eq 0 ]]; then
        echo "[INFO] No year subdirectories under $var_name; syncing variable directory directly."
        mkdir -p "$DEST/$var_name"
        rsync "${RSYNC_OPTS[@]}" "$var_dir/" "$DEST/$var_name/"
        find "$var_dir" -type d -empty -delete || true
        continue
    fi

    for year_dir in "${year_dirs[@]}"; do
        [[ -d "$year_dir" ]] || continue
        year_name=$(basename "$year_dir")

        echo "[INFO] Syncing $var_name/$year_name ..."
        mkdir -p "$DEST/$var_name/$year_name"

        rsync "${RSYNC_OPTS[@]}" "$year_dir/" "$DEST/$var_name/$year_name/"

        # Clean up empty directories after successful transfer of this subtree.
        find "$year_dir" -type d -empty -delete || true
    done

    # Clean up empty variable dir if all year dirs were emptied.
    find "$var_dir" -type d -empty -delete || true

done

# Final cleanup of any empty directories left behind in source tree.
find "$SRC" -type d -empty -delete || true

echo "[INFO] DONE: $(date)"