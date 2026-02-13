#!/usr/bin/env bash
#SBATCH --job-name=split_zpl
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=era5_logs/split_zpl_%j.out
#SBATCH --error=era5_logs/split_zpl_%j.err
#SBATCH -D /scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/npz

# ---- settings you can tweak ----
SRC="/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/npz/z_pl"
DEST_PARENT="/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/npz"
# Dry run if set to 1
DRY_RUN="${DRY_RUN:-0}"   # override with:  sbatch --export=DRY_RUN=1 split_zpl_levels.sbatch
# --------------------------------

set -euo pipefail
shopt -s nullglob
mkdir -p era5_logs

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "SRC=$SRC"
echo "DEST_PARENT=$DEST_PARENT"
echo "DRY_RUN=$DRY_RUN"
echo

count=0
# Find all NPZ files under .../z_pl/<year>/
while IFS= read -r -d '' f; do
  base=$(basename "$f")                  # z_pl_1000_hPa_589x789_19920101.npz (for example)
  parent_dir=$(dirname "$f")
  year_dir=$(basename "$parent_dir")     # usually YEAR

  # Extract pressure level from filename (2nd underscore field), drop non-digits (e.g. strip "hPa")
  lvl=""
  if [[ "$base" =~ ^z_pl_([0-9]+) ]]; then
    lvl="${BASH_REMATCH[1]}"             # e.g. 1000
  fi
  if [[ -z "$lvl" ]]; then
    echo "WARN: cannot extract level from '$base' -> got '$lvl'. Skipping." >&2
    continue
  fi

  # --- determine year (prefer parent dir; else from trailing YYYYMMDD in filename) ---
  if [[ "$year_dir" =~ ^[0-9]{4}$ ]]; then
    year="$year_dir"
  elif [[ "$base" =~ _([0-9]{8})\.npz$ ]]; then
    year="${BASH_REMATCH[1]:0:4}"
  else
    echo "WARN: cannot determine year for '$base'. Skipping." >&2
    continue
  fi

  target="$DEST_PARENT/z_pl_${lvl}/${year}"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "mkdir -p '$target'"
    echo "mv -n '$f' '$target/'"
  else
    mkdir -p "$target"
    mv -n "$f" "$target/"
  fi
  ((count++)) || true
done < <(find "$SRC" -type f -name 'z_pl_*.npz' -print0)

echo
echo "Done. Processed $count files."

# Optional cleanup of empty dirs under SRC:
# if [[ "$DRY_RUN" != "1" ]]; then
#   find "$SRC" -type d -empty -delete
# fi