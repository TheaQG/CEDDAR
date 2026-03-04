#!/bin/bash
#SBATCH --job-name=move_era5_raw
#SBATCH --output=logs/move_era5_raw_%j.log
#SBATCH --error=logs/move_era5_raw_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#!/usr/bin/env bash
set -euo pipefail

#############################################
# USER SETTINGS
#############################################

# OLD project (source)
SRC_BASE="/scratch/project_465001695/quistgaa/Data/Data_ERA5_tmp/raw"

# NEW project (destination) — 
# Suggestion: keep a raw/ folder so you can re-run preprocessing cleanly.
DST_BASE="/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/raw"

# If 1: do a dry run (no changes). If 0: actually copy+delete source.
DRY_RUN="${DRY_RUN:-1}"

# If 1: don't overwrite existing destination files (safer).
# If 0: allow overwrite.
NO_OVERWRITE="${NO_OVERWRITE:-1}"

# Optional: limit bandwidth (e.g. "200m"). Empty means unlimited.
BW_LIMIT="${BW_LIMIT:-}"

#############################################
# INTERNALS
#############################################
timestamp="$(date +%Y%m%dT%H%M%S)"
LOG="move_era5_raw_${timestamp}.log"

echo "[INFO] SRC_BASE=${SRC_BASE}" | tee -a "$LOG"
echo "[INFO] DST_BASE=${DST_BASE}" | tee -a "$LOG"
echo "[INFO] DRY_RUN=${DRY_RUN}  NO_OVERWRITE=${NO_OVERWRITE}" | tee -a "$LOG"
echo "[INFO] Log: ${LOG}" | tee -a "$LOG"

if [[ ! -d "$SRC_BASE" ]]; then
  echo "[ERROR] Source directory does not exist: $SRC_BASE" | tee -a "$LOG"
  exit 1
fi

mkdir -p "$DST_BASE"

RSYNC_OPTS=(-aH --info=progress2 --human-readable)
# Keep partial files so you can restart after interruption
RSYNC_OPTS+=(--partial --partial-dir=".rsync-partial")

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

if [[ "$NO_OVERWRITE" == "1" ]]; then
  RSYNC_OPTS+=(--ignore-existing)
fi

if [[ -n "$BW_LIMIT" ]]; then
  RSYNC_OPTS+=(--bwlimit="$BW_LIMIT")
fi

# We implement "move" as: rsync copy + verify exit code + remove source files
# --remove-source-files removes files that were successfully transferred
RSYNC_OPTS+=(--remove-source-files)

echo "[INFO] rsync options: ${RSYNC_OPTS[*]}" | tee -a "$LOG"

# Copy each top-level variable directory (cape, msl, pev, ...)
# This avoids one giant rsync operation and gives clearer progress.
mapfile -t TOPDIRS < <(find "$SRC_BASE" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

if [[ "${#TOPDIRS[@]}" -eq 0 ]]; then
  echo "[WARN] No subdirectories found under $SRC_BASE" | tee -a "$LOG"
  exit 0
fi

for d in "${TOPDIRS[@]}"; do
  src="${SRC_BASE}/${d}/"
  dst="${DST_BASE}/${d}/"
  mkdir -p "$dst"

  echo "------------------------------------------------------------" | tee -a "$LOG"
  echo "[INFO] Moving: $src  ->  $dst" | tee -a "$LOG"

  # Main transfer
  rsync "${RSYNC_OPTS[@]}" "$src" "$dst" | tee -a "$LOG"

  # If not dry-run: delete now-empty dirs in source
  if [[ "$DRY_RUN" != "1" ]]; then
    # Remove any empty directories left behind after --remove-source-files
    find "${SRC_BASE}/${d}" -type d -empty -print -delete | tee -a "$LOG" || true
  fi
done

# Final cleanup: remove empty dirs under SRC_BASE (if real run)
if [[ "$DRY_RUN" != "1" ]]; then
  find "$SRC_BASE" -type d -empty -print -delete | tee -a "$LOG" || true
fi

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[INFO] Done." | tee -a "$LOG"
echo "[INFO] If DRY_RUN=1, nothing was changed. Re-run with DRY_RUN=0 to execute." | tee -a "$LOG"