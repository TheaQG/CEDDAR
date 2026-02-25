#!/bin/bash
#SBATCH --job-name=inode_cleanup
#SBATCH --output=logs/inode_cleanup_%j.log
#SBATCH --error=logs/inode_cleanup_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

echo "=== Job start: $(date) ==="
echo "Host: $(hostname)"
echo "PWD : $(pwd)"

########################################
# Configurable section
########################################

# DRY_RUN=1: print actions only, do not modify files. DRY_RUN=0: execute.
DRY_RUN=${DRY_RUN:-1}
# Number of CPUs for compression (zstd)
CPUS=${SLURM_CPUS_PER_TASK:-4}
# Root data directory
ROOT=/scratch/project_465002493/quistgaa/Data/Data_DiffMod

# Safety: only archive+remove if folder contains at least this many files
# (default 1 = skip truly empty dirs, but still archive small ones)
MIN_FILES_TO_ARCHIVE=${MIN_FILES_TO_ARCHIVE:-1}

# List of target directories to process (edit as needed)
TARGET_DIRS=(
  "$ROOT/data_ERA5/size_589x789/prcp_589x789"
  "$ROOT/data_ERA5/size_589x789/temp_589x789"
  "$ROOT/data_ERA5/size_589x789/msl_589x789"
  "$ROOT/data_ERA5/size_589x789/z_pl_500_589x789"
  "$ROOT/data_ERA5/size_589x789/cape_589x789"
  "$ROOT/data_ERA5/size_589x789/ewvf_589x789"
  "$ROOT/data_ERA5/size_589x789/nwvf_589x789"
  "$ROOT/data_DANRA/size_589x789/prcp_589x789"
  "$ROOT/data_DANRA/size_589x789/temp_589x789"
)

# Subdirectories to archive+remove (if they exist and are non-empty)
# Requested: include raw `all/` and `all_filtered/`.
ARCHIVE_SUBDIRS=(train valid test all all_filtered)

# Subdirectories to remove (if they exist)
REMOVE_SUBDIRS=(all_filtered_small all_filtered_small_zarr)

# Subdirectories to remove if they are empty (common legacy folders)
REMOVE_EMPTY_SUBDIRS=(eval train2 valid2 test2)

# Remove legacy zarr stores inside zarr_files/ (if CLEAN_LEGACY_ZARR=1)
CLEAN_LEGACY_ZARR=1
# Remove any zarr stores in zarr_files/ that are NOT in ZARR_PROTECTED (if CLEAN_REDUNDANT_ZARR=1)
CLEAN_REDUNDANT_ZARR=1
# Legacy zarr store names (relative to zarr_files/)
ZARR_LEGACY_NAMES=("eval.zarr" "valid2.zarr" "train2.zarr" "test2.zarr")

# Never delete these zarr stores (critical)
ZARR_PROTECTED=("train.zarr" "valid.zarr" "test.zarr")

########################################
# Utility functions
########################################

count_files() {
  # count_files <dir>
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return
  fi
  find "$dir" -type f 2>/dev/null | wc -l
}

is_empty_dir() {
  # is_empty_dir <dir> : returns 0 if dir exists and has no entries
  local dir="$1"
  [ -d "$dir" ] || return 1
  [ -z "$(ls -A "$dir" 2>/dev/null)" ]
}

run_cmd() {
  # Usage: run_cmd <command...>
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN: $*"
  else
    "$@"
  fi
}

print_dir_summary() {
  local dir="$1"
  echo "----- Directory summary for: $dir -----"
  if [ ! -d "$dir" ]; then
    echo "  [MISSING]"
    return
  fi
  echo "  [ls -lah $dir]"
  ls -lah "$dir"
  echo "  [find $dir -maxdepth 2 -mindepth 1 -type d | sort]"
  find "$dir" -maxdepth 2 -mindepth 1 -type d | sort
  echo "  [File counts per direct child dir:]"
  for sub in "$dir"/*; do
    if [ -d "$sub" ]; then
      echo -n "    files($(basename "$sub"))="
      find "$sub" -type f 2>/dev/null | wc -l
    fi
  done
  if [ -d "$dir/zarr_files" ]; then
    echo "  [ls -lah $dir/zarr_files]"
    ls -lah "$dir/zarr_files"
  fi
  echo "---------------------------------------"
}

archive_and_remove_subdir() {
  # archive_and_remove_subdir <base_dir> <subdir_name>
  local base_dir="$1"
  local subdir_name="$2"
  local archive_dir="$base_dir/archives"
  local subdir_path="$base_dir/$subdir_name"
  local out="$archive_dir/${subdir_name}.tar.zst"

  if [ ! -d "$subdir_path" ]; then
    echo "  [SKIP] $subdir_path does not exist."
    return
  fi

  local nfiles
  nfiles=$(count_files "$subdir_path")
  if [ "$nfiles" -lt "$MIN_FILES_TO_ARCHIVE" ]; then
    echo "  [SKIP] $subdir_path has $nfiles files (< MIN_FILES_TO_ARCHIVE=$MIN_FILES_TO_ARCHIVE)."
    return
  fi

  run_cmd mkdir -p "$archive_dir"

  if [ -f "$out" ]; then
    echo "  Archive already exists: $out (skipping tar)"
  else
    echo "  Archiving $subdir_name ($nfiles files) -> $out"
    run_cmd tar -I "zstd -12 -T${CPUS}" -cf "$out" -C "$base_dir" "$subdir_name"
    if [ "$DRY_RUN" -eq 0 ]; then
      echo "  Archive created: $(ls -lh "$out")"
      echo "  Sanity check (first entries):"
      tar -tf "$out" | head -n 5
    fi
  fi

  # Remove the subdir after successful archive (not in DRY_RUN)
  if [ "$DRY_RUN" -eq 0 ] && [ -f "$out" ]; then
    echo "  Removing $subdir_path"
    run_cmd rm -rf "$subdir_path"
  elif [ "$DRY_RUN" -eq 1 ]; then
    echo "  DRY-RUN: Would remove $subdir_path"
  fi
}

cleanup_legacy_zarr() {
  # cleanup_legacy_zarr <base_dir>
  local base_dir="$1"
  local zarr_dir="$base_dir/zarr_files"
  if [ ! -d "$zarr_dir" ]; then
    echo "  [SKIP] $zarr_dir does not exist."
    return
  fi

  for name in "${ZARR_LEGACY_NAMES[@]}"; do
    local target="$zarr_dir/$name"
    # Never delete protected zarr stores
    for prot in "${ZARR_PROTECTED[@]}"; do
      if [ "$name" = "$prot" ]; then
        continue 2
      fi
    done
    if [ -e "$target" ]; then
      echo "  Removing legacy zarr: $target"
      run_cmd rm -rf "$target"
    fi
  done

  # Remove any *_eval.zarr in zarr_files, but protect train/valid/test
  find "$zarr_dir" -maxdepth 1 -type d -name '*_eval.zarr' | while read -r legacy; do
    local bn
    bn="$(basename "$legacy")"
    for prot in "${ZARR_PROTECTED[@]}"; do
      if [ "$bn" = "$prot" ]; then
        continue 2
      fi
    done
    echo "  Removing legacy zarr: $legacy"
    run_cmd rm -rf "$legacy"
  done
}

cleanup_redundant_zarr() {
  # cleanup_redundant_zarr <base_dir>
  # Removes all *.zarr directories in zarr_files/ except those listed in ZARR_PROTECTED.
  local base_dir="$1"
  local zarr_dir="$base_dir/zarr_files"
  if [ ! -d "$zarr_dir" ]; then
    echo "  [SKIP] $zarr_dir does not exist."
    return
  fi

  # Delete any *.zarr store not in the protected list
  find "$zarr_dir" -maxdepth 1 -type d -name '*.zarr' 2>/dev/null | while read -r z; do
    local bn keep
    bn="$(basename "$z")"
    keep=0
    for prot in "${ZARR_PROTECTED[@]}"; do
      if [ "$bn" = "$prot" ]; then
        keep=1
        break
      fi
    done
    if [ "$keep" -eq 0 ]; then
      echo "  Removing redundant zarr store: $z"
      run_cmd rm -rf "$z"
    fi
  done
}

remove_subdirs() {
  # remove_subdirs <base_dir>
  local base_dir="$1"
  for subdir in "${REMOVE_SUBDIRS[@]}"; do
    local subdir_path="$base_dir/$subdir"
    if [ -d "$subdir_path" ]; then
      echo "  Removing $subdir_path"
      run_cmd rm -rf "$subdir_path"
    fi
  done
}

remove_empty_subdirs() {
  # remove_empty_subdirs <base_dir>
  local base_dir="$1"
  for subdir in "${REMOVE_EMPTY_SUBDIRS[@]}"; do
    local subdir_path="$base_dir/$subdir"
    if [ -d "$subdir_path" ] && is_empty_dir "$subdir_path"; then
      echo "  Removing empty legacy dir: $subdir_path"
      run_cmd rm -rf "$subdir_path"
    fi
  done
}

process_target() {
  local base_dir="$1"
  echo ""
  echo "========== Processing: $base_dir =========="
  print_dir_summary "$base_dir"
  echo ""
  echo "---- PLAN for $base_dir ----"

  # Plan: which ARCHIVE_SUBDIRS exist and are non-empty
  local to_archive=()
  local to_archive_empty=()
  for subdir in "${ARCHIVE_SUBDIRS[@]}"; do
    local p="$base_dir/$subdir"
    if [ -d "$p" ]; then
      local nf
      nf=$(count_files "$p")
      if [ "$nf" -ge "$MIN_FILES_TO_ARCHIVE" ]; then
        to_archive+=("$subdir($nf files)")
      else
        to_archive_empty+=("$subdir($nf files)")
      fi
    fi
  done
  if [ "${#to_archive[@]}" -gt 0 ]; then
    echo "  Will archive+remove: ${to_archive[*]}"
  else
    echo "  No non-empty archive subdirs present."
  fi
  if [ "${#to_archive_empty[@]}" -gt 0 ]; then
    echo "  Archive candidates present but empty/small (will skip): ${to_archive_empty[*]}"
  fi

  # Plan: which REMOVE_SUBDIRS exist
  local to_remove=()
  for subdir in "${REMOVE_SUBDIRS[@]}"; do
    if [ -d "$base_dir/$subdir" ]; then
      to_remove+=("$subdir")
    fi
  done
  if [ "${#to_remove[@]}" -gt 0 ]; then
    echo "  Will remove: ${to_remove[*]}"
  else
    echo "  No remove subdirs present."
  fi

  # Plan: which empty legacy dirs exist
  local to_remove_empty=()
  for subdir in "${REMOVE_EMPTY_SUBDIRS[@]}"; do
    if [ -d "$base_dir/$subdir" ] && is_empty_dir "$base_dir/$subdir"; then
      to_remove_empty+=("$subdir")
    fi
  done
  if [ "${#to_remove_empty[@]}" -gt 0 ]; then
    echo "  Will remove empty legacy dirs: ${to_remove_empty[*]}"
  else
    echo "  No empty legacy dirs to remove."
  fi

  # Plan: which legacy zarr items will be removed
  if [ "$CLEAN_LEGACY_ZARR" -eq 1 ] && [ -d "$base_dir/zarr_files" ]; then
    local found_legacy=()
    for name in "${ZARR_LEGACY_NAMES[@]}"; do
      local target="$base_dir/zarr_files/$name"
      local skip=0
      for prot in "${ZARR_PROTECTED[@]}"; do
        if [ "$name" = "$prot" ]; then skip=1; fi
      done
      if [ "$skip" -eq 0 ] && [ -e "$target" ]; then
        found_legacy+=("$name")
      fi
    done
    # *_eval.zarr
    local eval_legacy=()
    while IFS= read -r p; do
      local bn skip
      bn="$(basename "$p")"
      skip=0
      for prot in "${ZARR_PROTECTED[@]}"; do
        if [ "$bn" = "$prot" ]; then skip=1; fi
      done
      if [ "$skip" -eq 0 ]; then
        eval_legacy+=("$bn")
      fi
    done < <(find "$base_dir/zarr_files" -maxdepth 1 -type d -name '*_eval.zarr' 2>/dev/null)
    if [ "${#found_legacy[@]}" -gt 0 ] || [ "${#eval_legacy[@]}" -gt 0 ]; then
      echo "  Will remove legacy zarr stores: ${found_legacy[*]} ${eval_legacy[*]}"
    else
      echo "  No legacy zarr stores to remove."
    fi
  fi

  # Plan: which redundant zarr stores will be removed (keep only protected)
  if [ "$CLEAN_REDUNDANT_ZARR" -eq 1 ] && [ -d "$base_dir/zarr_files" ]; then
    local redundant=()
    while IFS= read -r p; do
      local bn keep
      bn="$(basename "$p")"
      keep=0
      for prot in "${ZARR_PROTECTED[@]}"; do
        if [ "$bn" = "$prot" ]; then
          keep=1
          break
        fi
      done
      if [ "$keep" -eq 0 ]; then
        redundant+=("$bn")
      fi
    done < <(find "$base_dir/zarr_files" -maxdepth 1 -type d -name '*.zarr' 2>/dev/null)

    if [ "${#redundant[@]}" -gt 0 ]; then
      echo "  Will remove redundant zarr stores (keep: ${ZARR_PROTECTED[*]}): ${redundant[*]}"
    else
      echo "  No redundant zarr stores to remove (only protected present)."
    fi
  fi

  echo "-----------------------------"

  # Actions
  for subdir in "${ARCHIVE_SUBDIRS[@]}"; do
    archive_and_remove_subdir "$base_dir" "$subdir"
  done
  remove_subdirs "$base_dir"
  remove_empty_subdirs "$base_dir"
  if [ "$CLEAN_LEGACY_ZARR" -eq 1 ]; then
    cleanup_legacy_zarr "$base_dir"
  fi
  if [ "$CLEAN_REDUNDANT_ZARR" -eq 1 ]; then
    cleanup_redundant_zarr "$base_dir"
  fi

  print_dir_summary "$base_dir"
  echo "========== Done: $base_dir =========="
}

########################################
# Main script logic
########################################

echo "=== BEFORE: df -i /scratch/project_465002493 ==="
df -i /scratch/project_465002493 || true

for d in "${TARGET_DIRS[@]}"; do
  process_target "$d"
done

echo "=== AFTER: df -i /scratch/project_465002493 ==="
df -i /scratch/project_465002493 || true

echo ""
echo "Run as dry-run (default):"
echo "  sbatch bash_paper2/lumi/cleanup_inode.sh"
echo "Run for real (execute):"
echo "  sbatch --export=ALL,DRY_RUN=0 bash_paper2/lumi/cleanup_inode.sh"
echo ""
echo "=== Job end: $(date) ==="