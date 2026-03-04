#!/usr/bin/env bash
#SBATCH --job-name=archive_bad_era5_zarr
#SBATCH --output=logs/archive_bad_era5_zarr_%j.log
#SBATCH --error=logs/archive_bad_era5_zarr_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

# Print failing command + line number into the log.
trap 'echo "[ERR ] Failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

#############################################
# Safe archiver for bad ERA5 Zarr outputs
#
# Features:
# - DRY_RUN=1 by default (prints actions, does not modify filesystem)
# - Archives each train/valid/test .zarr directory into a .tar.zst (preferred) or .tar.gz
# - Writes a per-archive manifest with: timestamp, bytes, nfiles, capped hash, and first entries list
# - Never deletes originals unless REMOVE=1 (default 0)
# - Never overwrites archives unless ALLOW_OVERWRITE=1 (default 0)
# - Stops on any error (set -euo pipefail) and logs clearly
# - Portable compression path: (tar -cf - ...) | zstd/gzip -> file.part then mv
#
# Typical submission:
#   sbatch --export=ALL,DRY_RUN=1,REMOVE=0 bash_paper2/lumi/archive_bad_era5_zarr.sh
# Real run (no deletion):
#   sbatch --export=ALL,DRY_RUN=0,REMOVE=0 bash_paper2/lumi/archive_bad_era5_zarr.sh
# Optional deletion after successful archive (be careful):
#   sbatch --export=ALL,DRY_RUN=0,REMOVE=1 bash_paper2/lumi/archive_bad_era5_zarr.sh
#############################################

# -----------------------------
# User settings (override via env)
# -----------------------------
BASE="${BASE:-/scratch/project_465002493/quistgaa/Data/Data_DiffMod/data_ERA5/size_589x789}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${BASE}/__archives_bad_regrid__}"

# Safety defaults
DRY_RUN="${DRY_RUN:-1}"            # 1 = print only, 0 = execute
REMOVE="${REMOVE:-0}"              # 1 = delete originals after successful archive
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"  # 1 = overwrite existing archive+manifest

# Splits to archive
SPLITS_DEFAULT="train valid test"
SPLITS="${SPLITS:-$SPLITS_DEFAULT}"

# Hash / manifest settings
MAX_HASH_FILES="${MAX_HASH_FILES:-20000}"   # cap for hashing directory listing
FIRST_ENTRIES_N="${FIRST_ENTRIES_N:-200}"   # lines of first entries in manifest

# Compression
ZSTD_LEVEL="${ZSTD_LEVEL:-12}"
ZSTD_THREADS="${ZSTD_THREADS:-8}"

# Parallel archiving
MAX_JOBS="${MAX_JOBS:-3}"   # number of archives to run concurrently

# Misc
UMASK="${UMASK:-0027}"

umask "$UMASK"
export LC_ALL=C

log()  { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
err()  { echo "[ERR ] $*" >&2; }

run() {
  # Run command unless DRY_RUN=1. Print the command either way.
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: $*"
  else
    eval "$@"
  fi
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

now_stamp() {
  date +%Y%m%dT%H%M%S
}

# Return directory size in bytes (best-effort).
dir_size_bytes() {
  local d="$1"
  if have_cmd du; then
    # GNU du supports -sb. If not available, fallback to KiB.
    if du -sb "$d" >/dev/null 2>&1; then
      du -sb "$d" | awk '{print $1}'
      return 0
    fi
    du -sk "$d" | awk '{print $1 * 1024}'
    return 0
  fi
  echo 0
}

# Return number of files under directory.
dir_nfiles() {
  local d="$1"
  find "$d" -type f 2>/dev/null | wc -l | tr -d ' '
}

# Produce a stable capped hash of a directory tree, based on relative path + size + mtime.
# Output: single hash token (md5 or sha256), or NA if hashing tool missing.
manifest_hash_dir() {
  local d="$1"

  # We hash a capped, sorted listing to keep runtime bounded.
  # NOTE: this is not a cryptographic guarantee of full-content equality; it is a quick fingerprint.

  local sum_tool=""
  if have_cmd md5sum; then
    sum_tool="md5sum"
  elif have_cmd sha256sum; then
    sum_tool="sha256sum"
  else
    echo "NA"
    return 0
  fi

  # Use -printf for a deterministic record. No sort, just head for speed.
  # shellcheck disable=SC2016
  find "$d" -type f -printf '%P|%s|%T@\n' 2>/dev/null \
    | sort \
    | head -n "$MAX_HASH_FILES" \
    | "$sum_tool" \
    | awk '{print $1}' \
    | head -n 1 \
    | tr -d ' \t\r\n'
}

write_manifest() {
  local out_manifest="$1"
  local zarr_dir="$2"
  local bytes="$3"
  local nfiles="$4"
  local hash="$5"

  local ts
  ts="$(now_stamp)"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: write manifest -> $out_manifest"
    return 0
  fi

  {
    echo "timestamp=$ts"
    echo "base=$BASE"
    echo "zarr_dir=$zarr_dir"
    echo "bytes=$bytes"
    echo "nfiles=$nfiles"
    echo "hash_capped=$hash"
    echo "hash_capped_max_files=$MAX_HASH_FILES"
    echo "---- first_entries (path|size|mtime_epoch) ----"

    # First entries: do NOT sort (cheaper); these are just a quick peek.
    find "$zarr_dir" -type f -printf '%P|%s|%T@\n' 2>/dev/null \
      | head -n "$FIRST_ENTRIES_N" || true
  } > "$out_manifest"
}

# Create archive from a zarr directory. Writes archive to out_tar (.tar.zst preferred).
create_archive() {
  local zarr_dir="$1"
  local out_tar="$2"

  local parent_dir name
  parent_dir="$(dirname "$zarr_dir")"
  name="$(basename "$zarr_dir")"

  local tmp_tar
  tmp_tar="${out_tar}.part"

  if have_cmd zstd; then
    log "Archiving -> $out_tar"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY-RUN: (cd \"$parent_dir\" && tar -cf - \"$name\") | zstd -${ZSTD_LEVEL} -T${ZSTD_THREADS} -o \"$out_tar\""
    else
      rm -f "$tmp_tar"
      (cd "$parent_dir" && tar -cf - "$name") \
        | zstd -${ZSTD_LEVEL} -T${ZSTD_THREADS} -o "$tmp_tar"
      mv -f "$tmp_tar" "$out_tar"
    fi
  else
    warn "zstd not found; falling back to gzip (bigger + slower)."
    out_tar="${out_tar%.zst}.gz"
    tmp_tar="${out_tar}.part"
    log "Archiving -> $out_tar"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY-RUN: (cd \"$parent_dir\" && tar -cf - \"$name\") | gzip -c > \"$out_tar\""
    else
      rm -f "$tmp_tar"
      (cd "$parent_dir" && tar -cf - "$name") | gzip -c > "$tmp_tar"
      mv -f "$tmp_tar" "$out_tar"
    fi
  fi

  # Return the final path (handles gzip fallback)
  echo "$out_tar"
}

sanity_check_archive() {
  local out_tar="$1"
  log "Sanity check (first members):"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: tar -tf \"$out_tar\" | head -n 10"
  else
    tar -tf "$out_tar" | head -n 10
  fi
}

maybe_remove_original() {
  local zarr_dir="$1"
  if [[ "$REMOVE" != "1" ]]; then
    return 0
  fi
  warn "REMOVE=1 enabled: will delete original after successful archive: $zarr_dir"
  run "rm -rf \"$zarr_dir\""
}

archive_one_zarr() {
  local zarr_dir="$1"

  # Build archive destination paths. We keep per-variable structure under ARCHIVE_ROOT.
  # Example:
  #   BASE/var_589x789/zarr_files/train.zarr
  # -> ARCHIVE_ROOT/var_589x789/zarr_files/train.zarr_YYYYmmddTHHMMSS.tar.zst

  local rel
  rel="${zarr_dir#${BASE}/}"  # relative path under BASE
  local ts
  ts="$(now_stamp)"

  run "mkdir -p \"$ARCHIVE_ROOT/$(dirname "$rel")\""

  local out_tar out_manifest
  out_tar="$ARCHIVE_ROOT/${rel}_${ts}.tar.zst"
  out_manifest="$ARCHIVE_ROOT/${rel}_${ts}.manifest.txt"

  if [[ -e "$out_tar" || -e "$out_manifest" ]]; then
    if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
      err "Archive or manifest already exists for: $out_tar (set ALLOW_OVERWRITE=1 to override)"
      return 1
    fi
    warn "ALLOW_OVERWRITE=1: overwriting existing archive/manifest for: $rel"
  fi

  local bytes nfiles hash
  bytes="$(dir_size_bytes "$zarr_dir" 2>/dev/null || echo 0)"
  nfiles="$(dir_nfiles "$zarr_dir" 2>/dev/null || echo 0)"
  hash="$(manifest_hash_dir "$zarr_dir" 2>/dev/null || echo NA)"
  hash="$(echo "$hash" | head -n 1 | tr -d ' \t\r\n')"
  [[ -n "$hash" ]] || hash="NA"

  log "--------------------------------------------"
  log "ZARR:  $zarr_dir"
  log "  size:  $(awk -v b="$bytes" 'BEGIN{printf("%.1fGiB", b/1024/1024/1024)}')"
  log "  files: $nfiles"
  log "  hash:  $hash (capped to $MAX_HASH_FILES entries)"

  write_manifest "$out_manifest" "$zarr_dir" "$bytes" "$nfiles" "$hash"

  local final_tar
  final_tar="$(create_archive "$zarr_dir" "$out_tar")"

  # If gzip fallback happened, also align manifest base name (optional). We keep manifest as-is.
  sanity_check_archive "$final_tar"

  maybe_remove_original "$zarr_dir"
}

find_zarr_targets() {
  local split
  for split in $SPLITS; do
    # Find directories that end with /zarr_files/<split>.zarr
    find "$BASE" -type d -path "*/zarr_files/${split}.zarr" 2>/dev/null || true
  done
}

main() {
  log "BASE=$BASE"
  log "ARCHIVE_ROOT=$ARCHIVE_ROOT"
  log "DRY_RUN=$DRY_RUN  REMOVE=$REMOVE  ALLOW_OVERWRITE=$ALLOW_OVERWRITE"
  log "SPLITS=$SPLITS"
  log "ZSTD_LEVEL=$ZSTD_LEVEL  ZSTD_THREADS=$ZSTD_THREADS"
  log "MAX_HASH_FILES=$MAX_HASH_FILES  FIRST_ENTRIES_N=$FIRST_ENTRIES_N"

  if [[ ! -d "$BASE" ]]; then
    err "BASE does not exist or is not a directory: $BASE"
    exit 2
  fi

  run "mkdir -p \"$ARCHIVE_ROOT\""

  mapfile -t targets < <(find_zarr_targets)

  if [[ ${#targets[@]} -eq 0 ]]; then
    warn "No zarr targets found under BASE with splits: $SPLITS"
    exit 0
  fi

  log "Found ${#targets[@]} zarr directories to archive."

  local t
  for t in "${targets[@]}"; do
    archive_one_zarr "$t" &

    # limit number of concurrent jobs
    while [[ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]]; do
      sleep 1
    done
  done

  # Wait for all background jobs to finish
  wait

  log "Done."
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1: nothing was changed. Re-run with DRY_RUN=0 to execute."
  else
    log "DRY_RUN=0: archives/manifests were created under: $ARCHIVE_ROOT"
    if [[ "$REMOVE" == "0" ]]; then
      log "REMOVE=0: originals were NOT deleted."
    fi
  fi
}

main "$@"