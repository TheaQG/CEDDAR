#!/usr/bin/env bash

# rename_zpl.sh - convert filenames that got the wrong name during download:
#           z_pl_<YEAR>_<PLEV>.nc -> z_pl_<PLEV>_<YEAR>.nc
#   Usage:
#      ./rename_zpl.sh /absolute/or/relative/path/to/data/raw/z_pl
#      #       └── this folder contains level sub-dirs 1000/ 850/ …
#
#   Add -n for a dry-run (no actual mv) e.g.:
#      ./rename_zpl.sh -n /scratch/.../raw/z_pl
# 

# Set strict mode for better error handling
set -euo pipefail

#  DRYRUN flag: 0 --> real run, 1 --> dry-run (no mv)
#  Usage: ./rename_zpl.sh [-n] /path/to/z_pl
DRYRUN=0
# Check for dry-run option
# If the first argument is -n, set DRYRUN to 1 and shift arguments
[[ ${1-} == "-n" ]] && { DRYRUN=1; shift; }


# Check if the correct number of arguments is provided
[[ $# -eq 1 ]] || { echo "Usage: $0 [-n] /path/to/z_pl"; exit 1; }
ROOT=$1

# Check if the provided path is a directory
[[ -d $ROOT ]] || { echo "Error: $ROOT is not a directory"; exit 1; }


echo "Scanning $ROOT for z_pl files to rename..."
# Find all files matching the pattern z_pl_<YEAR>_<PLEV>.nc
find "$ROOT" -type f -name 'z_pl_[0-9][0-9][0-9][0-9]_[0-9]*.nc' | while read -r f; do
    file=$(basename "$f")
    dir=$(dirname "$f")

    if [[ $file =~ ^z_pl_([0-9]{4})_([0-9]+)\.nc$ ]]; then
        YEAR="${BASH_REMATCH[1]}"
        PLEV="${BASH_REMATCH[2]}"
        new="z_pl_${PLEV}_$YEAR.nc"

        if [[ -e $dir/$new ]]; then
            echo "Warning: $dir/$new already exists, skipping rename for $file"
            continue
        fi
        if [[ $DRYRUN -eq 1 ]]; then
            echo "Dry-run: would rename $file to $new"
        else
            mv "$dir/$file" "$dir/$new"
            echo "Renamed $file to $new"
        fi
    else
        echo "Skipping $file, does not match expected pattern"
    fi
done