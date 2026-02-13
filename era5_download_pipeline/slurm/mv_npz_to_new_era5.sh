#!/usr/bin/env bash
#SBATCH --job-name=mv_npz
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=era5_logs/mv_npz_%j.out
#SBATCH --error=era5_logs/mv_npz_%j.err

SRC=/scratch/project_465002493/quistgaa/Data/Data_ERA5_tmp/npz/
DEST=/scratch/project_465002493/quistgaa/Data/Data_DiffMod/data_ERA5/size_589x789/new_ERA5/

mkdir -p "$DEST"
rsync -avh --info=progress2 --remove-source-files "$SRC" "$DEST"
find "$SRC" -type d -empty -delete
echo "Done."