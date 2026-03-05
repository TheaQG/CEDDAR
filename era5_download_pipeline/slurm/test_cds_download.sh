#!/bin/bash
#SBATCH --job-name=cds_smoke_test
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=era5_logs/cds_smoke_%j.out
#SBATCH --error=era5_logs/cds_smoke_%j.err

set -eo pipefail

ACCOUNT="project_465002493"
USER_BASE="/scratch/${ACCOUNT}/${USER}"
CONT_BASE="/scratch/${ACCOUNT}/containers"
ROOT_DIR="${USER_BASE}/Code/CEDDAR"
CONTAINER="${CONT_BASE}/images/my_torch_container_with_plotting.sif"
OVERLAY="${CONT_BASE}/overlays/my_overlay.img"

ERA5_TMP_DIR="${USER_BASE}/Data/Data_ERA5_tmp"
OUTDIR="${ERA5_TMP_DIR}/test_cds_download"

mkdir -p "$OUTDIR"
mkdir -p "$ROOT_DIR/era5_download_pipeline/era5_logs"

echo "Running CDS smoke test"
echo "Output dir: $OUTDIR"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

MMB=/pfs/lustrep2/scratch/project_465002493/micromamba/bin/micromamba
export MAMBA_ROOT_PREFIX=/pfs/lustrep2/scratch/project_465002493/micromamba

srun singularity exec \
    --overlay "$OVERLAY":ro \
    --bind "$ROOT_DIR:/workspace" \
    "$CONTAINER" \
    bash -eu <<INNER

export MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX
MMB=$MMB

echo "Testing cdsapi import..."

\$MMB run -n era5 python - <<PY
import cdsapi, os

out = "${OUTDIR}/era5_test.nc"
print("Downloading to:", out)

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": "1994",
        "month": "01",
        "day": "01",
        "time": "00:00",
        "format": "netcdf",
        "area": [70, -20, 45, 35],
    },
    out,
)

print("Download finished:", out)
PY

INNER

echo "Finished: $(date)"