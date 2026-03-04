#!/bin/bash
#SBATCH --job-name=era5_regrid_test
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/test_regrid_%j.out
#SBATCH --error=logs/test_regrid_%j.err

set -euo pipefail

mkdir -p logs

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# --- Base paths (match lumi_process.sh style) ---
ACCOUNT="${SLURM_JOB_ACCOUNT:-project_465002493}"
USER_BASE="/scratch/${ACCOUNT}/${USER}"
ROOT_DIR="${ROOT_DIR:-${USER_BASE}/Code/CEDDAR}"
HOST_CODE="$ROOT_DIR"

# Data root used by the Python script
export ERA5_TMP_DIR="${ERA5_TMP_DIR:-${USER_BASE}/Data/Data_ERA5_tmp}"

# Optional test parameters
export TEST_VAR="${TEST_VAR:-cape}"
export TEST_YEAR="${TEST_YEAR:-1994}"

# Container + overlay (match your working setup)
CONTAINER="/scratch/${ACCOUNT}/containers/images/my_torch_container_with_plotting.sif"
OVERLAY="/scratch/${ACCOUNT}/containers/overlays/my_overlay.img"

echo "[INFO] USER_BASE=$USER_BASE"
echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] ERA5_TMP_DIR=$ERA5_TMP_DIR"
echo "[INFO] TEST_VAR=$TEST_VAR  TEST_YEAR=$TEST_YEAR"
echo "[INFO] Container: $CONTAINER"
echo "[INFO] Overlay  : $OVERLAY"

# Early check for overlay existence
if [[ ! -f "$OVERLAY" ]]; then
  echo "[ERR ] Overlay not found: $OVERLAY" >&2
  echo "[ERR ] Listing /scratch/${ACCOUNT}/containers/overlays:" >&2
  ls -la "/scratch/${ACCOUNT}/containers/overlays" | sed -n '1,200p' >&2 || true
  exit 2
fi

# Path to the script inside the container (since we bind ROOT_DIR -> /workspace)
PY_CONT="/workspace/era5_download_pipeline/pipeline/test_regrid_weights.py"

srun singularity exec \
     --cleanenv \
     --env ERA5_TMP_DIR="$ERA5_TMP_DIR" \
     --env TEST_VAR="$TEST_VAR" \
     --env TEST_YEAR="$TEST_YEAR" \
     --overlay "$OVERLAY":ro \
     --bind "$HOST_CODE:/workspace" \
     "$CONTAINER" \
     bash -eu <<INNER
MMB=/users/${USER}/micromamba/bin/micromamba
if [[ ! -x "\$MMB" ]]; then
    echo "ERROR: micromamba not found at \$MMB" >&2
    exit 1
fi

# Sanity: ensure required env vars made it through --cleanenv
if [[ -z "${ERA5_TMP_DIR:-}" ]]; then
  echo "ERROR: ERA5_TMP_DIR is not set inside container. Did --env forwarding fail?" >&2
  exit 3
fi

echo "[INFO] Inside container env: ERA5_TMP_DIR=$ERA5_TMP_DIR"
echo "[INFO] Inside container env: TEST_VAR=${TEST_VAR:-} TEST_YEAR=${TEST_YEAR:-}"

echo "[INFO] Inside container: listing pipeline dir"
ls -la /workspace/era5_download_pipeline/pipeline | sed -n '1,80p'

echo "[INFO] Running test script in micromamba env: era5"
"\$MMB" run -n era5 python "$PY_CONT"
INNER