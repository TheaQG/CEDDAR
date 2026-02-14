#!/bin/bash
#SBATCH --job-name=era5_preprocess
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=era5_logs/era5_preprocess_%j.out
#SBATCH --error=era5_logs/era5_preprocess_%j.err


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

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=${USER_BASE}/images/my_torch_container_with_plotting.sif
OVERLAY=${USER_BASE}/overlays/my_overlay.img
HOST_CODE=$ROOT_DIR

CFG_IN_WORKSPACE=/workspace/era5_download_pipeline/cfg/era5_pressure_pipeline.yaml
TS=$(date +%Y%m%d_%H%M%S)
LOG_HOST=$HOST_CODE/era5_download_pipeline/era5_logs/era5_processing_lumi_${TS}.log
LOG_CONT=/workspace/era5_download_pipeline/era5_logs/era5_processing_lumi_${TS}.log
mkdir -p "$(dirname "$LOG_HOST")"

echo "Job   : $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Log   : $LOG_HOST"
echo "Start : $(date)"

srun singularity exec \
     --cleanenv \
     --overlay "$OVERLAY":ro \
     --bind "$HOST_CODE:/workspace" \
     "$CONTAINER" \
     bash -eu <<INNER
MMB=/users/${USER}/micromamba/bin/micromamba
if [[ ! -x \$MMB ]]; then
    echo "ERROR: micromamba not found at \$MMB" >&2
    exit 1
fi

cd /workspace
\$MMB run -n era5 python -m era5_download_pipeline.cli.run_lumi \
          --cfg "$CFG_IN_WORKSPACE" \
          --log-level INFO \
          --n-workers \${SLURM_CPUS_PER_TASK:-\$(nproc)} \
          > "$LOG_CONT" 2>&1
INNER

echo "Finish: $(date)"
