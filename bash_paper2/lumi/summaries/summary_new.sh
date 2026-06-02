#!/bin/bash
#SBATCH --job-name=paper2_summary
#SBATCH --output=logs/slurm_paper2_summary_%x_%j.log
#SBATCH --error=logs/slurm_paper2_summary_%x_%j.err
#SBATCH --account=project_465002737
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2 # 7 # 56 # 7 * 8 cores per GPU
#SBATCH --mem-per-gpu=4G
#SBATCH --time=00:05:00

set -eo pipefail

# --------------------------------------------------------------------------
# User-editable selections
# --------------------------------------------------------------------------
ROOT_DIR="/scratch/project_465002493/quistgaa/Code/CEDDAR"

# --- Modules ---
# If you want a clean env, force purge; otherwise you can skip this block.
module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# --- Container ---
CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif

# --- Paths ---
SCRATCH="/scratch/project_465002493" #"/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR="$SCRATCH/$USER"
ROOT_DIR="$USER_DIR/Code/CEDDAR"

CONFIG_DIR="$ROOT_DIR/sbgm/config/paper2/summaries"
CONFIG_PATH="$CONFIG_DIR/p2_summary.yaml"

SCRIPT_PATH="$ROOT_DIR/sbgm/evaluate/evaluation_summary/paper2_figures_new.py"

# Now it’s safe to enable -u
set -u

# Export env; guard PYTHONPATH with a default
export ROOT_DIR CONFIG_DIR CONFIG_PATH SCRIPT_PATH
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Example prefix selections
MODEL_PREFIXES=("V0" "C_" "D_")
BASELINE_PREFIX="V0__"
BASELINE_SEED_PREFIXES=("V0__" "V0__seed")
POOL_MODE="all"
OUTPUT_DIR="$ROOT_DIR/paper2_summary_outputs"

# --------------------------------------------------------------------------
# Build CLI arguments
# --------------------------------------------------------------------------

ARGS=(
    "${SCRIPT_PATH}"
    --config "${CONFIG_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --baseline-prefix "${BASELINE_PREFIX}"
)

if [ ${#MODEL_PREFIXES[@]} -gt 0 ]; then
    ARGS+=(--model-prefixes "${MODEL_PREFIXES[@]}")
fi

if [ ${#BASELINE_SEED_PREFIXES[@]} -gt 0 ]; then
    ARGS+=(--baseline-seed-prefixes "${BASELINE_SEED_PREFIXES[@]}")
fi

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

echo "Running Paper 2 summary build..."
echo "Script: ${SCRIPT_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

singularity exec "${CONTAINER}" python "${ARGS[@]}"