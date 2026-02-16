#!/bin/bash
#SBATCH --job-name=sampler_grid_eval
#SBATCH --output=logs/slurm_sampler_grid_eval_%A_%a.out
#SBATCH --error=logs/slurm_sampler_grid_eval_%A_%a.err
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-59   # same number of combos as generation


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

set -eo pipefail

module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
module load lumi-tools || true

CONTAINER=${USER_BASE}/images/my_torch_container_with_plotting.sif

SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR="$SCRATCH/$USER"
ROOT_DIR="$USER_DIR/Code/CEDDAR"
CONFIG_DIR="$ROOT_DIR/sbgm/config"
DATA_DIR="$USER_DIR/Data/Data_DiffMod"
SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"
STATS_LOAD_DIR="$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/stats"
EVAL_DIR="$ROOT_DIR/evaluate_sbgm/results"
LOG_DIR="$ROOT_DIR/sbgm/logs"
EXP_DATE="$(date +%d_%m_%Y)"

set -u

export ROOT_DIR CONFIG_DIR DATA_DIR SAMPLE_DIR CKPT_DIR STATS_LOAD_DIR EVAL_DIR LOG_DIR EXP_DATE
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p logs

echo "[INFO] ROOT_DIR   = $ROOT_DIR"
echo "[INFO] SAMPLE_DIR = $SAMPLE_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

TASK_ID=${SLURM_ARRAY_TASK_ID}
echo "[INFO] This is sampler-grid EVAL combo index ${TASK_ID}"

srun singularity exec "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${PYTHONPATH}'
  export SAMPLER_COMBO_INDEX_START=${TASK_ID}
  export SAMPLER_COMBO_INDEX_END=${TASK_ID}

  python -m sbgm.cli.main_app \
    --mode sampler_grid_evaluation \
    --config_path $CONFIG_DIR/new_eval_setup_test.yaml \
    --make_plots
"
