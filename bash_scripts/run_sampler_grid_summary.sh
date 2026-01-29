#!/bin/bash
#SBATCH --job-name=sampler_grid_summary
#SBATCH --output=logs/slurm_sampler_grid_summary_%j.out
#SBATCH --error=logs/slurm_sampler_grid_summary_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard           # CPU-only is enough for summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -eo pipefail

# --- Modules & container (same style as sampler_grid_generation_array.sh) ---
module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
module load lumi-tools || true

CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif

# --- Paths (same pattern as your generation script) ---
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR="$SCRATCH/$USER"
ROOT_DIR="$USER_DIR/Code/SBGM_SD"
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
echo "[INFO] CKPT_DIR   = $CKPT_DIR"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- MODEL KEY: set this to the model you just evaluated with sampler_grid ---
# Example:
# MODEL_KEY="EDM_final_run__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56"
MODEL_KEY="EDM_final_run__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56"

SAMPLER_GRID_ROOT_HOST="$SAMPLE_DIR/evaluation/${MODEL_KEY}/sampler_grid"

echo "[INFO] MODEL_KEY            = $MODEL_KEY"
echo "[INFO] SAMPLER_GRID_ROOT    = $SAMPLER_GRID_ROOT_HOST"

if [[ ! -d "$SAMPLER_GRID_ROOT_HOST" ]]; then
  echo "[ERROR] sampler_grid directory does not exist:"
  echo "        $SAMPLER_GRID_ROOT_HOST"
  echo "        Did you run sampler_grid_generation + sampler_grid_evaluation first?"
  exit 1
fi

# --- Run summary inside the same container style as your other scripts ---
srun singularity exec "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${PYTHONPATH}'
  echo '[INFO] Running sampler-grid summary inside container'
  python -m sbgm.evaluate.summarize_sampler_grid \
    --sampler_grid_root '$SAMPLER_GRID_ROOT_HOST'
"

STATUS=$?

if [[ ${STATUS} -ne 0 ]]; then
  echo "[ERROR] Sampler-grid summary FAILED with status ${STATUS}"
  exit ${STATUS}
fi

echo "[INFO] Sampler-grid summary DONE."
echo "[INFO] Outputs should be at:"
echo "       $SAMPLER_GRID_ROOT_HOST/sampler_grid_summary.csv"
echo "       $SAMPLER_GRID_ROOT_HOST/sampler_grid_summary.json"