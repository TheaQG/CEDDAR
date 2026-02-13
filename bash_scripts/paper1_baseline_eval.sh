#!/bin/bash
#SBATCH --job-name=eval_baselines
#SBATCH --output=logs/slurm_eval_baselines_%x_%j.log
#SBATCH --error=logs/slurm_eval_baselines_%x_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --time=02:00:00


# Fail fast, but set -u only after we’ve safely handled env defaults
set -eo pipefail

# --- Modules ---
# If you want a clean env, force purge; otherwise you can skip this block.
module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits
# If your site expects lumi-tools present, reload it explicitly:
module load lumi-tools || true

# --- Container ---
CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif

# --- Paths ---
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
USER_DIR="$SCRATCH/$USER"
ROOT_DIR="$USER_DIR/Code/CEDDAR"
CONFIG_DIR="$ROOT_DIR/sbgm/config/full_runs_old"
DATA_DIR="$USER_DIR/Data/Data_DiffMod"
SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"
STATS_LOAD_DIR="$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/stats"
EVAL_DIR="$ROOT_DIR/evaluate_sbgm/results"
LOG_DIR="$ROOT_DIR/sbgm/logs"
EXP_DATE="$(date +%d_%m_%Y)"

CFG=${1:-paper1_baselines.yaml}  # config file
# SPLIT=${2:-test}  # train, valid, test
# TYPE=${3:-qm}   # bilinear | qm | unet_sr

# Now it’s safe to enable -u
set -u

# Export env; guard PYTHONPATH with a default
export ROOT_DIR CONFIG_DIR DATA_DIR SAMPLE_DIR CKPT_DIR STATS_LOAD_DIR EVAL_DIR LOG_DIR EXP_DATE
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

mkdir -p logs

echo "[INFO] Date of experiment = $EXP_DATE"
echo "[INFO] ROOT_DIR   = $ROOT_DIR"
echo "[INFO] DATA_DIR   = $DATA_DIR"
echo "[INFO] SAMPLE_DIR = $SAMPLE_DIR"
echo "[INFO] CKPT_DIR   = $CKPT_DIR"

# Threading caps inside container
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# echo "[INFO] Launching baselines ($TYPE)"
srun singularity exec "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${PYTHONPATH}'
  python -m sbgm.cli.main_app --config_path '$CONFIG_DIR/$CFG' --mode baseline_eval
"

