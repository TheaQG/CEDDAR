#!/bin/bash
#SBATCH --job-name=B0_S_EMA
#SBATCH --output=logs/slurm_B0_S_EMA_%x_%j.log
#SBATCH --error=logs/slurm_B0_S_EMA_%x_%j.err
#SBATCH --account=project_465002493
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56 # 7 * 8 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=08:00:00


# ===================================================================
# B0_S_EMA: BASELINE MODEL WITH SEASONAL INFORMATION AND EMA CONFIGURATION FOR ABLATION STUDIES 
# ===================================================================


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
CONFIG_DIR="$ROOT_DIR/sbgm/config/ablations/B0s"
DATA_DIR="$USER_DIR/Data/Data_DiffMod"
SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"
STATS_LOAD_DIR="$ROOT_DIR/data_analysis_pipeline/saved/statistics_run/stats"
EVAL_DIR="$ROOT_DIR/evaluate_sbgm/results"
LOG_DIR="$ROOT_DIR/sbgm/logs"
EXP_DATE="$(date +%d_%m_%Y)"

# Now it’s safe to enable -u
set -u

# Export env; guard PYTHONPATH with a default
export ROOT_DIR CONFIG_DIR DATA_DIR SAMPLE_DIR CKPT_DIR STATS_LOAD_DIR EVAL_DIR LOG_DIR EXP_DATE
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# === Optional: create logs directory if it doesn't exist ===
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

# --- MIOpen workaround: per-job DB ---
MIOPEN_DB_DIR="$SCRATCH/$USER/miopen_db_${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_DB_DIR"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB_DIR/userdb.sql"
export MIOPEN_SYSTEM_DB_PATH="$MIOPEN_DB_DIR/systemdb.sql"

CFG="$CONFIG_DIR/B0_S_EMA.yaml"

echo "[INFO] Running ablation B0 + S + EMA: training → generation → quicklook → evaluation (inside container)"

srun singularity exec "$CONTAINER" bash -lc "
  set -euo pipefail
  export PYTHONPATH='${PYTHONPATH}'
  python -m sbgm.cli.main_app --mode full_pipeline --config_path $CFG --make_plots
"
  # python -m sbgm.cli.main_app --mode quicklook --config_path $CFG --make_plots &&
  # python -m sbgm.cli.main_app --mode generate --config_path $CFG --make_plots &&
  # python -m sbgm.cli.main_app --mode evaluate --config_path $CFG --make_plots

  # python -m sbgm.cli.main_app --mode generate --config_path $CONFIG_DIR/ablation_basic_geo_SDF.yaml --make_plots &&
  # python -m sbgm.cli.main_app --mode sigma_star_generation --config_path $CONFIG_DIR/ablation_basic_geo_SDF.yaml --make_plots &&
  # python -m sbgm.cli.main_app --mode sigma_star_evaluation --config_path $CONFIG_DIR/ablation_basic_geo_SDF.yaml --make_plots  
  # python -m sbgm.cli.main_app --mode quicklook --config_path '$CONFIG_DIR/ablation_basic_geo_SDF.yaml' --make_plots &&
  # python -m sbgm.cli.main_app --mode full_pipeline --config_path $CONFIG_DIR/ablation_basic_geo_SDF.yaml --make_plots






# # === Launch the training ===
# echo "[INFO] Launching the full training-generation-evaluation pipeline..."
# srun singularity exec $CONTAINER \
#     python -m sbgm.cli.main_app --mode full_pipeline --config_path $CONFIG_DIR/paper1_expA.yaml --make_plots
