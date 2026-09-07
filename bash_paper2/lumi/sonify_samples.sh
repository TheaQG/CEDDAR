#!/bin/bash
#SBATCH --job-name=V0_sonify
#SBATCH --output=logs/slurm_V0_sonify_%x_%j.log
#SBATCH --error=logs/slurm_V0_sonify_%x_%j.err
#SBATCH --account=project_465002737
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=28 # 7 # 56 # 7 * 8 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=08:00:00


# ===================================================================
# V0 Noise-to-Rain trajectory extraction
# ===================================================================
# Runs the final V0 checkpoint through the EDM sampler with trajectory
# capture enabled. This does not generate videos/audio on LUMI; it only
# writes compact trajectory.npz + metadata.json packages for local use
# in NoiseToRain.ipynb.

set -eo pipefail

# --- Modules ---
module --force purge || true
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# --- Container ---
CONTAINER=/scratch/project_465002493/containers/images/my_torch_container_with_plotting.sif

# --- Paths ---
SCRATCH="/scratch/project_465002493"
USER_DIR="$SCRATCH/$USER"
ROOT_DIR="$USER_DIR/Code/CEDDAR"
CONFIG_DIR="$ROOT_DIR/sbgm/config/paper2"
DATA_DIR="$USER_DIR/Data/Data_DiffMod"
SAMPLE_DIR="$ROOT_DIR/models_and_samples/generated_samples"
CKPT_DIR="$ROOT_DIR/models_and_samples/trained_models"
STATS_LOAD_DIR="$ROOT_DIR/data_analysis_pipeline_private/saved/statistics_run/stats"
EVAL_DIR="$ROOT_DIR/evaluate_sbgm/results"
LOG_DIR="$ROOT_DIR/sbgm/logs"
EXP_DATE="$(date +%d_%m_%Y)"

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
echo "[INFO] STATS_LOAD_DIR = $STATS_LOAD_DIR"

# Threading caps inside container
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- MIOpen workaround: per-job DB ---
MIOPEN_DB_DIR="$SCRATCH/$USER/miopen_db_${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_DB_DIR"
export MIOPEN_USER_DB_PATH="$MIOPEN_DB_DIR/userdb.sql"
export MIOPEN_SYSTEM_DB_PATH="$MIOPEN_DB_DIR/systemdb.sql"

CFG="$CONFIG_DIR/baseline/V0.yaml"
OUT_DIR="$ROOT_DIR/models_and_samples/noise_to_rain/V0"

# User-facing controls. Override at sbatch time, e.g.:
#   sbatch --export=ALL,NUM_SAMPLES=8,MEMBERS_PER_SAMPLE=2,START_INDEX=25 bash_paper2/lumi/sonify_samples.sh
export NUM_SAMPLES="${NUM_SAMPLES:-60}"
export MEMBERS_PER_SAMPLE="${MEMBERS_PER_SAMPLE:-32}"
export START_INDEX="${START_INDEX:-120}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export BASE_SEED="${BASE_SEED:-504}"
export CAPTURE_EVERY="${CAPTURE_EVERY:-1}"
export CAPTURE_DTYPE="${CAPTURE_DTYPE:-float32}"
export SPLIT="${SPLIT:-test}"

echo "[INFO] CFG       = $CFG"
echo "[INFO] CKPT      = derived from V0.yaml via get_model_string(cfg)"
echo "[INFO] OUT_DIR   = $OUT_DIR"
echo "[INFO] SPLIT     = $SPLIT"
echo "[INFO] NUM_SAMPLES = $NUM_SAMPLES"
echo "[INFO] MEMBERS_PER_SAMPLE = $MEMBERS_PER_SAMPLE"
echo "[INFO] START_INDEX = $START_INDEX"
echo "[INFO] BATCH_SIZE = $BATCH_SIZE"
echo "[INFO] BASE_SEED = $BASE_SEED"
echo "[INFO] CAPTURE_EVERY = $CAPTURE_EVERY"
echo "[INFO] CAPTURE_DTYPE = $CAPTURE_DTYPE"

if [[ ! -f "$CFG" ]]; then
    echo "[ERROR] Config file not found: $CFG"
    exit 1
fi


mkdir -p "$OUT_DIR"

export TMPDIR="$SCRATCH/$USER/tmp"
mkdir -p "$TMPDIR"
unset MAMBA_EXE
unset MAMBA_ROOT_PREFIX

export SBGM_DISTRIBUTED=0
export DEBUG_SINGLE_GPU=1
export DDP_MULTI_GPU=0

COMMON_CONTAINER_ENV=$(cat <<EOF
set -euo pipefail
export TMPDIR='$TMPDIR'
unset MAMBA_EXE
unset MAMBA_ROOT_PREFIX
export PYTHONPATH='${PYTHONPATH}'
export ROOT_DIR='${ROOT_DIR}'
export CONFIG_DIR='${CONFIG_DIR}'
export DATA_DIR='${DATA_DIR}'
export SAMPLE_DIR='${SAMPLE_DIR}'
export CKPT_DIR='${CKPT_DIR}'
export STATS_LOAD_DIR='${STATS_LOAD_DIR}'
export EVAL_DIR='${EVAL_DIR}'
export LOG_DIR='${LOG_DIR}'
export EXP_DATE='${EXP_DATE}'
export SBGM_DISTRIBUTED=0
export DEBUG_SINGLE_GPU=1
export DDP_MULTI_GPU=0
EOF
)

echo "[INFO] Launching V0 Noise-to-Rain trajectory extraction inside container"

srun singularity exec "$CONTAINER" bash --noprofile --norc -lc "
    ${COMMON_CONTAINER_ENV}
    python -m sbgm.tools.sonify_sampling \
        --config '$CFG' \
        --output-dir '$OUT_DIR' \
        --split '$SPLIT' \
        --num-samples '$NUM_SAMPLES' \
        --members-per-sample '$MEMBERS_PER_SAMPLE' \
        --start-index '$START_INDEX' \
        --batch-size '$BATCH_SIZE' \
        --seed '$BASE_SEED' \
        --capture-every '$CAPTURE_EVERY' \
        --capture-dtype '$CAPTURE_DTYPE'
"

echo "[INFO] Done. Trajectory packages written to: $OUT_DIR"