#!/bin/bash
#SBATCH --job-name=D_MZP_1__seed2
#SBATCH --output=logs/slurm_D_MZP_1__seed2_%x_%j.log
#SBATCH --error=logs/slurm_D_MZP_1__seed2_%x_%j.err
#SBATCH --account=project_465002737
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=28 # 7 # 56 # 7 * 8 cores per GPU
#SBATCH --mem-per-gpu=60G
#SBATCH --time=20:00:00


# ===================================================================
# D_MZP_1__seed2: Simplest synoptic overview, no water vapour fluxes, (prcp + msl + z_pl_500), large domain context
# ===================================================================

# Fail fast, but set -u only after we’ve safely handled env defaults
set -eo pipefail

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

# === Optional: create logs directory if it doesn't exist ===
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

CFG="$CONFIG_DIR/physics_based/D_MZP_1__seed2.yaml"

echo "[INFO] Running D_MZP_1__seed2: training → generation → quicklook → evaluation (inside container)"


export TMPDIR="$SCRATCH/$USER/tmp"
mkdir -p "$TMPDIR"
unset MAMBA_EXE
unset MAMBA_ROOT_PREFIX

# ------------------------------------------------------------------
# Runtime mode selection
# ------------------------------------------------------------------
export DEBUG_SINGLE_GPU=0 #"${DEBUG_SINGLE_GPU:-1}" 
export DDP_MULTI_GPU=1 #"${DDP_MULTI_GPU:-0}"
export ENABLE_ROCM_MONITORING="${ENABLE_ROCM_MONITORING:-0}"
export ROCM_MONITOR_INTERVAL_SEC="${ROCM_MONITOR_INTERVAL_SEC:-60}"

if [[ "$DDP_MULTI_GPU" == "1" && "${SLURM_GPUS_ON_NODE:-1}" == "1" ]]; then
    echo "[WARN] DDP_MULTI_GPU=1 but allocation only exposes 1 GPU. Falling back to single-process launch."
    export DDP_MULTI_GPU=0
fi

echo "[INFO] DEBUG_SINGLE_GPU = $DEBUG_SINGLE_GPU"
echo "[INFO] DDP_MULTI_GPU = $DDP_MULTI_GPU"
echo "[INFO] ENABLE_ROCM_MONITORING = $ENABLE_ROCM_MONITORING"

ROCM_MONITOR_PID=""
if [[ "$ENABLE_ROCM_MONITORING" == "1" ]] && command -v rocm-smi >/dev/null 2>&1; then
    ROCM_LOG="logs/rocm_smi_${SLURM_JOB_ID}.log"
    echo "[INFO] Starting rocm-smi monitor -> $ROCM_LOG"
    (
        while true; do
            echo "===== $(date '+%F %T') =====" >> "$ROCM_LOG"
            rocm-smi --showuse --showmemuse --showpower --showclocks >> "$ROCM_LOG" 2>&1 || true
            sleep "$ROCM_MONITOR_INTERVAL_SEC"
        done
    ) &
    ROCM_MONITOR_PID=$!
fi

cleanup() {
    if [[ -n "${ROCM_MONITOR_PID:-}" ]]; then
        kill "$ROCM_MONITOR_PID" > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

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
EOF
)

if [[ "$DDP_MULTI_GPU" == "1" ]]; then
    export SBGM_DISTRIBUTED=1
    NPROC=4
    echo "[INFO] Detected ${NPROC} visible GPU(s) inside container"
    if [[ -z "${NPROC}" || "${NPROC}" == "0" ]]; then
        echo "[ERROR] No visible GPUs detected inside container. Aborting DDP launch."
        exit 1
    fi
    echo "[INFO] Launching DDP with torchrun on ${NPROC} processes"
    srun singularity exec "$CONTAINER" bash --noprofile --norc -lc "
        ${COMMON_CONTAINER_ENV}
        torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
        -m sbgm.cli.main_app --mode full_pipeline --config_path $CFG --make_plots
    "
else
    export SBGM_DISTRIBUTED=0
    echo "[INFO] Launching single-process run on 1 GPU"
    srun singularity exec "$CONTAINER" bash --noprofile --norc -lc "
        ${COMMON_CONTAINER_ENV}
        python -m sbgm.cli.main_app --mode full_pipeline --config_path $CFG --make_plots
    "
fi
  