#!/bin/bash
#SBATCH --job-name=era5_thetae850
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=era5_logs/thetae850_%j.out
#SBATCH --error=era5_logs/thetae850_%j.err

set -euo pipefail

module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

CONTAINER=/scratch/project_xxxxx/containers/images/my_torch_container_with_plotting.sif
OVERLAY=/scratch/project_xxxxx/containers/overlays/my_overlay.img
HOST_CODE=/scratch/project_xxxxx/USER/Code/CEDDAR

# Raw input dirs (as produced by your ERA5 pressure-level pipeline)
RAW_ROOT=/scratch/project_xxxxx/USER/Data/Data_ERA5_tmp/raw
IN_T_DIR=${RAW_ROOT}/t_pl/850
IN_Q_DIR=${RAW_ROOT}/q_pl/850

# Output dir for derived variable
OUT_DIR=${RAW_ROOT}/thetae_pl/850

# Year range (inclusive)
YEAR_START=1991
YEAR_END=2020

TS=$(date +%Y%m%d_%H%M%S)
LOG_HOST=$HOST_CODE/era5_download_pipeline/era5_logs/thetae850_lumi_${TS}.log
LOG_CONT=/workspace/era5_download_pipeline/era5_logs/thetae850_lumi_${TS}.log
mkdir -p "$(dirname "$LOG_HOST")"

echo "Job   : $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Log   : $LOG_HOST"
echo "Start : $(date)"

echo "IN_T_DIR: $IN_T_DIR"
echo "IN_Q_DIR: $IN_Q_DIR"
echo "OUT_DIR : $OUT_DIR"

echo "NOTE: This job runs CDO inside the container (not on login nodes)."

srun singularity exec \
     --cleanenv \
     --overlay "$OVERLAY":ro \
     --bind "$HOST_CODE:/workspace" \
     "$CONTAINER" \
     bash -eu <<'INNER'
MMB=/users/USER/micromamba/bin/micromamba
if [[ ! -x $MMB ]]; then
    echo "ERROR: micromamba not found at $MMB" >&2
    exit 1
fi

cd /workspace

# Run everything inside the micromamba env so cdo and libs are available
$MMB run -n era5 bash -eu <<'RUN'
set -euo pipefail

RAW_ROOT=/scratch/project_xxxxx/USER/Data/Data_ERA5_tmp/raw
IN_T_DIR=${RAW_ROOT}/t_pl/850
IN_Q_DIR=${RAW_ROOT}/q_pl/850
OUT_DIR=${RAW_ROOT}/thetae_pl/850

YEAR_START=1991
YEAR_END=2020

mkdir -p "$OUT_DIR"

# Make sure cdo is available in the env
if ! command -v cdo >/dev/null 2>&1; then
  echo "ERROR: cdo not found in PATH inside the container/env." >&2
  echo "Tip: add cdo to the 'era5' micromamba env, or use a container that includes it." >&2
  exit 1
fi

# Detect variable names from one sample file (ERA5 often uses 't' and 'q', but your files may differ)
sample_t=$(ls -1 "$IN_T_DIR"/t_pl_850_*.nc 2>/dev/null | head -n 1 || true)
sample_q=$(ls -1 "$IN_Q_DIR"/q_pl_850_*.nc 2>/dev/null | head -n 1 || true)
if [[ -z "$sample_t" || -z "$sample_q" ]]; then
  echo "ERROR: Could not find sample input files in $IN_T_DIR and/or $IN_Q_DIR" >&2
  exit 1
fi

# cdo showname may return multiple names; take the first token
TVAR=$(cdo -s showname "$sample_t" | awk '{print $1}')
QVAR=$(cdo -s showname "$sample_q" | awk '{print $1}')

if [[ -z "$TVAR" || -z "$QVAR" ]]; then
  echo "ERROR: Failed to detect variable names via 'cdo showname'." >&2
  exit 1
fi

echo "Detected variable names: TVAR=$TVAR, QVAR=$QVAR"

echo "Computing theta_e at 850 hPa for years ${YEAR_START}..${YEAR_END}"

# Constants for approximation:
#   r = q/(1-q)
#   theta = T * (1000/850)^0.2854
#   theta_e = theta * exp( (Lv * r) / (Cp * T) )
# Output in Kelvin.
for y in $(seq "$YEAR_START" "$YEAR_END"); do
  tfile="$IN_T_DIR/t_pl_850_${y}.nc"
  qfile="$IN_Q_DIR/q_pl_850_${y}.nc"
  ofile="$OUT_DIR/thetae_pl_850_${y}.nc"

  if [[ ! -f "$tfile" ]]; then
    echo "[WARN] Missing T file: $tfile (skipping year $y)" >&2
    continue
  fi
  if [[ ! -f "$qfile" ]]; then
    echo "[WARN] Missing q file: $qfile (skipping year $y)" >&2
    continue
  fi

  echo "[${y}] -> $ofile"

  # Right-to-left operator evaluation in CDO:
  # merge(T, q) -> expr(thetae_pl_850=...) -> setattribute(units)
  cdo -L -O -f nc4 -z zip_4 \
    -setattribute,thetae_pl_850@units="K" \
    -expr,"thetae_pl_850=${TVAR}*pow(1000/850,0.2854)*exp((2.5e6*(${QVAR}/(1-${QVAR})))/(1004*${TVAR}))" \
    -merge "$tfile" "$qfile" \
    "$ofile"

done

echo "Done. Output written to: $OUT_DIR"
RUN
INNER

# Copy container log back to host log location if it exists
if [[ -f "$LOG_HOST" ]]; then
  :
fi

echo "Finish: $(date)"