#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
    echo 'Usage: bash run_recovery.sh SETTINGS_FILE plan|run|evaluate [task indices]' >&2
    exit 2
fi
source "$1"
shift
: "${RECOVERY_DIR:?Add a fresh RECOVERY_DIR to the settings file}"
: "${INSPECTION_REPORT:?Set INSPECTION_REPORT to extended-recovery-inspection.json}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEVICE=cpu MPLBACKEND=Agg
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
exec "${PYTHON:-python}" "$HERE/recovery.py" "$@"
