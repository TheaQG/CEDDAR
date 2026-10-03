#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then
    echo 'Usage: bash diagnose_inputs.sh settings.recovery.env' >&2
    exit 2
fi
source "$1"
: "${REPO_DIR:?}" "${CAMPAIGN_DIR:?}" "${RECOVERY_DIR:?}" "${INSPECTION_REPORT:?}" "${CPU_THREADS:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEVICE=cpu MPLBACKEND=Agg
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
exec "${PYTHON:-python}" "$HERE/diagnose_inputs.py"
