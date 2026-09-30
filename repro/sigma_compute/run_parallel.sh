#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
    echo 'Usage: bash run_parallel.sh SETTINGS_FILE init|run|worker|evaluate|audit [indices...]' >&2
    exit 2
fi
settings="$1"
shift
source "$settings"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export DEVICE=cpu MPLBACKEND=Agg
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
exec "$PYTHON" "$HERE/campaign.py" "$@"
