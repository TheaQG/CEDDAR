#!/usr/bin/env bash
# Run from tcsh using bash. Reuse saved ensembles and Group 1's frozen dates.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# Conservative CPU default; no GPU, checkpoint or inference is required.
export OMP_NUM_THREADS="${CPU_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

exec "${PYTHON:-python}" -m revision_evaluation.run_dry_bias "$@"
