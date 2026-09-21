#!/usr/bin/env bash
# Table-only plots. Usage: bash repro/plot_revision.sh probabilistic --input-dir ...
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg

case "${1:-}" in
    deterministic|probabilistic|dry_bias|morphology) GROUP="$1"; shift ;;
    *) echo "Usage: $0 {deterministic|probabilistic|dry_bias|morphology} --input-dir METRIC_DIRECTORY [--output-dir NEW_EXTERNAL_DIRECTORY]" >&2; exit 2 ;;
esac

exec "${PYTHON:-python}" -m "revision_evaluation.plot_${GROUP}" "$@"
