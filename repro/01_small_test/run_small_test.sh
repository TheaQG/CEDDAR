#!/usr/bin/env bash
# Setup/inference only: never trains. All artifacts go to an external run directory.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -m repro.smoke "$@"
