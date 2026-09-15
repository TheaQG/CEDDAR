#!/usr/bin/env bash
# Reserved level: never report an empty script as a successful smoke test.
set -euo pipefail
printf '%s\n' 'Level 03 is not implemented. Use level 01/02 for inference checks; level 04 deliberately trains.' >&2
exit 2
