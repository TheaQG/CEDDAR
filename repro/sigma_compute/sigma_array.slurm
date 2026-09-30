#!/usr/bin/env bash
# Optional adapter only. Your current server uses the Bash launcher instead.
# Submit with --array=0-9%2 --cpus-per-task=4 and site-specific memory/time/partition.
# First initialize the shared campaign with run_parallel.sh SETTINGS init.
# Example: sbatch [site options] sigma_array.slurm /absolute/sigma_compute /absolute/settings.env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=sigma-%A_%a.log
set -euo pipefail
export CPU_THREADS="${SLURM_CPUS_PER_TASK:?}"
export CPU_BUDGET="$CPU_THREADS"
export MAX_JOBS=1
srun --cpu-bind=cores bash "$1/run_parallel.sh" "$2" worker "${SLURM_ARRAY_TASK_ID:?}"
