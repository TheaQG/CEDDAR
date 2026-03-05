#!/usr/bin/env bash
#SBATCH --job-name=net_test
#SBATCH --account=project_465002493
#SBATCH --partition=small
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=net_test.out
#SBATCH --error=net_test.err

module purge
module load lumi-container-wrapper

srun bash -lc 'curl -I https://cds.climate.copernicus.eu'