#!/bin/bash
#SBATCH --job-name=install_pkgs
#SBATCH --account=<set-at-submit-time>
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=install_pkgs_%j.out
#SBATCH --error=install_pkgs_%j.err


set -eo pipefail

# --- User/site configuration (override when submitting) ---
# Example:
#   sbatch --account=project_xxxxx \
#     --export=ALL,ROOT_DIR=/scratch/project_xxxxx/$USER/Code/CEDDAR,DATA_DIR=/scratch/project_xxxxx/$USER/Data/Data_DiffMod_small \
#     <script>.sh
ACCOUNT="${SLURM_JOB_ACCOUNT:-${ACCOUNT:-project_xxxxx}}"
USER_BASE="/scratch/${ACCOUNT}/${USER}"
ROOT_DIR="${ROOT_DIR:-${USER_BASE}/Code/CEDDAR}"
CEDDAR_RUNS="${CEDDAR_RUNS:-${USER_BASE}/runs/CEDDAR}"
DATA_BASE="${DATA_BASE:-${USER_BASE}/Data}"
DATA_DIR="${DATA_DIR:-${DATA_BASE}/Data_DiffMod_small}"
CONTAINER="${CONTAINER:-${USER_BASE}/containers/ceddar.sif}"
export ACCOUNT USER_BASE ROOT_DIR CEDDAR_RUNS DATA_BASE DATA_DIR CONTAINER

# === Load Singularity Environment ===
module purge
module use /appl/local/training/modules/AI-20240529/
module load singularity-userfilesystems singularity-CPEbits

# === Define paths ===
CONTAINER="/scratch/project_xxxxxxxxx/containers/images/my_torch_container_with_plotting.sif"
OVERLAY_DIR="/scratch/project_xxxxxxxxx/containers/overlays"
OVERLAY_IMG="$OVERLAY_DIR/my_overlay.img"
OVERLAY_SIZE_MB=5000

# === Create overlay if it doesn't exist ===
mkdir -p "$OVERLAY_DIR"
if [ ! -f "$OVERLAY_IMG" ]; then
    echo "Creating overlay image..."
    singularity overlay create --size $OVERLAY_SIZE_MB "$OVERLAY_IMG"
else
    echo "Overlay already exists."
fi

# === 3. Install packages in the container using the overlay ===
singularity exec --overlay "$OVERLAY_IMG":rw "$CONTAINER" bash -c "
    set -e 
    pip install --upgrade pip && pip install scikit-learn omegaconf"
