tmux new-session -s ceddar-sigma /bin/tcsh

cd /home/theaqg/CEDDAR_GMD_revision
source /home/theaqg/envs/ceddar-revision/bin/activate.sh

# Create fresh timestamped output directory
unsetenv COMPARISON_DIR

# Explicit settings for full comparison
setenv MAX_DATES 1000
setenv ENSEMBLE_SIZE 32
setenv SIGMA_SEED 504
setenv CPU_THREADS 4
setenv PYTHONUNBUFFERED 1
setenv SIGMA_STAR_GRID "0.95 1.00 1.05"

bash repro/run_sigma_initialization_comparison.sh