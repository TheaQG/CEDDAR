#!/usr/bin/env bash

set -euo pipefail


# =============================================================================
# CEDDAR GMD revision — manuscript figures
# Generates:
#   Main Figures 02-07
#   Supplementary Figures S05 - S10
# =============================================================================

# Usage (activate the figure environment first):
#   DATA_ROOT=/path/to/CEDDAR_manuscript_figure_data \
#   OUT=/path/to/output bash /path/to/CEDDAR/bash_paper1/run_manuscript_figures.sh
# REPO defaults to this checkout; OUT defaults to a sibling output directory.
# Relative overrides are resolved from the directory where this script is invoked.
REPO="$(cd "${REPO:-$(dirname "${BASH_SOURCE[0]}")/..}" && pwd)"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the manuscript figure-data directory}"
OUT="${OUT:-$REPO/../CEDDAR_manuscript_figures}"

case "$DATA_ROOT" in
    /*) ;;
    *) DATA_ROOT="$PWD/$DATA_ROOT" ;;
esac
case "$OUT" in
    /*) ;;
    *) OUT="$PWD/$OUT" ;;
esac

LEGACY_ROOT="$DATA_ROOT/P1_legacy"
REVISION="$DATA_ROOT/P1_revision/evaluation"
SAMPLES_ROOT="$DATA_ROOT/P1_F_samples"
SIGMA_ROOT="$DATA_ROOT/P1_sigma_star"

MODEL="B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56"

F_EVAL="$LEGACY_ROOT/F_evaluation/prcp"
BILINEAR_EVAL="$LEGACY_ROOT/baselines/bilinear/test/prcp"
QM_EVAL="$LEGACY_ROOT/baselines/qm/test/prcp"

GENERATION_ROOT="$SAMPLES_ROOT/SBGM_SD/models_and_samples/generated_samples/generation"
GENERATION_DIR="$GENERATION_ROOT/$MODEL"

DETERMINISTIC="$REVISION/deterministic"
DRY_BIAS="$REVISION/dry_bias"
MORPHOLOGY="$REVISION/morphology"
PROBABILISTIC="$REVISION/probabilistic"
ISS="$REVISION/iss"
DAILY_MEAN_BIAS="$REVISION/daily_mean_bias"

TEMPORAL="$F_EVAL/temporal/tables"

SIGMA_TEST="$SIGMA_ROOT/full_test"
SIGMA_VALID="$SIGMA_ROOT/extended_validation"

CACHE="$OUT/_derived"
DISTRIBUTIONS="$REVISION/distributions"
ENSEMBLE_HIST="$DISTRIBUTIONS/ceddar_member_histograms.npz"
QM_HIST="$DISTRIBUTIONS/qm_land_histograms.npz"

DERIVED_SPATIAL="$DRY_BIAS/derived_spatial"


require_dir () {
    if [[ ! -d "$1" ]]; then
        echo
        echo "ERROR: Required directory does not exist:"
        echo "  $1"
        echo
        exit 1
    fi
}
require_file () {
    if [[ ! -f "$1" ]]; then
        echo
        echo "ERROR: Required file does not exist:"
        echo "  $1"
        echo
        exit 1
    fi
}

cd "$REPO"

mkdir -p "$OUT"
mkdir -p "$CACHE"


# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------

require_dir "$F_EVAL"
require_dir "$BILINEAR_EVAL"
require_dir "$QM_EVAL"
require_dir "$GENERATION_DIR"

require_dir "$DISTRIBUTIONS"
require_file "$ENSEMBLE_HIST"
require_file "$QM_HIST"

require_dir "$DERIVED_SPATIAL"

require_dir "$ISS"
require_file "$ISS/iss_summary.csv"
require_file "$ISS/iss_threshold_counts.csv"

require_dir "$TEMPORAL"
require_file "$TEMPORAL/temporal_metrics_2019.npz"
require_file "$TEMPORAL/temporal_series_2019.npz"

require_dir "$DAILY_MEAN_BIAS"
require_file "$DAILY_MEAN_BIAS/daily_mean_bias.csv"
require_file "$DAILY_MEAN_BIAS/daily_mean_bias_summary.csv"

require_dir "$DETERMINISTIC"
require_dir "$DRY_BIAS"
require_dir "$MORPHOLOGY"
require_dir "$PROBABILISTIC"

require_dir "$SIGMA_TEST"
require_dir "$SIGMA_VALID"


echo
echo "============================================================"
echo "Generating CEDDAR manuscript figures"
echo "Data root: $DATA_ROOT"
echo "Output:    $OUT"
echo "============================================================"
echo

# =============================================================================
# Figure 02 - samples and distributions
# =============================================================================

echo "[1/6] Figure 02 - samples and distributions"

python -m revision_manuscript_plots.figures.fig_samples_distributions \
  --legacy-eval "$F_EVAL" \
  --generation-dir "$GENERATION_DIR" \
  --ensemble-histograms "$ENSEMBLE_HIST" \
  --qm-histograms "$QM_HIST" \
  --tails-mode bars \
  --output-dir "$OUT"

echo "      done."


# =============================================================================
# Figure 03 - deterministic performance
# =============================================================================

echo "[2/6] Figure 03 - deterministic performance"

python -m revision_manuscript_plots.figures.fig_deterministic \
  --revision-dir "$DETERMINISTIC" \
  --output-dir "$OUT"

echo "      done."


# =============================================================================
# Figure 04 - dry-bias decomposition
# =============================================================================

echo "[3/6] Figure 04 - dry-bias decomposition"

python -m revision_manuscript_plots.figures.fig_dry_bias \
  --revision-dir "$DRY_BIAS" \
  --spatial-dir "$F_EVAL/spatial/tables" \
  --qm-spatial-dir "$QM_EVAL/spatial/tables" \
  --derived-spatial-dir "$DERIVED_SPATIAL" \
  --generation-dir "$GENERATION_DIR" \
  --spatial-mode sum \
  --bias-type absolute \
  --output-dir "$OUT"

echo "      done."


# =============================================================================
# Figure 05 - probabilistic and calibration
# =============================================================================

echo "[4/6] Figure 05 - probabilistic evaluation"

python -m revision_manuscript_plots.figures.fig_probabilistic \
  --revision-dir "$PROBABILISTIC" \
  --histogram rank \
  --output-dir "$OUT"

echo "      done."


# =============================================================================
# Figure 06 - morphology
# =============================================================================

echo "[5/6] Figure 06 - morphology"

python -m revision_manuscript_plots.figures.fig_spatial_morphology \
  --legacy-eval "$F_EVAL" \
  --qm-eval "$QM_EVAL" \
  --revision-dir "$MORPHOLOGY" \
  --sal-threshold 1.0 \
  --output-dir "$OUT"

echo "      done."


# =============================================================================
# Figure 07 - sigma star
# =============================================================================

echo "[6/6] Figure 07 - sigma* sensitivity"

python -m revision_manuscript_plots.figures.fig_sigma_star \
  --eval-dir "$SIGMA_TEST" \
  --example-category heavy_mean \
  --example-sigmas \
      0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 \
  --spatial-mode outline \
  --output "$OUT/fig_sigma_star"

echo "      done."


# =============================================================================
# Supplementary Figure S5 - extended-validation sigma* sweep
# =============================================================================

echo "[supplement] Figure S5 - extended-validation sigma* sweep"

python -m revision_manuscript_plots.figures.fig_sigma_star \
  --eval-dir "$SIGMA_VALID" \
  --example-category heavy_mean \
  --example-sigmas \
      0.55 0.70 0.85 1.00 1.15 1.30 1.45 \
  --spatial-mode outline \
  --output "$OUT/figS05_sigma_star_extended_validation"

echo "      done."


# =============================================================================
# Supplementary Figure S6 - sigma* uncertainty
# =============================================================================

echo "[supplement] Figure S6 - sigma* uncertainty"

python -m revision_manuscript_plots.figures.figS_sigma_star_uncertainty \
  --eval-dir "$SIGMA_VALID" \
  --output "$OUT/figS06_sigma_star_uncertainty"

echo "      done."


# =============================================================================
# Supplementary Figure S7 - intensity-scale skill
# =============================================================================

echo "[supplement] Figure S7 - intensity-scale skill"

python -m revision_manuscript_plots.figures.figS_iss \
  --revision-dir "$ISS" \
  --output "$OUT/figS07_iss"

echo "      done."


# =============================================================================
# Supplementary Figure S8 - seasonal CRPS
# =============================================================================

echo "[supplement] Figure S8 - seasonal CRPS"

python -m revision_manuscript_plots.figures.figS_seasonal_crps \
  --revision-dir "$PROBABILISTIC" \
  --output "$OUT/figS08_seasonal_crps"

echo "      done."


# =============================================================================
# Supplementary Figure S9 - temporal persistence
# =============================================================================

echo "[supplement] Figure S9 - temporal persistence"

python -m revision_manuscript_plots.figures.figS_temporal_persistence \
  --temporal-dir "$TEMPORAL" \
  --year 2019 \
  --wet-threshold-mm 1.0 \
  --output "$OUT/figS09_temporal_persistence"

echo "      done."


# =============================================================================
# Supplementary Figure S10 - daily mean precipitation bias
# =============================================================================

echo "[supplement] Figure S10 - daily mean precipitation bias"

python -m revision_manuscript_plots.figures.figS_daily_mean_bias \
  --revision-dir "$DAILY_MEAN_BIAS" \
  --rolling-window 14 \
  --output "$OUT/figS10_daily_mean_bias"

echo "      done."


# -----------------------------------------------------------------------------
# Finished
# -----------------------------------------------------------------------------

echo
echo "============================================================"
echo "All manuscript figures generated successfully."
echo "Output directory:"
echo "$OUT"
echo "============================================================"
