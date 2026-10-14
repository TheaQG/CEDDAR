from pathlib import Path

import numpy as np

from revision_manuscript_plots.data import legacy


legacy_eval = Path(
    "/Users/au728490/Downloads/"
    "B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__"
    "LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56/prcp"
)

qm_generation = Path(
    "/Users/au728490/Downloads/"
    "samples__final_model/SBGM_SD/models_and_samples/"
    "generated_samples/generation/baselines/qm/test"
)

output = Path(
    "/Users/au728490/ceddar_runs/"
    "CEDDAR_figure_tests/qm_land_histograms.npz"
)

# ------------------------------------------------------------------
# Use the existing legacy distribution product so that the corrected
# QM histograms use exactly the same dates and bin edges as Figure 2.
# ------------------------------------------------------------------

dist = legacy.load_seasonal_distributions(
    legacy_eval
)

daily = dist["arrays"]["dist_daily"]

dates = np.asarray(
    daily["dates"]
).astype(str)

bins = np.asarray(
    daily["bins"],
    dtype=float,
)

# ------------------------------------------------------------------
# Use the QM evaluation land mask.
# ------------------------------------------------------------------

land = legacy.load_land_mask(
    qm_generation
)

# ------------------------------------------------------------------
# Rebuild QM histograms using Danish land pixels only.
# ------------------------------------------------------------------

product = legacy.build_qm_histograms(
    dates=dates,
    qm_generation_dir=qm_generation,
    bins=bins,
    land_mask=land,
)

legacy.save_qm_histograms(
    product,
    output,
)

print(
    "dates:",
    len(product["dates"]),
)

print(
    "bins:",
    (
        product["bins"][0],
        product["bins"][-1],
    ),
)

print(
    "daily count range:",
    (
        product["counts"].sum(axis=1).min(),
        product["counts"].sum(axis=1).max(),
    ),
)

print(
    "saved:",
    output,
)