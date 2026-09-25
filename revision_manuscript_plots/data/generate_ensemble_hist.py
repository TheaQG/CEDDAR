from pathlib import Path

from revision_manuscript_plots.data import legacy

legacy_eval = Path(
    "/Users/au728490/Downloads/"
    "B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__"
    "LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56/prcp"
)

generation = Path(
    "/Users/au728490/Downloads/"
    "samples__final_model/SBGM_SD/models_and_samples/"
    "generated_samples/generation/"
    "B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__"
    "LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56"
)

dist = legacy.load_seasonal_distributions(legacy_eval)

saved = dist["arrays"]["dist_daily"]

product = legacy.build_ensemble_histograms(
    dates=saved["dates"],
    generation_dir=generation,
    bins=saved["bins"],
    land_only=True,
)

output = Path(
    "/Users/au728490/ceddar_runs/"
    "CEDDAR_figure_tests/"
    "ceddar_member_histograms.npz"
)

legacy.save_ensemble_histograms(product, output,)

print(output)
print("dates:", product["counts_by_member"].shape[0],)
print("members:", product["counts_by_member"].shape[1],)
print("bins:", product["counts_by_member"].shape[2],)
