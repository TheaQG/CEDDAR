from pathlib import Path
from revision_manuscript_plots.data import legacy


generation = Path(
    "/Users/au728490/Downloads/"
    "samples__final_model/SBGM_SD/models_and_samples/"
    "generated_samples/generation/"
    "B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56"
)

output = Path("/Users/au728490/ceddar_runs/"
              "CEDDAR_figure_tests/"
              "derived_spatial"
              )

product = (legacy.build_ceddar_spatial_summaries(generation_dir=generation, year="2019"))

legacy.save_ceddar_spatial_summaries(product, output, year="2019")

print(f"dates: {len(product['dates'])}")
print("Saved to:", output)