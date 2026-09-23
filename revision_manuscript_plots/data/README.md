# Manuscript data loaders

These loaders only read saved artifacts. They do not generate samples, rerun
evaluation, inverse-transform fields, change masks, or apply the figure style.
Call them from the repository root with imports starting `revision_manuscript_plots`.
Panel/figure files are still placeholders; these readers provide their inputs.

## Paths and return values

`paths.py` supplies `CEDDAR_PAPER1_ORIGINAL` and `CEDDAR_PAPER1_REVISION` roots.
The legacy evaluation default is the supplied ATMO path under
`paper1_original/legacy_evaluation/SBGM_SD/models_and_samples/generated_samples/`
`evaluation/B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56/prcp`.
Original generation defaults use the corresponding `revision_inputs/.../generation/`
tree. Revision stage defaults are `paper1_revision/evaluation/<stage>`.

Each evaluation loader returns:

- `tables`: file stem → list of CSV row dictionaries. Values remain strings;
  empty numeric cells are missing, **not zero**. This matches the existing
  `revision_evaluation.plot_common.read_table` / `number` interface.
- `arrays`: NPZ stem → dictionary of NumPy arrays, retaining saved names/shapes.
- `metadata`: metadata file path → parsed JSON/YAML or plain sidecar text.
- `sources`: loaded filenames/paths → absolute paths for figure provenance.
- `directory`, `origin`; revision stages additionally expose `manifest` and,
  when saved, `dates` from `dates.txt`.

No silent date intersection, member pooling or metric renaming occurs. Check
recorded dates, thresholds and subsets before combining legacy and revision panels.
Missing optional artifacts remain absent; required missing files raise an error.
Revision stage loaders require a complete manifest for the requested stage.

```python
from revision_manuscript_plots.data import legacy, revision

psd = legacy.load_psd()
k = psd['arrays']['scale_psd_curves']['k']
daily = revision.load_deterministic()['tables']['daily_continuous_metrics']
crps = revision.load_probabilistic()['tables']['crps']
objects = revision.load_morphology()['tables']['objects_absolute']
dry = revision.load_dry_bias()['tables']['ensemble_member_decomposition']

examples = legacy.load_example_fields(
    ['20190103', '20190104'], baseline_dirs=legacy.DEFAULT_BASELINES)
fields = examples['dates']['20190103']['fields']
# fields: danra, era5_condition, ceddar_mean, ceddar_median, ceddar_pmm,
#         era5_bilinear and qm (the last two require baseline_dirs).
members = examples['dates']['20190103']['ensemble']  # [M,H,W], all saved members
valid = examples['dates']['20190103']['valid']
```

Example fields use physical mm/day artifacts and preserve the saved PMM. Mean
and median are calculated from physical members. Reference arrays are checked
against requested baselines with rtol=atol=0.001, matching the existing revision
configuration. No regridding is performed and no latitude/longitude is invented.
Land and finite-support masks are returned separately; arrays are not masked in
place. Dates must actually exist in the selected generation folder.

For baseline PSD/distributions/spatial maps, call the same legacy loader with the
baseline's **evaluation** directory. A legacy loader accepts a model evaluation
root, `prcp/`, the particular metric-family directory, or its `tables/` directory.
Revision stage overrides point directly to a raw stage directory, not a figure folder.

```python
qm_psd = legacy.load_psd('/actual/QM/evaluation/prcp')
sigma_old = legacy.load_sigma_star()  # original sigma_control tables + metadata
sigma_new = revision.load_sigma_star('/actual/sigma_comparison/legacy')
# Select a specific mode/run: no latest-run guessing or cross-mode merging.
```

## Inputs for the sketched figures

| Figure content | Loader |
|---|---|
| Example maps / reference differences | `legacy.load_example_fields(dates, ...)` |
| Seasonal histograms | `legacy.load_seasonal_distributions()`; `season_indices` selects rows in `dist_daily` |
| Tails and wet-event summaries | `legacy.load_extremes()`; retain threshold/basis metadata |
| Daily deterministic errors / detection | `revision.load_deterministic()` |
| PSD plus object morphology / SAL | `legacy.load_psd()` and `revision.load_morphology()` |
| CRPS / coverage / reliability / rank / binned spread-skill | `revision.load_probabilistic()` |
| Historical continuous PIT / saved spread-skill scatter inputs | `legacy.load_probabilistic()` |
| Annual maps and dry-bias decomposition | `legacy.load_spatial()` and `revision.load_dry_bias()` |
| Sigma* main, extremes and STD/SEM panels | `legacy.load_sigma_star()` or `revision.load_sigma_star(run_dir)` |

Important distinctions when writing the panels:

- Legacy `scale_psd_curves.psd_gen` is PMM PSD; `psd_gen_ens_mean` is the
  member-averaged PSD, **not** PSD of the ensemble mean. Sigma* PSD is of the
  ensemble mean. Do not give these the same label or directly mix their estimators.
- Legacy `dist_daily.counts_gen` contains **PMM** histograms. The saved pooled
  ensemble histogram is separate and has no seasonal member breakdown. A seasonal
  ensemble curve needs additional aggregation from members; the loader does not
  invent one or relabel PMM as ensemble.
- Revision rank histograms do not supply continuous PIT values, and binned
  spread-skill does not supply the original scatterplot's individual points.
- Annual QM/median/PMM maps are returned only if present in the chosen spatial
  tables. A saved accumulation may cover missing dates rather than a full year;
  preserve `*.meta.txt` sample counts. Do not label absent maps as available.
- Historical sigma* metadata may describe requested settings, not the executed
  sampler. Revision metadata keeps `generation` and `requested_control` distinct.
- For morphology, calculate per-member absolute count error **before** averaging
  members within dates. Undefined object fractions and SAL are not zero errors.
- The supplied preprocessing illustration is an external PNG, not evaluation data;
  read it directly with `matplotlib.image.imread(path)` in the supplementary figure.
  The data-overview dates must come from a folder that actually contains them.
