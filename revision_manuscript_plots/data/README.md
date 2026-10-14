# Final manuscript data inputs

The final figure runner is
[`bash_paper1/run_manuscript_figures.sh`](../../bash_paper1/run_manuscript_figures.sh).
It passes explicit paths from `DATA_ROOT` to the saved-data loaders. Use the
[compact archive workflow](../../repro/MANUSCRIPT_FIGURE_DATA.md) for checksums and
verified reproduction of Figures 2-7 and S5-S10. Figure 1 and S1-S4 were manually
prepared.

## Input layout

| Archive subtree | Final figure inputs |
| --- | --- |
| `P1_F_samples/` | Two selected physical example ensembles/references/PMM and saved land masks |
| `P1_legacy/` | Daily distributions, CEDDAR/QM PSD, 2019 accumulation maps and temporal tables |
| `P1_revision/evaluation/` | Deterministic, dry-bias, probabilistic, morphology, common-event ISS, daily mean bias and corrected histograms |
| `P1_sigma_star/` | Full-test/extended-validation metrics, PSD, metadata and selected example bundles |

The unchanged loaders may read more saved tables/sidecars than a panel displays.
The archive retains those dependencies and shell preflight inputs. Only the
sigma* example bundles actually selected by the full-data run are retained;
output equality was checked after staging.

## Loader interface

`legacy.py` and `revision.py` read saved artifacts; they do not generate model
samples or rerun the evaluation stages. CSV bundles retain strings and empty
numeric cells (undefined values); NPZ bundles retain saved array keys/shapes.
They also return `metadata`, `sources`, `directory` and `origin`. Revision stages
require a complete stage manifest and expose its frozen dates when present.
No silent date intersection, metric renaming, field conversion or regridding occurs.
Physical example loaders calculate mean/median from the saved members and retain
saved PMM. Land and finite-support masks are returned separately.

For standalone use, supply explicit paths:

```python
from pathlib import Path
from revision_manuscript_plots.data import legacy, revision

root = Path('/absolute/path/to/CEDDAR_manuscript_figure_data')
deterministic = revision.load_deterministic(root / 'P1_revision/evaluation/deterministic')
generation = (root / 'P1_F_samples/SBGM_SD/models_and_samples/generated_samples/generation'
              / legacy.MODEL_NAME)
examples = legacy.load_example_fields(['20191011', '20200926'], generation_dir=generation)
```

Legacy evaluation readers accept a model/prcp root, metric-family directory or
its `tables/` directory. Revision readers take the raw stage directory. Without
explicit paths, historical defaults come from `CEDDAR_PAPER1_ORIGINAL` and
`CEDDAR_PAPER1_REVISION` in `paths.py`; setting `DATA_ROOT` alone configures the
shell runner, not those standalone Python defaults. Baseline example overrides
must be a method-to-generation-directory mapping, not an evaluation directory.

## Scientific distinctions to preserve

- Legacy PSD `psd_gen` is PMM PSD; `psd_gen_ens_mean` is the member-averaged PSD,
  not the PSD of the ensemble mean. Sigma* PSD is of the ensemble mean.
- Legacy `dist_daily.counts_gen` contains PMM histograms. Final ensemble/QM
  distributions use their separately saved histogram products. Do not relabel
  PMM histograms as ensemble distributions or invent seasonal member breakdowns.
- Rank histograms and PIT are different saved products. The final spread-skill
  panel uses saved pooled bins and the separately derived daily scatter inputs.
- Accumulation maps cover their recorded available dates; preserve saved sample
  counts and metadata rather than assuming a complete year. Missing QM/median/PMM
  products are not substituted automatically.
- Saved examples use physical mm/day. Reference tolerances document storage
  differences without modifying fields. Land/finite masks remain separate;
  unsupported grid shapes fail rather than triggering automatic regridding.
- Sigma* requested settings are not proof of effective sampler behavior. Preserve
  generation/evaluation metadata, explicit run selection and saved grids.
- Morphology takes absolute object-count errors before averaging members; undefined
  fractions/SAL are not zero errors. See [morphology details](../../revision_evaluation/MORPHOLOGY.md).

The `generate_*.py` scripts derive additional diagnostics from full saved fields.
They are separate from archive-only plotting; inspect their input/output options
and any historical local paths before use. `investigate_qm_bias.py` is retained as
an investigation utility. Existing scientific calculations are unchanged.
