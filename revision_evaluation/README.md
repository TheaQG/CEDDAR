# Revised evaluation of saved precipitation fields

This package computes deterministic, probabilistic, dry-bias and morphology
products from saved physical CEDDAR and benchmark fields. Model inference,
checkpoint loading and QM fitting are separate workflows. The scientific
implementations, metric definitions and manuscript samples are frozen.

For **final manuscript figures**, use
[`run_manuscript_figures.sh`](../bash_paper1/run_manuscript_figures.sh) and the
[compact figure-data archive](../repro/MANUSCRIPT_FIGURE_DATA.md). Recomputing the
diagnostics below requires the original generated fields, which are not all
included in that compact archive.

## Inputs and date selection

Copy [`atmo.yaml`](atmo.yaml) to your run directory and configure its input and
output paths for your machine. Preserve the scientific settings for manuscript
reproduction. Each method declares its physical units and layout:

- CEDDAR: `ensembles_phys/`, `lr_hr_phys/` and saved `pmm_phys/` fields.
- Baselines: `physical` uses `pmm_phys/` and `lr_hr_phys/`;
  `legacy_physical` uses `pmm/` and `lr_hr/`. Directory names alone do not prove units.
- Land masks: dated `lsm/YYYYMMDD.npz` takes precedence over `meta/land_mask.npz`.
  Accepted keys are `lsm_hr`, `lsm`, `mask`, `roi`; masks must be finite binary 0/1.
  A corrupt dated mask fails rather than falling back. An optional ROI must share
  the grid. All methods' masks and saved DANRA references must agree.

Preflight checks every common date and member. It records input hashes, units,
reference tolerances, finite support and missing dates without computing metrics:

```bash
bash repro/run_revision_deterministic.sh \
  --config /absolute/path/to/evaluation.yaml --preflight-only
```

The configured period is 2019-2020 (731 calendar dates). The actual shared date
set is recorded; missing dates are reported. Freeze the preflight `dates.txt` for
the deterministic pass, then reuse it for all later stages:

```bash
bash repro/run_revision_deterministic.sh \
  --config /absolute/path/to/evaluation.yaml \
  --dates-file /absolute/path/to/preflight_TIMESTAMP/dates.txt

bash repro/run_revision_probabilistic.sh \
  --config /absolute/path/to/evaluation.yaml \
  --dates-file /absolute/path/to/deterministic/dates.txt

bash repro/run_revision_dry_bias.sh \
  --config /absolute/path/to/evaluation.yaml \
  --dates-file /absolute/path/to/deterministic/dates.txt

bash repro/run_revision_morphology.sh \
  --config /absolute/path/to/evaluation.yaml \
  --dates-file /absolute/path/to/deterministic/dates.txt
```

Relative paths resolve against the selected YAML directory. A supplied date list
must match the complete current common set; there is no silent per-method or
per-date omission. The manuscript expects 32 members. Smaller ensembles are for
explicit synthetic fixtures, not replacement manuscript samples.

## Products and interpretation

| Stage | Principal saved products |
| --- | --- |
| Deterministic | Daily bias/MAE/RMSE/correlation; pooled occurrence and event-detection metrics; daily sufficient counts |
| Probabilistic | CRPS, central coverage/width, reliability, randomized ranks and pooled spread-skill tables |
| Dry bias | Own-wet-mask occurrence/intensity and quantiles, including separate member/season summaries |
| Morphology | Absolute/equal-area object tables and physical-threshold SAL; see [details](MORPHOLOGY.md) |

Each stage retains `manifest.json`, `dates.txt`, `resolved_config.yaml`,
`daily_support.csv` and `input_files.csv` as applicable. Successful manifests say
`status: complete`; failed runs retain partial diagnostics and failure status.
Existing stage directories are refused. Use a fresh external `--output-root`
for another run, preserving the intended dates. Inputs are checked for size/mtime
changes before completion, with SHA256 recorded for subsequent verification.

All precipitation is mm/day. Daily support intersects land/ROI with finite DANRA,
all selected predictions and every ensemble member. Methods therefore use the
same pixels each day. Undefined statistics remain NaN (empty numeric CSV cells),
not zero. Negative precipitation is retained and reported, not clipped. Reference
matching uses the configured tolerances without changing values or reprojecting;
matching arrays are evidence of alignment, not a coordinate-system proof.

Continuous errors describe dates. Occurrence/detection pool valid pixel-days,
with inclusive thresholds and NaN for zero denominators; daily ratios are not
averaged. Empty seasons retain zero counts and undefined statistics. Physical
ensemble mean/median use float64, with the linear median for even member counts.
PMM is loaded as saved: the original producer constructs it in model space and
back-transforms it. Recomputing PMM from physical members would change the method.
The manuscript and Supplementary Information provide the full scientific methods.

Additional daily correlation, daily spread-skill, common-event ISS, histogram and
spatial products are generated under
[`revision_manuscript_plots/data/`](../revision_manuscript_plots/data/).

## Optional diagnostic plots

`plot_probabilistic.py` retains interval-width, observed-wet binned spread-skill
and bin-population diagnostics beyond the final manuscript panels.
`plot_morphology.py` retains object/SAL diagnostics and exports date-level summaries.
These are standalone diagnostic layouts, separate from the final figure runner:

```bash
bash repro/plot_revision.sh probabilistic \
  --input-dir /absolute/path/to/evaluation/probabilistic \
  --output-dir /absolute/path/to/new-diagnostic-plots
```

Use `morphology` for object/SAL plots. Outputs must be new external directories.
The final manuscript morphology panels also import `date_summaries` and
`common_dates` from `plot_morphology.py`; keep that module and `plot_common.py`.

## Lightweight checks

```bash
python -m unittest revision_evaluation.tests.test_deterministic \
  revision_evaluation.tests.test_probabilistic \
  revision_evaluation.tests.test_dry_bias \
  revision_evaluation.tests.test_object_metrics -v
```

These checks cover numerical definitions, empty populations, shared masks and
dates, physical-only input validation, source preservation, output guards and
retained launcher/plot integration. They use small synthetic fixtures and need
no archived checkpoint, source reanalysis download or GPU.
