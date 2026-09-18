# GMD revision evaluation — Group 1

This package evaluates saved precipitation fields. It does not load a model,
generate samples, fit QM, alter physical values, or call the legacy evaluation
or plotting runners. Group 1 implements deterministic metrics and common input
validation; probabilistic calibration, dry-bias decomposition, morphology and
figures will be separate additions. No new dependencies are required.

## Run on ATMO

Use the existing environment and the `paper1-revision-2026` checkout. The supplied
`atmo.yaml` contains the three original input paths provided for this revision.
Review its unit declarations and reference tolerances before the full evaluation.

From a **tcsh** terminal (preferably inside tmux):

```tcsh
cd /home/theaqg/CEDDAR_GMD_revision
source /home/theaqg/envs/ceddar-revision/bin/activate.csh
bash repro/run_revision_deterministic.sh --preflight-only
```

Preflight reads **every common date**, including all ensemble members, without
calculating metric tables. It hashes the files read and checks arrays, keys,
member counts, shapes, saved DANRA references and binary land masks. It creates
`evaluation/preflight_<UTC timestamp>/` and prints its full path. Inspect:

- `manifest.json`: completion status, Git/runtime, methods, thresholds, mask
  policy, member count, reference tolerances, units declarations, date/pixel counts.
- `date_inventory.json`: common 2019–2020 dates, dates missing from each artifact
  folder, and dates outside the evaluation period.
- `daily_support.csv`: valid/common land pixels and excluded nonfinite pixels.
- `input_files.csv`: exact source paths, byte sizes, modification times and SHA256.
- `dates.txt` and `resolved_config.yaml`: inputs for reproducing the same analysis.

Freeze that date selection for the calculation (replace the example path):

```tcsh
setenv REVISION_DATES /home/theaqg/CEDDAR_runs/paper1_revision/evaluation/preflight_TIMESTAMP/dates.txt
bash repro/run_revision_deterministic.sh --dates-file "$REVISION_DATES"
```

Results go to
`/home/theaqg/CEDDAR_runs/paper1_revision/evaluation/deterministic/`.
The final manifest must say `"status": "complete"`. A failed run retains its error
and partial diagnostics. The runner **refuses an existing deterministic directory**;
use a new `--output-root /home/theaqg/CEDDAR_runs/paper1_revision/evaluation_retry`
for another attempt. Do not delete original generation results.

`--config /path/to/settings.yaml` selects a different configuration. Relative
paths in that YAML resolve against its directory. `--output-root` and `--dates-file`
also resolve against the YAML directory if relative; absolute ATMO paths avoid
ambiguity. An input method can be omitted explicitly from `methods`; there is no
silent per-date omission. CEDDAR remains the reference/ensemble source. If no
`--dates-file` is supplied, the runner selects the current complete common date
intersection and writes it. Later analyses should reuse that list, not rescan a
different method/date population. A supplied list must match the current common
set exactly; added or missing files require deliberate review.

## Definitions and output tables

All precipitation is mm/day. The selected methods are ERA5 bilinear, QM, CEDDAR
ensemble mean, CEDDAR ensemble median and saved CEDDAR PMM. Exactly 32 members are
expected by default. A declared smaller `expected_members` supports synthetic
fixtures only; the runner warns that it is not the manuscript ensemble.

The shared daily support is land (intersected with any explicit ROI), finite DANRA,
finite predictions from **all selected methods**, and finite values in **all ensemble
members**. Thus every method uses the same pixels that day. Pixel counts can still
vary between days. There is no renormalization, clipping or NaN filling. Negative
valid field values are retained, counted and warned about. Zero valid cases yield
NaN statistics, saved as **empty numeric CSV cells**, with explicit zero counts.

| Table | Definition |
|---|---|
| `daily_continuous_metrics.csv` | One row per date/season/method: spatial bias (prediction minus DANRA), MAE, RMSE, Pearson correlation and valid pixel count. No space–time pooling. Constant-field or fewer-than-two-pixel correlation is NaN. |
| `occurrence_metrics.csv` | DANRA and every selected method: pooled >=1 mm/day frequency, DANRA frequency and their difference, wet/valid pixel-day counts; ALL/DJF/MAM/JJA/SON. |
| `event_detection_metrics.csv` | Each method at >=1/10/20 mm/day: pooled hits, misses, false alarms, correct negatives, POD/FAR/CSI and sample/event/date counts. No averaging daily ratios. |
| `occurrence_daily_counts.csv` | Daily sufficient counts, retained for shared dry-bias calculations and possible date-level resampling. |
| `event_detection_daily_counts.csv` | Daily contingency counts for possible paired, whole-date bootstrap calculations. |
| `daily_support.csv` | Per-date land and finite common support, including excluded nonfinite pixels. |

POD = H/(H+M), FAR = F/(H+F), CSI = H/(H+M+F); zero denominators yield NaN.
`n_dates` includes dates with no finite support; `n_valid_dates` counts dates with
positive support. Occurrence pools valid land **pixel-days**, not daily frequencies.
An absent season is retained with zero counts and NaN frequency. No confidence
intervals are calculated in Group 1. Pixels are not treated as independent
replicates; the primary continuous-metric distributions are across dates. The two
years provide descriptive seasonal diagnostics, not climatological inference.

Mean and median are calculated from physical members using float64 arithmetic.
The median uses the linear 0.5 quantile (average of the two middle members for
M=32). PMM is loaded unchanged: the producer constructs PMM in model space and
then back-transforms it. Recomputing PMM from physical members would change that
method definition. Saved baseline fields are already on the target grid and are
not interpolated again.

## Input validation and its limits

The existing `EvalDataResolver` and `BaselineDataResolver` load fields; strict
prechecks disable their ambiguous model-space fallbacks. CEDDAR requires
`ensembles_phys/`, `lr_hr_phys/`, and `pmm_phys/` when PMM is selected. Baselines
explicitly declare either `physical` (`pmm_phys/`, `lr_hr_phys/`) or
`legacy_physical` (`pmm/`, `lr_hr/`). The latter is appropriate for the original
unscaled baseline producer; directory names alone do not establish units.

Land masks are checked separately because of legacy resolver bugs. Per-date
`lsm/YYYYMMDD.npz` takes precedence over `meta/land_mask.npz`; accepted keys are
`lsm_hr`, `lsm`, `mask`, `roi`. A supplied ROI is a separate NPZ on the same grid.
Only finite binary 0/1 masks are accepted. A corrupt existing per-date mask is an
error, not an invitation to fall back. All methods' land masks must agree.

Saved DANRA references must have matching shapes, finite support and values within
the configured tolerances. The example permits rtol=0.001 and atol=0.001 mm/day for
float16 versus float32 storage; these tolerances never modify a field. Matching
references/masks are evidence for alignment, **not a coordinate-system proof**.
The reader does not reproject data or infer units from its magnitude. Configuration
unit declarations and source metadata hashes document the assumptions; review the
original generation configuration when provenance is uncertain. Model-space-only
artifacts require a separate explicit conversion with original statistics, not an
automatic fallback here.

The common dates are the intersection of all required artifact folders within
20190101–20201231 (731 calendar days). Missing dates are listed and warned about;
no claim of all 731 days is made if fewer are available. Invalid date filenames,
missing masks, wrong member counts and inconsistent reference fields fail clearly.
Read input files are checked for size/mtime changes before completion; their hashes
are recorded for later verification. Inputs must remain unchanged during a run.

## Tests

```bash
python -m unittest revision_evaluation.tests.test_deterministic -v
```

Tests use tiny synthetic physical NPZs with the same folder/key conventions as
CEDDAR and the baselines. They check definitions, common masks, ties at physical
thresholds, empty subsets, member counts, even-member median, physical-only loading,
reference/land alignment, common/frozen dates, source-file preservation, external
output guards, failure status, and the Bash preflight entry point. No real data,
checkpoint, GPU, server access or plotting is required.
