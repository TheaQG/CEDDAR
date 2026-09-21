# Group 4: QM and precipitation morphology

This analysis tests differences in precipitation objects without assuming that QM
performs poorly. It evaluates DANRA, bilinear ERA5, QM and each CEDDAR member.
Ensemble mean/median/PMM are not primary morphology fields. Their saved inputs remain
part of the existing common-date and finite-pixel checks, unchanged from Groups 1–3.
No training, generation, regridding, smoothing or minimum-object-size filtering occurs.

## Review of the existing code

The dry-bias implementation is consistent with the requested member analysis:
`run_dry_bias.py` keys accumulators by `(member, season)` and retains wet values
separately for each member. `field_components` applies that field's own `P >= 1`
mask. ALL and seasonal groups are separate summaries, not summed together later.
`plot_dry_bias.member_summary` takes median and IQR across member-level statistics,
not pooled member pixel values. Annual member P50/P90/P99 are evaluated separately;
seasonal wet frequency/conditional means come from separate member-season counts.
Member wet values are stored as float32 before quantile calculation; sums/counts
use float64. No numerical change to these working routines was needed.

Reuse: `RevisionInputs.load_date` exposes physical observations, fields, all members
and the common valid land mask. The existing CSV, manifest and plot helpers supply
I/O. Connected objects use the existing SciPy dependency's `ndimage.label` directly.
Legacy object extraction is nested inside the SAL helper and hard-wires different
threshold/filtering behaviour, so it is not a reusable unfiltered object routine.

Legacy `sbgm/evaluate/evaluate_prcp/eval_features/metrics_features.py` has two SAL
variants: `compute_sal` uses spatial standard deviation for S and only centroid
displacement for L; `compute_sal_object` uses largest-mass fraction or Herfindahl
concentration for S. These are not the standard SAL structure component. The legacy
feature runner evaluates time-mean maps. Its final-test YAML selects object mode,
quantile 0.85, smoothing sigma 1, minimum area 16 and Herfindahl concentration.
These routines and historical results remain unchanged.

## Definitions

- Absolute objects: `P >= q`, q = 1, 5, 10, 20 mm/day, on valid land, with
  8-neighbour connectivity. Sea/invalid pixels break connectivity. Areas are pixel
  counts; boundaries can truncate objects, consistently across all methods.
- Outputs retain date, season, method, zero-based member ID, threshold and valid
  support. Deterministic member IDs are blank. Event-free fields have zero objects
  and wet area, but NaN object sizes, largest-object fraction and intensities.
  With no valid pixels, object count/fractions are undefined rather than zero.
- Equal area: derive `f_obs` from DANRA at q; select the other field at its linear
  quantile `Q(1-f_obs)`, including all ties. A zero target explicitly selects an empty
  mask, with no effective threshold (NaN); a unit target selects all valid pixels.
  DANRA retains its original physical mask. No random tie-breaking occurs.
  Save q, effective q*, target/achieved fractions and their difference. Large tie
  effects, especially zero ties, can prevent a useful area match; inspect the
  achieved fractions before attributing remaining differences to organisation.
  Equal-area wet area is a control diagnostic, not a performance score.
- SAL uses the standard components from [Wernli et al. (2008), Eqs. 2, 4–9](https://doi.org/10.1175/2008MWR2415.1)
  with the **fixed physical object thresholds above**, not the original paper's
  fraction-of-maximum thresholds or the legacy YAML's quantile/smoothing settings.
  For object mass m and peak p, scaled volume is m/p; S compares mass-weighted
  scaled volumes. A compares full valid-domain means. L = L1 + L2 compares full-field
  centres and mass-weighted object-centre scatter. Distance normalization uses the
  maximum separation of common valid grid-cell centres. Equal grid spacing is
  assumed, so its physical scale cancels. Negative precipitation makes SAL undefined
  for that comparison, with counts recorded; fields are never silently clipped.
  Missing objects make S and L undefined; A can still exist (e.g. −2 for a dry
  prediction and wet reference). Both completely dry fields give undefined A.
  No combined SAL norm or equal-area SAL is calculated.

## ATMO evaluation and plots

From tcsh, preferably in tmux, after transferring the code to the revision branch:

```tcsh
cd /home/theaqg/CEDDAR_GMD_revision
source /home/theaqg/envs/ceddar-revision/bin/activate.csh

bash repro/run_revision_morphology.sh --dates-file /home/theaqg/CEDDAR_runs/paper1_revision/evaluation/deterministic/dates.txt

bash repro/plot_revision.sh morphology --input-dir /home/theaqg/CEDDAR_runs/paper1_revision/evaluation/morphology
```

The existing `atmo.yaml` is reused. The full 644-date list is retained. Metrics go
to `evaluation/morphology/`: `objects_absolute.csv`, `objects_equal_area.csv`,
`sal_absolute.csv`, `daily_support.csv`, `input_files.csv`, `dates.txt`, resolved
configuration and manifest. These are new outputs; an existing output directory
is refused. Use a new `--output-root` for a rerun, retaining the original dates file.

Figures go to `evaluation/manuscript_ready/morphology/` as PNG and vector PDF:

- `group4_objects`: paired differences from DANRA in object count and largest-object
  area fraction, for absolute and equal-area masks. Lines are medians across dates;
  CEDDAR first uses the member median for each date. Its blue band joins the medians
  across dates of the within-date member lower/upper quartiles. It is **not** a
  confidence interval or an IQR of all pooled members. Baseline lines have no bands.
- `group4_sal_q1`: daily S/A/L distributions at 1 mm/day, with CEDDAR represented by
  one member median per date. Boxes span the IQR across dates; all outliers remain.
  `--sal-threshold 5` (or 10/20) selects another saved threshold; also supply a new
  `--output-dir` when plotting again.

Each plotted metric/threshold uses dates defined for all three comparison methods.
Within-date member summaries use finite members only. `date_level_summary.csv`
retains their quartiles and valid-member counts, including undefined dates.
`plot_sample_counts.csv` records each plot's common date count and minimum valid
member count. Undefined object fractions at high thresholds can reduce the plotted
sample substantially. Object count retains event-free days. No ratios to zero
object counts, member-as-date replication or bootstrap intervals are used.
Small median differences alone do not prove equivalence; raw date/member results
are preserved. Object counts/sizes alone do not test geographical event placement.

## Minimal checks

```bash
python -m unittest revision_evaluation.tests.test_dry_bias revision_evaluation.tests.test_object_metrics -v
```

Only three new small tests: connectivity/object sizes (including SAL volume and
location identities), equal-area endpoints/ties, and event-free/undefined metrics.
No synthetic directory pipeline, preflight framework or launcher tests were added.
