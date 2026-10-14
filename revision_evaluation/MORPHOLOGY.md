# Precipitation morphology and SAL

`run_morphology.py` evaluates DANRA, bilinear ERA5, QM and each CEDDAR member on
the frozen common dates and finite land support. Saved ensemble summaries remain
part of the common input checks. There is no new generation, regridding,
smoothing or minimum-object-size filtering. See the [evaluation guide](README.md)
for configuration, launch and output handling.

## Saved definitions

- Absolute objects use `P >= q` at q = 1, 5, 10, 20 mm/day, with 8-neighbour
  connectivity. Invalid/sea cells break connectivity; areas count grid cells.
  Domain boundaries may truncate objects consistently across methods.
- Event-free fields have zero objects and wet area, but undefined sizes,
  largest-object fraction and intensities. With no valid pixels, object count
  and fractions are also undefined. These values are not filled with zeros.
- Equal-area controls use DANRA's observed wet fraction at each reference
  threshold, selecting other fields at the linear quantile `Q(1-f_obs)` with all
  ties included. Zero target fraction gives an empty mask and undefined effective
  threshold; unit fraction selects all valid pixels. DANRA keeps its physical
  mask. Saved target/achieved fractions expose tie effects, especially at zero.
  Equal-area wet area is a control diagnostic, not a performance score.
- SAL follows [Wernli et al. (2008), Eqs. 2, 4-9](https://doi.org/10.1175/2008MWR2415.1)
  using the fixed physical thresholds above. Structure compares mass-weighted
  scaled volumes (object mass/peak); amplitude compares valid-domain means;
  location combines field-centre displacement and object-centre scatter.
  Distance normalization uses the maximum separation of valid grid-cell centres;
  equal spacing is assumed. Negative precipitation makes SAL undefined for that
  comparison and is counted, not clipped. Missing objects make S/L undefined;
  A may remain defined (for example, -2 for a dry prediction and wet reference).
  Both fully dry fields have undefined A. There is no combined SAL norm or
  equal-area SAL.

The legacy `sbgm/evaluate/evaluate_prcp/eval_features/metrics_features.py` SAL
variants use different structure/location definitions and historical filtering.
Their outputs are not interchangeable with these revised physical-threshold
products. Both the legacy implementation and revised metric code remain unchanged.

## Tables and plotting aggregation

The stage saves `objects_absolute.csv`, `objects_equal_area.csv`,
`sal_absolute.csv`, daily support, frozen dates, resolved configuration, input
hashes and a completion manifest. Rows retain date, season, method, threshold and
zero-based member ID; deterministic member IDs are blank.

The final manuscript panels reuse `date_summaries` and `common_dates` from
`plot_morphology.py`. Object-count error takes the absolute paired difference
**before** averaging members within each date, then averages across common dates.
For largest-object fraction and SAL, finite members are summarized by their
within-date median. Each metric/threshold compares the three prediction methods
on the same dates where it is defined. Undefined fractions/SAL stay undefined;
event-free object counts remain in count diagnostics.

The standalone morphology plotter additionally saves `date_level_summary.csv`,
`object_count_summary.csv` and `plot_sample_counts.csv`, including member quartiles,
valid-member counts and event/event-free cohorts. Its member bands describe
within-date variability, not confidence intervals. These diagnostics do not treat
members as independent dates, apply bootstrap intervals, or establish geographical
event placement from object counts alone. Small median differences do not prove
equivalence. Raw date/member tables remain the scientific record.

## Dry-bias relationship

Dry-bias metrics use each field's own `P >= 1` mask. Member wet values are retained
separately, including annual and seasonal summaries. Wet quantiles use float32
stored values; sums/counts use float64. The final dry-bias panels show median/IQR
across member statistics, without pooling member pixels. ALL and seasonal groups
are separate summaries. The [final data-loader guide](../revision_manuscript_plots/data/README.md)
explains related histogram and PSD distinctions.
