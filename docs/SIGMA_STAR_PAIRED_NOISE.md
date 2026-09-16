# Paired-noise sigma* pilots

Use these pilots to separate the response to sigma* from differences in random
realizations. They require no training and work on CPU with the existing checkpoint.
The default remains `noise_mode: sequential`, preserving the previous RNG behavior.
The explicit option below changes the experiment's random draws, not its EDM
schedule, churn amplitudes, network, transforms or metric definitions.

## What is paired

`full_gen_eval.sigma_control.noise_mode: paired` uses the experiment seed
(`full_gen_eval.seed`), date identifier and draw role to seed local PyTorch
generators. Churn draws are indexed by sampling step. Initial Gaussian noise and
noise-based classifier-free null conditions have separate roles. Thus a churn
branch active in only one variant does not shift subsequent common draws. The
protocol is `ceddar-indexed-noise-v1`; `sbgm/sampling_noise.py` defines its SHA-256
seed derivation. Process RNG state is not advanced by these sampler draws.

Pairing means the **standard normal tensors** match at shared roles/step indices.
Their scaled amplitudes, initial states and resulting samples can differ because
sigma* changes the schedule. Keep the checkpoint, conditioning, dates, member
count, resolution, steps, seed, dtype, device and software fixed. Different devices
or PyTorch versions need not give identical tensors. Changing member count is not
promised to preserve the earlier members' draws. Random data transforms are not
paired by this option; input hashes allow that mismatch to be detected.

Each generated date saves `meta/noise/<date>.json` inside its sigma directory:
actual draw seeds, shapes, dtypes and hashes, plus conditioning/reference hashes.
Generation provenance also records the protocol and effective sampler seed.
`repro.check_paired_noise` checks input hashes and shared draw hashes across all
supplied variants. It permits unshared roles (e.g. different active churn steps),
and reports their count. It does not verify checkpoint identity, model parameters,
completion or sample equality; inspect the usual generation manifests as well.

## ATMO commands

Use the active environment and the same `DATA_DIR`, `STATS_LOAD_DIR`,
`PUBLISHED_CHECKPOINT` and `DEVICE=cpu` as the successful pilots. These commands
work when launched from the repository in `tcsh`. Use new run directories.

```tcsh
setenv SIGMA_GLOBAL /home/theaqg/CEDDAR_runs/paper1_revision/sigma_global_paired_s504
setenv SIGMA_LATE /home/theaqg/CEDDAR_runs/paper1_revision/sigma_late_paired_s504

python -m unittest repro.test_paired_noise repro.test_sigma_plots repro.test_sigma_star repro.test_provenance repro.test_workflows repro.test_runtime -v

bash repro/run_sigma_star.sh prepare --run-dir "$SIGMA_GLOBAL" --noise-mode paired --seed 504 --sigma-star-mode global --initial-state schedule --steps 56 --ensemble-size 2 --max-dates 1 --split valid
bash repro/run_sigma_star.sh prepare --run-dir "$SIGMA_LATE" --noise-mode paired --seed 504 --sigma-star-mode late_ramp --initial-state schedule --steps 56 --ensemble-size 2 --max-dates 1 --split valid

bash repro/run_sigma_star.sh generate --run-dir "$SIGMA_GLOBAL"
bash repro/run_sigma_star.sh generate --run-dir "$SIGMA_LATE"
python -m repro.check_paired_noise "$SIGMA_GLOBAL" "$SIGMA_LATE"
bash repro/run_sigma_star.sh evaluate --run-dir "$SIGMA_GLOBAL"
bash repro/run_sigma_star.sh evaluate --run-dir "$SIGMA_LATE"
```

Inspect both resolved YAMLs before generation. The default preparation grid is
0.95/1.00/1.05 and the F config's late ramp is 0.60–0.85. Use `--config` during
preparation if the previous pilot used a different config. A saved resolved YAML
can be supplied: output roots are redirected to the new run while its data and
statistics paths remain the supplied config's values. Check those inputs explicitly.
Preparation loads no checkpoint and generates no samples.

At sigma*=1, both modes have the same schedule and initial amplitude. With matching
inputs/draws and deterministic execution their generated arrays should match;
matching aggregate metrics alone is a weaker check. Non-unit differences measure
the response for these realizations, not a general monotonic relation. Start with
this one-date pilot; only after the pairing check passes expand dates/members or
repeat with `--seed 505` in fresh directories. Two members cannot establish reliable
ensemble calibration or spread estimates.

To compare legacy initialization, prepare a third run with the same paired settings
and `--sigma-star-mode global --initial-state legacy_sigma_max`. Label it a **paired
legacy-initialization experiment**, not a reproduction of historical random draws.
An old sequential run must not serve as the paired comparator. Even sigma*=1
can change relative to the old sequential run because it now receives different
draws. Preserve all old outputs.

## Plot corrections and reuse of existing outputs

```tcsh
bash repro/run_sigma_star.sh plot --run-dir /absolute/path/to/existing/sigma_run
```

This reads that run's saved config, evaluation tables/PSD arrays and generated
example fields. It rewrites figures with the same filenames; it does not generate
samples, alter CSV values or recompute evaluation metrics. Preserve copies of old
figures first if their appearance is needed as audit evidence.

Changes:

- Autoscaled axes retain the reported pilot's low correlation, CRPS and power-ratio
  values. Unavailable values remain missing and all-NaN panels are labelled.
- Error bars use finite daily values separately for each metric. Nominal SEM is
  sample SD divided by sqrt(number of finite dates); it assumes independent dates.
  STD uses the existing population-SD convention. Both are omitted with fewer than
  two finite dates. Neither represents spread between ensemble members.
- The shaded PSD interval is labelled **slope-fit band**, not “sigma* control (late)”.
  A ramp in sampler time does not prescribe a spatial wavelength cutoff.
- Recorded global mode has no ramp-fraction annotation. Initial-state policy is
  displayed when recorded; unknown mode remains unknown.
- Correlation is labelled PMM–LR; PSD is labelled ensemble mean. PSD-curve slopes
  are explicitly slopes of the plotted date-mean spectrum, which need not equal
  the mean of daily slopes in the summary plot. The fit band is read from the saved
  PSD artifact rather than assumed from the requested config.

Existing metric code falls back to 5–20 km unless `psd_band_km` is a Python list or
tuple; an OmegaConf `ListConfig` can therefore fall back even with a custom request.
This change does not alter that metric behavior. Before using a custom band, fix
and test that configuration handling separately; the saved `psd_band_km` is the
authoritative evidence of what the existing metric actually used.

## Proposed complementary metric (not implemented here)

Report **RMS ensemble spread over a fixed land mask**, in mm/day. For physical
member fields P_m(x), M >= 2, and the same land pixels L for every sigma*:

    mean(x) = sum_m P_m(x) / M
    variance(x) = sum_m (P_m(x) - mean(x))^2 / (M - 1)
    spread_RMS = sqrt(sum_{x in L} variance(x) / |L|)

Calculate one value per date and sigma*. Retain dry land pixels; do not inherit the
HR-rainy CRPS filter. Require finite values for all members on the common mask and
keep M fixed. Report paired changes relative to sigma*=1, alongside absolute spread
and CRPS. Larger spread alone is not evidence of improved calibration.

This distinguishes **disagreement between members** from spatial variability
within a field. PSD of the ensemble mean can lose power through member cancellation;
it cannot establish ensemble spread. If individual-field texture is the next
question, mean member PSD is a useful additional spectral diagnostic, also distinct
from PSD of the mean. Neither proposal changes the existing metric definitions.

The distinction between ensemble-mean error and ensemble spread is standard in
[ECMWF's discussion of ensemble verification](https://www.ecmwf.int/en/newsletter/166/news/new-tool-understand-changes-ensemble-forecast-skill).
The fixed-land aggregation above is a proposed CEDDAR diagnostic, not a claim that
this source prescribes that exact mask or formula.

## Validation

32 CPU unit/workflow tests passed on 15 September 2026, including the frozen
sequential reference, alpha=1 mode agreement, paired grid-order invariance, churn
branch independence, actual saved noise/input hashes, provenance mismatch rejection,
preparation from a resolved config, and plotting/plot-only regressions. Changed
Python files parse; launcher syntax and `git diff --check` pass. Rendered plots were
visually inspected using the reported one-date metrics and synthetic PSD fixtures.
The tests use a toy denoiser; no real checkpoint generation was run here. GPU
execution remains untested. Matplotlib emitted an existing `get_cmap` deprecation
warning; it did not prevent plotting or test completion.
