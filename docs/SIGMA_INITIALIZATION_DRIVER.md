# Run the LEGACY–MATCHED–LATE_RAMP comparison on ATMO

Keep the existing runs. Duplicate pasted tables do not establish that generation
was wrong. This driver creates a new timestamped parent, prepares all three configurations,
generates all three ensembles, checks their provenance, and evaluates each separately.
It uses the existing sampler and evaluation implementations without changing them.

After transferring these files to the ATMO checkout, run from your **tcsh** terminal:

```tcsh
cd /home/theaqg/CEDDAR_GMD_revision
source /home/theaqg/envs/ceddar-revision/bin/activate.csh
bash repro/run_sigma_initialization_comparison.sh
```

Defaults: first 24 validation dates, 8 members, 56 steps, seed 504, paired noise,
and sigma* = 0.95, 1.00, 1.05. The three variants are:

| Run directory | Scaling mode | Initial state |
|---|---|---|
| `legacy/` | `global` | `legacy_sigma_max` |
| `matched/` | `global` | `schedule` |
| `late_ramp/` | `late_ramp` | `schedule` |

The late ramp inherits its parameters from the base YAML: the default
`F_final_test_eval.yaml` has `ramp_start_frac: 0.60` and `ramp_end_frac: 0.85`.
Check the resolved configuration if supplying another YAML. The global runs retain
these settings for provenance, but do not use the ramp. This adds a late-stage
schedule sensitivity experiment; it does not introduce frequency-selective noise.

These 24 dates are a pilot, not a seasonally representative sample. All three runs
execute sequentially on CPU with one thread by default. The third run adds another
full generation and evaluation; it does not reuse ensembles from either global run.

The script contains the current ATMO data/checkpoint paths and bundled statistics
path. Existing `DATA_DIR`, `STATS_LOAD_DIR`, and `PUBLISHED_CHECKPOINT` environment
variables override those defaults. Other overrides are `CPU_THREADS`, `SIGMA_SEED`,
`MAX_DATES`, `ENSEMBLE_SIZE`, `SIGMA_CONFIG`, `PYTHON`, and `COMPARISON_DIR`.
For example, `setenv MAX_DATES 1` and `setenv ENSEMBLE_SIZE 2` make a smaller test.
Inspect all resolved configurations if you override the base YAML.

Outputs default to:

```text
/home/theaqg/CEDDAR_runs/paper1_revision/sigma_init_d24_m8_s504_<UTC timestamp>/
    legacy/                 # resolved config, samples, evaluation, manifests
    matched/
    late_ramp/
    <stage>.log             # separate preparation/generation/evaluation logs
    comparison/
        legacy_metrics_by_sigma.csv
        matched_metrics_by_sigma.csv
        late_ramp_metrics_by_sigma.csv
        sources.json        # original CSV paths/hashes and verification results
```

The checks reject differing scientific configurations beyond scaling mode and initialization,
incorrect output paths, missing provenance, different checkpoint hashes, or
inconsistent paired inputs/draws. They require identical saved physical samples
at sigma*=1 across all supplied runs, and matching baseline metrics (including matching NaNs). They check
date/grid completeness before copying tables. Identical complete CSVs produce a
note, rather than a failure: a nonzero sensitivity is not a requirement for success.

`--prepare-only` creates and checks configurations without inference. A comparison
directory must be new; the driver does not resume or overwrite it. If a stage fails,
its log and partial outputs remain for inspection. For a fresh complete attempt,
omit `COMPARISON_DIR` or choose a new one. To continue prepared runs manually,
use `repro/run_sigma_star.sh generate --run-dir <run-directory>` and then
`evaluate --run-dir <run-directory>` for each run.

You can check the existing runs **without rerunning inference**:

```bash
python -m repro.check_sigma_initialization \
  /home/theaqg/CEDDAR_runs/paper1_revision/sigma_global_legacy_paired_d24_m8_s504 \
  /home/theaqg/CEDDAR_runs/paper1_revision/sigma_global_matched_paired_d24_m8_s504
```

To include a late-ramp run in the checks, append its run directory as the third
positional argument. The original two-directory command remains supported.

Add `--config-only` for a quick configuration check. Add `--output <new external
directory>` to check and export the existing evaluation tables with distinct names.
Neither command edits the original runs. The helper checks recorded checkpoint
hashes; it does not rehash the current checkpoint file or rerun the network.
