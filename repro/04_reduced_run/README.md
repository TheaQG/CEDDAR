# 04_reduced_run: explicit reduced training experiment

Run [the smoke test](../README.md) first. This workflow trains, generates and evaluates
using real train/valid/test data. Its scientific YAML settings are unchanged.
It is not a bounded installation check or a reproduction of manuscript results.

```tcsh
setenv DATA_DIR /home/theaqg/CEDDAR_migration/Data/Data_DiffMod
setenv CEDDAR_RUNS /home/theaqg/CEDDAR_runs/reduced_trial_01
bash repro/04_reduced_run/run_reduced_local.sh
```

CPU is the default; `setenv DEVICE cuda` selects GPU in tcsh. The wrapper works from any directory.
Checkpoints, samples, evaluation and logs are under the external run root (or explicit
`CKPT_DIR`, `SAMPLE_DIR`, `EVAL_DIR`, `LOG_DIR` overrides). Choose a fresh destination.
Use `--dry_run` to check config/log setup without training. See [provenance details](../README.md).

The old LUMI launcher is a historical template requiring site-specific paths,
container/modules and external Slurm log destinations. It was not validated by this
CPU infrastructure revision. Previous runtime/metric ranges were not verified
acceptance criteria and have been removed. Real-data training and GPU validation
remain separate from the synthetic smoke test.

A full run can consume substantial CPU time. Test `--dry_run` first; it checks
configuration/log setup but does not load data, train or evaluate.
