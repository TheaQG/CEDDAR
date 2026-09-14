# 02_reduced_run: explicit reduced training experiment

Run [the smoke test](../README.md) first. This workflow trains, generates and evaluates
using real train/valid/test data. Its scientific YAML settings are unchanged.
It is not a bounded installation check or a reproduction of manuscript results.

```bash
export DATA_DIR=/external/Data_DiffMod_small
export CEDDAR_RUNS=/external/CEDDAR_runs/reduced_trial_01
bash repro/02_reduced_run/run_reduced_local.sh
```

CPU is the default; `DEVICE=cuda` selects GPU. The wrapper works from any directory.
Checkpoints, samples, evaluation and logs are under the external run root (or explicit
`CKPT_DIR`, `SAMPLE_DIR`, `EVAL_DIR`, `LOG_DIR` overrides). Choose a fresh destination.
Use `--dry_run` to check config/log setup without training. See [provenance details](../README.md).

The old LUMI launcher is a historical template requiring site-specific paths,
container/modules and external Slurm log destinations. It was not validated by this
CPU infrastructure revision. Previous runtime/metric ranges were not verified
acceptance criteria and have been removed. Real-data training and GPU validation
remain separate from the synthetic smoke test.

Purpose: Does a reduced but scientifically meaningful CEDDAR experiment behave sensibly?

Runs with:
    - Real data
    - Trains model from scratch
    - Generation
    - Evaluation
    - 15 epochs
    - 8 ensemble members
    - Up to 20 dates
    - Several evaluation tasks
Is a potentially substantial CPU job