# 02_real_artifact_smoke

This test verifies portability of the published CEDDAR Paper I model.

It uses:
- real DANRA/ERA5 input data,
- the archived final B1_GSDF_RGBCE checkpoint,
- stored training normalization statistics,
- CPU by default,
- one test date,
- two ensemble members,
- two EDM sampling steps.

It does not train a model and is not intended to reproduce manuscript metrics.
Its purpose is to verify that the published artifact can be loaded and executed
in a clean environment.

Example:

```tcsh
setenv DATA_DIR /home/theaqg/CEDDAR_migration/Data/Data_DiffMod
setenv PUBLISHED_CHECKPOINT /path/to/final_checkpoint.pth.tar
setenv CEDDAR_RUNS /home/theaqg/CEDDAR_runs/repro/02_real_artifact_smoke

bash repro/02_real_artifact_smoke/run_real_artifact_smoke.sh