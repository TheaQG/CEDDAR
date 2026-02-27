# 01_small_test

This is a minimal end-to-end reproducibility test for CEDDAR that runs on a **CPU** laptop/desktop.
It is designed to verify that the full pipeline (training, generation, evaluation) works:
- Data loading + scaling/transformation
- Training loop (with 2 iterations for speed, and batch size 2)
- Generation of the mock test dataset (10 sampling steps, ensemble size 4)
- Evaluation (probabilistic metrics + a few date plots only)

This test is not meant to reproduce paper-quality skills or results, only to verify that the workflow is functional and produces sensible outputs. 

---

## Expected runtime
- **CPU**: < 5 minutes (depending on hardware)

---

## Prerequisites
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Download Data_DiffMod_small from Zenodo: ___INSERT LINK___ and place it **next to** the CEDDAR/ repository folder:
```<project_root>/
    CEDDAR/
    Data_DiffMod_small/
```
(Alternatively, you can place it anywhere and update the `DATA_DIR` variable in `run_small_test.sh` accordingly (see below), but placing it next to CEDDAR is the simplest approach.)

## Running the test

From anywhere inside the CEDDAR repository, run:
```bash repro/01_small_test/run_small_test.sh```

### Optional: custom locations

If you place the dataset elsewhere:
```bash
export DATA_DIR="/path/to/Data_DiffMod_small"
bash repro/01_small_test/run_small_test.sh
```

If you want the outputs to go somewhere else, you can also set the following variables before running the script:
```bash
export CEDDAR_RUNS="/path/to/ceddar_runs" # Base directory for all CEDDAR runs (default: <project_root>/ceddar_runs)
bash repro/01_small_test/run_small_test.sh
```

---

## Outputs

By default, outputs are written outside the repo to avoid cluttering the repository with generated files. The expected output structure is:

```~/ceddar_runs/
    repro/
        01_small_test/
            outputs/
                checkpoints/        # Model checkpoints saved during training
                samples/            # Generated samples, evaluation results, and plots
                    evaluation/     # Evaluation results (e.g., JSON files with metrics)
                    generation/     # Generated samples (.npz files)
                    losses/         # Training loss curves (e.g., CSV or JSON files)
                    quicklooks/     # Quicklook plots for generated samples
                    samples/        # Samples from training iterations (e.g., for monitoring training progress)
                logs/          # Training logs (e.g., TensorBoard files)
```

This script also prints a short evaluation summary at the end by reading the generated prob_summary.csv file (CRPS and MAE for the test set).

---

## Expected results (sanity check)

Exact values may vary slightly due to randomness and hardware/platform differences, but the following should be in a similar range:
- CRPS_ensemble: typically ~15-20 
- PMM_MAE: typically ~20-45

If the run completes without errors and metrics are within these ranges, the pipeline is considered functional. If you see errors, or metrics that are orders of magnitude off (e.g., CRPS in the hundreds), there may be an issue with the setup or code. In that case, please check the logs for errors and verify that you have the latest version of the repo with all necessary files included.

---

## Notes

- This test uses precomputed statistics for all datasets (train/val/test) that are included in the repo to ensure reproducibility. They are comitted under: `repro/assets/stats/statistics_run/stats/`.
- Evaluation is intentionally limited to a small number of dates, a small ensemble, and a subset of the metrics to keep runtime short.

