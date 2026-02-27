# 02_reduced_run - Reuced scientific reproducibility run

This run provides a compact but scientifically meaningful reproduction of the CEDDAR training-generation-evaluation pipeline, using the small dataset Data_DiffMod_small.

It is intended for:
- Reviewers who want to quickly verify that the code runs and produces reasonable outputs.
- Reproducibility validation
- Cross-machine verification (CPU/GPU/HPC)
- Sanity checking training stability and evaluation outputs

This run is NOT a full reproduction of the main paper results, and is not intended for benchmarking or performance evaluation but it exercises the following key scientific aspects:
- Training of the model on a small dataset (Data_DiffMod_small) 
- Ensemble generation using the trained model
- Evaluation of the generated samples using a reduced set of evaluation metrics (subset of the full evaluation suite, but full set available by adjusting the config)
    - Probabilistic
    - Distributional
    - Date-based visual diagnostics

---

## Overview
Configuration:
`repro/02_reduced_run/reduced_run_config.yaml`

Runners:
`run_reduced_local.sh` - for local execution (CPU or GPU)
`run_reduced_lumi.sh` - for execution on Lumi HPC (GPU)

Dataset:
`Data_DiffMod_small` - a small subset of the full dataset

Statistics:
`repro/assets/stats/statistics_run/stats` - precomputed statistics for evaluation

---

## Expected runtime:

| Environment | Runtime (approx.) |
|-------------|-------------------|
| Local CPU   | ~5-15 minutes     |
| Local GPU   | NOT TESTED        |
| Lumi (1 GPU)| ~2-5 minutes      |

The run is designed to complete well under 1 hour on any of these environments.

---

## Prerequisites

### 1. Install the code

From repository root:
```bash 
pip install -r requirements.txt
```

### 2. Download the dataset 

Download the `Data_DiffMod_small` dataset from Zenodo and place it next to the repository:
```
project_root/
├── CEDDAR/  # repository code
├── Data_DiffMod_small/  # downloaded dataset
```

Alternatively, set manually:
```bash
export DATA_DIR="/path/to/Data_DiffMod_small"
```

### 3. No additional stats download required
Statistics (for the full dataset) JSON files are commited in:
```repro/assets/stats/statistics_run/stats```

These are used automatically by the config.

---
---

## Running the reduced experiment

### Local run
```bash
bash repro/02_reduced_run/run_reduced_local.sh
```

Force GPU:
```bash
DEVICE=cuda bash repro/02_reduced_run/run_reduced_local.sh
```

Force CPU:
```bash
DEVICE=cpu bash repro/02_reduced_run/run_reduced_local.sh
```

Override output dir:
```bash
export CEDDAR_RUNS="/path/to/output_dir"
bash repro/02_reduced_run/run_reduced_local.sh
```

---

### LUMI (HPC example)
```bash
cd /scratch/project_12345/USER/Code/CEDDAR  # adjust to your project scratch dir
sbatch /path/to/repro/02_reduced_run/run_reduced_lumi.sh
```

Optional overrides:
```bash
CONTAINER=/path/to/container.sif \
DATA_DIR=/path/to/Data_DiffMod_small \
CEDDAR_RUNS=/path/to/output_dir \
sbatch /path/to/repro/02_reduced_run/run_reduced_lumi.sh
```

---

## Outputs

Outputs are written outside the repository:

### Local default
```$HOME/ceddar_runs/repro/02_reduced_run/outputs/ ```

### LUMI default
```/scratch/<project>/<user>/runs/CEDDAR/repro/02_reduced_run/outputs/```

Main outputs:
- checkpoints/ - model checkpoints
- samples/samples - training time samples
- samples/generation - generated samples for evaluation
- samples/evaluation - evaluation outputs (metrics, diagnostics)

Key summary of CRPS/MAE metrics:
- ```.../samples/evaluation/<MODEL_NAME>/prcp/probabilistic/tables/prob_summary.csv ```

---

## Expected results (Sanity check)

Exact values vary slightly due to randomness and hardware differences, but the following ranges are expected for the key evaluation metrics.

### 1. Training behaviour
Training and validation losses should:
- Start around 0.8-1.0 
- Decrease steadily over epochs
- End around (after 10 epochs):
    - Training loss: 0.15-0.25
    - Validation loss: 0.10-0.20

Small oscillations are normal, but no divergence or large spikes should occur.

Failure indicators:
- Loss diverges
- Loss > 2.0 at any point
- NaNs
- No decreasing trend

### 2. Dates evaluation
For example 4 plotted dates:
#### Dry/light precipitation
- CRPS: ~0.2 - 0.4
- MAE: ~0.3 - 0.7

#### Moderate/wet precipitation
- CRPS: ~4 - 5
- MAE: ~5 - 6

Generated fields should show much stochasticity and scattered patterns due to the small dataset and short training, but should not be completely random noise - and should not be identical to the LR input. They should also clearly exhibit different patterns/amplitudes for the example dates.

### 3. Probabilistc evaluation
From ../probabilistic/tables/prob_summary.csv and CRPS mean map (../probabilistic/figures/crps_mean_map.png):
- Overall mean CRPS (per pixel, over time: ~1.5 - 2.5
- CRPS spatial map should:
    - Not be spatially uniform
    - Not contain NaNs
    - Have higher values in western Denmark and over Sweden (due to more complex terrain, more precipitation on average, and weather patterns)

---

## What this run verifies
This run should confirm the following:
- Scaling statistics and transforms/back-transforms work correctly
- Diffusion training is stable
- Ensemble generation works and produces non-trivial samples
- Evaluation pipeline runs end-to-end
- Device selection works (CPU/GPU/HPC)
- Stats JSON loading and usage works
- No silent shape mismatches occur

---

## What this run does NOT cover
This is a reduced run and therefore does not:
- Perform full sigma-star grid sweeps (as described in the paper)
- Train large models for publication-scale results
- Run all ablation
- Reproduce full paper figures.

For full reproduction, refer to Zenodo for pretrained models, and the full training and evaluation scripts with the original configs.

For the full dataset, follow the data download and preprocessing instructions in the paper, and the README, to generate the full dataset and statistics.

---

This reduced run serves as a compact, cross-platform validation of the CEDDAR diffusion downscaling pipeline.