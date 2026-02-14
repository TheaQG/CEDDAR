# CEDDAR

## Controllable Ensemble Diffusion Downscaling for Atmospheric Rainfall

CEDDAR is a diffusion-based statistical downscaling framework for generating high-resolution precipitation ensembles over Denmark from low-resolution ERA5 precipitation.

It is built around the Elucidated Diffusion Model (EDM) and includes a modular evaluation suite tailored to precipitation (probabilistic, spatial, distributional, and temporal diagnostics).

---

## Overview
![alt text](figures/introduction_long.png)

CEDDAR learns a conditional generative mapping:

ERA5 (low resolution) → DANRA-scale (high resolution)

Key ideas:
- Diffusion-based generative modelling of precipitation fields
- Ensemble generation for uncertainty quantification
- Scale control through sampling parameters (e.g., $\sigma^{*}$)
- Evaluation suite focused on precipitation skill (CRPS, MAE, spatial structure)

---

## Example outputs
![alt text](figures/Dates_final.png)

CEDDAR produces:
- Spatially structured high-resolution precipitation fields
- Ensemble variability across members
- Probabilistic evaluation maps (e.g., mean CRPS)
- Date-based diagnostics with CRPS and MAE

The reduced reproducibility run (repro/02_reduced_run) generates figures similar to the examples above.

---

## Quickstart (Reproducibility)

Install locally:
```bash
pip install -r requirements.txt
pip install -e .
```

Download `Data_DiffMod_small` from Zenodo (link to be added) and place it next to the repository:
```
project_root/
    CEDDAR/
    Data_DiffMod_small/
```

### 01_small_test (CPU, smoke test)
```bash
bash repro/01_small_test/run_small_test.sh
```

### 02_reduced_run (CPU/GPU/LUMI HPC)
Local (CPU/GPU):
```bash
bash repro/02_reduced_run/run_reduced_local.sh  # Local CPU/GPU
```

LUMI HPC (GPU):
```bash
sbatch repro/02_reduced_run/run_reduced_lumi.sh  # LUMI HPC (GPU)
```
See the respective README files in repro/ for expected runtimes and sanity ranges.

---

## Usage

CEDDAR is configuration-driven, with modular runners for local and HPC execution, with example usage:
```bash
python -m sbgm.cli.main_app --config path/to/config.yaml --mode training
```
For working examples, use the reproducibility workflowa in repro/ or the full ablation and model bash scripts in bash_scripts/ or bash_ablations/.

---

## Data

Full datasets are not included due to size and licensing.
A small example dataset (Data_DiffMod_small) is available on Zenodo for testing and reproducibility purposes.
The full dataset used in the paper can be reproduced by following the data processing steps outlined in the paper and code, starting from ERA5 precipitation data or by contacting the authors for access to the processed dataset.

---

## Citation
```bibtex
@misc{CEDDAR,
    author          = {Quistgaard, Thea and collaborators},
    title           = {CEDDAR: Controllable Ensemble Diffusion Downscaling for Atmospheric Rainfall},
    year            = {2026},
    howpublished    = {GitHub repository}, 
}
```

---

## License

MIT License. See LICENSE file for details.