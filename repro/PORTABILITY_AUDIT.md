# v1.0.2 portability audit

Baseline: `a349ffc` (`v1.0.2`), branch `paper1-revision-2026`.
This audit concerns tracked source, not private/ignored scripts or historical run outputs.
No change to the EDM equations, schedule, initial state, weights or metric formulas is
part of the infrastructure revision. In particular the known ignored late-ramp settings
remain unchanged; provenance must report the effective global default.

## Output map and minimal migration

| Area | Writers / path owners | v1.0.2 issue / migration |
|---|---|---|
| Training checkpoints | `sbgm/training_main.py`, `training.py`, `training_utils.py` | `paths.checkpoint_dir` is joined to `paths.path_save`. Absolute external roots avoid ambiguous relative joins. Existing filenames are retained. |
| Training samples, losses, metrics, figures | `training.py`, `training_main.py`, `plotting_utils.py`, `monitoring.py` | Derived from sample/path_save/log settings. Put these roots outside source. |
| Generation / PMM / masks / manifests | `sbgm/generate/{generation,generation_main,generation_sigma_grid_main,generation_sampler_grid_main}.py` | Derived from `paths.sample_dir`; filenames do not distinguish every run. Use a fresh external experiment root; provenance alone does not prevent overwriting samples. |
| Quicklooks | `sbgm/generate/quicklook.py` | Under sample root; retain layout. |
| Evaluation tables, figures, caches | `sbgm/evaluate/`, `sbgm/evaluate_sbgm/` | Usually under sample root, including `sample_dir/evaluation`; `evaluation_dir` is not universally consumed. Do not promise that changing EVAL_DIR alone moves all evaluations. |
| Baseline models/outputs/evaluation | `baselines/` | Configured roots and derived paths; some evaluation APIs use `paths.root`. Keep the whole experiment root external. |
| Application logs/config snapshots | `sbgm/logging_utils.py`, `sbgm/cli/main_app.py` | Fixed RUN filenames overwrite prior provenance. Use invocation-specific names and record effective generation parameters at the actual call. |
| Statistics/comparisons/correlations | `data_analysis_pipeline/{stats_analysis,comparison,correlations}/` | Environment roots exist, but fallbacks include `.`, `./correlation_outputs`, `./correlation_stats`. Run legacy preprocessing from an external working directory with explicit output settings. |
| Data preprocessing/splits | `data_analysis_pipeline/{preprocess,splits}/`, `sbgm/data/`, `sbgm/utils.py` | Writes beside data (filtered files, NPZ/Zarr, splits); these are data transformations, not read-only inference. Keep DATA_DIR external; do not run against the only archived input copy. |
| ERA5 download/regridding | `era5_download_pipeline/{cli,pipeline,cfg,slurm}/` | `/tmp/era5_downloads`, `../era5_logs`, scratch roots, remote copy destinations. External data/temp/log roots and CDO executable needed; separate from model smoke test. |
| Historical LUMI jobs | `bash_scripts/`, `bash_component_study/`, `lumi/` | Many scripts overwrite portable defaults with `$ROOT_DIR/models_and_samples`, `$ROOT_DIR/sbgm/logs`, repository `saved/`; Slurm `logs/...` is relative to submission cwd. These remain historical templates, not the portable ATMO entry point. Submit future GPU jobs with absolute `--output/--error` paths (directories created before submission). |
| Library caches/temp files | Matplotlib, Python bytecode, PyTorch, tempfile | Explicit external cache/temp settings in the portable launcher; small bundled statistics remain read-only source assets. |

`python repro/audit_repository.py --output /external/audit` produces a per-file,
per-line inventory of candidate path settings and write/copy sites, plus actual AST
imports. Comments/examples are marked; shell heredoc imports require manual review.
This avoids treating every `savefig` as a hard-coded path. Dynamically constructed
paths require following the owning root in the table above. Historical templates are
not globally rewritten: changing dozens of unrelated HPC/download jobs would obscure
the small, reviewable model-infrastructure change.

Concrete baseline examples (line numbers in `v1.0.2`):

- `bash_component_study/B1s/run_B1_G.sh:51-60`: reconstructs `/scratch/.../Code/CEDDAR`
  and places samples/checkpoints/evaluation/logs underneath it; line 71 creates `logs`
  relative to the submission directory.
- `sbgm/logging_utils.py:106,129,132`: opens the same RUN config/JSON/Markdown paths
  with `w`, replacing provenance from previous invocations.
- `data_analysis_pipeline/stats_analysis/data_stats_pipeline.py:34-35`: figure/statistics
  fallbacks are `.`.
- `data_analysis_pipeline/correlations/correlation_pipeline.py:97-106`: figure/statistics
  fallbacks are relative paths followed by directory creation.

The baseline inventory contains 3,776 candidate matches across 220 files (including
comments and repeated classifications), not 3,776 independently confirmed writers.
The inventory script reads Git blobs from `--ref v1.0.2`, so revision edits do not
shift its evidence lines. `revision.txt` records the full commit it inspected.

## Dependencies

Actual model/data/evaluation imports: torch, torchvision, numpy, scipy, zarr,
netCDF4, omegaconf, PyYAML, matplotlib, tqdm. `mpl_toolkits` comes with matplotlib.
Torchvision is used by dataset transforms and the U-Net; it is not optional here.
netCDF4 is imported by `sbgm.utils`, even for Zarr inference.

- `xarray`: no tracked Python import; unnecessary in the model environment.
- Python package `cdo`: no import; preprocessing invokes the **CDO executable**
  using subprocess. Installing the Python wrapper does not install that executable.
- `cdsapi`: only ERA5 downloads; separate optional requirements.
- `scikit-learn`: only `data_analysis_pipeline/correlations/data_correlations.py`;
  optional preprocessing/analysis requirements.
- `pandas`: missing for a Python heredoc in the old small-test shell script, not
  needed by core Python evaluation. The replacement smoke test uses stdlib JSON/CSV.
- `numcodecs`: Zarr dependency worth constraining explicitly with Zarr 2; an
  unconstrained Zarr 3 upgrade is not a reproducible v1.0.2 environment.
- Optuna shell jobs refer to absent `sbgm.sweep.run_optuna`; installing Optuna alone
  cannot restore that workflow. No active WandB/TensorBoard import was found.
- Local-looking imports `utils` and `plot_utils` in legacy data scripts are not
  package dependencies to install from PyPI.

Version pins document the local Python 3.11 reference environment used for validation;
they do not claim to reconstruct the historical LUMI environment. The ATMO Linux
environment must be installed and smoke-tested separately. Record the complete
installed package set per run; CPU/CUDA/ROCm wheels are platform-specific.
Local `pip check` additionally found missing Torch transitive dependency `fsspec`
and unrelated `xmip`/`pysal` dependency conflicts. The functional checks passed in
that environment, but it is not a clean installation validation. Normal installation
of the requirements resolves Torch's declared dependencies; run `pip check` in a
fresh ATMO environment and preserve its resolved freeze before scientific runs.

## Existing repro workflows

01 launches a full training pipeline, not a bounded smoke test. It omits EXP_DATE,
does not establish a cwd/PYTHONPATH, and ends successfully when summary CSV is absent.
Its `--config` relies on argparse abbreviation of `--config_path`. Its numerical
skill ranges and runtime promises are not portable acceptance tests.

02 is a reduced training experiment, not an installation check. It has similar cwd
and abbreviated-flag issues. Keep its scientific YAML settings unchanged and offer
the bounded smoke test as a preflight. Do not reinterpret a successful randomly
initialized smoke run as checkpoint validation or scientific reproduction.

Acceptance: data and model shapes, finite outputs, inverse-transform round trip,
saved-array evaluation, checkpoint provenance and actual sampler arguments. No
training convergence, physical skill threshold or sigma-star method fix belongs here.
