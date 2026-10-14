# Environment and reproduction guide

Use the source checkout matching your selected data/model archive. The historical
LUMI scripts are templates; portable model entry points are `run_model.sh` and
`01_small_test/run_small_test.sh`. The final figure runner is
[`../bash_paper1/run_manuscript_figures.sh`](../bash_paper1/run_manuscript_figures.sh).
See [PORTABILITY_AUDIT.md](PORTABILITY_AUDIT.md) for the path and import audit.

For a separately logged LEGACY–MATCHED–LATE_RAMP sigma* comparison on ATMO, see
[the comparison driver instructions](../docs/SIGMA_INITIALIZATION_DRIVER.md).

## Environment

Use **Python 3.11** and keep the environment outside the Git checkout. From the
repository root, create and activate an environment (bash/zsh):

```bash
python3.11 -m venv ../ceddar-env
source ../ceddar-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

On **Linux CPU**, install the CPU wheels before the remaining requirements:

```bash
python -m pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```

On **macOS**, skip that CPU-wheel command. For **GPU** installations, select the
appropriate wheel index from [PyTorch's v2.2.0 instructions](https://pytorch.org/get-started/previous-versions/#v220),
keeping Torch 2.2.0 and Torchvision 0.17.0 paired. Keep an existing working research
environment if it already supplies the required versions.

Then install and record the environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
mkdir -p ../CEDDAR_runs/environment
python -m pip freeze > ../CEDDAR_runs/environment/requirements-resolved.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA build:', torch.version.cuda)"
```

The explicit `+cpu` versions select Linux CPU wheels while allowing dependencies
from PyPI; expected versions are `2.2.0+cpu` / `0.17.0+cpu`, with CUDA build `None`.
On ATMO's **tcsh**, activate with `source ../ceddar-env/bin/activate.csh` instead.
Use `setenv NAME value` for environment overrides there; bash/zsh uses
`export NAME=value`.

`requirements.txt` includes the model and figure dependencies, including
**pandas 2.0.3** for saved-table loaders. The compact archive was validated with
Python 3.11.4, NumPy 1.25.2, SciPy 1.11.4, Matplotlib 3.7.1, pandas 2.0.3,
Torch 2.2.0, PyYAML 6.0.1 and Pillow 10.2.0. All 12 PNGs matched the existing final
figures exactly; PDFs matched after excluding timestamp metadata. Keep the saved
environment and rendering resources when exact figure appearance matters.

These are revision validation pins, including NumPy 1 and Zarr 2, rather than a
complete transitive lock or a reconstruction of the original LUMI environment.
Preserve a resolved package list for each environment. Optional correlation and
ERA5 download dependencies live in [`../requirements-preprocess.txt`](../requirements-preprocess.txt);
CDO is a separate host executable. Figure reproduction uses saved products and
does not need CDO, source-data downloads or checkpoint access.

## Manuscript figures

With this environment active and the compact archive extracted outside the
checkout, run from the repository root:

```bash
DATA_ROOT=/absolute/path/to/CEDDAR_manuscript_figure_data \
OUT=/absolute/path/to/figures \
bash bash_paper1/run_manuscript_figures.sh
```

On tcsh, set `DATA_ROOT` and `OUT` with `setenv` first, then invoke the same Bash
runner. It creates Figures 2-7 and S5-S10 as PNG/PDF. Figure 1 and S1-S4 were
manually prepared. `REPO` defaults to this checkout; `OUT` defaults to a sibling
`CEDDAR_manuscript_figures` directory. Relative overrides resolve from the
invocation directory. See [the archive guide](MANUSCRIPT_FIGURE_DATA.md) for
checksums, dependency tracing and the guarded full-data/compact-data comparison.

## Paths and launch

```tcsh
setenv CEDDAR_RUNS /absolute/path/to/CEDDAR_runs/my_experiment
setenv DATA_DIR /absolute/path/to/Data_DiffMod
bash /path/to/CEDDAR/repro/run_model.sh \
  --config_path /path/to/CEDDAR/sbgm/config/component_study/B1s/B1_GSDF_RGBCE.yaml \
  --mode generate --device cpu
```

For bash/zsh use `export NAME=value` instead of `setenv NAME value`.

Defaults are sibling `CEDDAR_runs/` and `Data_DiffMod_small/`. `CKPT_DIR`,
`SAMPLE_DIR`, `LOG_DIR`, `EVAL_DIR`, `STATS_LOAD_DIR`, `TMPDIR`, `XDG_CACHE_HOME`,
`MPLCONFIGDIR` and `TORCH_HOME` can be overridden. Checkpoints default to
`CEDDAR_RUNS/checkpoints`; put the selected checkpoint there under the name derived
by the existing model/config naming function, or set `CKPT_DIR` to its directory.
Small bundled statistics are read-only defaults. Most evaluation outputs remain
under `SAMPLE_DIR/evaluation`; `EVAL_DIR` is not used by every evaluator.

Explicit YAML paths still apply. The model CLI rejects resolved output roots inside
the source tree, including symlinks; it does not silently relocate hard-coded inputs.
Use `--dry_run` to resolve configuration and write its snapshot without model work.
Choose a new experiment root for each scientific run: existing sample/checkpoint
filenames are retained and can overwrite files if you reuse their destinations.
The portable shell wrappers disable bytecode writes and work from any directory.
Legacy preprocessing/download scripts require their own explicit roots; this change
does not rewrite their filesystem behavior. No archived outputs are moved/deleted.

## Repro levels

| Level | Purpose | Status |
|---|---|---|
| [01_small_test](01_small_test/README.md) | Synthetic inputs/random weights; inference, transforms, saved arrays and CRPS | Runnable; CPU by default |
| [02_real_artifact_smoke](02_real_artifact_smoke/README.md) | Same checks with explicit real data, archived checkpoint and matching statistics | Runnable; artifact must be supplied |
| [04_reduced_run](04_reduced_run/README.md) | Deliberate training/generation/evaluation experiment | Existing workflow; potentially expensive |

```tcsh
env PYTHONDONTWRITEBYTECODE=1 python -m unittest repro.test_runtime repro.test_provenance repro.test_workflows -v
bash repro/01_small_test/run_small_test.sh
```

The default level-01 smoke creates synthetic precipitation Zarr/static inputs and
random weights, then samples one date/two members at 32×32 with two EDM steps.
It checks inverse-transform round trips, reloads physical outputs and evaluates CRPS.
Missing/nonfinite arrays or failed checks exit nonzero. Runtime is measured, not a
pass criterion. Do not run Python with `-O` or `PYTHONOPTIMIZE`; assertions are required.

Level 02 retains the configured spatial size (128×128 in the supplied B1_GSDF_RGBCE
candidate). Follow its README to set artifact paths. Missing real artifacts never
fall back to fixtures. `--preflight-only` checks data/weights/transforms without
sampling; it writes `preflight_result.json`, not a full smoke-pass result.

Both smoke levels accept `--device cuda`, `--threads`, `--steps` and `--output`.
Outputs must use a new external directory; by default a timestamped directory is
created under `CEDDAR_RUNS/smoke/`. Preserve `smoke_result.json` and YAML manifests.
Random-weight metrics are not skill scores. A two-step run is not a σ* experiment
or a scientific evaluation of the checkpoint. No level-01/02 run trains a model.

## Provenance

Each CLI invocation gets a unique log/config/entry manifest. Generation writes a
unique YAML in its `meta/` directory at the first actual sampler call, with checkpoint
path/SHA256/weight key, resolved config, Git branch/commit/dirty state, host, Python,
Torch/device, installed packages and **effective sampler arguments including defaults**.
Quicklooks also record that call. Training records its initial checkpoint/device.
Missing/unobserved values are null; entry/training manifests do not claim a sampler
was executed. The generation YAML is authoritative for sampler settings; existing
legacy JSON metadata is retained for readers but may only describe requested settings.

In the original v1.0.2 archive, the standard generation runner omitted mode/ramp
arguments. This checkout retains the completed sigma* audit fixes and forwards
the full controls. The `schedule` and `legacy_sigma_max` initial-state policies
are distinct; preserve the policy and settings recorded for the experiment being
reproduced. See [the audit corrections](../docs/SIGMA_STAR_REVISION.md) and
[comparison workflows](../docs/SIGMA_INITIALIZATION_DRIVER.md).
Requested generation seed is recorded separately from a CPU RNG-state digest: the
process RNG is not reset at each sweep point. Paired-noise mode instead keys draws
by seed/date/role/step and records their hashes. These records are not full RNG-state
checkpoints.
Preserve output metadata/date lists, input data/statistics versions and original
checkpoints separately; the manifest hashes the checkpoint, not the entire dataset.

## Sigma* audit and revision

See [the frozen-code audit](../docs/SIGMA_STAR_AUDIT.md) and [revision/ATMO instructions](../docs/SIGMA_STAR_REVISION.md). Prepare an isolated pilot with `bash repro/run_sigma_star.sh prepare --run-dir /external/runs/sigma_star_pilot`; generation and evaluation are separate explicit actions.

For paired random draws across sigma* values/modes, plot-only refreshes, and the proposed spread metric, see [paired-noise pilots](../docs/SIGMA_STAR_PAIRED_NOISE.md). Sequential sampling remains the default.

## Revision deterministic evaluation

For the original 32-member 2019–2020 test outputs and baseline fields, see
[Group 1 evaluation instructions](../revision_evaluation/README.md). Start with
`bash repro/run_revision_deterministic.sh --preflight-only`; this performs no inference.
