# Portable model setup

Use branch `paper1-revision-2026`. The historical LUMI scripts are templates;
the supported portable entry points are `run_model.sh` and `01_small_test/run_small_test.sh`.
See [PORTABILITY_AUDIT.md](PORTABILITY_AUDIT.md) for the path and import audit.

For a separately logged LEGACY–MATCHED–LATE_RAMP sigma* comparison on ATMO, see
[the comparison driver instructions](../docs/SIGMA_INITIALIZATION_DRIVER.md).

## Environment (ATMO Linux CPU, Python 3.11)

Keep the environment outside the Git checkout. ATMO uses **tcsh**; these paths
use your home directory and do not require a literal `/external` directory:

```tcsh
python3.11 -m venv /home/theaqg/envs/ceddar-revision
source /home/theaqg/envs/ceddar-revision/bin/activate.csh
python -m pip install --upgrade pip setuptools wheel --index-url https://pypi.org/simple
python -m pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
mkdir -p /home/theaqg/CEDDAR_runs/environment
python -m pip freeze > /home/theaqg/CEDDAR_runs/environment/requirements-resolved.txt
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA build:', torch.version.cuda)"
```

Run the install commands from the repository root. For bash/zsh, activate with
`source /home/theaqg/envs/ceddar-revision/bin/activate` instead. The explicit `+cpu`
versions select CPU wheels while allowing dependencies to come from PyPI. Expected
versions are `2.2.0+cpu` / `0.17.0+cpu`, with CUDA build `None`. An already working
ATMO environment does not need to be recreated; keep its resolved package list.

The CPU index and version pair are documented in [PyTorch's v2.2.0 instructions](https://pytorch.org/get-started/previous-versions/#v220).
On macOS omit the CPU-wheel command and install `requirements.txt` directly. GPU installations need the matching platform wheel
from that page; do not install the CPU wheel for a GPU test. `requirements.txt` pins
the directly used model packages against the available Python 3.11 reference
environment, including NumPy 1 and Zarr 2. It is **not** a complete transitive lock or
a reconstruction of the submitted LUMI environment. Preserve the resolved freeze
on ATMO; each run also records installed package versions. Optional correlation and
ERA5 download dependencies live in `requirements-preprocess.txt`; the download client
is not locally validated/pinned and CDO is a separate host executable.

## Paths and launch

```tcsh
setenv CEDDAR_RUNS /home/theaqg/CEDDAR_runs/my_experiment
setenv DATA_DIR /home/theaqg/CEDDAR_migration/Data/Data_DiffMod
bash /path/to/CEDDAR/repro/run_model.sh \
  --config_path /path/to/CEDDAR/sbgm/config/component_study/B1s/B1.yaml \
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
| [03_end_to_end_smoke](03_end_to_end_smoke/README.md) | Future bounded training lifecycle test | Not implemented; exits nonzero |
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

In v1.0.2, GenerationRunner forwards `sigma_star` but omits the mode/ramp arguments.
The manifest therefore reports `global` even if YAML requests `late_ramp`. This
infrastructure change preserves that behavior and the unscaled initial state.
Requested generation seed is recorded separately from a CPU RNG-state digest: the
runner does not reseed each sweep point. This is not a full RNG-state checkpoint.
Preserve output metadata/date lists, input data/statistics versions and original
checkpoints separately; the manifest hashes the checkpoint, not the entire dataset.

## Sigma* audit and revision

See [the frozen-code audit](../docs/SIGMA_STAR_AUDIT.md) and [revision/ATMO instructions](../docs/SIGMA_STAR_REVISION.md). Prepare an isolated pilot with `bash repro/run_sigma_star.sh prepare --run-dir /external/runs/sigma_star_pilot`; generation and evaluation are separate explicit actions.

For paired random draws across sigma* values/modes, plot-only refreshes, and the proposed spread metric, see [paired-noise pilots](../docs/SIGMA_STAR_PAIRED_NOISE.md). Sequential sampling remains the default.
