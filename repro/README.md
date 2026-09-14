# Portable model setup

Use branch `paper1-revision-2026`. The historical LUMI scripts are templates;
the supported portable entry points are `run_model.sh` and `01_small_test/run_small_test.sh`.
See [PORTABILITY_AUDIT.md](PORTABILITY_AUDIT.md) for the path and import audit.

## Environment (ATMO Linux CPU, Python 3.11)

Keep both the environment and outputs outside the Git checkout. Set these example
paths for your account; run the install commands from the repository root:

```bash
python3.11 -m venv /external/envs/ceddar-revision
source /external/envs/ceddar-revision/bin/activate.csh
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.0 torchvision==0.17.0 --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
mkdir -p /external/CEDDAR_runs/environment
python -m pip freeze > /external/CEDDAR_runs/environment/requirements-resolved.txt
```

The CPU wheel command follows [PyTorch's v2.2.0 instructions](https://pytorch.org/get-started/previous-versions/#v220).
On macOS omit the CPU-index step. GPU installations need the matching platform wheel
from that page; do not install the CPU wheel for a GPU test. `requirements.txt` pins
the directly used model packages against the available Python 3.11 reference
environment, including NumPy 1 and Zarr 2. It is **not** a complete transitive lock or
a reconstruction of the submitted LUMI environment. Preserve the resolved freeze
on ATMO; each run also records installed package versions. Optional correlation and
ERA5 download dependencies live in `requirements-preprocess.txt`; the download client
is not locally validated/pinned and CDO is a separate host executable.

## Paths and launch

```bash
export CEDDAR_RUNS=/external/CEDDAR_runs/my_experiment
export DATA_DIR=/external/Data_DiffMod_small
bash /path/to/CEDDAR/repro/run_model.sh \
  --config_path /path/to/CEDDAR/sbgm/config/component_study/B1s/B1.yaml \
  --mode generate --device cpu
```

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

## Smoke tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest repro.test_runtime repro.test_provenance -v
bash repro/01_small_test/run_small_test.sh
# Real inputs and a trusted trained checkpoint matching this exact configuration:
bash repro/01_small_test/run_small_test.sh \
  --config /path/to/model.yaml --data-root /external/data \
  --checkpoint /external/checkpoints/model.pth.tar --device cpu --steps 2
```

The default smoke test creates synthetic precipitation Zarr/static inputs, constructs
the configured network, saves/reloads random weights, samples one date/two members
at 32×32, checks inverse-transform round trips, reloads physical outputs and evaluates
CRPS. Missing/nonfinite arrays or failed assertions produce a nonzero exit. Real-input
mode retains the config's spatial size; missing real data never falls back to synthetic.
`--device cuda` runs the same check on an available GPU; `--threads` controls CPU
threads. `--output` must be a new external directory; otherwise a timestamped one is
created under `CEDDAR_RUNS/smoke`. Keep `smoke_result.json` and the manifests.

Random-weight metrics are **not skill scores**. This is not a training test, a full
evaluation-suite test, or a σ* experiment. `02_reduced_run` remains an explicit
training/generation/evaluation experiment with its original scientific YAML.

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
