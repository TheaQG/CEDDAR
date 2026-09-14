# Infrastructure validation, 2026-09-14

Baseline: `v1.0.2` / `a349ffce2e959d8ad42facd8c9e31f7cf47206b6`.
Local reference: Python 3.11.4, Torch 2.2.0, CPU, one Torch thread for smoke inference.

| Check | Result |
|---|---|
| Path/provenance unit tests | Five passed: defaults/overrides, source/symlink rejection, resolved config guard, actual sampler defaults, checkpoint hash/RNG preservation/unique manifests. |
| Synthetic end-to-end smoke | Passed: actual Zarr loader, configured network, checkpoint round trip, 2 members × 2 EDM steps at 32×32, inverse round trip, physical NPZ reload and finite CRPS. |
| Launcher from unrelated cwd | Passed for smoke and reduced-config CLI `--dry_run`; no training launched. |
| Frozen runner comparison | Same model/random checkpoint, synthetic batch and seed 7382. Baseline generation module loaded directly from `git show v1.0.2:sbgm/generate/generation.py`. Model/physical ensembles, model/physical PMM, HR and LR physical arrays bitwise identical to revised runner. |
| Scientific source/config diff | Sampler, network, data modules, transforms, loss/metric formulas and scientific YAML files unchanged. Model construction only adds explicit CPU placement. |
| Shell syntax / diff whitespace | Passed for portable wrappers / revision diff. |
| Existing Conda environment `pip check` | Failed: missing `fsspec` (Torch dependency), missing `xarrayutils`/`xgcm` (xmip), incompatible giddy (pysal). User environment not modified. |

Synthetic timing was approximately 2–3 seconds after imports for this tiny case.
Random-weight scores are not physical skill evidence or runtime estimates for the
full experiment. The comparison covers one input/seed; it does not prove cross-platform
bitwise reproducibility. A fresh Linux requirements installation, real-data/checkpoint
inference, training, the full evaluation suite and GPU execution remain untested here.

The revised smoke and unit tests can be run using [README.md](README.md). Preserve
ATMO's smoke JSON/YAML and resolved environment alongside that machine's run outputs.
