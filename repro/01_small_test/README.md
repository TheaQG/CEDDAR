# 01_small_test: CPU/GPU setup and inference check

See [the setup guide](../README.md) for environment, paths and real-checkpoint options.

```bash
bash repro/01_small_test/run_small_test.sh
```

Synthetic installation smoke test. Tests if CEDDAR can import and execute its core generation/evaluation plumbing on this machine.

Pass criteria: all imports work, generation returns finite arrays, inverse transform works, CRPS is finite, outputs stay outside the repository, provenance is written.

This now runs a bounded inference smoke test, not training. Defaults: synthetic Zarr
inputs, random weights, one date, two ensemble members, two EDM steps, 32×32 crop.
It checks data loading, model/checkpoint construction, inverse transforms, saved
physical arrays and CRPS. Artifacts are outside source under `CEDDAR_RUNS/smoke/`.
`smoke_result.json` is written only on success; failures exit nonzero.

Use [level 02](../02_real_artifact_smoke/README.md) for real artifacts with an
explicit checkpoint, matching statistics and configuration. Use `--device cuda` for the same test on a GPU. There is no training/skill
acceptance threshold or hardware-independent runtime promise. The original YAML
is retained unchanged as the default architecture/configuration source; smoke-only
overrides are recorded in the resolved run manifest.
