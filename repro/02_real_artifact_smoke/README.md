# 02_real_artifact_smoke

Loads one paired real DANRA/ERA5 date, the explicit archived checkpoint and supplied
training statistics. Checks weight compatibility, inverse transforms, two-member
EDM inference, saved physical arrays and CRPS. Defaults: CPU, one thread, test split,
two steps, and the configured 128×128 crop. It never trains or substitutes fixtures.

The supplied YAML retains the architecture/conditioning/normalization settings of
repository `B1_GSDF_RGBCE.yaml`. It is a **candidate**, not proof of the archived
run's provenance: compare it with the original training config and statistics from
LUMI. Strict weight loading detects incompatible weights, but cannot prove that
normalization statistics or other non-weight settings are historically correct.

After copying the checkpoint from LUMI, set these in **tcsh** from the repository root:

```tcsh
setenv DATA_DIR /home/theaqg/CEDDAR_migration/Data/Data_DiffMod
setenv PUBLISHED_CHECKPOINT /home/theaqg/CEDDAR_runs/checkpoints/EXACT_CHECKPOINT_FILENAME.pth.tar
setenv STATS_LOAD_DIR /path/to/matching/training/statistics/root
setenv CEDDAR_RUNS /home/theaqg/CEDDAR_runs/repro/02_real_artifact_smoke
bash repro/02_real_artifact_smoke/run_real_artifact_smoke.sh --preflight-only
bash repro/02_real_artifact_smoke/run_real_artifact_smoke.sh
```

Replace the checkpoint filename and statistics root with existing paths. The longer
`paper1_original/final_model/SBGM_SD/models_and_samples/trained_models/` location also
works. `PUBLISHED_CHECKPOINT` must name the actual file, including its extension.
Moving the checkpoint later only requires changing that variable. The checkpoint
name can retain `TIMESTEPS_56`; the smoke uses two steps without renaming the file.
No other file is selected through `CHECKPOINT_NAME` or a derived experiment name.

To use an archived YAML instead of the candidate, run
`setenv SMOKE_CONFIG /path/to/model.yaml` in tcsh, or pass `--config /path/to/model.yaml`.
The smoke overrides only its workload/device/output paths and uses a recorded seed
of 504; model and EDM parameters remain from the supplied config. The generation
manifest records actual sampler arguments, including the unchanged v1.0.2 mode default.

The statistics root must contain the usual `DANRA/prcp/train/` and `ERA5/prcp/train/`
JSON tree (plus any other variables in your config). Bundled statistics may be used
only after confirming they match the checkpoint's training setup. Static files default
to `DATA_DIR/data_lsm/truth_fullDomain/lsm_full.npz` and the analogous topography path.
The loader expects paired split Zarr stores under each model/variable directory.

Preflight writes `preflight_result.json` and a manifest; it does not prove inference
works. A complete successful run writes `smoke_result.json`, generation arrays and
manifests under a fresh timestamped directory. `--output /external/new-directory`
selects an explicit destination; existing destinations are rejected. `SAMPLE_DIR`
and `EVAL_DIR` do not control these smoke outputs. Missing inputs, incompatible weights
or failed checks return nonzero, and no unconditional PASS message is printed.

This checks portability, not predictive skill or historical artifact authenticity.
The CPU/GPU environment must first pass `pip check`; see [setup](../README.md).

Manifest source fields distinguish supplied artifacts from locally generated fixtures.
They do not label an arbitrary supplied checkpoint as trained or scientifically verified.
