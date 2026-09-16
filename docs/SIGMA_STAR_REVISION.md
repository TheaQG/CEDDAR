# Sigma* revision and ATMO usage

The historical findings are in [SIGMA_STAR_AUDIT.md](SIGMA_STAR_AUDIT.md). The frozen archive and manuscript PDFs have not been edited. These changes belong to `paper1-revision-2026`; they are not a reinterpretation of what v1.0.2 executed.

## Changes for review

1. **Sampler and forwarding.** `sbgm/sigma_control.py` contains the schedule construction and validation previously embedded in `edm_sampler`. Generation and training previews now forward the same complete set of sigma* arguments. Invalid modes, nonpositive parameters, unresolved/reversed ramps and increasing schedules fail explicitly. A ramp needs at least two distinct mapped indices: the default two-step smoke cannot resolve a late ramp; use 56 steps for that check.
2. **Initialization.** New default `sigma_star_initial_state: schedule` initializes with `sigma_max * f_0`. It uses the analytic first endpoint to preserve the alpha=1 baseline exactly, rather than introducing a float32 roundoff change from substituting the reconstructed first node. `legacy_sigma_max` retains the frozen initial amplitude. To compare with the frozen standard runner, explicitly choose **global + legacy_sigma_max**. This option does not reproduce invalid/degenerate frozen inputs, which are now rejected.
3. **Provenance and isolation.** The generation manifest binds the actual sampler arguments and records base/scaled nodes, factors, ramp indices, initial SD, effective churn bounds, gamma, nominal injected-noise SD and step deltas. No random draws are made for provenance. Sigma sweeps reject duplicate output names and existing sigma directories, so they cannot silently mix modes or overwrite a prior run. The original sequential RNG seeding across the grid remains the default. An explicit paired-noise option is now available; see [paired pilots](SIGMA_STAR_PAIRED_NOISE.md).
4. **Evaluation.** Sigma evaluation honors `paths.evaluation_dir`. Mode annotations come from generation manifests; requested config is stored separately. The ATMO workflow requires matching sampler provenance and a completion manifest, and fails on missing arrays or differing date lists. Historical evaluation without manifests can still be used only with the strict flag disabled; its mode is unknown and is not annotated from config. Metric definitions are unchanged: PSD of the physical ensemble mean, PMM/LR low-pass correlation, and the configured rainy-land CRPS. Plot limits, SEM/NaN handling and labels are now corrected; see [plot corrections and the proposed spread metric](SIGMA_STAR_PAIRED_NOISE.md). The spread metric is a proposal only.
5. **ATMO preparation.** `repro/run_sigma_star.sh` uses the active environment and the existing `F_final_test_eval.yaml` architecture. It writes a resolved config to a fresh external run directory. Preparation defaults to a small **validation** pilot: one date, two members, 56 steps, alpha `[0.95, 1.0, 1.05]`, late ramp .60–.85, schedule-matched initialization. These are explicit pilot overrides, not reconstructed manuscript settings. Preparation loads no checkpoint and generates no samples. An explicit checkpoint filename is supported through `paths.inference_checkpoint` for the sigma-grid path, including renamed archived files.

The F configuration's `sampler_grid.sigma_scale: [1.1]` is **not consumed by sigma-star generation**. This workflow retains `edm.sigma_min=0.002`, `sigma_max=80`, `rho=7`, churn `2/40/80/1`. Whether historical sigma* results used the tuned endpoints 0.0022/88 still needs run-specific evidence. Do not silently incorporate 1.1 into the revision.

No training/loss, network architecture, weights, transforms, physical clipping rule or scientific metric definitions were changed. Non-unit sigma* outputs will change when the previously ignored late ramp is honored or global initialization is corrected. Preserve and label historical and revised results separately.

## ATMO commands, after reviewing and transferring this diff

Run from the repository with the `ceddar-revision` environment active. In your `tcsh` shell, keep the input paths from the successful real-artifact smoke:

```tcsh
setenv DATA_DIR /home/theaqg/CEDDAR_migration/Data/Data_DiffMod
setenv STATS_LOAD_DIR /home/theaqg/CEDDAR_GMD_revision/repro/assets/stats/statistics_run/stats
setenv PUBLISHED_CHECKPOINT /home/theaqg/CEDDAR_runs/paper1_original/checkpoints_from_lumi/B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56.pth.tar
setenv DEVICE cpu
setenv SIGMA_RUN_DIR /home/theaqg/CEDDAR_runs/paper1_revision/sigma_star_pilot_001

python -m unittest repro.test_sigma_star repro.test_provenance repro.test_workflows repro.test_runtime -v
bash repro/run_sigma_star.sh prepare --run-dir "$SIGMA_RUN_DIR"
cat "$SIGMA_RUN_DIR/resolved_config.yaml"
```

The following are separate, explicit actions. They have **not** been run against the ATMO checkpoint in this revision:

```tcsh
bash repro/run_sigma_star.sh generate --run-dir "$SIGMA_RUN_DIR"
bash repro/run_sigma_star.sh evaluate --run-dir "$SIGMA_RUN_DIR"
```

Preparation overrides old smoke output/cache environment variables so outputs remain beneath the chosen run directory. Inputs are not copied or modified. Output locations are `samples/generation/<model-key>/sigma_star=.../`, `evaluation/<model-key>/prcp/sigma_control/`, and `logs/`. Generate/evaluate use the saved resolved config and reject new preparation overrides. For a different mode, initial-state policy, grid, split, member count or date cap, prepare a **new** run directory with the corresponding flags; use `--help` for their names. A failed/partial generation also requires a new directory; resume is intentionally not implemented. The LUMI Slurm script is not needed on ATMO.

## Validation scope

**Validation: 20 CPU unit/workflow tests passed on 15 September 2026; changed Python files parse, shell syntax and `git diff --check` pass.**

The regression suite checks frozen global reference outputs with a toy denoiser, the alpha=1 baseline, initial-state scaling, denoiser sigma arguments, terminal Euler behavior, default ramp indices, unchanged early churn, actual churn magnitude versus manifest, invalid settings before RNG draws, full runner argument forwarding, provenance mismatch/completion rejection, external output isolation and preparation without checkpoint loading. Existing portability/workflow tests are included. These are CPU software checks; they do not establish a physically admissible alpha interval, calibration, monotonicity, or historical figure provenance. The local Python environment emitted an existing NumPy binary-compatibility warning during an import; ATMO should run the same tests in its own environment.

Suggested minimal manuscript wording, subject to the chosen experiment and results: “We perturb the inference-time EDM noise schedule by a dimensionless factor alpha. In late-ramp mode the multiplier transitions smoothly from one to alpha over specified sampling indices. The modified noise levels enter denoiser conditioning, preconditioning and numerical integration. We evaluate the resulting changes in spatial structure and ensemble behavior empirically; the construction does not guarantee monotonic control or physical admissibility.”
