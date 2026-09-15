# Frozen v1.0.2 sigma* audit (15 September 2026)

## 1. Executive finding

In frozen v1.0.2, sigma* multiplies the numerical EDM noise schedule; it is not an independent multiplier of an additive Gaussian noise term. Consequently it changes the sigma supplied to the denoiser, its preconditioning, and the Euler/Heun integration path. The sampler supports global scaling and a smooth late ramp, but the generation runner forwards only `sigma_star`, so the published generation entry points use the sampler's **global default even when the YAML requests late_ramp**. Under global scaling, the code also scales the churn window and the amplitude of active churn perturbations. However, the initial Gaussian state retains standard deviation `sigma_max`, rather than `sigma_star * sigma_max`. With the default late ramp called directly, the initial state would be consistent and the high-sigma churn would precede the ramp. The main text and SI alternate between incompatible noise-only and schedule-scaling explanations, and their late-ramp description is not the behavior reached through the frozen runner. The historical figures cannot yet be linked to an exact executed config, checkpoint hash and source revision. The present evidence therefore warrants **D for historical result provenance**, with confirmed implementation inconsistencies that require new inference if the revised contribution is to describe the intended late-ramp method; no retraining is required.

## Evidence boundary

Implementation evidence is the local `code/CEDDAR-1.0.2` archive, not the revision branch. All 366 release-tracked files were checked against the Git blobs and are byte-identical to commit `a349ffce2e959d8ad42facd8c9e31f7cf47206b6` (`v1.0.2^{commit}`; the annotated tag object is `55442dddaae1649c676a7403eff415be5c2fb388`). Paths and line ranges below refer **only to this frozen commit**. They can be inspected with `git show v1.0.2:<path>`; line numbers on the revision branch will differ.

The supplied PDFs were inspected through text extraction and rendered equations/figures. Main manuscript PDF pages coincide with printed pages; SI references use printed S-page numbers, which also equal PDF page numbers for the cited pages.

| Source | SHA256 |
| --- | --- |
| `manuscript/manuscript_GMD.pdf` | `8bda3a78e16c76746e243fbf755932abb91e60ccfc9923eae02bf6d4e521f879` |
| `manuscript/SI_GMD.pdf` | `ac1918a674f3dc970073291b6c4fada20c9837c62f1773929f8a48123c6a6fd1` |

This audit does not independently verify LUMI runs. The user confirmed the ATMO statistics and supplied a passing CPU smoke result: 20190101, two members, 56 steps, CRPS 0.6296967, 11.29 seconds. That establishes a working current inference path, not historical sigma* provenance or scientific calibration. The frozen tree and PDFs remain unchanged. Subsequently authorized revision changes and toy regression tests are described separately in [SIGMA_STAR_REVISION.md](SIGMA_STAR_REVISION.md).

## 2. Code provenance map

| Class | Frozen file/function | Relevant evidence |
| --- | --- | --- |
| A: sampler | `sbgm/score_sampling.py::edm_sampler`, 14–43, 125–190, 210–291 | Defaults, base schedule, factors, churn window, fixed initial amplitude, Euler/Heun |
| A: denoiser | `sbgm/score_unet.py::EDMPrecondUNet._precond/forward`, 104–165 | Actual sigma embedding and preconditioning |
| B: CLI | `sbgm/cli/main_app.py`, 209–217; `cli/launch_generation_sigma_star.py::run/main`, 31–39, 101 onward | `--mode sigma_star_generation` dispatch; standalone grid/M/max-dates/seed overrides |
| B: sweep | `sbgm/generate/generation_sigma_grid_main.py::generation_sigma_grid_main`, 62–203 | Seeds, checkpoint, split, grid, requested mode/ramp copied into `cfg.edm` |
| B: runner | `sbgm/generate/generation.py::GenerationRunner.run`, 522–622 | Actual sampler kwargs omit mode and every ramp argument |
| B: training previews | `sbgm/training.py`, 1504–1523 | Also forwards alpha without mode/ramp; not a training-loss use of alpha |
| C: parsing | `sbgm/utils.py::load_config`, 498–513 | OmegaConf/environment resolution; sampler defaults remain Python defaults |
| C: runner config | `generation.py::GenerationConfig`, 36–65; sigma-grid builder, 29–50 | N fallback 40; M/full seed supplied separately; no ramp fields |
| D: final test candidate | `bash_component_study/F_final_test_eval.sh`, 90–103; `sbgm/config/component_study/F_final_test_eval.yaml`, 202–218, 314–356 | Test split; sigma commands commented; requested late ramp; ten-value grid |
| D: validation candidate | `bash_component_study/B1s/run_B1_GSDF_RGBCE.sh`, 90–100; corresponding YAML, 153–218, 314–347 | Active sigma commands; validation split; same model and ten-value grid |
| D: incomplete candidate | `bash_scripts/paper1_final.sh`, 83–86 | Refers to absent `sbgm/config/paper1_final_config.yaml` |
| E: sigma evaluation | `sbgm/evaluate/evaluate_prcp/eval_sigma_star/evaluate_sigma_control.py::run`, 13–69 | Evaluation orchestration; metadata copies **requested** ramp from config |
| E: metrics | `.../metrics_sigma_control.py::evaluate_sigma_control`, 400–674 | Date loading, PSD of ensemble mean, PMM correlation, masked CRPS, aggregation |
| E: figures | `.../plot_sigma_control.py::plot_sigma_control`, 23 onward; PSD plot, 567–576, 683 onward | STD/SEM; figure ramp annotation comes from config-derived JSON |

Other aliases are distinct mechanisms: `generation_sampler_grid_main.py`, 204–205, 238–260, multiplies both **input endpoints** by `sigma_scale`. This scales the initial amplitude but does not scale the churn window with that factor. It is not interchangeable with `sigma_star`. Its evaluations/summaries are in `evaluate_sampler_grid_main.py` and `summarize_sampler_grid*.py`. `scale_utils.py::sigma_star_from_preserve_scale`, 131–145, maps wavelength to a blur-equivalent value in pixels; `monitoring.py::compute_sigma_star_from_loader`, 215 onward, and the commented integration at 284–294 do not connect this spatial proxy to the active dimensionless sigma* sampler control. No active spatial-frequency filter is applied by sigma*.

## 3. Exact implemented sampler behavior

### Call chain and configuration precedence

`F_final_test_eval.sh` (after manually enabling its commented sigma command) -> `main_app --mode sigma_star_generation --config_path ...` -> `launch_generation_sigma_star.run(cfg)` -> `generation_sigma_grid_main(cfg)` -> `GenerationRunner.run(...)` -> `edm_sampler(...)` -> `EDMPrecondUNet.forward(...)` -> inverse transforms and saved daily ensembles/PMM.

The sweep seeds Torch/NumPy once, before constructing the model and processing the whole grid (69–72). It loads `network_params` from the derived model filename, sets eval mode (82–88), resolves `val/valid/validation` to `valid` (91–123), and reads `full_gen_eval.sigma_star_grid` (125–130). Mode/ramp precedence is `full_gen_eval.sigma_control` over `edm` over defaults (133–140). Each alpha is written into `cfg.edm` along with the requested ramp (146–187), but the runner passes **only alpha** at 600–622. Logging the requested mode in the grid launcher does not override the omitted sampler argument. The sampler's own log at 76–81 is evidence of the arguments actually received.

Outputs are under `<paths.sample_dir>/generation/<model-key>/sigma_star=<alpha to 2 decimals>/`. Physical inverse transformation and optional extreme clipping occur after sampling (`generation.py`, 626–688); they also affect the fields later evaluated. PMM is computed in model space before inversion. The legacy `meta/manifest.json` (815–830) records M, N, seed and date count, but not the actual sigma mode, checkpoint hash or trajectory. The RNG is not reset per alpha: equal initial seed does not mean identical member noise across grid points, and grid order can affect results.

### Mathematics reconstructed from the code

Let alpha denote the dimensionless `sigma_star`, N the number of positive sigma nodes, and i=0,...,N-1. Ignoring float32 roundoff,

\[
q_i=\left[\sigma_{\max}^{1/\rho}+\frac{i}{N-1}
(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho})\right]^\rho,
\qquad s_i=f_i q_i,\quad s_N=0.
\]

In global mode, `f_i = alpha`. In late-ramp mode,

\[
i_0=\operatorname{round}(r_{start}(N-1)),\quad
i_1=\operatorname{round}(r_{end}(N-1)),\quad
t_i=\operatorname{clip}\left(\frac{i-i_0}{\max(i_1-i_0,1)},0,1\right),
\quad f_i=1+(\alpha-1)t_i^2(3-2t_i).
\]

The fractional indices are clamped to valid indices and `i1 >= i0`. Python `round`, including ties-to-even, is used, **not floor**. For N=56, the defaults give i0=33 and i1=47 (zero-based); scaling first differs from one at index 34. If both sigma thresholds are supplied, the first **base-schedule** crossings replace these indices. Unreached thresholds fall back to truncated `int(0.6*(N-1))` / `int(0.85*(N-1))`, not the configured fractions. With coincident indices the factor jumps after that index; a final-index coincidence can mean no control at all. Reversed thresholds, partial thresholds, invalid modes and nonpositive alpha are not properly rejected in the frozen implementation: any non-`global` mode string enters the ramp branch, and a partial threshold pair falls back to fractions.

Define churn bounds `(L,U)=(alpha*S_min, alpha*S_max)` globally, or `(S_min,S_max)` for a ramp. At each step,

\[
\gamma_i=\mathbf{1}_{L\le s_i\le U}\min(S_{churn}/N,\sqrt2-1),\quad
\hat s_i=(1+\gamma_i)s_i,\quad
\hat x_i=x_i+S_{noise}\sqrt{\hat s_i^2-s_i^2}\,z_i,
\]

where independent Gaussian `z_i` is drawn only when churn is active. Then, writing D for the denoiser,

\[
d_i=\frac{\hat x_i-D(\hat x_i,\hat s_i,c)}{\hat s_i},\quad
x_E=\hat x_i+(s_{i+1}-\hat s_i)d_i.
\]

For i<N-1, Heun evaluates `D(x_E,s_{i+1},c)` and averages the two derivatives with the same step delta. The final iteration uses Euler to zero without a network call at zero; algebraically the final output is the last denoised prediction. The initial state in frozen code is **`x_0 = sigma_max * z_0`**, irrespective of alpha. Thus global initialization has standard deviation / nominal first sigma = `1/alpha`. This is a demonstrable inconsistency with schedule-matched initialization, not proof by itself of physical invalidity.

| Quantity | Actual effect of alpha |
| --- | --- |
| Numerical sigma nodes | All positive nodes globally; a suffix under a directly called ramp; terminal zero unchanged |
| Network sigma and drift | Both Euler and Heun network calls receive modified sigmas; drift denominators also change |
| Denoiser coefficients | `c_in=1/sqrt(s²+sigma_data²)`, `c_skip=sigma_data²/(s²+sigma_data²)`, `c_out=s*sigma_data/sqrt(s²+sigma_data²)`, `c_noise=0.5*log(s)` change (`score_unet.py`, 104–163) |
| Churn | Global active indices are preserved mathematically by scaling the bounds; active injection SD scales by alpha. Boundary membership is subject to float32 roundoff |
| Default late-ramp churn | Window 40–80 occurs before indices 33–47; for the candidate alpha grid the injected churn perturbations are unchanged |
| Other random terms | No independent alpha multiplier. Initial Gaussian amplitude is unchanged in frozen code; optional CFG noise-null conditioning is not scaled directly |
| Integration | Modified sigmas change step deltas and evaluated states. Under uniform scaling, delta/sigma-hat is invariant when gamma matches; bigger absolute deltas do not establish stronger denoising |
| Initial/last positive level | Global nominal first and last positive levels are alpha*80 and alpha*0.002, while initial state still has SD 80; ramp normally retains first 80 and ends at alpha*0.002 |
| Model/training | Weights stay fixed. `losses.py::EDMLoss.sample_sigma`, 28–29, samples a lognormal distribution; 47–51 adds sigma-scaled noise. It does not use alpha or a fixed inference timestep schedule |

Optional classifier-free guidance would add sigma-dependent guidance weighting and may draw null-condition noise (`score_sampling.py`, 53–66, 96–99, 216–252). It is disabled in the candidate configs. There is no basis for interpreting a changed sigma embedding as a measured change in denoiser confidence.

## 4. Manuscript-code comparison

Classification: **I** directly implied by implementation; **E** empirical observation/figure evidence only (historical data not independently re-evaluated); **H** reasonable hypothesis; **U** unsupported or incorrect as a mechanistic claim. Severity concerns interpretation/provenance, not numerical effect size.

| Location | Claim | Frozen evidence / agreement | Class; severity | Recommended action |
| --- | --- | --- | --- | --- |
| Abstract p1; conclusion p26 | Inference control without retraining; predictable variability | Inference-only yes; predictable metric direction not guaranteed | I for inference-only, E/H for response; moderate | Retain inference-only diagnostic, qualify response by actual tested cases |
| Main 2.4 pp7–8 | Only injected noise is multiplied | Schedule, denoiser input sigma and integration all change | U; high | Replace noise-only explanation |
| Main 2.4 p7 | Late-stage rescaling was implemented/used | Direct sampler supports it; standard runner drops mode | U for published call chain; high | Establish run source or regenerate explicitly wired late-ramp results |
| Main 2.4 p7 | Network trained on a fixed sigma schedule | Training samples a lognormal noise distribution | U; moderate | Distinguish training noise distribution from inference discretization |
| Main 2.4 pp7–8 | Lower alpha increases confidence/gradients; higher alpha gives smaller, conservative updates | Coefficients change, but sign/magnitude of nonlinear response not implied | H/U as assertion; high | Present as hypotheses; remove deterministic causal language |
| Main 3.2 pp10–11 | Final tuned sampler; final grid 0.90:0.05:1.15; preliminary .70/.85/1/1.15/1.30 | F config has .80:0.05:1.25 and separate sampler-grid scale 1.1, unused by sigma launcher | Unclear historical agreement; high | Recover resolved configs and figure source data; list base endpoints actually used |
| Main 4.5 pp19–20; Fig9 p22 | Lower alpha shows more fine texture, higher alpha smoother fields/spread changes | Qualitative examples exist, but no guaranteed mechanism; examples show five values .8/.9/1/1.1/1.2 | E, H for causal account; moderate | Describe observations for identified runs and quantify spread independently |
| Main 4.5 p20 | Ensemble spread (CRPS) | CRPS is an ensemble score, not a spread statistic | U; high | Separate CRPS from spread; report member variability directly |
| Main 4.5 p20; Fig10 p23 | Excess PSD proves off-manifold behavior; red region outside HR range | Higher PSD than an HR reference curve is not a test of manifold membership or full physical admissibility | H/U; high | Say excess power relative to the reference; avoid manifold proof |
| Fig10 p23; S4 pS20 | Annotation says late_ramp | Plot reads JSON copied from requested config, not the sampler | U as provenance; high | Use actual sampler manifest/log; regenerate labels and possibly samples |
| Fig10 p23 | Ensemble PSD/correlation/CRPS | Metrics use PSD of **ensemble mean**, low-pass **PMM** vs LR correlation, and HR>1 mm/day land CRPS in F config | Incomplete labels; high | Specify reductions, mask and rainy selection in caption; PSD(mean) is not mean(PSD) |
| Main discussion 5.1 p23 | Predictable, monotonic control; avoids mode collapse | No guarantee; Fig10 itself has a U-shaped PSD slope over its shown range | U for general guarantee; high | Restrict to observed metric-specific trends; remove claim of avoiding collapse |
| SI S1.7 pS7 Eq13; pS9 | Scale only stochastic forcing, leaving drift unchanged | Contradicts actual drift and schedule changes | U; high | Replace with reconstructed schedule/churn/Heun equations |
| SI S1.7 pS8 Eqs14–16 | Smoothstep schedule and floor indices | Factors agree for supported ramp; indices use round, not floor; N=56 end index 47 rather than floor 46 | Partial; high mode / low indexing | Correct discrete construction and identify whether it ran |
| SI S1.7 ppS8–9 | More injected noise causes stronger drift/smoothing; less causes weaker drift/more detail | Default late ramp changes no high-sigma churn; nonlinear denoiser response not established; main text gives different stronger/weaker account | U/H; high | Remove noise-forcing explanation; use measured trajectories and metric observations |
| SI S1.7 pS9 | Temperature analogy, nearby manifold vs off-manifold | Qualitative analogy only; not sampling a proven temperature family; lognormal training has no hard sigma interval | H; moderate | Label analogy explicitly and define empirical validity checks |
| SI S4.3 ppS17/S20 | All diagnostics monotonic; correlation increases when alpha decreases | Shown correlation increases with alpha, while slope is nonmonotonic | U for direction/general claim; high | Correct from metric tables, not narrative expectation |
| SI S3/S4/S6 ppS19–21 | Exploratory grid and validation split | Broad five-value grid appears in figures; exact executed config absent; validation is a caption claim | E/unconfirmed provenance; high | Recover these runs separately from final test sweep |
| SI S5 caption pS21 and prose pS20 | STD includes inter-ensemble variability; SEM shows response uncertainty | Code aggregates one metric per date; STD is across dates, not a member-spread decomposition; nominal SEM ignores temporal dependence | U/incomplete; moderate | Label date variation correctly; avoid equating it to ensemble spread |

Additional evaluation limitations to retain in interpretation: PSD/slope are computed on whole fields without the land mask passed to those routines, whereas correlation/CRPS can be land masked. Missing folders/dates/arrays are skipped; per-metric NaNs can also make sample counts differ. Plot SEM uses total rows per alpha rather than finite counts per metric. These are reasons to inspect saved metric tables and date membership before relying on uncertainty bars. They were not silently redefined in the sampler revision.

## 5. Experiment provenance

“Confirmed configuration” below confirms what the frozen file specifies, not that the historical job executed it. F denotes `sbgm/config/component_study/F_final_test_eval.yaml`; B denotes `.../B1s/B1_GSDF_RGBCE.yaml`.

| Quantity | Value | Evidence | Confidence |
| --- | --- | --- | --- |
| Architecture/model key | `B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56` | F architecture plus `utils.get_model_string`; `training_utils.py`, 1015–1018 | Confirmed config/derived key; figure linkage unclear |
| Checkpoint | Derived key + `.pth.tar`; `network_params`, not EMA | Sigma grid 82–88 | Confirmed loader behavior; exact historical bytes/hash not recoverable |
| Training seed | 504 | F/B `training.seed`, 153 | Confirmed config; actual checkpoint training provenance unclear |
| Sampler / N | EDM Euler/Heun, 56 positive steps + terminal zero | F 202–218; sampler 257–291 | Confirmed config/code; run-specific unclear |
| rho / sigma_min / sigma_max | 7 / 0.002 / 80 | F 207–216 | Confirmed sigma-sweep config |
| S_churn / S_min / S_max / S_noise | 2 / 40 / 80 / 1 | F 212–215 | Confirmed config |
| Separate endpoint scale | `sampler_grid.sigma_scale=[1.1]`; would give 0.0022 and 88 in the **sampler-grid** launcher | F 331–334; sampler-grid 238–260 | Confirmed separate control; tuned historical baseline unclear |
| Main candidate alpha grid | .80,.85,.90,.95,1,1.05,1.10,1.15,1.20,1.25 | F/B 320; Fig10 axes resemble this range | Confirmed config; plausible figure source, not proven |
| Exploratory alpha grid | .70,.85,1,1.15,1.30 | Main p11; SI pS17; S3/S4/S6 | Confirmed paper claim/figure labels; exact config not recoverable |
| Requested mode / ramp | late_ramp / .60–.85; no sigma thresholds | F/B 322–329 | Confirmed config |
| Effective mode via frozen runner | global; ramp fractions inactive | `generation.py`, 600–622; sampler defaults 36–42 | Confirmed call-chain behavior; historical run source remains unclear |
| Initial state | SD 80 independent of alpha | Sampler 210–211 | Confirmed frozen behavior |
| Ensemble size | 32 | F/B 346 | Confirmed config; historical membership unclear |
| Generation seed | 504, once for entire process/sweep | F/B 317; sigma-grid 69–72 | Confirmed config/seed scope; exact historical RNG stream unclear |
| Split | F=test; B=val mapped to valid | F/B 315; grid 100–110 | Confirmed config; exact figure input list unavailable |
| Evaluation period | Paper: test 2019–2020, validation 2016–2018 | Main Table1 p10 | Confirmed paper claim; actual date inventories unavailable |
| Date cap | 1000 | F/B 347 | Confirmed config; not proof that all dates exist/finish |
| Grid/output roots | `/scratch/<account>/<user>/Code/CEDDAR/models_and_samples/{trained_models,generated_samples}` | F shell 51–59; grid output resolver 14–27 | Confirmed script templates; actual account/job paths unclear |
| Evaluation roots | `<sample_dir>/evaluation/<model-key>/prcp/sigma_control` | Frozen evaluator 17 | Confirmed code; separate `EVAL_DIR` was not honored here |
| Metrics/figures | `tables/metrics_by_sigma.csv`, `tables/agg_summary.csv`, `sigma_psd_curves.npz`, `sigma_control_meta.json`, figure PNGs | Metrics/evaluation/plot modules | Confirmed naming; original files absent |
| Figure linkage | Fig10/S4 display config-derived late-ramp annotation; F sigma commands commented | Plot 567–576; F shell 99–103 | Insufficient to identify an executed run |
| Stats/checkpoint on ATMO | User reports correct stats and successfully loaded original checkpoint | User-provided 56-step smoke log | User-confirmed current smoke; not independent historical verification |

## 6. External evidence still required

Retrieve from LUMI, separately for exploratory validation and final test figures:

1. Submitted Slurm script, exact command line, exported paths, job ID and stdout/stderr; particularly `[sampler] sigma*: ... mode=...` and base EDM settings. Grid-launcher mode logs alone are insufficient.
2. Resolved generation YAML and original training YAML; include the missing `paper1_final_config.yaml` if used, all CLI overrides, and whether the 1.1 endpoint scale was copied into `edm`.
3. Exact checkpoint file/hash, checkpoint metadata and training seed history (including W&B run/config if available); source Git commit plus any uncommitted sampler/runner diff.
4. Generation folder `meta/manifest.json`, all available generation YAML/JSON, per-date file inventory and member counts; source data split inventories and matching normalization statistics.
5. Figure source CSV/NPZ/JSON, plotting commands/scripts, source paths, timestamps and original figure files. Preserve config-derived metadata, but do not treat it as proof of actual sampler arguments.

## 7. Rerun recommendation

**D is the best overall historical classification:** exact figure provenance is insufficient. B is ruled out by demonstrable inconsistencies. A remains possible only for results that can be established as an intentional, explicitly described global sampler; it cannot justify claiming late-ramp results. **C applies if the contribution is to remain the intended late-ramp, schedule-matched method:** the frozen standard runner does not execute that method, and global initialization is inconsistent with its scaled first node. New inference is then required. Preserve historical outputs as evidence and generate revision outputs separately. All these sampler experiments use a fixed checkpoint and can run after training on CPU; no new training is implied.

## 8. Minimal verification plan

After the audit, the user authorized bug fixes and toy CPU tests. The revision notes distinguish tests already performed from these outstanding scientific checks:

1. Same-seed toy calls: verify alpha=1 against frozen behavior, explicit legacy global reproduction, first-state amplitude, denoiser sigma arguments and terminal Euler behavior.
2. Log the actual base/scaled sigma arrays, ramp indices, churn-active steps/noise SD and step deltas; verify N=56 ramp indices 33/47 and unchanged early churn in the default ramp.
3. Exercise the real generation runner with controlled inputs to confirm all mode/ramp arguments and initialization policy reach the sampler and manifest.
4. After review, generate one validation date with the known checkpoint, two members, 56 steps and a small alpha grid; inspect finite physical fields, data/date matching, saved metadata and runtime. This is a pipeline check, not a validated scientific range.
5. Only then choose a small, declared validation subset and enough members to examine PSD(mean) versus mean(member PSD), explicit ensemble spread, CRPS and PMM/LR correlation. Check direction/monotonicity per metric without assuming it. A controlled common-noise comparison would require an explicit new RNG protocol; the current revision preserves sequential sweep seeding.
6. Confirm the tuned base endpoints before any publication sweep. If changing from 80/0.002 to 88/0.0022, record that as an additional experiment choice, not as a synonym for alpha.
