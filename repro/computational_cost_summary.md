# CEDDAR computational-cost audit

All three supplied logs contain complete 400-epoch training runs. Only seed 1 contains complete full-generation and evaluation stages. Seed labels follow the user's mapping; the first experiment is named V0, not V0_seed1, and explicit numerical training seeds are not recorded.

## Training

Times are copied from the log bodies without timezone conversion. Training is measured from the timestamp attached to “Starting epoch 1/400” to “Epoch 400/400 completed”. This includes validation, checkpointing and training diagnostics, but excludes initial setup and subsequent quicklook/full generation.

| Seed | Date | Training start | Training end | Wall time | Wall hours | Accelerator-device hours |
|---|---|---|---|---|---|---|
| 1 | 2026-04-11 | 16:59:53 | 21:00:57 | 4 h 1 min 4 s | 4.0178 | 16.0711 |
| 2 | 2026-04-14 | 11:08:04 | 14:59:27 | 3 h 51 min 23 s | 3.8564 | 15.4256 |
| 3 | 2026-04-14 | 11:08:07 | 15:03:53 | 3 h 55 min 46 s | 3.9294 | 15.7178 |

Mean: **3.9345 h**; sample standard deviation: **0.0808 h** (4 min 51 s); range: **3.8564–4.0178 h**. Mean training cost: **15.7381 accelerator-device hours**; sample SD: **0.3233**; sum across the three training runs: **47.2144 device hours**. The longest run is 4.2% longer than the shortest. These are observed differences between runs, not an isolated causal effect of the random seed.

All runs log world_size=4, distributed data-parallel training, AMD Instinct MI250X devices, 68.70 GB memory per reported device, 18,101,315 trainable parameters, and zero non-trainable parameters. All start from scratch and contain 400 distinct epoch-completion records. Shared setup includes 128×128 precipitation fields, batch size 16 per rank (nominal global batch 64), 9,131 training samples, 1,096 validation samples, seasonal sin/cos conditioning, SDF weighting and RainGate. Initial application/data/model setup adds 19 seconds in each run.

## Inference

All three full-generation configurations request 731 days, 32 members and 56 EDM steps. The shared test period is 2019–2020; the completed seed-1 manifest confirms all 731 days and 23,392 generated member fields. Non-main ranks exit and the serial pipeline continues on rank 0 using one exposed accelerator device.

| Seed | First daily ensemble starts | Full generation ends | Last timestamp in file | Complete-run generation time |
|---|---|---|---|---|
| 1 | 2026-04-11 21:01:10 | 2026-04-11 21:17:06 | 2026-04-11 21:54:41 | 15 min 56 s |
| 2 | 2026-04-14 14:59:39 | Not recorded | 2026-04-14 15:11:35 | Not available |
| 3 | 2026-04-14 15:04:05 | Not recorded | 2026-04-14 15:11:35 | Not available |

The configuration messages precede the first daily ensemble by 1, 0 and 1 seconds respectively. Timing from those messages would give 957 rather than 956 seconds for seed 1. The adopted definition excludes checkpoint/data-loader setup and the preceding quicklook, but includes generation-loop data handling, inverse transforms, deterministic summaries, output writes and final manifest handling.

| Seed | Observed interval | Completed daily ensembles in interval | Seconds per 32-member daily ensemble | Daily ensembles per device-hour |
|---|---|---|---|---|
| 1 | complete generation; 956 s | 731 | 1.3078 | 2752.7 |
| 2 | partial interval: start of day 1 to start of day 500; 653 s | 499 | 1.3086 | 2751.0 |
| 3 | partial interval: start of day 1 to start of day 300; 391 s | 299 | 1.3077 | 2752.9 |

For seed 2, day 500 begins at 15:10:32, so 499 preceding daily ensembles span 653 seconds. For seed 3, day 300 begins at 15:10:36, so 299 preceding ensembles span 391 seconds. These partial intervals corroborate approximately 1.31 seconds per daily ensemble; they are not completed 731-day benchmarks. Do not divide their truncated durations by 731 or report a three-run complete-inference mean. Sampler records continue after these last dated progress messages, but do not establish final completion.

Seed 1 uses 956/3600 = **0.2656 device-hours** for generation. Its effective amortized cost is **0.04087 seconds per member**; this is throughput, not independently measured single-member latency.

## Evaluation

Only seed 1 records evaluation: **21:17:06–21:54:41 on 2026-04-11**, or **37 min 35 s** (2,255 seconds). The following approximate stage durations use successive task-dispatch boundaries and include plotting/I/O:

| Evaluation task | Wall time |
|---|---|
| Probabilistic | 36 s |
| Scale | 12 min 26 s |
| Extremes | 20 min 43 s |
| Distributional | 54 s |
| Spatial | 2 min 31 s |
| Temporal | 11 s |
| Features/objects | 11 s |
| Per-date | 3 s |

Seeds 2 and 3 contain no evaluation stage. Evaluation device utilization is not established, so evaluation wall time should not be described as measured accelerator computation. Seed 1's application spans 16:59:34–21:54:41 = **4 h 55 min 7 s**, including initial setup, training, quicklook, generation and evaluation.

## Differences and limits

- Seeds 2 and 3 stop at the same wall-clock second during generation, with no failure traceback or completion marker. This is consistent with copies captured while running; the files alone cannot establish the cause or whether the jobs later completed.
- Logging is interleaved between ranks, with damaged and nonchronological lines near initialization. A non-main-rank exit message appears near the beginning despite its later timestamp. Timing therefore uses explicit rank-0 training milestones, not file position alone.
- Seed 1 logs every generated day and detailed save operations. Seeds 2 and 3 log the first few days and then every 50th day, while retaining sampler messages. Matching reported model settings does not prove identical code/configuration versions; config paths are N/A.
- Training seed integers are not explicitly confirmed. The seed-1 generation manifest gives sampling seed 504; quicklook seeds are 882055, 308582 and 272051. These should not be confused with training seeds.
- Each run first generates a separate quicklook of six days × five members, using 56 steps. Its generation lasts approximately 5, 6 and 6 seconds. It is excluded from full-generation throughput.
- Log-body times are three hours ahead of the UTC-style Z timestamps in filenames. The body does not specify a timezone, so these are reported as logged. Durations are unaffected.
- Device hours are wall time × participating device count, not measured utilization or scheduler billing. LUMI documentation states that an MI250X module contains two separately exposed compute dies (GCDs). The logged 68.70 GB is consistent with 64 GiB per GCD, so four logged devices likely mean four GCDs rather than four whole MI250X modules. Confirm allocation/binding from the job script/accounting before giving card counts or billed GPU-hours. Idle devices may remain reserved after non-main ranks exit.
- Seed 1 has plotting-layout warnings; no fatal error is recorded. Missing optional LR/residual fields are described as expected in these configurations.

Hardware reference: [LUMI-G documentation](https://docs.lumi-supercomputer.eu/hardware/lumig/). Billing distinction: [LUMI billing policy](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/).

## Suggested manuscript wording

Across three training-seed runs, the 18.1-million-parameter CEDDAR configuration required 3.93 ± 0.08 h for 400 epochs (mean ± sample standard deviation), using four exposed AMD MI250X accelerator devices in distributed data-parallel mode. This corresponds to 15.74 ± 0.32 accelerator-device hours per training run. In one fully recorded inference run, generation of 32-member ensembles for all 731 test days using a 56-step EDM sampler on one exposed accelerator device required 15 min 56 s, or 1.31 s per daily ensemble. Generation timings include data handling and output writing; initialization and subsequent evaluation are excluded. Partial generation intervals from the other two runs showed comparable throughput.

## Selected source evidence

### Seed 1 — V0__d0ca0d42__20260411T135934Z.log

Source: [V0__d0ca0d42__20260411T135934Z.log](/var/folders/nj/qv09n8sx0tsfmf6z0ktk_pwjx80ybt/T/codex-file-preview-Vx3f8t/V0__d0ca0d42__20260411T135934Z.log)

```text
L14: 2026-04-11 16:59:34 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.training_main |           Runtime: distributed=True rank=2 local_rank=2 world_size=4 requested_device=cuda:2
L137: 2026-04-11 16:59:53 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.training_main |           ▸ Number of trainable parameters in model: 18,101,315
L188: 2026-04-11 16:59:53 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.training | 
L190:       ▸ Starting epoch 1/400...
L7136: 2026-04-11 21:00:57 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.training | → Epoch 400/400 completed. 
L7313: 2026-04-11 21:01:09 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.generate.generation | [generation] cfg: ensemble_size=32, steps=56, save_space=physical, predict_residual=False, out_root=/scratch/project_465002493/quistgaa/Code/CEDDAR/models_and_samples/generated_samples/generation/V0__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56
L7316: 2026-04-11 21:01:10 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.generate.generation | [generation] Generating for date 20190101 (1/731) with ensemble size 32.
L13158: 2026-04-11 21:17:04 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.generate.generation | [generation] Generating for date 20201231 (731/731) with ensemble size 32.
L13167: 2026-04-11 21:17:06 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.generate.generation | [generation] Done. Wrote 731 days to /scratch/project_465002493/quistgaa/Code/CEDDAR/models_and_samples/generated_samples/generation/V0__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56. Manifest: {'model': 'V0__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56', 'ensemble_size': 32, 'sampler_kind': 'edm', 'sampler_steps': 56, 'seed': 504, 'n_days': 731, 'save_space': 'physical', 'physical_dtype': 'float32', 'stationary_cutout_cfg': True, 'lsm_stationary_observed': True}
L13183: 2026-04-11 21:17:06 | INFO    | V0__d0ca0d42__20260411T135934Z | sbgm.evaluate.evaluation | [EvaluationRunner] Starting evaluation with tasks: ['prcp_probabilistic', 'prcp_scale', 'prcp_extremes', 'prcp_distributional', 'prcp_spatial', 'prcp_temporal', 'prcp_features', 'prcp_dates']
L14725: 2026-04-11 21:54:41 | INFO    | V0__d0ca0d42__20260411T135934Z | __main__ | === SBGM_SD MAIN APP DONE ===
```

### Seed 2 — V0_seed2__32db94fa__20260414T080745Z.log

Source: [V0_seed2__32db94fa__20260414T080745Z.log](/var/folders/nj/qv09n8sx0tsfmf6z0ktk_pwjx80ybt/T/codex-file-preview-4BFpE1/V0_seed2__32db94fa__20260414T080745Z.log)

```text
L21: 2026-04-14 11:07:45 | INFO    | V0_seed2__32db94fa__20260414T080745Z | sbgm.training_main |           Runtime: distributed=True rank=0 local_rank=0 world_size=4 requested_device=cuda:0
L142: 2026-04-14 11:08:04 | INFO    | V0_seed2__32db94fa__20260414T080745Z | sbgm.training_main |           ▸ Number of trainable parameters in model: 18,101,315
L193: 2026-04-14 11:08:04 | INFO    | V0_seed2__32db94fa__20260414T080745Z | sbgm.training | 
L195:       ▸ Starting epoch 1/400...
L7149: 2026-04-14 14:59:27 | INFO    | V0_seed2__32db94fa__20260414T080745Z | sbgm.training | → Epoch 400/400 completed. 
L7308: 2026-04-14 14:59:39 | INFO    | V0_seed2__32db94fa__20260414T080745Z | sbgm.generate.generation | [generation] cfg: ensemble_size=32, steps=56, save_space=physical, predict_residual=False, out_root=/scratch/project_465002493/quistgaa/Code/CEDDAR/models_and_samples/generated_samples/generation/V0_seed2__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56
L7311: 2026-04-14 14:59:39 | INFO    | V0_seed2__32db94fa__20260414T080745Z | sbgm.generate.generation | [generation] Generating for date 20190101 (1/731) with ensemble size 32.
L8321: 2026-04-14 15:10:32 | INFO    | V0_seed2__32db94fa__20260414T080745Z | sbgm.generate.generation | [generation] Generating for date 20200514 (500/731) with ensemble size 32.
L8419: 2026-04-14 15:11:35 | INFO    | V0_seed2__32db94fa__20260414T080745Z | sbgm.score_sampling | [sampler] sigma*: 1.000, mode=global, ramp_frac=(0.60,0.85), ramp_sigma=(None,None)
```

### Seed 3 — V0_seed3__974b9364__20260414T080748Z.log

Source: [V0_seed3__974b9364__20260414T080748Z.log](/var/folders/nj/qv09n8sx0tsfmf6z0ktk_pwjx80ybt/T/codex-file-preview-tHKSOi/V0_seed3__974b9364__20260414T080748Z.log)

```text
L21: 2026-04-14 11:07:48 | INFO    | V0_seed3__974b9364__20260414T080748Z | sbgm.training_main |           Runtime: distributed=True rank=0 local_rank=0 world_size=4 requested_device=cuda:0
L142: 2026-04-14 11:08:07 | INFO    | V0_seed3__974b9364__20260414T080748Z | sbgm.training_main |           ▸ Number of trainable parameters in model: 18,101,315
L193: 2026-04-14 11:08:07 | INFO    | V0_seed3__974b9364__20260414T080748Z | sbgm.training | 
L195:       ▸ Starting epoch 1/400...
L7142: 2026-04-14 15:03:53 | INFO    | V0_seed3__974b9364__20260414T080748Z | sbgm.training | → Epoch 400/400 completed. 
L7301: 2026-04-14 15:04:04 | INFO    | V0_seed3__974b9364__20260414T080748Z | sbgm.generate.generation | [generation] cfg: ensemble_size=32, steps=56, save_space=physical, predict_residual=False, out_root=/scratch/project_465002493/quistgaa/Code/CEDDAR/models_and_samples/generated_samples/generation/V0_seed3__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56
L7304: 2026-04-14 15:04:05 | INFO    | V0_seed3__974b9364__20260414T080748Z | sbgm.generate.generation | [generation] Generating for date 20190101 (1/731) with ensemble size 32.
L7910: 2026-04-14 15:10:36 | INFO    | V0_seed3__974b9364__20260414T080748Z | sbgm.generate.generation | [generation] Generating for date 20191027 (300/731) with ensemble size 32.
L8002: 2026-04-14 15:11:35 | INFO    | V0_seed3__974b9364__20260414T080748Z | sbgm.score_sampling | [sampler] sigma*: 1.000, mode=global, ramp_frac=(0.60,0.85), ramp_sigma=(None,None)
```
