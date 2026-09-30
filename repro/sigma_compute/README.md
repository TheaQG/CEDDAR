# Parallel legacy σ* runs

One complete σ* value per process is supported by the inspected implementation. Use a bounded Bash launch on your compute server. No scheduler is required. No SSH connection, remote launch, or full inference run has been performed by this task.

The supplied scripts keep **global scaling, legacy_sigma_max initialization, paired noise, seed 504, 56 steps, 32 members, and the validation split**. Production defaults also retain the existing 1000-date cap and dense grid 0.80–1.25 in increments of 0.05. The actual number of dates can be smaller than 1000 because the loader uses the available HR/LR date intersection.

## Run on one compute server

Copy this directory to the server, activate the existing CEDDAR Python environment, and edit `settings.env`. In particular, `REPO_DIR` is a placeholder and must point to the actual checkout. Prefer an absolute path for `PYTHON`. Check the data, statistics, checkpoint, configuration and campaign paths. The campaign must be NEW and outside the CEDDAR repository.

First run a small pilot with a separate settings file/campaign: grid `0.95 1.00 1.05`, `MAX_DATES=3`, the same 32 members and 56 steps. This is a launch/pairing check, not a replacement for the production sweep.

From this directory on the compute server:

```bash
bash run_parallel.sh settings.env init
nohup bash run_parallel.sh settings.env run > launcher.log 2>&1 &
```

`init` resolves the full configuration without loading the checkpoint or generating samples. `run` runs at most `MAX_JOBS` processes at once. Each process completes all dates and all 32 members for one σ*. It starts the next value when a slot becomes free. Leave the repository, checkpoint and input data unchanged throughout the campaign.

After the launcher finishes successfully:

```bash
bash run_parallel.sh settings.env audit
nohup bash run_parallel.sh settings.env evaluate > evaluation-launcher.log 2>&1 &
```

Evaluation repeats the audit, links completed σ* directories into the combined layout, then invokes the original evaluator once. It retains the original generation manifests and noise records. It does not regenerate samples or edit input files. Combined results are under `CAMPAIGN_DIR/combined/evaluation/`.

Monitor `launcher.log` and `CAMPAIGN_DIR/tasks/NN/generate.log`. Each successful task writes `done.json` with its output path, elapsed seconds and completed date count; `runtime.json` identifies its host and settings. A task directory by itself does not mean completion. One failed task makes the launcher return nonzero; other submitted tasks still finish. Evaluation refuses incomplete campaigns.

## Split across servers

Initialize once. With the repository, data and campaign available at the same absolute paths on both servers, run disjoint index subsets:

```bash
# Server A
nohup bash run_parallel.sh settings.env run 0 1 2 3 4 > server-a.log 2>&1 &
# Server B
nohup bash run_parallel.sh settings.env run 5 6 7 8 9 > server-b.log 2>&1 &
```

For the default dense grid, index 0 is 0.80, index 4 is 1.00, and index 9 is 1.25. Jobs may finish in any order. `MAX_JOBS`, `CPU_THREADS`, and `CPU_BUDGET` apply separately on each host. After both launchers succeed, evaluate once on either host. Atomic task-directory creation prevents duplicate index launches from writing the same outputs.

This version assumes shared paths. For servers without a shared filesystem, arrange explicit copying and path remapping before using combined evaluation; the absolute links and saved paths are not a portable archive.

## CPU settings

Start by measuring one job at 1, 2, 4 and 8 threads. Use the same small date subset, same σ*, 32 members and 56 steps in separate fresh campaigns. Use enough dates to distinguish steady inference time from checkpoint loading and imports; a three-date smoke run alone is a weak performance benchmark. Record wall time and peak resident memory, then test two concurrent jobs at the best thread count.

The initial settings are **two processes × four threads = eight CPUs**. They are a starting experiment, not a measured optimum or an assertion that eight CPUs are available to you. Set `CPU_BUDGET` to the CPUs you may use. The launcher checks `MAX_JOBS × CPU_THREADS <= CPU_BUDGET`, and on Linux also checks the process's CPU affinity mask. This is not a resource reservation: coordinate with other users of the server.

Each inference subprocess sets:

| Setting | Value |
|---|---|
| PyTorch intra-op threads | `CPU_THREADS` |
| PyTorch inter-op threads | 1 |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | `CPU_THREADS` |
| `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` | 1 |
| Dynamic OpenMP/MKL thread adjustment | disabled |
| Generation DataLoader workers | unchanged at 0 |

The thread settings are applied before importing the CEDDAR CLI. Setting only a shell variable for PyTorch inter-op threading is not sufficient; the entrypoint calls its API explicitly. PyTorch documents the distinction and the need to configure threads before work starts: [intra-op API](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html), [inter-op API](https://docs.pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html), [environment variables](https://docs.pytorch.org/docs/stable/threading_environment_variables.html).

Use physical cores as the initial planning unit. Avoid crossing CPU sockets with a single process until measured. Increasing concurrent processes also duplicates model/activation memory and multiplies reads from the data server. Stop increasing concurrency if memory pressure, swapping or I/O contention reduces total throughput. Keep any node-local data copy identical and preserve it for the whole campaign.

The existing legacy wrapper sets OpenMP/MKL threads to `CPU_THREADS`, defaulting to **one**. Merely moving it to a many-core compute server does not make it use all cores. The generation DataLoader also explicitly has zero workers; that governs data loading, not the parallelism of PyTorch's CPU operations. The 32 ensemble members are already one inference batch. Preserve that shape for this audit.

For the current Heun sampler, 56 steps normally mean 111 batched model evaluations per date (without classifier-free guidance). At 1000 dates, one σ* therefore entails approximately 111,000 forward calls with a batch of 32. Parallel σ* jobs reduce sweep elapsed time when resources are available; they do not remove this work or necessarily accelerate any individual value.

## Scientific and output constraints

* **Noise pairing:** the runner derives a seed from root seed 504 and the date. The sampler uses separate indexed streams for `initial` and `churn:<step>` (and conditional-null streams when relevant). Neither σ* nor job index enters those keys. Keep seed 504 in every process. Identical seed/stream/shape/device/dtype and compatible PyTorch implementation reproduce the same standard-normal draws, independently of process or sweep order.
* **Member batching:** the noise tensor is drawn for all 32 members at once. The current API does not provide global member offsets. Splitting into four 8-member jobs, or simply offsetting their seeds, would not establish the same paired experiment. Date sharding may be feasible with further implementation, but the present CLI only caps dates; it does not expose a safe date-range sharding interface. This bundle splits only σ*.
* **Inputs:** noise pairing is insufficient if dates, crop bounds, conditioning, masks or references differ. The local final loader is sequential, batch size one date, and the inspected YAML enables fixed evaluation cutouts. The audit compares per-date conditioning/reference hashes as well as noise hashes.
* **Legacy meaning:** global scaling multiplies the positive schedule and churn-window endpoints by σ*, then appends zero. `legacy_sigma_max` keeps initial standard deviation at the unscaled σ_max, which is 80 in the inspected configuration. Thus σ*=0.80 starts from noise with standard deviation 80 even though the first schedule value is approximately 64. This intentional historical mismatch is preserved. The default `schedule` or `late_ramp` settings would describe a different experiment.
* **Random draws versus amplitudes:** paired standard normals are identical across σ*. Churn amplitudes and trajectories change with σ*, as intended. Generated precipitation should not be identical across σ*. At σ*=1, both initialization conventions coincide, but paired noise does not necessarily reproduce the old sequential-RNG realization.
* **Reproducibility:** do not mix CPU and GPU jobs in this paired sweep or silently change precision, ensemble batching or PyTorch versions. Identical input/noise hashes do not imply bitwise-identical neural-network outputs across CPUs, libraries or thread counts. Choose one thread setting for the final sweep after the pilot.
* **Shared outputs:** the original grid driver writes distinct σ* subdirectories, but `repro.sigma_star` also sets logs, caches and temporary directories from its run directory. Running several full sweeps against one run directory is unsafe. This bundle gives each task a complete isolated run directory, sharing inputs only. It then creates read-through links for one evaluator.
* **Completion and provenance:** the audit checks completion, physical-ensemble/PMM date files, matching date sets, all recorded inputs/noise draws, identical checkpoint hashes, validation split, ensemble/step/seed settings, and the original evaluator's observed-sampler checks. Unexpected configuration differences also stop collection. For the inspected default grid, all values activate churn at steps 0–6. The audit intentionally stops if stream sets differ, so an unexpected boundary/branch difference can be inspected rather than silently accepted.

## Failed runs and optional scheduler use

The original generation driver explicitly refuses existing σ* output directories; it does not resume a partly generated value. This bundle preserves that behavior. It never deletes partial results. You can run only indices that have not started, for example `run 6 7 8 9`. To retry a failed index, first confirm no process is still writing, then retain its old task directory under a different name outside its original `tasks/NN` path before retrying that index. Do not mark partial runs complete manually. Alternatively, initialize a new campaign.

Evaluation has a persistent `evaluation_started` reservation directory to prevent concurrent writers and accidental reruns. After a failed evaluation, inspect the log and confirm it has stopped before explicitly removing that empty reservation directory and retrying. Never remove it while evaluation is active. Links remain valid only while the original task directories remain in place.

`sigma_array.slurm` is supplied only for a future scheduler-managed server. Initialize first, then submit one array element per σ*, with site-specific time, memory, account and partition options. For example, the main grid uses `--array=0-9%2 --cpus-per-task=4`; pass the absolute bundle directory and settings file as the two script arguments. An evaluation job should depend on successful completion of the whole array using `afterok:<array-job-id>`. Slurm's `%2` limits concurrent array elements; `%A_%a` makes logs distinct. See [Slurm job arrays and dependencies](https://slurm.schedmd.com/job_array.html). Each element requests one node, but that does not require a separate physical node for every element; Slurm may pack them when resources permit.

## Inspection and validation scope

The nine uploaded files were inspected. Additional implementation was read from `/Users/au728490/Code/CEDDAR`; the uploaded generation driver and `repro/sigma_star.py` exactly match that checkout by SHA-256. The noise, sampler, loader and provenance conclusions rely on that local checkout. The compute server must use the same compatible implementation. The bundle records hashes of key source files at initialization and rejects changes in those files between campaign operations; this is not a complete environment or repository snapshot.

Validation performed locally: shell/Python syntax checks; all 16 existing paired-noise and sigma-control tests passed; the default 56-step dense grid uses the same active churn steps at every σ*. A separate bundle integration check uses the real preparation CLI and a small toy denoiser with 32 members and 56 steps to exercise task isolation, gathering, and rejection of duplicate tasks and mismatched conditioning hashes. These checks do not establish full-model runtime, memory use, cluster access, or a successful production run. The production checkpoint and validation dataset were not loaded for inference.
