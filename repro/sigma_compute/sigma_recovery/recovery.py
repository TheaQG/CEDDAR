"""Recover a paired, CPU, non-residual CEDDAR sweep without editing its source.

Old files are read only. New sampling is isolated. Combined provenance explicitly
describes an assembled dataset and links to both original sampler invocations.
"""
import copy
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(os.environ['RECOVERY_DIR']).expanduser().resolve()
OLD = Path(os.environ['CAMPAIGN_DIR']).expanduser().resolve()
REPO = Path(os.environ['REPO_DIR']).expanduser().resolve()
SCRIPT = Path(__file__).resolve()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def isolate(run):
    env = dict(CEDDAR_RUNS=run, SAMPLE_DIR=run/'samples', EVAL_DIR=run/'evaluation',
               LOG_DIR=run/'logs', TMPDIR=run/'tmp', XDG_CACHE_HOME=run/'cache',
               MPLCONFIGDIR=run/'cache/matplotlib', TORCH_HOME=run/'cache/torch',
               CKPT_DIR=run/'unused-checkpoints', DEVICE='cpu')
    os.environ.update({key: str(value) for key, value in env.items()})


def libraries():
    import torch
    torch.set_num_threads(int(os.environ['CPU_THREADS']))
    torch.set_num_interop_threads(1)
    return torch


def cli(action, run, extra=(), log=None):
    command = [sys.executable, str(SCRIPT), '_cli', action, '--run-dir', str(run), *map(str, extra)]
    if log:
        with Path(log).open('x') as stream:
            subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(command, cwd=REPO, check=True)


def prepare(run, config, grid, state, log):
    cli('prepare', run, ['--config', config, '--sigma-star-grid', *grid,
        '--sigma-star-mode', 'global', '--initial-state', 'legacy_sigma_max',
        '--noise-mode', 'paired', '--seed', 504, '--steps', 56, '--ensemble-size', 32,
        '--max-dates', state['max_dates'], '--split', state.get('split', 'valid')], log)


def check_sources(state):
    for name, expected in state['source_sha256'].items():
        require(digest(REPO/name) == expected, f'Source changed: {name}')


def configuration(path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    full, controls = cfg.full_gen_eval, cfg.full_gen_eval.sigma_control
    require(full.seed == 504 and full.ensemble_size == 32, 'Expected seed 504 and 32 members')
    require(cfg.edm.sampling_steps == 56, 'Expected 56 steps')
    require(controls.noise_mode == 'paired' and controls.sigma_star_mode == 'global', 'Wrong noise/sigma mode')
    require(controls.sigma_star_initial_state == 'legacy_sigma_max', 'Wrong initialization')
    require(cfg.training.device == 'cpu', 'Recovery requires original CPU generation')
    require(not cfg.edm.get('predict_residual', False), 'This recovery path does not support residual prediction')
    require(not cfg.get('classifier_free_guidance', {}).get('enabled', False), 'Guided recovery is not supported')
    return cfg


def comparable_config(cfg, run):
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(cfg, resolve=True)
    data['full_gen_eval']['sigma_star_grid'] = []
    data['full_gen_eval']['sigma_control']['example_sigma_subset'] = []
    return json.dumps(data, sort_keys=True).replace(str(run), '<RUN>')


def batch_record(batch):
    import torch
    from sbgm.utils import extract_samples
    from sbgm.sampling_noise import tensor_sha256
    from sbgm.generate.generation import _repeat_to_M
    date = str(batch['date'][0])
    x, y, cond, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(batch, 'cpu')
    values = dict(y=_repeat_to_M(y, 32), cond_img=_repeat_to_M(cond, 32),
                  lsm_cond=_repeat_to_M(lsm, 32), topo_cond=_repeat_to_M(topo, 32))
    hashes = {key: tensor_sha256(value) for key, value in values.items() if torch.is_tensor(value)}
    hashes['hr_reference'] = tensor_sha256(x[:1]) if x is not None else None
    return date, hashes


def products(output, date):
    return [output/folder/f'{date}.npz' for folder in
            ('ensembles_phys', 'pmm_phys', 'lr_hr_phys', 'lsm')] + [output/'meta/noise'/f'{date}.json']


def original_metadata(output, sigma, cfg):
    import yaml
    from omegaconf import OmegaConf
    from sbgm.provenance import sigma_generation_metadata
    # This invocation was interrupted. Check its actual sampler parameters without
    # claiming it has a completion manifest. No saved configuration is changed.
    partial = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    partial.full_gen_eval.sigma_control.require_generation_manifest = False
    metadata = sigma_generation_metadata(output.parent, [sigma], partial)
    path = Path(metadata[0]['manifest'])
    return path, yaml.safe_load(path.read_text())


def plan():
    import torch
    from sbgm.training_utils import get_final_gen_dataloader
    require(ROOT != OLD and OLD not in ROOT.parents and ROOT not in OLD.parents,
            'Recovery and original campaign must be separate directories')
    require(ROOT != REPO and REPO not in ROOT.parents, 'Recovery output must be outside repository')
    require(not ROOT.exists(), 'RECOVERY_DIR already exists; choose a fresh directory')
    state = read_json(OLD/'campaign.json')
    report = read_json(os.environ['INSPECTION_REPORT'])
    check_sources(state)
    require(Path(report['campaign']).resolve() == OLD, 'Inspection belongs to a different campaign')
    require(all(not report[key] for key in ('source_issues', 'pairing_issues', 'metadata_issues')),
            'Inspection has unresolved issues')
    require(len(report['tasks']) == len(state['grid']), 'Wrong number of inspected tasks')
    require(report['max_dates'] == state['max_dates'] > 0, 'A positive, matching date cap is required')
    root_cfg = configuration(OLD/'combined/resolved_config.yaml')
    checkpoint = root_cfg.paths.inference_checkpoint
    require(Path(os.environ['PUBLISHED_CHECKPOINT']).resolve() == Path(checkpoint).resolve(),
            'PUBLISHED_CHECKPOINT differs from original saved checkpoint')
    checkpoint_sha = digest(checkpoint)
    reference_cfg = comparable_config(root_cfg, OLD/'combined')
    records, tasks = {}, []
    for index, row in enumerate(report['tasks']):
        sigma = state['grid'][index]
        require(row['index'] == index and row['sigma'] == sigma and not row['issues'], 'Invalid inspection task')
        task_cfg = configuration(OLD/'tasks'/f'{index:02d}'/'run/resolved_config.yaml')
        require(comparable_config(task_cfg, OLD/'tasks'/f'{index:02d}'/'run') == reference_cfg,
                f'Task {index} configuration differs')
        require(task_cfg.full_gen_eval.split == state.get('split', 'valid'), 'Split mismatch')
        require(task_cfg.full_gen_eval.max_dates == state['max_dates'], 'Date cap mismatch')
        output = Path(row['output']).resolve()
        require(OLD in output.parents, 'Original output lies outside campaign')
        manifest, observed = original_metadata(output, sigma, task_cfg)
        require(observed['checkpoint']['sha256'] == checkpoint_sha == row['checkpoint_sha256'], 'Checkpoint changed')
        require(observed['data']['split'] == task_cfg.full_gen_eval.split, 'Observed split mismatch')
        require(str(observed['torch']['version']) == str(torch.__version__), 'Use the original PyTorch version')
        complete = row['complete_candidates']
        require(complete and len(complete) < state['max_dates'], 'Expected a nonempty partial prefix')
        snapshot = {str(manifest): digest(manifest)}
        print(f'Fingerprinting saved files for sigma={sigma:.2f}', flush=True)
        for date in complete:
            for path in products(output, date):
                snapshot[str(path)] = digest(path)
            noise = read_json(output/'meta/noise'/f'{date}.json')
            require(noise['root_seed'] == 504 and noise['device'] == 'cpu', 'Wrong saved noise settings')
            require(noise['torch_version'] == str(torch.__version__), 'Saved noise uses a different PyTorch version')
            if date in records:
                require(records[date] == noise, f'Pairing mismatch for {date}')
            records[date] = noise
        tasks.append(dict(index=index, sigma=sigma, original_output=str(output), reused=complete,
                          canary=complete[-1], snapshot=snapshot, original_provenance=str(manifest)))
    print('Checking original dataset order and conditioning hashes; no inference.', flush=True)
    root_cfg.data_handling.update(split=root_cfg.full_gen_eval.split, shuffle=False, drop_last=False)
    loader = get_final_gen_dataloader(root_cfg, split=root_cfg.full_gen_eval.split)
    dates, inputs = [], {}
    for batch in itertools.islice(loader, state['max_dates']):
        date, hashes = batch_record(batch)
        require(date not in inputs, f'Duplicate dataset date: {date}')
        if date in records:
            require(hashes == records[date]['inputs_sha256'], f'Dataset inputs changed: {date}')
        dates.append(date)
        inputs[date] = hashes
        if len(dates) % 100 == 0:
            print(f'  checked {len(dates)} dates', flush=True)
    require(len(dates) == state['max_dates'], 'Dataset contains fewer dates than the requested cap; inspect before recovery')
    for task in tasks:
        n = len(task['reused'])
        require(task['reused'] == dates[:n], f"Task {task['index']}: saved dates are not the exact dataset prefix")
        task['missing'] = dates[n:]
    ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT/'remaining').mkdir()
    prepare(ROOT/'combined', OLD/'combined/resolved_config.yaml', state['grid'], state, ROOT/'prepare.log')
    data = dict(state=state, dates=dates, inputs=inputs, tasks=tasks,
                original_campaign=str(OLD), checkpoint_sha256=checkpoint_sha,
                inspection_sha256=digest(os.environ['INSPECTION_REPORT']),
                script_sha256=digest(SCRIPT), created=time.time(), planning_host=socket.gethostname())
    save_json(ROOT/'recovery_plan.json', data)
    for task in tasks:
        print(f"sigma={task['sigma']:.2f}: reuse {len(task['reused'])}, generate {len(task['missing'])}, plus 1 repeat-date check")
    print(f'PLAN READY: {ROOT}. No model inference performed.', flush=True)


def load_plan():
    data = read_json(ROOT/'recovery_plan.json')
    require(data['original_campaign'] == str(OLD), 'Original campaign changed')
    require(data['script_sha256'] == digest(SCRIPT), 'Recovery script changed after planning')
    check_sources(data['state'])
    return data


def verify_snapshot(task):
    for path, expected in task['snapshot'].items():
        require(digest(path) == expected, f'Original file changed since planning: {path}')


def repeat_check(task, fresh, target):
    import numpy as np
    date = task['canary']
    original = Path(task['original_output'])
    require(read_json(original/'meta/noise'/f'{date}.json') == read_json(fresh/'meta/noise'/f'{date}.json'),
            'Repeat-date noise/conditioning differs on this host; stopping before missing dates')
    with np.load(original/'ensembles_phys'/f'{date}.npz') as a, np.load(fresh/'ensembles_phys'/f'{date}.npz') as b:
        x, y = a['ens'].astype('float64'), b['ens'].astype('float64')
        require(x.shape == y.shape and np.isfinite(y).all(), 'Invalid repeat-date output')
        check = dict(date=date, noise_and_inputs_equal=True, bitwise_equal=bool(np.array_equal(x, y)),
                     max_absolute_difference_mm=float(np.max(np.abs(x-y))),
                     rmse_difference_mm=float(np.sqrt(np.mean((x-y)**2))),
                     close_rtol_1e4_atol_1e4=bool(np.allclose(x, y, rtol=1e-4, atol=1e-4)))
    save_json(target, check)
    print('Repeat-date check:', check, flush=True)
    # Do not silently mix materially different numerical behavior across hosts.
    require(check['close_rtol_1e4_atol_1e4'],
            'Repeat-date output exceeds numerical tolerance. Inspect canary.json before proceeding; originals are intact.')


class SelectedLoader:
    def __init__(self, loader, data, task, fresh, check_path):
        self.loader, self.data, self.task = loader, data, task
        self.fresh, self.check_path = fresh, check_path
        self.selected = {task['canary'], *task['missing']}

    def __len__(self):
        return len(self.selected)

    def __iter__(self):
        seen = []
        for index, batch in enumerate(itertools.islice(self.loader, len(self.data['dates']))):
            date, hashes = batch_record(batch)
            require(date == self.data['dates'][index], 'Dataset order changed since planning')
            require(hashes == self.data['inputs'][date], f'Dataset inputs changed since planning: {date}')
            seen.append(date)
            if date in self.selected:
                yield batch
                if date == self.task['canary']:
                    repeat_check(self.task, self.fresh, self.check_path)
        require(seen == self.data['dates'], 'Dataset ended early')


def worker(index, data):
    from omegaconf import OmegaConf
    from sbgm.utils import get_model_string
    task = data['tasks'][index]
    output = ROOT/'remaining'/f'{index:02d}'
    output.mkdir(exist_ok=False)
    save_json(output/'runtime.json', dict(host=socket.gethostname(), pid=os.getpid(), start=time.time(),
                                         threads=int(os.environ['CPU_THREADS']), sigma=task['sigma']))
    verify_snapshot(task)
    run = output/'run'
    prepare(run, OLD/'tasks'/f'{index:02d}'/'run/resolved_config.yaml', [task['sigma']],
            data['state'], output/'prepare.log')
    cfg = OmegaConf.load(run/'resolved_config.yaml')
    require(digest(cfg.paths.inference_checkpoint) == data['checkpoint_sha256'], 'Checkpoint changed since planning')
    fresh = run/'samples/generation'/get_model_string(cfg)/f"sigma_star={task['sigma']:.2f}"
    module = importlib.import_module('sbgm.generate.generation_sigma_grid_main')
    original_loader = module.get_final_gen_dataloader
    module.get_final_gen_dataloader = lambda cfg, split: SelectedLoader(
        original_loader(cfg, split=split), data, task, fresh, output/'canary.json')
    print(f"Recover sigma={task['sigma']:.2f}: {len(task['missing'])} missing dates + repeat check", flush=True)
    sys.argv = ['repro.sigma_star', 'generate', '--run-dir', str(run)]
    runpy.run_module('repro.sigma_star', run_name='__main__')
    expected = {task['canary'], *task['missing']}
    require(read_json(fresh/'meta/manifest.json')['n_days'] == len(expected), 'Recovery completion count differs')
    require({p.stem for p in (fresh/'ensembles_phys').glob('*.npz')} == expected, 'Unexpected recovery dates')
    save_json(output/'done.json', dict(output=str(fresh), new_dates=len(task['missing']),
                                     plan_sha256=digest(ROOT/'recovery_plan.json'), finished=time.time()))
    print(f"DONE sigma={task['sigma']:.2f}", flush=True)


def launch(index):
    log = ROOT/f'recover-{index:02d}.log'
    with log.open('x') as stream:
        subprocess.run([sys.executable, str(SCRIPT), '_worker', str(index)], cwd=REPO,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f'DONE task {index:02d}', flush=True)


def assemble(data):
    import numpy as np
    import yaml
    from omegaconf import OmegaConf
    from sbgm.provenance import sigma_generation_metadata, write_provenance
    from sbgm.utils import get_model_string
    from inspect_partial import read_npz
    cfg = OmegaConf.load(ROOT/'combined/resolved_config.yaml')
    model = get_model_string(cfg)
    all_noise, entries = {}, []
    for task in data['tasks']:
        index, sigma = task['index'], task['sigma']
        verify_snapshot(task)
        done = read_json(ROOT/'remaining'/f'{index:02d}'/'done.json')
        require(done['plan_sha256'] == digest(ROOT/'recovery_plan.json'), 'Recovery was run with a different plan')
        fresh, old = Path(done['output']), Path(task['original_output'])
        newmeta = sigma_generation_metadata(fresh.parent, [sigma], cfg)[0]
        newprov = yaml.safe_load(Path(newmeta['manifest']).read_text())
        oldprov = yaml.safe_load(Path(task['original_provenance']).read_text())
        require(newprov['checkpoint']['sha256'] == data['checkpoint_sha256'], 'Recovery checkpoint mismatch')
        require(newprov['data']['split'] == data['state'].get('split', 'valid'), 'Recovery split mismatch')
        old_sampler, new_sampler = copy.deepcopy(oldprov['sampler']), copy.deepcopy(newprov['sampler'])
        old_sampler.pop('noise_seed', None)
        new_sampler.pop('noise_seed', None)
        require(old_sampler == new_sampler, 'Original and recovery sampler settings differ')
        require(read_json(ROOT/'remaining'/f'{index:02d}'/'canary.json')['close_rtol_1e4_atol_1e4'], 'Repeat check failed')
        origins, first_mask, stationary = {}, None, True
        reused = set(task['reused'])
        print(f'Checking and assembling sigma={sigma:.2f}', flush=True)
        for date in data['dates']:
            source = old if date in reused else fresh
            ensemble = read_npz(source/'ensembles_phys'/f'{date}.npz', ['ens'])['ens']
            shape = tuple(cfg.highres.data_size)
            require(ensemble.shape == (32, 1, *shape), f'Wrong ensemble shape: {date}')
            pmm = read_npz(source/'pmm_phys'/f'{date}.npz', ['pmm'])['pmm']
            refs = read_npz(source/'lr_hr_phys'/f'{date}.npz', ['hr', 'lr'])
            mask = read_npz(source/'lsm'/f'{date}.npz', ['lsm_hr'])['lsm_hr'] > 0.5
            for array in (pmm, refs['hr'], refs['lr'], mask):
                require(array.shape[-2:] == shape, f'Wrong field shape: {date}')
            if first_mask is None:
                first_mask = mask
            stationary = stationary and bool(np.array_equal(first_mask, mask))
            noise = read_json(source/'meta/noise'/f'{date}.json')
            require(noise['date'] == date and noise['root_seed'] == 504, 'Wrong date/seed in noise record')
            require(noise['inputs_sha256'] == data['inputs'][date], 'Conditioning differs from planned inputs')
            if date in all_noise:
                require(all_noise[date] == noise, f'Paired noise differs at {date}')
            else:
                all_noise[date] = noise
            origins[date] = str(source)
        entries.append((task, fresh, oldprov, newprov, newmeta['manifest'], origins, stationary))
    # No aggregate directories are created until all source datasets pass.
    (ROOT/'assembly_started').mkdir(exist_ok=False)
    for task, fresh, oldprov, newprov, newmanifest, origins, stationary in entries:
        sigma = task['sigma']
        out = ROOT/'combined/samples/generation'/model/f'sigma_star={sigma:.2f}'
        out.mkdir(parents=True, exist_ok=False)
        for date, origin in origins.items():
            source = Path(origin)
            for path in products(source, date):
                dest = out/path.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(path)
        first_source = Path(origins[data['dates'][0]])
        canonical = first_source/'meta/land_mask.npz'
        if canonical.is_file():
            (out/'meta/land_mask.npz').symlink_to(canonical)
        assembly = dict(record_kind='assembled_dataset_not_a_single_sampler_invocation',
                        original_provenance=task['original_provenance'],
                        original_provenance_sha256=digest(task['original_provenance']),
                        recovery_provenance=newmanifest, recovery_provenance_sha256=digest(newmanifest),
                        reused_dates=len(task['reused']), new_dates=len(task['missing']),
                        canary=read_json(ROOT/'remaining'/f"{task['index']:02d}"/'canary.json'),
                        plan_sha256=digest(ROOT/'recovery_plan.json'), date_sources=origins)
        save_json(out/'meta/recovery_assembly.json', assembly)
        completion = read_json(fresh/'meta/manifest.json')
        completion.update(n_days=len(data['dates']), lsm_stationary_observed=stationary,
                          record_kind='assembled_dataset', recovery_assembly='recovery_assembly.json')
        save_json(out/'meta/manifest.json', completion)
        local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        local_cfg.edm.sigma_star = sigma
        local_cfg.data_handling.split = local_cfg.full_gen_eval.split
        write_provenance(out/'meta', local_cfg, stage='generation_assembly', device='cpu',
                         checkpoint=oldprov['checkpoint'], sampler=oldprov['sampler'],
                         record_kind='assembled_dataset', first_date=data['dates'][0],
                         ensemble_size=32, recovery=assembly)
    sigma_generation_metadata(ROOT/'combined/samples/generation'/model, data['state']['grid'], cfg)
    save_json(ROOT/'assembly_done.json', dict(plan_sha256=digest(ROOT/'recovery_plan.json'),
                                            dates=len(data['dates']), variants=len(entries)))
    print('ASSEMBLY PASS: full date sets, products, paired noise/inputs and sampler provenance verified.', flush=True)


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else ''
    if action == '_cli':
        # repro.sigma_star establishes the specific run's isolated environment.
        libraries()
        sys.argv = ['repro.sigma_star', *sys.argv[2:]]
        runpy.run_module('repro.sigma_star', run_name='__main__')
        return
    require(action in ('plan', 'run', '_worker', 'evaluate'), 'Use plan, run or evaluate')
    threads, jobs, budget = [int(os.environ[k]) for k in ('CPU_THREADS', 'MAX_JOBS', 'CPU_BUDGET')]
    require(min(threads, jobs, budget) > 0, 'CPU settings must be positive')
    needed = jobs*threads if action == 'run' else threads
    require(needed <= budget, 'CPU budget exceeded')
    if hasattr(os, 'sched_getaffinity'):
        require(needed <= len(os.sched_getaffinity(0)), 'CPU request exceeds affinity mask')
    # Controller/import caches must not alter the original campaign or create ROOT before plan.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='ceddar-recovery-') as cache:
        isolate(Path(cache))
        libraries()
        if action == 'plan':
            require(len(sys.argv) == 2, 'plan takes no indices')
            plan()
            return
        data = load_plan()
        if action == '_worker':
            require(len(sys.argv) == 3, '_worker requires one index')
            index = int(sys.argv[2])
            require(0 <= index < len(data['tasks']), 'Invalid task index')
            worker(index, data)
        elif action == 'run':
            indices = [int(s) for s in sys.argv[2:]] or list(range(len(data['tasks'])))
            require(len(set(indices)) == len(indices) and all(0 <= i < len(data['tasks']) for i in indices), 'Invalid indices')
            require(all(not (ROOT/'remaining'/f'{i:02d}').exists() and not (ROOT/f'recover-{i:02d}.log').exists()
                        for i in indices), 'Selected recovery tasks already exist; no overwrite/resume is attempted')
            failed = []
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures = {pool.submit(launch, i): i for i in indices}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        failed.append(futures[future])
                        print(f'FAILED task {futures[future]:02d}: {exc}', flush=True)
            require(not failed, f'Failed recovery tasks: {failed}; all outputs retained')
        else:
            require(len(sys.argv) == 2, 'evaluate takes no indices')
            require(not (ROOT/'assembly_started').exists(), 'Assembly already started; inspect existing results before retrying')
            assemble(data)
            cli('evaluate', ROOT/'combined', log=ROOT/'evaluate.log')


if __name__ == '__main__':
    main()
