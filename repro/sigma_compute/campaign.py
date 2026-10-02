"""Isolated sigma jobs using the existing CEDDAR CLI; no sampler modifications."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ['CAMPAIGN_DIR']).expanduser().resolve()
REPO = Path(os.environ['REPO_DIR']).expanduser().resolve()
SOURCES = ['repro/sigma_star.py', 'sbgm/generate/generation_sigma_grid_main.py',
           'sbgm/generate/generation.py', 'sbgm/sampling_noise.py',
           'sbgm/sigma_control.py', 'sbgm/score_sampling.py',
           'sbgm/training_utils.py', 'sbgm/provenance.py', 'sbgm/runtime.py']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def fingerprints():
    return {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in SOURCES}


def invoke(action, run, extra=(), log=None):
    command = [sys.executable, str(HERE / 'threaded_entry.py'), action,
               '--run-dir', str(run), *map(str, extra)]
    if log:
        with Path(log).open('x') as stream:
            subprocess.run(command, cwd=REPO, check=True, stdout=stream,
                           stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, cwd=REPO, check=True)


def campaign_split(state):
    # Campaigns created by v1 always used validation, regardless of current env.
    split = state.get('split', 'valid')
    require(split in ('valid', 'test'), f'Invalid saved campaign split: {split}')
    return split


def verify_saved_split(run, expected):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(run / 'resolved_config.yaml')
    actual = cfg.full_gen_eval.split
    require(actual == expected,
            f'Saved configuration split {actual!r} differs from campaign split {expected!r}: {run}')


def prepare(run, config, grid, max_dates, log, split='valid'):
    require(split in ('valid', 'test'), f'Invalid split: {split}')
    invoke('prepare', run, ['--config', config, '--sigma-star-grid', *grid,
                           '--sigma-star-mode', 'global',
                           '--initial-state', 'legacy_sigma_max',
                           '--noise-mode', 'paired', '--seed', '504',
                           '--steps', '56', '--ensemble-size', '32',
                           '--max-dates', max_dates, '--split', split], log)
    verify_saved_split(run, split)


def load():
    state = json.loads((ROOT / 'campaign.json').read_text())
    require(state['source_sha256'] == fingerprints(),
            'CEDDAR source differs from campaign initialization. Use a new campaign.')
    verify_saved_split(ROOT / 'combined', campaign_split(state))
    return state


def init():
    require(ROOT != REPO and REPO not in ROOT.parents, 'Outputs must be outside the repository')
    grid = [float(s) for s in os.environ['SIGMA_STAR_GRID'].replace(',', ' ').split()]
    require(grid and all(math.isfinite(s) and s > 0 for s in grid), 'Invalid sigma grid')
    require(len({f'{s:.2f}' for s in grid}) == len(grid), 'Sigma names collide at two decimal places')
    max_dates = int(os.environ['MAX_DATES'])
    require(max_dates == -1 or max_dates > 0, 'MAX_DATES must be positive or -1')
    split = os.environ.get('SIGMA_SPLIT', 'valid').strip().lower()
    require(split in ('valid', 'test'), 'SIGMA_SPLIT must be valid or test')
    for key in ('DATA_DIR', 'STATS_LOAD_DIR', 'PUBLISHED_CHECKPOINT', 'SIGMA_CONFIG'):
        require(Path(os.environ[key]).exists(), f'Missing {key}: {os.environ[key]}')
    source_sha = fingerprints()
    ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT / 'tasks').mkdir()
    prepare(ROOT / 'combined', os.environ['SIGMA_CONFIG'], grid, max_dates,
            ROOT / 'prepare.log', split=split)
    state = dict(grid=grid, max_dates=max_dates, split=split, source_sha256=source_sha,
                 repo=str(REPO), created=time.time())
    (ROOT / 'campaign.json').write_text(json.dumps(state, indent=2))
    print(f'Prepared {ROOT}; split={split}; max_dates={max_dates}; '
          f'grid={grid}; no inference performed.', flush=True)


def worker(index, state):
    require(0 <= index < len(state['grid']), f'Invalid index {index}')
    task = ROOT / 'tasks' / f'{index:02d}'
    # Atomic reservation across hosts. Never silently resume partial generation.
    task.mkdir(exist_ok=False)
    sigma = state['grid'][index]
    started = time.time()
    (task / 'runtime.json').write_text(json.dumps(dict(
        host=socket.gethostname(), launcher_pid=os.getpid(), start=started,
        cpu_threads=int(os.environ['CPU_THREADS']), sigma=sigma,
        split=campaign_split(state)), indent=2))
    print(f'Start index {index}, sigma*={sigma:.2f}, split={campaign_split(state)}; '
          f'log: {task / "generate.log"}', flush=True)
    prepare(task / 'run', ROOT / 'combined/resolved_config.yaml', [sigma],
            state['max_dates'], task / 'prepare.log', split=campaign_split(state))
    invoke('generate', task / 'run', log=task / 'generate.log')
    matches = list((task / 'run/samples/generation').glob(f'*/sigma_star={sigma:.2f}'))
    require(len(matches) == 1, f'Expected one sigma output at {task}')
    manifest = json.loads((matches[0] / 'meta/manifest.json').read_text())
    require(manifest.get('n_days', 0) > 0, 'No completed dates')
    (task / 'done.json').write_text(json.dumps(dict(
        output=str(matches[0]), seconds=time.time() - started,
        n_days=manifest['n_days']), indent=2))
    print(f'Done index {index}, sigma*={sigma:.2f}', flush=True)


def normalized_config(path, run):
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    cfg['full_gen_eval']['sigma_star_grid'] = []
    cfg['full_gen_eval']['sigma_control']['example_sigma_subset'] = []
    return json.dumps(cfg, sort_keys=True).replace(str(run), '<RUN>')


def audit(state, link=False):
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'audit-cache/matplotlib')
    os.environ['XDG_CACHE_HOME'] = str(ROOT / 'audit-cache')
    import yaml
    from omegaconf import OmegaConf
    from sbgm.provenance import sigma_generation_metadata
    from sbgm.utils import get_model_string

    combined = ROOT / 'combined'
    expected_split = campaign_split(state)
    verify_saved_split(combined, expected_split)
    cfg = OmegaConf.load(combined / 'resolved_config.yaml')
    model = get_model_string(cfg)
    reference_cfg = normalized_config(combined / 'resolved_config.yaml', combined)
    first_records = None
    checkpoint_sha = None
    sources = []
    for index, sigma in enumerate(state['grid']):
        task = ROOT / 'tasks' / f'{index:02d}'
        done = json.loads((task / 'done.json').read_text())
        source = Path(done['output'])
        require(source.parent.name == model, f'Model name differs: {source}')
        require(normalized_config(task / 'run/resolved_config.yaml', task / 'run') == reference_cfg,
                f'Unexpected configuration difference in {task}')
        manifest = json.loads((source / 'meta/manifest.json').read_text())
        for key, expected in dict(ensemble_size=32, sampler_steps=56, seed=504,
                                  noise_mode='paired', noise_protocol='ceddar-indexed-noise-v1').items():
            require(manifest.get(key) == expected, f'{source}: wrong {key}')
        # Existing provenance validation checks observed sampler controls, not just YAML requests.
        metadata = sigma_generation_metadata(source.parent, [sigma], cfg)
        observed = yaml.safe_load(Path(metadata[0]['manifest']).read_text())
        sha = observed['checkpoint']['sha256']
        require(sha and (checkpoint_sha is None or sha == checkpoint_sha), 'Checkpoint mismatch')
        checkpoint_sha = sha
        require(observed['data']['split'] == expected_split,
                f'Generation did not use campaign split {expected_split}: {source}')
        records = {p.stem: json.loads(p.read_text()) for p in sorted((source / 'meta/noise').glob('*.json'))}
        require(len(records) == manifest['n_days'] > 0, f'Noise audit count mismatch: {source}')
        for date, record in records.items():
            require(record.get('root_seed') == 504 and record.get('date') == date,
                    f'Wrong seed/date in {source}: {date}')
            initial = record.get('draws', {}).get('initial', {})
            require(initial.get('shape', [None])[0] == 32 and initial.get('standard_normal_sha256'),
                    f'Missing 32-member initial-noise audit: {source}: {date}')
        for folder in ('ensembles_phys', 'pmm_phys'):
            dates = {p.stem for p in (source / folder).glob('*.npz')}
            require(dates == set(records), f'Incomplete {folder} dates: {source}')
        # Compares date, root seed, device/version, input hashes and every consumed draw hash.
        # Unexpected churn-branch differences deliberately stop evaluation for inspection.
        if first_records is None:
            first_records = records
        else:
            require(records.keys() == first_records.keys(), f'Date sets differ: {source}')
            for date in records:
                require(records[date] == first_records[date],
                        f'Paired inputs/noise differ for {date}: {source}')
        sources.append(source)
    if link:
        base = combined / 'samples/generation' / model
        base.mkdir(parents=True, exist_ok=True)
        for source in sources:
            target = base / source.name
            if target.is_symlink():
                require(target.resolve() == source.resolve(), f'Wrong existing link: {target}')
            else:
                require(not target.exists(), f'Existing output: {target}')
                target.symlink_to(source, target_is_directory=True)
    print(f'PASS: split={expected_split}; {len(sources)} sigma values, {len(first_records)} matching dates; '
          'paired input/noise hashes, checkpoint and legacy controls agree.', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['init', 'run', 'worker', 'audit', 'evaluate'])
    parser.add_argument('indices', nargs='*', type=int)
    args = parser.parse_args()
    threads, jobs, budget = (int(os.environ[k]) for k in ('CPU_THREADS', 'MAX_JOBS', 'CPU_BUDGET'))
    require(min(threads, jobs, budget) > 0, 'CPU settings must be positive')
    concurrent = jobs if args.action == 'run' else 1
    require(concurrent * threads <= budget, 'Jobs × threads exceeds CPU_BUDGET')
    if hasattr(os, 'sched_getaffinity'):
        require(concurrent * threads <= len(os.sched_getaffinity(0)), 'CPU request exceeds process affinity')
    if args.action == 'init':
        require(not args.indices, 'init takes no indices')
        init()
        return
    state = load()
    if args.action == 'worker':
        require(len(args.indices) == 1, 'worker needs exactly one index')
        worker(args.indices[0], state)
    elif args.action == 'run':
        indices = args.indices or list(range(len(state['grid'])))
        require(len(indices) == len(set(indices)), 'Duplicate indices')
        require(all(0 <= i < len(state['grid']) for i in indices), 'Index outside grid')
        require(all(not (ROOT / 'tasks' / f'{i:02d}').exists() for i in indices),
                'One or more tasks already exist. Select unstarted indices or use a new campaign.')
        failures = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(worker, i, state): i for i in indices}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(futures[future])
                    print(f'FAILED index {futures[future]}: {exc}', file=sys.stderr, flush=True)
        require(not failures, f'Failed indices: {failures}; partial outputs retained')
    else:
        require(not args.indices, f'{args.action} takes no indices')
        audit(state, link=args.action == 'evaluate')
        if args.action == 'evaluate':
            # One writer; all sigma directories are read through links, with original metadata intact.
            (ROOT / 'evaluation_started').mkdir(exist_ok=False)
            invoke('evaluate', ROOT / 'combined', log=ROOT / f'evaluate-{time.time_ns()}.log')


if __name__ == '__main__':
    main()
