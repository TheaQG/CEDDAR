"""Check legacy/matched runs and an optional late-ramp run; export source-labelled tables."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from repro.check_paired_noise import check_runs
from sbgm.provenance import sigma_generation_metadata
from sbgm.runtime import external_output

POLICIES = ('legacy_sigma_max', 'schedule', 'schedule')
MODES = ('global', 'global', 'late_ramp')
LABELS = ('legacy', 'matched', 'late_ramp')


def check_configs(roots):
    if len(roots) not in (2, 3) or len(set(roots)) != len(roots):
        raise ValueError('Supply distinct legacy, matched and optionally late-ramp directories')
    configs, normalized = [], []
    for root, policy, mode in zip(roots, POLICIES, MODES):
        cfg = yaml.safe_load((root / 'resolved_config.yaml').read_text())
        full = cfg['full_gen_eval']
        control = full['sigma_control']
        expected = dict(noise_mode='paired', sigma_star_mode=mode, sigma_star_initial_state=policy)
        for key, value in expected.items():
            if control.get(key) != value:
                raise ValueError(f'{root}: expected {key}={value}, got {control.get(key)}')
        if not {0.95, 1.0, 1.05}.issubset(set(full['sigma_star_grid'])):
            raise ValueError(f'{root}: expected comparison grid including 0.95, 1, 1.05')
        cleaned = copy.deepcopy(cfg)
        cleaned['full_gen_eval']['sigma_control'].pop('sigma_star_initial_state')
        cleaned['full_gen_eval']['sigma_control'].pop('sigma_star_mode')
        for key, suffix in dict(sample_dir='samples', evaluation_dir='evaluation',
                                log_dir='logs', path_save='samples').items():
            if Path(cfg['paths'][key]).resolve() != root / suffix:
                raise ValueError(f'{root}: {key} points outside this run')
            cleaned['paths'][key] = suffix
        if Path(cfg['diagnostics']['histogram_path']).resolve() != root / 'logs/histograms':
            raise ValueError(f'{root}: histogram_path points outside this run')
        cleaned['diagnostics']['histogram_path'] = 'logs/histograms'
        # Date labels need not match for separately prepared historical pilots.
        cleaned.get('experiment', {}).pop('date', None)
        configs.append(cfg)
        normalized.append(cleaned)
    if any(cfg != normalized[0] for cfg in normalized[1:]):
        raise ValueError('Configs differ beyond scaling mode, initialization, output paths and experiment date')
    return configs


def check_generation(roots, configs):
    report = check_runs(roots)
    checkpoints, bases = [], []
    for root, cfg in zip(roots, configs):
        candidates = sorted((root / 'samples/generation').iterdir())
        candidates = [p for p in candidates if p.is_dir()]
        if len(candidates) != 1:
            raise ValueError(f'{root}: expected exactly one model directory')
        base = candidates[0]
        records = sigma_generation_metadata(base, cfg['full_gen_eval']['sigma_star_grid'], cfg)
        for record in records:
            if not record.get('manifest'):
                raise ValueError('Missing observed generation provenance')
            manifest = yaml.safe_load(Path(record['manifest']).read_text())
            digest = manifest['checkpoint'].get('sha256')
            if not digest:
                raise ValueError('Missing checkpoint hash')
            checkpoints.append(digest)
        bases.append(base)
    if len(set(checkpoints)) != 1:
        raise ValueError('Checkpoint hashes differ')
    expected_dates = configs[0]['full_gen_eval']['max_dates']
    if expected_dates > 0 and report['dates'] != expected_dates:
        raise ValueError(f"Expected {expected_dates} dates, found {report['dates']}")
    dates = sorted(p.stem for p in (bases[0] / 'sigma_star=1.00/meta/noise').glob('*.json'))
    for date in dates:
        arrays = []
        for base in bases:
            with np.load(base / f'sigma_star=1.00/ensembles_phys/{date}.npz') as data:
                arrays.append(data['ens'])
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError(f'{date}: non-finite baseline samples')
        if any(not np.array_equal(arrays[0], array) for array in arrays[1:]):
            raise ValueError(f'{date}: sigma*=1 physical samples differ; inspect execution settings')
    return {**report, 'checkpoint_sha256': checkpoints[0], 'baseline_samples_equal': True, 'date_list': dates}


def export_tables(roots, configs, output, report):
    sources, tables = {}, []
    for label, root, cfg in zip(LABELS, roots, configs):
        paths = list((root / 'evaluation').glob('*/prcp/sigma_control/tables/metrics_by_sigma.csv'))
        if len(paths) != 1:
            raise ValueError(f'{root}: expected exactly one metrics_by_sigma.csv')
        path = paths[0]
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        table = {(r['date'], float(r['sigma_star'])): r for r in rows}
        grid = cfg['full_gen_eval']['sigma_star_grid']
        if len(table) != len(rows) or len(table) != report['dates'] * len(grid):
            raise ValueError(f'{root}: missing or duplicate metric rows')
        dates = {d for d, _ in table}
        if dates != set(report['date_list']) or set(table) != {(d, float(a)) for d in dates for a in grid}:
            raise ValueError(f'{root}: inconsistent metric grid')
        tables.append(table)
        sources[label] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if any(set(table) != set(tables[0]) for table in tables[1:]):
        raise ValueError('Metric dates/grid differ')
    for key in tables[0]:
        if key[1] == 1.0:
            for metric, value in tables[0][key].items():
                for table in tables[1:]:
                    if metric != 'date' and not np.array_equal(float(value), float(table[key][metric]), equal_nan=True):
                        raise ValueError(f'{key}: baseline metric differs: {metric}')
    if len({source['sha256'] for source in sources.values()}) < len(sources):
        print('NOTE: some entire metric tables are identical; inspect outputs before interpreting sensitivity.')
    output = external_output(output)
    output.mkdir(parents=True, exist_ok=False)
    for label, source in sources.items():
        shutil.copyfile(source['path'], output / f'{label}_metrics_by_sigma.csv')
    (output / 'sources.json').write_text(json.dumps({**report, 'tables': sources}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('legacy', type=Path)
    parser.add_argument('matched', type=Path)
    parser.add_argument('late_ramp', nargs='?', type=Path, help='Optional third run: late_ramp with schedule initialization')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--config-only', action='store_true')
    mode.add_argument('--generation-only', action='store_true')
    mode.add_argument('--output', type=Path, help='New directory for named CSV copies and source hashes')
    args = parser.parse_args()
    roots = [args.legacy.resolve(), args.matched.resolve()]
    if args.late_ramp is not None:
        roots.append(args.late_ramp.resolve())
    configs = check_configs(roots)
    print('Configs match except for scaling mode, initialization, output paths and experiment date.')
    if args.config_only:
        return
    report = check_generation(roots, configs)
    print(json.dumps(report, indent=2))
    if args.output:
        export_tables(roots, configs, args.output, report)


if __name__ == '__main__':
    main()
