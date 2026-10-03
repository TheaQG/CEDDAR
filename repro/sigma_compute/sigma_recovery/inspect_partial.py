"""Read-only inventory of interrupted CEDDAR outputs; does not resume generation.

Run in the existing CEDDAR environment. Requires NumPy and PyYAML.
Only the explicitly requested report file is written, outside the campaign.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_npz(path, required_keys):
    # Reading all members also exercises ZIP decompression/CRC checking.
    with np.load(path, allow_pickle=False) as archive:
        require(set(required_keys) <= set(archive.files), f'missing keys in {path.name}')
        arrays = {key: archive[key] for key in archive.files}
    for key, array in arrays.items():
        require(array.dtype.kind in 'biuf', f'non-numeric {path.name}:{key}')
        require(np.isfinite(array).all(), f'nonfinite values in {path.name}:{key}')
    return arrays


def inspect(campaign):
    state = json.loads((campaign / 'campaign.json').read_text())
    split = state.get('split', 'valid')
    report = dict(campaign=str(campaign), inspection_host=socket.gethostname(),
                  split=split, max_dates=state['max_dates'], tasks=[],
                  source_issues=[], pairing_issues=[], metadata_issues=[])
    repo = Path(state['repo'])
    for name, expected in state['source_sha256'].items():
        path = repo / name
        if not path.is_file():
            report['source_issues'].append(f'Missing: {path}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            report['source_issues'].append(f'Changed: {path}')
    references = {}
    checkpoint_hashes = set()
    for index, sigma in enumerate(state['grid']):
        task = campaign / 'tasks' / f'{index:02d}'
        row = dict(index=index, sigma=sigma, complete_candidates=[], incomplete={}, issues=[])
        report['tasks'].append(row)
        try:
            cfg = yaml.safe_load((task / 'run/resolved_config.yaml').read_text())
            full = cfg['full_gen_eval']
            controls = full['sigma_control']
            require(full['split'] == split, 'configuration split differs from campaign')
            require(full['max_dates'] == state['max_dates'], 'max_dates differs from campaign')
            require(full['seed'] == 504 and full['ensemble_size'] == 32, 'wrong seed or member count')
            require(cfg['edm']['sampling_steps'] == 56, 'wrong step count')
            require(controls['noise_mode'] == 'paired', 'noise is not paired')
            require(controls['sigma_star_mode'] == 'global', 'wrong sigma mode')
            require(controls['sigma_star_initial_state'] == 'legacy_sigma_max', 'wrong initialization')
            outputs = list((task / 'run/samples/generation').glob(f'*/sigma_star={sigma:.2f}'))
            require(len(outputs) == 1, 'expected one sigma output directory')
            output = outputs[0]
            row['output'] = str(output)
            provenance_paths = list((output / 'meta').glob('*_generation_*.yaml'))
            require(len(provenance_paths) == 1, 'missing or ambiguous original provenance')
            provenance = yaml.safe_load(provenance_paths[0].read_text())
            require(provenance['data']['split'] == split, 'observed split differs')
            checkpoint = provenance['checkpoint']['sha256']
            require(bool(checkpoint), 'missing checkpoint hash')
            checkpoint_hashes.add(checkpoint)
            row['checkpoint_sha256'] = checkpoint
            row['completion_manifest_exists'] = (output / 'meta/manifest.json').is_file()
            folders = {'ensembles_phys': '.npz', 'pmm_phys': '.npz',
                       'lr_hr_phys': '.npz', 'lsm': '.npz', 'meta/noise': '.json'}
            sets = {folder: {p.stem for p in (output / folder).glob('*' + suffix)}
                    for folder, suffix in folders.items()}
            row['file_counts'] = {folder: len(dates) for folder, dates in sets.items()}
            dates = sorted(set().union(*sets.values()))
            print(f'Checking task {index:02d}, sigma={sigma:.2f}: {len(dates)} saved dates', flush=True)
            for n, date in enumerate(dates, 1):
                try:
                    missing = [folder for folder, names in sets.items() if date not in names]
                    require(not missing, 'missing files: ' + ', '.join(missing))
                    ensemble = read_npz(output / 'ensembles_phys' / f'{date}.npz', ['ens'])['ens']
                    require(ensemble.ndim == 4 and ensemble.shape[:2] == (32, 1), 'wrong ensemble shape')
                    shape = ensemble.shape[-2:]
                    require(tuple(cfg['highres']['data_size']) == shape, 'wrong spatial shape')
                    pmm = read_npz(output / 'pmm_phys' / f'{date}.npz', ['pmm'])['pmm']
                    refs = read_npz(output / 'lr_hr_phys' / f'{date}.npz', ['hr', 'lr'])
                    mask = read_npz(output / 'lsm' / f'{date}.npz', ['lsm_hr'])['lsm_hr']
                    for array in (pmm, refs['hr'], refs['lr'], mask):
                        require(array.shape[-2:] == shape, 'wrong field shape')
                    noise = json.loads((output / 'meta/noise' / f'{date}.json').read_text())
                    require(noise['date'] == date and noise['root_seed'] == 504, 'wrong date/seed')
                    require(noise['protocol'] == 'ceddar-indexed-noise-v1', 'wrong noise protocol')
                    initial = noise['draws']['initial']
                    require(initial['shape'] == list(ensemble.shape), 'wrong initial-noise shape')
                    require(bool(initial['standard_normal_sha256']), 'missing initial-noise hash')
                    if date in references and noise != references[date]:
                        report['pairing_issues'].append(f'task {index:02d}, date {date}: noise/input records differ')
                    else:
                        references[date] = noise
                    row['complete_candidates'].append(date)
                except Exception as exc:
                    row['incomplete'][date] = str(exc)
                if n % 100 == 0:
                    print(f'  inspected {n}/{len(dates)}', flush=True)
        except Exception as exc:
            row['issues'].append(str(exc))
        print(f"Task {index:02d}: {len(row['complete_candidates'])} readable complete candidates; "
              f"{len(row['incomplete'])} incomplete/invalid; {len(row['issues'])} task issues", flush=True)
    if len(checkpoint_hashes) > 1:
        report['metadata_issues'].append('Checkpoint hashes differ across tasks')
    report['scope'] = ('File-integrity inventory only. Does not certify membership/order in the original '
                       'first max_dates dataset entries, unchanged data, or equivalence of restarted model outputs. '
                       'No completion markers are created and no files in the campaign are modified.')
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.expanduser().resolve()
    target = args.report.expanduser().resolve()
    require(target != campaign and campaign not in target.parents, 'Report must be outside campaign')
    require(not target.exists(), 'Report exists; choose a new report filename')
    report = inspect(campaign)
    with target.open('x') as stream:
        json.dump(report, stream, indent=2)
    print('Source issues:', len(report['source_issues']))
    print('Pairing issues:', len(report['pairing_issues']))
    print('Metadata issues:', len(report['metadata_issues']))
    print('Report:', target)
    print('Inspection only: existing generation files were not changed. No jobs were launched.')


if __name__ == '__main__':
    main()
