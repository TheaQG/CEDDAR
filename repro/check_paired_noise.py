"""Compare actual Gaussian draw hashes across paired sigma* run directories."""
import argparse
import json
from pathlib import Path


def check_runs(roots):
    groups = []
    for root in roots:
        by_variant = {}
        for path in sorted(Path(root).glob('samples/generation/*/sigma_star=*/meta/noise/*.json')):
            by_variant.setdefault(str(path.parents[2]), {})[path.stem] = json.loads(path.read_text())
        if not by_variant:
            raise ValueError(f'No paired-noise records under {root}')
        groups.extend(by_variant.values())
    if len(groups) < 2:
        raise ValueError('Need at least two sigma/run variants to compare')
    reference = groups[0]
    if any(set(g) != set(reference) for g in groups[1:]):
        raise ValueError('Variants do not contain the same dates')
    unmatched = 0
    for date, ref in reference.items():
        for other in groups[1:]:
            record = other[date]
            for key in ('protocol', 'root_seed', 'noise_seed', 'device', 'torch_version', 'inputs_sha256'):
                if ref[key] != record[key]:
                    raise ValueError(f'{date}: different {key}')
            shared = ref['draws'].keys() & record['draws'].keys()
            if 'initial' not in shared:
                raise ValueError(f'{date}: missing initial draw')
            for stream in shared:
                if ref['draws'][stream] != record['draws'][stream]:
                    raise ValueError(f'{date}: different standard normals for {stream}')
            unmatched += len(ref['draws'].keys() ^ record['draws'].keys())
    return dict(variants=len(groups), dates=len(reference), unshared_draws=unmatched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    result = check_runs(args.run_dirs)
    print('Paired inputs and shared standard-normal draws match:', result)
    if result['unshared_draws']:
        print('Unshared draws reflect different active churn/null branches; shared step indices remain aligned.')
    print('This checks noise and inputs, not checkpoint identity, calibration or sample equality.')


if __name__ == '__main__':
    main()
