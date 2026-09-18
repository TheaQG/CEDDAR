"""Evaluate saved CEDDAR ensemble members on Group 1's frozen dates and support."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from sbgm.runtime import external_output
from .common_io import RevisionInputs, resolve_inputs, write_table, write_run_metadata
from .probabilistic_calibration import (COVERAGES, PROBABILITY_EDGES, SUBSETS, finite_cases,
    coverage_rows, empirical_crps, rank_counts, reliability_counts, spread_edges,
    spread_skill_counts, pool_bins, summarize_daily, ratio)
from .thresholds import EVENT_THRESHOLDS, WET_THRESHOLD, season_of

logger = logging.getLogger(__name__)


def run(config, rank_seed=504, spread_bins=10):
    if rank_seed < 0 or spread_bins < 1:
        raise ValueError('rank_seed must be nonnegative and spread_bins must be positive')
    config = dict(config)
    frozen = Path(config['dates_file'] or Path(config['output_root']) / 'deterministic/dates.txt')
    if not frozen.is_file():
        raise FileNotFoundError('Reuse Group 1 dates.txt via --dates-file (or deterministic/dates.txt)')
    config['dates_file'] = str(frozen.resolve())
    directory = external_output(Path(config['output_root']) / 'probabilistic')
    for spec in config['inputs'].values():
        root = Path(spec['root']).resolve()
        if directory == root or root in directory.parents:
            raise ValueError('Evaluation output cannot be inside an input generation directory')
    directory.mkdir(parents=True, exist_ok=False)
    settings = dict(stage='probabilistic', output=str(directory),
                    thresholds=list(EVENT_THRESHOLDS), wet_threshold=WET_THRESHOLD,
                    coverages=list(COVERAGES), quantile_method='linear; inclusive endpoints',
                    subsets=list(SUBSETS), rank_subset='all_land', rank_seed=rank_seed,
                    rank_definition='integer uniform from number less through number less + number equal',
                    rank_rng='NumPy default_rng(SeedSequence([rank_seed, YYYYMMDD]))',
                    crps_definition='empirical ensemble CRPS, M² pairwise denominator, float64',
                    reliability_edges=PROBABILITY_EDGES.tolist(),
                    bin_closure='left closed, right open; final bin includes right endpoint',
                    spread_definition='sample standard deviation (ddof=1)',
                    skill_definition='RMSE of physical ensemble mean from pooled squared errors',
                    spread_bins_requested=spread_bins, spread_edge_population='all_land, all dates',
                    uncertainty='none; daily summaries describe variation across dates')
    resolved = dict(config, probabilistic=settings)
    (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    manifest = directory / 'manifest.json'
    write_run_metadata(manifest, config, status='running', **settings)
    logger.info('Output: %s; Git: %s', directory, json.loads(manifest.read_text())['git'])
    logger.info('Input roots: %s', {k: v['root'] for k, v in config['inputs'].items()})
    logger.info('Members: %d; rank seed: %d; thresholds: %s; coverage: %s',
                config['expected_members'], rank_seed, EVENT_THRESHOLDS, COVERAGES)
    try:
        reader = RevisionInputs(config)
        (directory / 'date_inventory.json').write_text(json.dumps(reader.inventory, indent=2)+'\n')
        (directory / 'dates.txt').write_text('\n'.join(reader.dates)+'\n')
        logger.info('Common available dates: %d (%s to %s); unavailable calendar dates: %d',
                    len(reader.dates), reader.dates[0], reader.dates[-1],
                    len(reader.inventory['missing_calendar_dates']))
        if config['expected_members'] != 32:
            logger.warning('Non-32-member configuration; use only for synthetic verification')
        # Retain one spread per valid pixel-day (~42 MB for this 644-date sample),
        # not all ensembles. The second pass uses these one-time pooled edges.
        spread_chunks = []
        for i, date in enumerate(reader.dates, 1):
            sample = reader.load_date(date)
            ens, obs = finite_cases(sample['ensemble'], sample['observation'], sample['valid'])
            spread_chunks.append(ens.std(axis=0, ddof=1))
            if i == 1 or i % 50 == 0 or i == len(reader.dates):
                logger.info('Spread pass: %d/%d dates', i, len(reader.dates))
        edges = spread_edges(np.concatenate(spread_chunks), spread_bins)
        del spread_chunks
        settings['spread_edges'] = edges.tolist()
        settings['spread_bins_actual'] = len(edges)-1
        (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(dict(config, probabilistic=settings), sort_keys=False))
        logger.info('Pooled spread edges (mm/day): %s', edges.tolist())
        logger.info('Land mask example: %s', next(r['path'] for r in reader.files.values() if r['role'] == 'land_mask'))
        coverage, crps, reliability, spread, ranks, support = [], [], [], [], [], []
        for i, date in enumerate(reader.dates, 1):
            sample = reader.load_date(date)
            ens, obs = finite_cases(sample['ensemble'], sample['observation'], sample['valid'])
            common = dict(date=date, season=season_of(date))
            sd, error2 = ens.std(axis=0, ddof=1), (ens.mean(axis=0)-obs)**2
            scores = empirical_crps(ens, obs)
            n = int(obs.size)
            negative = int(np.count_nonzero(ens < 0))
            support.append(dict(**common, n_land_pixels=int(sample['land'].sum()), n_pixels=n,
                                n_excluded_nonfinite=int(sample['land'].sum())-n,
                                n_negative_member_values=negative,
                                n_negative_observations=int(np.count_nonzero(obs < 0))))
            if negative or np.any(obs < 0):
                logger.warning('%s: negative physical values retained without clipping', date)
            for subset in SUBSETS:
                selected = np.ones(n, dtype=bool) if subset == 'all_land' else obs >= WET_THRESHOLD
                prefix = dict(**common, subset=subset)
                coverage.extend(dict(**prefix, **row) for row in coverage_rows(ens[:, selected], obs[selected]))
                count, total = int(selected.sum()), float(scores[selected].sum())
                crps.append(dict(**prefix, crps=ratio(total, count), crps_sum=total, n_pixels=count))
                spread.extend(dict(**prefix, **row) for row in spread_skill_counts(sd[selected], error2[selected], edges))
            rng = np.random.default_rng(np.random.SeedSequence([rank_seed, int(date)]))
            counts = rank_counts(ens, obs, rng)
            ranks.extend(dict(**common, subset='all_land', rank=j, count=int(c), n_cases=n)
                         for j, c in enumerate(counts))
            for threshold in EVENT_THRESHOLDS:
                reliability.extend(dict(**common, subset='all_land', **row)
                                   for row in reliability_counts(ens, obs, threshold))
            if i == 1 or i % 25 == 0 or i == len(reader.dates):
                logger.info('Metrics pass: %d/%d dates; %s, %d valid land pixels', i, len(reader.dates), date, n)
        reader.check_unchanged()
        for name, rows in [('coverage_daily', coverage), ('crps', crps),
                           ('reliability_daily_counts', reliability), ('spread_skill_daily_counts', spread),
                           ('rank_daily_counts', ranks), ('daily_support', support),
                           ('input_files', list(reader.files.values()))]:
            write_table(directory / f'{name}.csv', rows)
        write_table(directory / 'coverage_summary.csv',
                    summarize_daily(coverage, 'empirical_coverage', ('nominal_coverage',)) +
                    summarize_daily(coverage, 'mean_interval_width', ('nominal_coverage',)))
        write_table(directory / 'crps_summary.csv', summarize_daily(crps, 'crps'))
        write_table(directory / 'reliability.csv', pool_bins(reliability,
                    ('subset', 'threshold', 'bin', 'bin_left', 'bin_right'),
                    ('n_cases', 'forecast_probability_sum', 'n_events'), 'reliability'))
        write_table(directory / 'spread_skill.csv', pool_bins(spread,
                    ('subset', 'bin', 'bin_left', 'bin_right'),
                    ('n_cases', 'spread_sum', 'squared_error_sum'), 'spread'))
        total = sum(row['n_pixels'] for row in support)
        write_table(directory / 'rank_histogram.csv', [dict(subset='all_land', rank=j,
                    count=sum(r['count'] for r in ranks if r['rank'] == j),
                    frequency=ratio(sum(r['count'] for r in ranks if r['rank'] == j), total),
                    n_cases=total, expected_frequency=1/(config['expected_members']+1),
                    n_dates=len(reader.dates), n_valid_dates=sum(r['n_pixels'] > 0 for r in support))
                    for j in range(config['expected_members']+1)])
        write_run_metadata(manifest, config, status='complete', dates=reader.dates,
                           n_dates=len(reader.dates), n_pixel_days=total,
                           n_valid_dates=sum(r['n_pixels'] > 0 for r in support), grid_shape=list(reader.shape),
                           land_mask_sources=sorted(r['path'] for r in reader.files.values() if r['role'] == 'land_mask'),
                           **settings)
    except Exception as error:
        write_run_metadata(manifest, config, status='failed', error=str(error), **settings)
        raise
    logger.info('Completed probabilistic evaluation: %s', directory)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('atmo.yaml'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dates-file', type=Path)
    parser.add_argument('--rank-seed', type=int, default=504)
    parser.add_argument('--spread-bins', type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    print(run(resolve_inputs(args.config, args.output_root, args.dates_file), args.rank_seed, args.spread_bins))


if __name__ == '__main__':
    main()
