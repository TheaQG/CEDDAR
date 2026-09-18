"""Evaluate saved fields or preflight them; never generate samples or figures."""
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from sbgm.runtime import external_output
from .common_io import RevisionInputs, resolve_inputs, write_table, write_run_metadata
from .deterministic_metrics import (daily_continuous, occurrence_counts, contingency_counts,
                                    aggregate_occurrence, aggregate_detection)
from .thresholds import EVENT_THRESHOLDS, WET_THRESHOLD, season_of

logger = logging.getLogger(__name__)


def run(config, preflight_only=False):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = external_output(Path(config['output_root']) / (f'preflight_{stamp}' if preflight_only else 'deterministic'))
    for spec in config['inputs'].values():
        root = Path(spec['root']).resolve()
        if directory == root or root in directory.parents:
            raise ValueError('Evaluation output cannot be inside an input generation directory')
    # Exclusive directory creation also prevents two processes mixing result files.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = directory / 'manifest.json'
    settings = dict(stage='deterministic_preflight' if preflight_only else 'deterministic',
                    output=str(directory), thresholds=list(EVENT_THRESHOLDS), wet_threshold=WET_THRESHOLD)
    write_run_metadata(manifest, config, status='running', **settings)
    logger.info('Output: %s; Git: %s', directory, json.loads(manifest.read_text())['git'])
    logger.info('Inputs: %s', {k: v['root'] for k, v in config['inputs'].items()})
    logger.info('Methods: %s; expected members: %s; event thresholds: %s mm/day',
                config['methods'], config['expected_members'], EVENT_THRESHOLDS)
    try:
        reader = RevisionInputs(config)
        (directory / 'date_inventory.json').write_text(json.dumps(reader.inventory, indent=2)+'\n')
        (directory / 'dates.txt').write_text('\n'.join(reader.dates)+'\n')
        logger.info('Common dates: %d (%s to %s); missing calendar dates: %d', len(reader.dates),
                    reader.dates[0], reader.dates[-1], len(reader.inventory['missing_calendar_dates']))
        if len(reader.dates) != 731:
            logger.warning('The common set does not cover all 731 calendar dates; see date_inventory.json')
        if config['expected_members'] != 32:
            logger.warning('Non-32-member configuration: this is not the specified manuscript ensemble')
        continuous, occurrence, detection, checks = [], [], [], []
        for index, date in enumerate(reader.dates, 1):
            sample = reader.load_date(date)
            obs, valid, fields = sample['observation'], sample['valid'], sample['fields']
            common = dict(date=date, season=season_of(date), subset='all_land')
            n_land, n_valid = int(sample['land'].sum()), int(valid.sum())
            if index == 1:
                logger.info('Land mask sources (first date): %s',
                            [r['path'] for r in reader.files.values() if r['role'] == 'land_mask'])
                logger.info('Reference tolerances: rtol=%g, atol=%g; ROI: %s',
                            config['reference_rtol'], config['reference_atol'], config['roi_mask'])
            if n_valid == 0:
                logger.warning('%s: no valid common land pixels; statistics will be NaN', date)
            checks.append(dict(date=date, n_land_pixels=n_land, n_pixels=n_valid,
                               n_excluded_nonfinite=n_land-n_valid))
            for method, field in {'danra': obs, **fields}.items():
                negatives = int(np.count_nonzero(valid & (field < 0)))
                if negatives:
                    logger.warning('%s/%s has %d negative valid values; retained without clipping', date, method, negatives)
                if not preflight_only:
                    occurrence.append(dict(**common, method=method, wet_threshold=WET_THRESHOLD,
                                           **occurrence_counts(field, valid), n_negative_pixels=negatives))
                    if method != 'danra':
                        continuous.append(dict(**common, method=method, **daily_continuous(field, obs, valid)))
                        for threshold in EVENT_THRESHOLDS:
                            detection.append(dict(**common, method=method, threshold=threshold,
                                                  **contingency_counts(field, obs, valid, threshold)))
            if index == 1 or index % 25 == 0 or index == len(reader.dates):
                logger.info('Checked %d/%d dates (%s); valid land pixels: %d', index, len(reader.dates), date, n_valid)
        reader.check_unchanged()
        write_table(directory / 'daily_support.csv', checks)
        write_table(directory / 'input_files.csv', list(reader.files.values()))
        if not preflight_only:
            write_table(directory / 'daily_continuous_metrics.csv', continuous)
            write_table(directory / 'occurrence_daily_counts.csv', occurrence)
            write_table(directory / 'occurrence_metrics.csv', aggregate_occurrence(occurrence, config['methods']))
            write_table(directory / 'event_detection_daily_counts.csv', detection)
            write_table(directory / 'event_detection_metrics.csv', aggregate_detection(detection, config['methods'], EVENT_THRESHOLDS))
        write_run_metadata(manifest, config, status='complete', n_dates=len(reader.dates),
                           n_valid_dates=sum(row['n_pixels'] > 0 for row in checks),
                           n_pixel_days=sum(row['n_pixels'] for row in checks),
                           grid_shape=list(reader.shape),
                           land_mask_sources=sorted(row['path'] for row in reader.files.values() if row['role'] == 'land_mask'),
                           dates=reader.dates, **settings)
    except Exception as error:
        write_run_metadata(manifest, config, status='failed', error=str(error), **settings)
        raise
    logger.info('Completed %s: %s', settings['stage'], directory)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('atmo.yaml'))
    parser.add_argument('--output-root', type=Path, help='Override external evaluation root (must be new for a repeated analysis)')
    parser.add_argument('--dates-file', type=Path, help='Previously saved dates.txt; require exactly the same common dates')
    parser.add_argument('--preflight-only', action='store_true', help='Validate all common dates and save an input report, without metrics')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    config = resolve_inputs(args.config, args.output_root, args.dates_file)
    print(run(config, args.preflight_only))


if __name__ == '__main__':
    main()
