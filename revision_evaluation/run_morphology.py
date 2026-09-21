"""Group 4: paired absolute/equal-area objects and SAL of individual realizations."""
import argparse
import json
import logging
from pathlib import Path

import yaml

from sbgm.runtime import external_output
from .common_io import RevisionInputs, resolve_inputs, write_table, write_run_metadata
from .object_metrics import THRESHOLDS, object_metrics, equal_area_mask
from .sal_analysis import domain_diameter, field_sal, sal_components
from .thresholds import season_of

logger = logging.getLogger(__name__)


def run(config):
    config = dict(config)
    frozen = Path(config['dates_file'] or Path(config['output_root']) / 'deterministic/dates.txt')
    if not frozen.is_file():
        raise FileNotFoundError('Reuse Group 1 dates.txt via --dates-file or <output_root>/deterministic/dates.txt')
    config['dates_file'] = str(frozen.resolve())
    if not {'qm', 'era5_bilinear'} <= set(config['methods']):
        raise ValueError('Keep QM and bilinear ERA5 in the common-input methods')
    output = external_output(Path(config['output_root']) / 'morphology')
    for spec in config['inputs'].values():
        root = Path(spec['root']).resolve()
        if output == root or root in output.parents:
            raise ValueError('Output cannot be inside a generation input directory')
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(stage='morphology', output=str(output), thresholds=list(THRESHOLDS),
                    methods=['danra', 'era5_bilinear', 'qm', 'ceddar_members'],
                    members=config['expected_members'], member_ids='zero-based',
                    connectivity=8, area_units='pixels', intensity_units='mm/day',
                    object_definition='P >= threshold on common valid land; no smoothing or size filter',
                    equal_area='linear Q_(1-f_obs), inclusive ties; zero target selects empty mask',
                    sal='Wernli components with fixed physical thresholds; not legacy structure proxies',
                    sal_distance='grid-cell-centre Euclidean distance / diameter of common valid land',
                    sal_negative_policy='negative precipitation makes SAL undefined; no clipping',
                    uncertainty='none; members remain nested within dates')
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(dict(config, morphology=settings), sort_keys=False))
    manifest = output / 'manifest.json'
    write_run_metadata(manifest, config, status='running', **settings)
    logger.info('Output: %s; Git: %s', output, json.loads(manifest.read_text())['git'])
    logger.info('Inputs: %s', {k: v['root'] for k, v in config['inputs'].items()})
    logger.info('Thresholds: %s; members: %s; 8-neighbour objects', THRESHOLDS, config['expected_members'])
    absolute, relative, sal, support = [], [], [], []
    try:
        reader = RevisionInputs(config)
        (output / 'dates.txt').write_text('\n'.join(reader.dates)+'\n')
        logger.info('Frozen available dates: %d (%s–%s)', len(reader.dates), reader.dates[0], reader.dates[-1])
        for index, date in enumerate(reader.dates, 1):
            sample = reader.load_date(date)
            obs, valid = sample['observation'], sample['valid']
            n = int(valid.sum())
            support.append(dict(date=date, n_pixels=n, n_land_pixels=int(sample['land'].sum())))
            diameter = domain_diameter(valid)
            references = {q: field_sal(obs, valid, q) for q in THRESHOLDS}
            # Keep the Group 1 input/mask policy, but do not use ensemble averages as morphology.
            fields = [('danra', '', obs), ('era5_bilinear', '', sample['fields']['era5_bilinear']),
                      ('qm', '', sample['fields']['qm'])]
            fields.extend(('ceddar_members', member, field) for member, field in enumerate(sample['ensemble']))
            for method, member, field in fields:
                for threshold in THRESHOLDS:
                    base = dict(date=date, season=season_of(date), method=method, member=member,
                                subset='all_land', reference_threshold=threshold)
                    mask = valid & (field >= threshold)
                    absolute.append(dict(**base, threshold=threshold, threshold_type='absolute',
                                         **object_metrics(field, valid, mask)))
                    if method == 'danra':
                        fraction = float(mask.sum()/n) if n else float('nan')
                        eq_mask, info = mask, dict(effective_threshold=threshold,
                            target_wet_fraction=fraction, achieved_wet_fraction=fraction,
                            wet_fraction_difference=0. if n else float('nan'), target_wet_pixel_count=int(mask.sum()))
                    else:
                        eq_mask, info = equal_area_mask(field, obs, valid, threshold)
                        sal.append(dict(**base, threshold_type='absolute', n_pixels=n,
                                        domain_diameter_pixels=diameter,
                                        **sal_components(references[threshold], field_sal(field, valid, threshold), diameter)))
                    relative.append(dict(**base, threshold_type='equal_area', **info,
                                         **object_metrics(field, valid, eq_mask)))
            if index == 1:
                logger.info('Land mask: %s', next(r['path'] for r in reader.files.values() if r['role'] == 'land_mask'))
            if index == 1 or index % 25 == 0 or index == len(reader.dates):
                logger.info('Processed %d/%d dates (%s), %d valid land pixels', index, len(reader.dates), date, n)
        reader.check_unchanged()
        for name, rows in [('objects_absolute', absolute), ('objects_equal_area', relative),
                           ('sal_absolute', sal), ('daily_support', support), ('input_files', list(reader.files.values()))]:
            write_table(output / f'{name}.csv', rows)
        write_run_metadata(manifest, config, status='complete', dates=reader.dates, n_dates=len(reader.dates),
                           n_valid_dates=sum(r['n_pixels'] > 0 for r in support),
                           n_pixel_days=sum(r['n_pixels'] for r in support), grid_shape=list(reader.shape), **settings)
    except Exception as error:
        write_run_metadata(manifest, config, status='failed', error=str(error), **settings)
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('atmo.yaml'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dates-file', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    print(run(resolve_inputs(args.config, args.output_root, args.dates_file)))


if __name__ == '__main__':
    main()
