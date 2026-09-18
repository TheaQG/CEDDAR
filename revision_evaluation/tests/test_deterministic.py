import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from sbgm.runtime import SOURCE_ROOT
from revision_evaluation.common_io import RevisionInputs, resolve_inputs
from revision_evaluation.common_masks import joint_land_mask, subset_mask
from revision_evaluation.deterministic_metrics import (
    daily_continuous, occurrence_counts, contingency_counts, detection_scores,
    aggregate_occurrence, aggregate_detection,
)
from revision_evaluation.run_deterministic import run
from revision_evaluation.thresholds import season_of


class MetricTests(unittest.TestCase):
    def test_perfect_and_constant_predictions(self):
        obs = np.array([[0., 1.], [10., 20.]])
        result = daily_continuous(obs, obs, np.ones_like(obs, dtype=bool))
        self.assertEqual([result[k] for k in ('bias', 'mae', 'rmse')], [0, 0, 0])
        self.assertAlmostEqual(result['pearson_r'], 1)
        constant = daily_continuous(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))
        self.assertTrue(np.isnan(constant['pearson_r']))

    def test_nonperfect_daily_errors(self):
        obs = np.array([[1., 3.]])
        pred = np.array([[3., 2.]])
        result = daily_continuous(pred, obs, np.ones_like(obs))
        self.assertEqual(result['bias'], 0.5)
        self.assertEqual(result['mae'], 1.5)
        self.assertAlmostEqual(result['rmse'], np.sqrt(2.5))
        self.assertAlmostEqual(result['pearson_r'], -1)

    def test_known_contingency_and_undefined_ratios(self):
        obs = np.array([[1., 1., 0., 0., 0.]])
        pred = np.array([[1., 0., 1., 1., 0.]])
        counts = contingency_counts(pred, obs, np.ones_like(obs), 1)
        self.assertEqual(counts, dict(hits=1, misses=1, false_alarms=2, correct_negatives=1, n_pixel_days=5))
        scores = detection_scores(1, 1, 2)
        self.assertEqual(scores['pod'], 0.5)
        self.assertAlmostEqual(scores['far'], 2/3)
        self.assertEqual(scores['csi'], 0.25)
        self.assertTrue(all(np.isnan(x) for x in detection_scores(0, 0, 0).values()))

    def test_pooled_detection_not_mean_daily_scores(self):
        rows = [dict(method='qm', threshold=1, hits=1, misses=0, false_alarms=0,
                     correct_negatives=0, n_pixel_days=1),
                dict(method='qm', threshold=1, hits=0, misses=9, false_alarms=0,
                     correct_negatives=0, n_pixel_days=9)]
        result = aggregate_detection(rows, ['qm'], [1])[0]
        self.assertEqual(result['pod'], 0.1)
        self.assertEqual(result['csi'], 0.1)
        self.assertEqual(result['n_events'], 10)

    def test_joint_land_finite_support_and_empty_mask(self):
        obs = np.array([[1., 1000.], [2., 3.]])
        pred = np.array([[1., -1000.], [np.nan, 3.]])
        ens = np.stack([obs, obs])
        ens[1, 1, 1] = np.inf
        valid = joint_land_mask([[1, 0], [1, 1]], obs, pred, ens)
        np.testing.assert_array_equal(valid, [[True, False], [False, False]])
        self.assertEqual(daily_continuous(pred, obs, valid)['n_pixels'], 1)
        self.assertEqual(daily_continuous(pred, obs, valid)['bias'], 0)
        empty = daily_continuous(pred, obs, np.zeros_like(obs))
        self.assertEqual(empty['n_pixels'], 0)
        self.assertTrue(all(np.isnan(empty[k]) for k in ('bias', 'mae', 'rmse', 'pearson_r')))
        with self.assertRaises(ValueError):
            joint_land_mask([[1, .5], [1, 1]], obs)
        with self.assertRaises(ValueError):
            joint_land_mask(np.ones((3, 3)), obs)
        np.testing.assert_array_equal(subset_mask(valid, obs, 'observed_wet'), valid)

    def test_occurrence_uses_pixel_days_and_inclusive_threshold(self):
        a = np.array([[0., 1., 2., np.nan]])
        self.assertEqual(occurrence_counts(a, np.ones_like(a)),
                         dict(n_wet_pixel_days=2, n_pixel_days=3))
        rows = []
        for method, wet in [('danra', 1), ('qm', 2)]:
            rows += [dict(method=method, season='DJF', n_wet_pixel_days=wet, n_pixel_days=3),
                     dict(method=method, season='JJA', n_wet_pixel_days=0, n_pixel_days=1)]
        summary = aggregate_occurrence(rows, ['qm'])
        qm = next(r for r in summary if r['method'] == 'qm' and r['season'] == 'ALL')
        self.assertEqual(qm['wet_frequency'], 0.5)
        self.assertEqual(qm['wet_frequency_bias'], 0.25)
        mam = next(r for r in summary if r['season'] == 'MAM')
        self.assertEqual(mam['n_pixel_days'], 0)
        self.assertTrue(np.isnan(mam['wet_frequency']))
        self.assertEqual([season_of(d) for d in ['20191231', '20200301', '20200601', '20200901']],
                         ['DJF', 'MAM', 'JJA', 'SON'])


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.obs = np.array([[0., 1., 10.], [20., .5, 2.]], dtype=np.float32)
        self.land = np.array([[1, 1, 1], [1, 0, 1]], dtype=bool)
        self.dates = ['20180101', '20190101', '20200701']
        inputs = {}
        for name in ('ceddar', 'bilinear', 'qm'):
            root = self.root / name
            suffix = '_phys' if name == 'ceddar' else ''
            for folder in (f'pmm{suffix}', f'lr_hr{suffix}', 'lsm', 'meta'):
                (root / folder).mkdir(parents=True)
            if name == 'ceddar':
                (root / 'ensembles_phys').mkdir()
            for date in self.dates:
                np.savez(root / f'pmm{suffix}/{date}.npz', pmm=self.obs[None, None])
                np.savez(root / f'lr_hr{suffix}/{date}.npz', hr=self.obs[None, None])
                np.savez(root / f'lsm/{date}.npz', lsm=self.land[None, None])
                if name == 'ceddar':
                    np.savez(root / f'ensembles_phys/{date}.npz', ens=np.repeat(self.obs[None, None], 32, axis=0))
            (root / 'meta/manifest.json').write_text(json.dumps({'fixture': True, 'units': 'mm/day'}))
            inputs[name] = dict(root=str(root), units='mm/day', units_evidence='Synthetic physical arrays',
                                layout='physical' if name == 'ceddar' else 'legacy_physical')
        self.config_path = self.root / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(dict(inputs=inputs, output_root=str(self.root / 'results'))))
        self.cfg = resolve_inputs(self.config_path)

    def test_inventory_and_legacy_physical_baselines(self):
        reader = RevisionInputs(self.cfg)
        self.assertEqual(reader.dates, ['20190101', '20200701'])
        self.assertEqual(reader.inventory['outside_period_by_artifact']['qm/prediction'], ['20180101'])
        sample = reader.load_date('20190101')
        self.assertEqual(sample['valid'].sum(), 5)
        for field in sample['fields'].values():
            np.testing.assert_array_equal(field, self.obs)
        self.assertTrue(any(r['role'] == 'land_mask' for r in reader.files.values()))
        self.assertTrue(all(len(r['sha256']) == 64 for r in reader.files.values()))

    def test_missing_dates_reported_and_frozen_list_enforced(self):
        (self.root / 'qm/pmm/20200701.npz').unlink()
        reader = RevisionInputs(self.cfg)
        self.assertEqual(reader.dates, ['20190101'])
        self.assertIn('20200701', reader.inventory['missing_by_artifact']['qm/prediction'])
        frozen = self.root / 'dates.txt'
        frozen.write_text('20190101\n20200701\n')
        self.cfg['dates_file'] = str(frozen)
        with self.assertRaisesRegex(ValueError, 'Frozen date list differs'):
            RevisionInputs(self.cfg)

    def test_model_space_fallback_is_rejected(self):
        (self.root / 'ceddar/ensembles_phys').rename(self.root / 'ceddar/ensembles')
        with self.assertRaisesRegex(FileNotFoundError, 'model-space fallback is disabled'):
            RevisionInputs(self.cfg)

    def test_missing_physical_key_cannot_fall_back(self):
        path = self.root / 'ceddar/lr_hr_phys/20190101.npz'
        np.savez(path, wrong_key=self.obs)
        with self.assertRaisesRegex(ValueError, 'missing required key'):
            RevisionInputs(self.cfg).load_date('20190101')

    def test_member_count_and_even_member_median(self):
        path = self.root / 'ceddar/ensembles_phys/20190101.npz'
        ens = np.zeros((32, 1, 2, 3), dtype=np.float32)
        ens[16:] = 10
        np.savez(path, ens=ens)
        sample = RevisionInputs(self.cfg).load_date('20190101')
        np.testing.assert_array_equal(sample['fields']['ceddar_median'], np.full((2, 3), 5))
        np.testing.assert_array_equal(sample['fields']['ceddar_pmm'], self.obs)  # Saved PMM preserved.
        np.savez(path, ens=ens[:8])
        with self.assertRaisesRegex(ValueError, 'expected 32 members'):
            RevisionInputs(self.cfg).load_date('20190101')

    def test_one_nonfinite_member_excludes_pixel_for_every_method(self):
        ens = np.repeat(self.obs[None, None], 32, axis=0)
        ens[0, 0, 0, 0] = np.nan
        np.savez(self.root / 'ceddar/ensembles_phys/20190101.npz', ens=ens)
        sample = RevisionInputs(self.cfg).load_date('20190101')
        self.assertFalse(sample['valid'][0, 0])
        self.assertEqual(sample['valid'].sum(), 4)

    def test_reference_and_land_mismatch_rejected(self):
        path = self.root / 'qm/lr_hr/20190101.npz'
        np.savez(path, hr=(self.obs+1)[None, None])
        with self.assertRaisesRegex(ValueError, 'DANRA reference values differ'):
            RevisionInputs(self.cfg).load_date('20190101')
        np.savez(path, hr=self.obs[None, None])
        np.savez(self.root / 'qm/lsm/20190101.npz', lsm=np.ones_like(self.land))
        with self.assertRaisesRegex(ValueError, 'land mask differs'):
            RevisionInputs(self.cfg).load_date('20190101')

    def test_missing_land_mask_and_invalid_roi_rejected(self):
        (self.root / 'qm/lsm/20190101.npz').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'land mask is required'):
            RevisionInputs(self.cfg).load_date('20190101')
        np.savez(self.root / 'qm/meta/land_mask.npz', mask=self.land)
        roi = self.root / 'roi.npz'
        np.savez(roi, mask=np.zeros_like(self.land))
        self.cfg['roi_mask'] = str(roi)
        sample = RevisionInputs(self.cfg).load_date('20190101')
        self.assertEqual(sample['valid'].sum(), 0)
        np.savez(roi, mask=np.full(self.land.shape, np.nan))
        with self.assertRaisesRegex(ValueError, 'finite 0/1'):
            RevisionInputs(self.cfg).load_date('20190101')

    def test_changed_input_and_invalid_units_rejected(self):
        reader = RevisionInputs(self.cfg)
        reader.load_date('20190101')
        np.savez(self.root / 'qm/pmm/20190101.npz', pmm=self.obs+2)
        with self.assertRaisesRegex(ValueError, 'Input changed'):
            reader.check_unchanged()
        config = yaml.safe_load(self.config_path.read_text())
        config['inputs']['qm']['units'] = 'model_space'
        self.config_path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, 'declare units'):
            resolve_inputs(self.config_path)

    def test_empty_land_output_has_nan_metrics_and_zero_counts(self):
        roi = self.root / 'roi.npz'
        np.savez(roi, mask=np.zeros_like(self.land))
        self.cfg['roi_mask'] = str(roi)
        output = run(self.cfg)
        with (output / 'daily_continuous_metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(r['n_pixels'] == '0' and r['rmse'] == '' for r in rows))
        with (output / 'event_detection_metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(r['hits'] == '0' and r['pod'] == '' for r in rows))

    def test_output_rejected_inside_source_or_inputs(self):
        with self.assertRaisesRegex(ValueError, 'outside the source'):
            resolve_inputs(self.config_path, output_root=SOURCE_ROOT / 'evaluation')
        self.cfg['output_root'] = str(self.root / 'ceddar')
        with self.assertRaisesRegex(ValueError, 'inside an input'):
            run(self.cfg)

    def test_preflight_then_evaluation_tables_and_no_overwrite(self):
        before = {str(p): p.read_bytes() for name in ('ceddar', 'qm', 'bilinear') for p in (self.root/name).rglob('*') if p.is_file()}
        preflight = run(self.cfg, preflight_only=True)
        self.assertFalse((preflight / 'daily_continuous_metrics.csv').exists())
        cfg = resolve_inputs(self.config_path, dates_file=preflight / 'dates.txt')
        output = run(cfg)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['n_dates'], 2)
        self.assertEqual(manifest['n_pixel_days'], 10)
        with (output / 'daily_continuous_metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(float(row['rmse']) == 0 and int(row['n_pixels']) == 5 for row in rows))
        with (output / 'occurrence_metrics.csv').open() as stream:
            occurrence = list(csv.DictReader(stream))
        self.assertEqual(len(occurrence), 30)
        self.assertTrue(all(row['wet_frequency'] == '' for row in occurrence if row['season'] == 'MAM'))
        with (output / 'event_detection_metrics.csv').open() as stream:
            detection = list(csv.DictReader(stream))
        self.assertEqual(len(detection), 15)
        self.assertTrue(all(float(row['csi']) == 1 for row in detection))
        for path, content in before.items():
            self.assertEqual(Path(path).read_bytes(), content)
        with self.assertRaises(FileExistsError):
            run(cfg)

    def test_cli_preflight_and_failed_status(self):
        result = subprocess.run(['bash', str(SOURCE_ROOT / 'repro/run_revision_deterministic.sh'),
                                 '--config', str(self.config_path), '--preflight-only'],
                                cwd=self.root, env={**os.environ, 'PYTHON': sys.executable},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Common dates: 2', result.stderr)
        np.savez(self.root / 'ceddar/ensembles_phys/20190101.npz', ens=np.zeros((2, 1, 2, 3)))
        with self.assertRaises(ValueError):
            run(self.cfg)
        manifest = json.loads((self.root / 'results/deterministic/manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertFalse((self.root / 'results/deterministic/daily_continuous_metrics.csv').exists())


if __name__ == '__main__':
    unittest.main()
