"""Numerical definitions, empty populations and saved-artifact/plot integration."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from revision_evaluation import probabilistic_calibration as metrics
from revision_evaluation.run_probabilistic import run
from revision_evaluation.run_deterministic import run as run_deterministic
from revision_evaluation.plot_probabilistic import run as plot
from revision_evaluation.tests import test_deterministic as fixtures


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


class MetricTests(unittest.TestCase):
    def test_linear_coverage_endpoints_and_width(self):
        ens = np.tile(np.array([0., 1., 2., 3.])[:, None], (1, 4))
        obs = np.array([.75, 2.25, 0., 3.])
        rows = metrics.coverage_rows(ens, obs)
        self.assertEqual(rows[0]['empirical_coverage'], .5)
        self.assertEqual(rows[0]['mean_interval_width'], 1.5)
        self.assertEqual(rows[0]['n_covered'], 2)
        self.assertEqual(rows[0]['n_pixels'], 4)
        self.assertTrue(all(b['mean_interval_width'] >= a['mean_interval_width']
                            and b['empirical_coverage'] >= a['empirical_coverage']
                            for a, b in zip(rows[:-1], rows[1:])))
        perfect = metrics.coverage_rows(np.repeat(obs[None], 32, axis=0), obs)
        self.assertTrue(all(r['empirical_coverage'] == 1 and r['mean_interval_width'] == 0 for r in perfect))

    def test_empirical_crps_against_brute_force(self):
        ens = np.array([[0., 1.], [2., 4.], [5., 8.]])
        obs = np.array([1., 7.])
        expected = np.mean(abs(ens-obs), axis=0) - .5*np.mean(abs(ens[:, None]-ens[None, :]), axis=(0, 1))
        np.testing.assert_allclose(metrics.empirical_crps(ens, obs), expected)
        # M² gives 0.5 here; replacing it by the fair M(M-1) denominator gives 0.
        self.assertEqual(metrics.empirical_crps(np.array([[0.], [2.]]), np.array([1.]))[0], .5)
        np.testing.assert_array_equal(metrics.empirical_crps(np.repeat(obs[None], 32, axis=0), obs), [0., 0.])

    def test_rank_ties_are_integer_uniform_and_seeded(self):
        ens, obs = np.zeros((3, 60000)), np.zeros(60000)
        counts = metrics.rank_counts(ens, obs, np.random.default_rng(504))
        self.assertEqual(counts.sum(), len(obs))
        np.testing.assert_allclose(counts/len(obs), np.full(4, .25), atol=.01)
        np.testing.assert_array_equal(counts, metrics.rank_counts(ens, obs, np.random.default_rng(504)))
        ens = np.tile(np.array([0., 1., 2.])[:, None], (1, 2))
        np.testing.assert_array_equal(metrics.rank_counts(ens, np.array([-1., 3.]), np.random.default_rng(1)), [1, 0, 0, 1])

    def test_reliability_keeps_zero_one_and_internal_edge(self):
        ens = np.array([[0., 0., 2., 2.], [0., 2., 0., 2.]])
        obs = np.array([0., 1., 0., 1.])
        rows = metrics.reliability_counts(ens, obs, 1)
        self.assertEqual(sum(r['n_cases'] for r in rows), 4)
        self.assertEqual([rows[i]['n_cases'] for i in [0, 5, 9]], [1, 2, 1])
        pooled = metrics.pool_bins(rows, ('threshold', 'bin', 'bin_left', 'bin_right'),
                                   ('n_cases', 'forecast_probability_sum', 'n_events'), 'reliability')
        self.assertEqual(pooled[0]['mean_forecast_probability'], 0)
        self.assertEqual(pooled[5]['observed_frequency'], .5)
        self.assertEqual(pooled[9]['mean_forecast_probability'], 1)
        self.assertTrue(np.isnan(pooled[1]['observed_frequency']))

    def test_spread_bins_and_pooled_rmse(self):
        ens = np.array([[0., 0.], [2., 4.]])
        np.testing.assert_allclose(ens.std(axis=0, ddof=1), [np.sqrt(2), np.sqrt(8)])
        edges = metrics.spread_edges(np.array([0., 0., 0., 1., 2.]), 10)
        self.assertTrue(np.all(np.diff(edges) > 0))
        rows = metrics.spread_skill_counts(np.array([0., 1., 2.]), np.array([0., 4., 9.]), edges)
        self.assertEqual(sum(r['n_cases'] for r in rows), 3)
        constant = metrics.spread_edges(np.zeros(5))
        np.testing.assert_array_equal(constant, [0., 0.])
        self.assertEqual(metrics.spread_skill_counts(np.zeros(5), np.ones(5), constant)[0]['n_cases'], 5)
        rows = [dict(bin=0, n_cases=1, spread_sum=1., squared_error_sum=0.),
                dict(bin=0, n_cases=3, spread_sum=9., squared_error_sum=12.)]
        pooled = metrics.pool_bins(rows, ('bin',), ('n_cases', 'spread_sum', 'squared_error_sum'), 'spread')[0]
        self.assertAlmostEqual(pooled['rmse'], np.sqrt(3))
        self.assertEqual(pooled['mean_spread'], 2.5)

    def test_finite_land_and_empty_populations(self):
        obs = np.array([[0., 1., 2., 3.]])
        ens = np.repeat(obs[None], 2, axis=0)
        ens[0, 0, 2] = np.nan
        e, o = metrics.finite_cases(ens, obs, np.array([[1, 1, 1, 0]]))
        np.testing.assert_array_equal(o, [0, 1])
        e, o = e[:, :0], o[:0]
        self.assertTrue(all(r['n_pixels'] == 0 and np.isnan(r['empirical_coverage'])
                            and np.isnan(r['mean_interval_width']) for r in metrics.coverage_rows(e, o)))
        self.assertEqual(metrics.empirical_crps(e, o).size, 0)
        self.assertEqual(metrics.rank_counts(e, o, np.random.default_rng(0)).sum(), 0)
        self.assertEqual(sum(r['n_cases'] for r in metrics.reliability_counts(e, o, 1)), 0)

    def test_summary_weights_dates_and_preserves_empty_seasons(self):
        rows = [dict(date='20190101', season='DJF', subset='all_land', crps=0., n_pixels=100),
                dict(date='20190102', season='DJF', subset='all_land', crps=2., n_pixels=1)]
        summary = metrics.summarize_daily(rows, 'crps')
        self.assertEqual(summary[0]['mean'], 1.)
        self.assertEqual(summary[0]['n_pixel_days'], 101)
        empty = next(r for r in summary if r['season'] == 'MAM')
        self.assertTrue(np.isnan(empty['mean']))
        self.assertEqual(empty['n_valid_dates'], 0)


class ArtifactTests(unittest.TestCase):
    setUp = fixtures.ArtifactTests.setUp

    def freeze(self):
        path = self.root / 'dates.txt'
        path.write_text('20190101\n20200701\n')
        self.cfg['dates_file'] = str(path)

    def test_full_run_tables_preserved_inputs_and_plots(self):
        self.freeze()
        before = {p: p.read_bytes() for name in ('ceddar', 'qm', 'bilinear') for p in (self.root/name).rglob('*') if p.is_file()}
        output = run(self.cfg)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['n_dates'], 2)
        self.assertEqual(manifest['n_pixel_days'], 10)
        self.assertEqual(manifest['spread_edges'], [0., 0.])
        self.assertTrue(all(float(r['empirical_coverage']) == 1 for r in table(output / 'coverage_daily.csv')))
        self.assertTrue(all(float(r['crps']) == 0 for r in table(output / 'crps.csv')))
        ranks = table(output / 'rank_histogram.csv')
        self.assertEqual(len(ranks), 33)
        self.assertEqual(sum(int(r['count']) for r in ranks), 10)
        for threshold in [1, 10, 20]:
            rows = [r for r in table(output / 'reliability.csv') if float(r['threshold']) == threshold]
            self.assertEqual(sum(int(r['n_cases']) for r in rows), 10)
            self.assertTrue(all(r['observed_frequency'] == r['mean_forecast_probability'] for r in rows))
        plots = plot(output)
        self.assertEqual(len(list(plots.glob('*.png'))), 5)
        self.assertEqual(len(list(plots.glob('*.pdf'))), 5)
        self.assertEqual(json.loads((plots / 'figure_provenance.json').read_text())['n_dates'], 2)
        with self.assertRaises(FileExistsError):
            plot(output)
        with self.assertRaises(FileExistsError):
            run(self.cfg)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_no_valid_pixels_and_empty_wet_subset(self):
        self.freeze()
        roi = self.root / 'roi.npz'
        np.savez(roi, mask=np.zeros_like(self.land))
        self.cfg['roi_mask'] = str(roi)
        output = run(self.cfg)
        self.assertTrue(all(r['n_pixels'] == '0' and r['crps'] == '' for r in table(output / 'crps.csv')))
        self.assertTrue(all(r['n_pixels'] == '0' and r['empirical_coverage'] == ''
                            for r in table(output / 'coverage_daily.csv')))
        self.assertTrue(all(r['rmse'] == '' for r in table(output / 'spread_skill.csv')))
        self.assertTrue(all(r['frequency'] == '' for r in table(output / 'rank_histogram.csv')))
        self.assertEqual(len(list(plot(output).glob('*.png'))), 5)

    def test_frozen_dates_required_and_failure_status(self):
        with self.assertRaisesRegex(FileNotFoundError, 'Reuse Group 1'):
            run(self.cfg)
        self.freeze()
        np.savez(self.root / 'ceddar/ensembles_phys/20190101.npz', ens=np.zeros((2, 1, 2, 3)))
        with self.assertRaisesRegex(ValueError, 'expected 32'):
            run(self.cfg)
        self.assertEqual(json.loads((self.root/'results/probabilistic/manifest.json').read_text())['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'complete probabilistic'):
            # Empty tables are not present after failed metrics; test guard directly.
            from revision_evaluation.plot_common import prepare_output
            prepare_output(self.root/'results/probabilistic', self.root/'plots',
                           {'status': 'failed', 'stage': 'probabilistic'}, 'probabilistic')

    def test_cli_launchers_and_group1_plots(self):
        deterministic = run_deterministic(self.cfg)
        source = Path(__file__).resolve().parents[2]
        env = {**os.environ, 'PYTHON': sys.executable}
        result = subprocess.run(['bash', str(source/'repro/run_revision_probabilistic.sh'),
                                 '--config', str(self.config_path)],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Metrics pass: 2/2', result.stderr)
        for group in ('deterministic', 'probabilistic'):
            result = subprocess.run(['bash', str(source/'repro/plot_revision.sh'), group,
                                     '--input-dir', str(deterministic.parent/group)],
                                    cwd=self.root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((self.root/'results/manuscript_ready/deterministic').glob('*.pdf'))), 2)


if __name__ == '__main__':
    unittest.main()
