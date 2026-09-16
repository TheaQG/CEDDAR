"""Plot regression checks using the reported one-date pilot and synthetic PSDs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sbgm.evaluate.evaluate_prcp.eval_sigma_star import plot_sigma_control as plots
from sbgm.evaluate.evaluate_prcp.eval_sigma_star import evaluate_sigma_control as evaluation

PILOT = np.array([
    [.95, .7093341785, -4.9901617751, -6.6152789976, 1.1558766365, .0028188297],
    [1., .5441676346, -5.2213827376, -6.6152789976, 1.0961328745, .1824964308],
    [1.05, .3223138305, -6.3243420165, -6.6152789976, .7765203118, .1981827912]])


def write_pilot(root, values=PILOT):
    tables = root / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    keys = ['sigma_star', 'r_lp', 'slope_gen', 'slope_hr', 'crps', 'hk_gain']
    np.savetxt(tables / 'agg_summary.csv', values, delimiter=',', comments='',
               header=','.join([keys[0]] + [k + '_mean' for k in keys[1:]]))
    np.savetxt(tables / 'metrics_by_sigma.csv', values, delimiter=',', comments='', header=','.join(keys))
    (root / 'sigma_control_meta.json').write_text(json.dumps({
        'metrics': {'psd_band_km': [5, 20], 'crps_rain_thresh': 1, 'eval_land_only': True},
        'ramp': {'mode': 'global', 'start_frac': .6, 'end_frac': .85, 'initial_state': 'schedule'}}))
    return tables / 'agg_summary.csv'


class SigmaPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_pilot_points_are_visible_without_single_date_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = write_pilot(root)
            figures = {}
            with patch.object(plots, '_savefig', side_effect=lambda f, p, **kw: figures.update({Path(p).name: f})):
                plots.plot_sigma_control(summary, root / 'figures', combined=True, error_mode='sem')
            fig = figures['sigma_control_overview_sem.png']
            for ax, col in zip(fig.axes, [1, 2, 4, 5]):
                low, high = ax.get_ylim()
                self.assertLess(low, PILOT[:, col].min())
                self.assertGreater(high, PILOT[:, col].max())
                self.assertFalse(ax.containers)  # no invented uncertainty for one date
            self.assertIn('PMM', fig.axes[0].get_title())
            self.assertIn('mm/day', fig.axes[2].get_ylabel())

    def test_sem_uses_finite_count_for_each_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = write_pilot(root)
            (root / 'tables/metrics_by_sigma.csv').write_text('sigma_star,crps\n1,1\n1,3\n1,nan\n')
            np.testing.assert_allclose(plots._date_errors(summary, np.array([1.]), 'crps', 'sem'), [1.])

    def test_single_sigma_and_nan_metric_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = PILOT[1:2].copy()
            values[0, 4] = np.nan
            summary = write_pilot(root, values)
            figures = []
            with patch.object(plots, '_savefig', side_effect=lambda f, *a, **kw: figures.append(f)):
                plots.plot_sigma_control(summary, root / 'figures', combined=True)
            self.assertIn('No finite values', [t.get_text() for t in figures[0].axes[2].texts])

    def test_psd_global_labels_and_unclipped_power(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pilot(root)
            k = np.array([.01, .06, .10, .15, .25])
            power = np.array([1e7, 1e-3, 1e-6, 1e-9, 1e-12])
            np.savez(root / 'tables/sigma_psd_curves.npz', k=k, sigma_vals=[1.],
                     psd_hr_mean=power, psd_hr_std=np.zeros(5), psd_lr_mean=power,
                     psd_lr_std=np.zeros(5), psd_gen_mean=power[None],
                     psd_gen_std=np.zeros((1, 5)), lr_nyquist=.02, psd_band_km=[5, 20])
            figures = []
            with patch.object(plots, '_savefig', side_effect=lambda f, *a, **kw: figures.append(f)):
                plots.plot_sigma_control_psd_curves(root)
            ax = figures[0].axes[0]
            labels = '\n'.join(ax.get_legend_handles_labels()[1])
            annotation = '\n'.join(t.get_text() for t in ax.texts)
            self.assertIn('Slope-fit band', labels)
            self.assertNotIn('control (late)', labels)
            self.assertIn('mode: global', annotation)
            self.assertIn('Initial state: schedule', annotation)
            self.assertNotIn('frac:', annotation)
            self.assertLess(ax.get_ylim()[0], 1e-12)
            self.assertGreater(ax.get_ylim()[1], 1e7)

    def test_plot_only_does_not_recompute_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_pilot(root)
            cfg = OmegaConf.create({'full_gen_eval': {'sigma_star_grid': [1.]},
                                    'paths': {'sample_dir': str(root)}})
            with patch.object(evaluation, 'evaluate_sigma_control') as compute, \
                 patch.object(evaluation, 'get_model_string', return_value='model'), \
                 patch.object(evaluation, 'plot_sigma_control'), \
                 patch.object(evaluation, 'plot_sigma_control_psd_curves'), \
                 patch.object(evaluation, 'plot_sigma_control_examples_grid'):
                evaluation.plot_saved_sigma_control(cfg, root)
            compute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
