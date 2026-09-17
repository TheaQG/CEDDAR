"""Check run separation, failure propagation and source-labelled exports without real inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from repro.check_sigma_initialization import check_configs, check_generation, export_tables

ROOT = Path(__file__).resolve().parents[1]


class ComparisonDriverTests(unittest.TestCase):
    def prepare(self, root):
        checkpoint = root / 'checkpoint.pth.tar'
        checkpoint.write_bytes(b'prepare must not load this checkpoint')
        env = {**os.environ, 'DATA_DIR': str(root), 'STATS_LOAD_DIR': str(root),
               'PUBLISHED_CHECKPOINT': str(checkpoint), 'PYTHON': sys.executable,
               'COMPARISON_DIR': str(root / 'comparison'), 'CPU_THREADS': '1',
               'MAX_DATES': '2', 'ENSEMBLE_SIZE': '2', 'SIGMA_SEED': '504'}
        env.pop('SIGMA_CONFIG', None)
        result = subprocess.run(['bash', str(ROOT / 'repro/run_sigma_initialization_comparison.sh'),
                                 '--prepare-only'], env=env, cwd=root, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        roots = [root / 'comparison' / label for label in ('legacy', 'matched', 'late_ramp')]
        return roots, env

    def test_prepare_no_inference_and_reject_collisions_or_wrong_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots, env = self.prepare(Path(tmp).resolve())
            configs = check_configs(roots)
            self.assertEqual(configs[0]['full_gen_eval']['sigma_control']['sigma_star_initial_state'], 'legacy_sigma_max')
            self.assertEqual(configs[1]['full_gen_eval']['sigma_control']['sigma_star_initial_state'], 'schedule')
            self.assertEqual(len(check_configs(roots[:2])), 2)  # Existing two-run checks still work.
            late = configs[2]['full_gen_eval']['sigma_control']
            self.assertEqual(late['sigma_star_mode'], 'late_ramp')
            self.assertEqual(late['sigma_star_initial_state'], 'schedule')
            self.assertEqual((late['ramp_start_frac'], late['ramp_end_frac']), (0.60, 0.85))
            self.assertTrue(all(not (root / 'samples/generation').exists() for root in roots))
            late['sigma_star_mode'] = 'global'
            (roots[2] / 'resolved_config.yaml').write_text(yaml.safe_dump(configs[2]))
            with self.assertRaisesRegex(ValueError, 'expected sigma_star_mode=late_ramp'):
                check_configs(roots)
            late['sigma_star_mode'] = 'late_ramp'
            (roots[2] / 'resolved_config.yaml').write_text(yaml.safe_dump(configs[2]))
            result = subprocess.run(['bash', str(ROOT / 'repro/run_sigma_initialization_comparison.sh')],
                                     env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            configs[1]['full_gen_eval']['sigma_control']['sigma_star_initial_state'] = 'legacy_sigma_max'
            (roots[1] / 'resolved_config.yaml').write_text(yaml.safe_dump(configs[1]))
            with self.assertRaisesRegex(ValueError, 'expected sigma_star_initial_state=schedule'):
                check_configs(roots)

    def test_generation_check_compares_actual_baseline_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp) / label for label in ('legacy', 'matched', 'late_ramp')]
            configs = [{'full_gen_eval': {'max_dates': 1, 'sigma_star_grid': [.95, 1., 1.05]}}] * 3
            records = []
            for root in roots:
                base = root / 'samples/generation/model/sigma_star=1.00'
                (base / 'meta/noise').mkdir(parents=True)
                (base / 'meta/noise/20160101.json').write_text('{}')
                (base / 'ensembles_phys').mkdir()
                np.savez(base / 'ensembles_phys/20160101.npz', ens=np.ones((2, 1, 4, 4)))
                manifest = base / 'meta/observed.yaml'
                manifest.write_text(yaml.safe_dump({'checkpoint': {'sha256': 'same'}}))
                records.append([{'manifest': str(manifest)}])
            with patch('repro.check_sigma_initialization.check_runs', return_value={'dates': 1, 'variants': 9}), \
                 patch('repro.check_sigma_initialization.sigma_generation_metadata', side_effect=records * 2):
                self.assertTrue(check_generation(roots, configs)['baseline_samples_equal'])
                np.savez(base / 'ensembles_phys/20160101.npz', ens=np.zeros((2, 1, 4, 4)))
                with self.assertRaisesRegex(ValueError, 'physical samples differ'):
                    check_generation(roots, configs)

    def test_named_exports_preserve_source_bytes_and_nan_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = [root / label for label in ('legacy', 'matched', 'late_ramp')]
            configs = [{'full_gen_eval': {'sigma_star_grid': [.95, 1., 1.05]}}] * 3
            for i, run in enumerate(roots):
                tables = run / 'evaluation/model/prcp/sigma_control/tables'
                tables.mkdir(parents=True)
                (tables / 'metrics_by_sigma.csv').write_text(
                    f'date,sigma_star,crps\n20160101,0.95,{i+1}\n20160101,1,nan\n20160101,1.05,2\n')
            output = root / 'export'
            export_tables(roots, configs, output, {'dates': 1, 'date_list': ['20160101']})
            sources = json.loads((output / 'sources.json').read_text())['tables']
            self.assertNotEqual(sources['legacy']['sha256'], sources['matched']['sha256'])
            for label in ('legacy', 'matched', 'late_ramp'):
                self.assertEqual((output / f'{label}_metrics_by_sigma.csv').read_bytes(),
                                 Path(sources[label]['path']).read_bytes())
            late_table = Path(sources['late_ramp']['path'])
            late_table.write_text(late_table.read_text().replace('20160101,1,nan', '20160101,1,5'))
            with self.assertRaisesRegex(ValueError, 'baseline metric differs'):
                export_tables(roots, configs, root / 'bad_export',
                              {'dates': 1, 'date_list': ['20160101']})

    def test_failed_prepare_stops_driver_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            roots, env = self.prepare(root)
            env['COMPARISON_DIR'] = str(root / 'bad')
            env['SIGMA_CONFIG'] = str(root / 'missing.yaml')
            result = subprocess.run(['bash', str(ROOT / 'repro/run_sigma_initialization_comparison.sh')],
                                     env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'bad/generate_legacy.log').exists())
            self.assertTrue((root / 'bad/prepare_legacy.log').exists())


if __name__ == '__main__':
    unittest.main()
