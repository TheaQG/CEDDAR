"""Noise pairing survives branch, grid-order and global-RNG changes."""
from pathlib import Path
import json
import tempfile
import unittest

import torch
from omegaconf import OmegaConf
from unittest.mock import patch

from sbgm.sampling_noise import PROTOCOL, indexed_seed, paired_normal
from sbgm.score_sampling import edm_sampler
from sbgm.provenance import effective_sampler_settings, sigma_generation_metadata, write_provenance
from repro.test_sigma_star import ToyDenoiser
from repro.check_paired_noise import check_runs


class PairedNoiseTests(unittest.TestCase):
    def test_repeatable_independent_draws_do_not_advance_rng(self):
        before = torch.get_rng_state().clone()
        seed = indexed_seed(504, 'date:20160101')
        a = paired_normal((2, 1, 4, 4), seed=seed, stream='churn:3', device='cpu')
        paired_normal((2, 1, 4, 4), seed=seed, stream='churn:1', device='cpu')
        b = paired_normal((2, 1, 4, 4), seed=seed, stream='churn:3', device='cpu')
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertNotEqual(seed, indexed_seed(504, 'date:20160102'))
        self.assertNotEqual(seed, indexed_seed(505, 'date:20160101'))

    def test_actual_sampler_aligns_churn_despite_different_active_windows(self):
        kwargs = dict(score_model=ToyDenoiser(), batch_size=2, num_steps=8, device='cpu',
                      img_size=4, noise_seed=123, S_churn=2)
        a, b = {}, {}
        before = torch.get_rng_state().clone()
        edm_sampler(**kwargs, S_min=0, S_max=float('inf'), noise_audit=a)
        edm_sampler(**kwargs, S_min=0, S_max=40, noise_audit=b)
        self.assertNotEqual(set(a), set(b))
        self.assertTrue(any(k.startswith('churn:') for k in set(a) & set(b)))
        for stream in set(a) & set(b):
            self.assertEqual(a[stream], b[stream])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_grid_order_and_alpha_one_invariance(self):
        for mode in ('global', 'late_ramp'):
            out = {}
            for values in ([.95, 1, 1.05], [1.05, 1, .95]):
                for alpha in values:
                    key = (mode, alpha)
                    result = edm_sampler(ToyDenoiser(), 2, 8, 'cpu', 4, sigma_star=alpha,
                                         sigma_star_mode=mode, noise_seed=504, S_churn=2)
                    if key in out:
                        self.assertTrue(torch.equal(out[key], result))
                    out[key] = result
            if mode == 'global':
                baseline = out[(mode, 1)]
            else:
                self.assertTrue(torch.equal(baseline, out[(mode, 1)]))

    def test_runner_passes_date_seed_and_saves_actual_draws(self):
        from sbgm.generate.generation import GenerationConfig, GenerationRunner
        root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(root / 'repro/01_small_test/small_test_config.yaml')
        # Resolve otherwise unused paths without reading real artifacts.
        cfg.paths = {'stats_load_dir': 'unused', 'data_dir': 'unused'}
        cfg.experiment.date = 'test'
        cfg.diagnostics.histogram_path = 'unused'
        cfg.edm.sampling_steps = 8
        cfg.full_gen_eval = {'seed': 504, 'sigma_control': {'noise_mode': 'paired'}}
        batch = {'date': ['20160101'], 'prcp_hr': torch.ones(1, 1, 4, 4),
                 'prcp_lr': torch.ones(1, 1, 4, 4), 'lsm_hr': torch.ones(1, 1, 4, 4)}
        identity = lambda x: x
        with tempfile.TemporaryDirectory() as tmp, \
             patch('sbgm.generate.generation._build_back_transforms',
                   return_value={'generated': identity, 'prcp_hr': identity, 'prcp_lr': identity}):
            gen = GenerationConfig(tmp, ensemble_size=2, max_dates=1)
            runner = GenerationRunner(ToyDenoiser(), cfg, 'cpu', Path(tmp), gen)
            runner.run([batch])
            record = json.loads((Path(tmp) / 'meta/noise/20160101.json').read_text())
            self.assertEqual(record['noise_seed'], indexed_seed(504, 'date:20160101'))
            self.assertIn('initial', record['draws'])
            self.assertIn('cond_img', record['inputs_sha256'])
            self.assertEqual(record['protocol'], PROTOCOL)

    def test_provenance_omits_mutable_audit(self):
        settings = effective_sampler_settings(edm_sampler, dict(score_model=None, batch_size=2,
                 num_steps=8, img_size=4, device='cpu', noise_seed=10, noise_audit={}))
        self.assertEqual(settings['noise_seed'], 10)
        self.assertNotIn('noise_audit', settings)

    def test_evaluation_rejects_different_pairing_mode_or_seed(self):
        cfg = {'edm': {'sampling_steps': 8}, 'full_gen_eval': {
            'seed': 504, 'sigma_control': {'noise_mode': 'paired'}}}
        with tempfile.TemporaryDirectory() as tmp:
            settings = effective_sampler_settings(edm_sampler, dict(score_model=None,
                batch_size=2, num_steps=8, device='cpu', img_size=4, noise_seed=10))
            write_provenance(Path(tmp) / 'sigma_star=1.00/meta', cfg,
                             stage='generation', sampler=settings)
            self.assertEqual(sigma_generation_metadata(tmp, [1.], cfg)[0]['sampler'], settings)
            cfg['full_gen_eval']['seed'] = 505
            with self.assertRaisesRegex(ValueError, 'paired seed mismatch'):
                sigma_generation_metadata(tmp, [1.], cfg)
            cfg['full_gen_eval']['sigma_control']['noise_mode'] = 'sequential'
            with self.assertRaisesRegex(ValueError, 'noise_mode mismatch'):
                sigma_generation_metadata(tmp, [1.], cfg)

    def test_saved_hash_checker_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = dict(protocol=PROTOCOL, root_seed=504, noise_seed=10, device='cpu',
                          torch_version=str(torch.__version__), inputs_sha256={'cond_img': 'same'},
                          draws={'initial': {'standard_normal_sha256': 'same'}})
            for alpha in ('.95', '1.00'):
                p = root / f'samples/generation/model/sigma_star={alpha}/meta/noise'
                p.mkdir(parents=True)
                (p / '20160101.json').write_text(json.dumps(record))
            self.assertEqual(check_runs([root])['variants'], 2)
            record['inputs_sha256']['cond_img'] = 'different'
            (p / '20160101.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'inputs_sha256'):
                check_runs([root])


if __name__ == '__main__':
    unittest.main()
