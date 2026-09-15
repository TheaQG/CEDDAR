"""Small CPU checks of sampler mechanics; no data, checkpoint or training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

from sbgm.score_sampling import edm_sampler
from sbgm.sigma_control import build_edm_schedule, sigma_star_kwargs
from sbgm.provenance import effective_sampler_settings, sigma_generation_metadata, write_provenance


class ToyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.calls = []

    def forward(self, x, sigma, **kwargs):
        self.calls.append((x.clone(), sigma.clone()))
        return self.weight * x + 0.05 * sigma[:, None, None, None]


class SigmaStarTests(unittest.TestCase):
    def test_frozen_global_reference(self):
        # Frozen v1.0.2 a349ffce, CPU float32, seed=504, ToyDenoiser above,
        # M=2, N=4, H=W=2, S_churn=2, window=[40,80]; no conditioning.
        expected = {
            1.0: [684.5534668, -70.2576294, 754.0041504, -566.2095947,
                  -535.9605713, 687.8698730, -302.6444092, -546.4719238],
            0.9: [689.8010254, -103.3912354, 770.4389648, -507.1842041,
                  -470.2545166, 643.1711426, -283.1344604, -500.3607178],
        }
        for alpha, reference in expected.items():
            torch.manual_seed(504)
            result = edm_sampler(ToyDenoiser(), 2, 4, 'cpu', 2, sigma_star=alpha,
                                 sigma_star_initial_state='legacy_sigma_max',
                                 S_churn=2, S_min=40, S_max=80)
            torch.testing.assert_close(result.flatten(), torch.tensor(reference), rtol=2e-6, atol=1e-4)
        torch.manual_seed(504)
        revised = edm_sampler(ToyDenoiser(), 2, 4, 'cpu', 2, S_churn=2, S_min=40, S_max=80)
        torch.testing.assert_close(revised.flatten(), torch.tensor(expected[1.0]), rtol=2e-6, atol=1e-4)

    def test_initial_state_network_levels_and_terminal_euler(self):
        for mode in ('global', 'late_ramp'):
            for alpha in (0.9, 1.0, 1.1):
                with self.subTest(mode=mode, alpha=alpha):
                    model = ToyDenoiser()
                    torch.manual_seed(12)
                    noise = torch.randn(2, 1, 2, 2)
                    torch.manual_seed(12)
                    result = edm_sampler(model, 2, 8, 'cpu', 2, sigma_star=alpha, sigma_star_mode=mode)
                    sigmas, details = build_edm_schedule(8, sigma_star=alpha, sigma_star_mode=mode)
                    self.assertEqual(len(model.calls), 15)  # 2N-1, no call at sigma=0
                    torch.testing.assert_close(model.calls[0][0], noise * details['initial_std'])
                    torch.testing.assert_close(model.calls[0][1], sigmas[0].expand(2))
                    x_last, s_last = model.calls[-1]
                    torch.testing.assert_close(result, 0.25*x_last + 0.05*s_last[:, None, None, None])
                    self.assertEqual(float(sigmas[-1]), 0)

    def test_ramp_indices_and_early_churn(self):
        for alpha in (0.8, 0.95, 1.0, 1.25):
            _, ramp = build_edm_schedule(56, sigma_star=alpha, sigma_star_mode='late_ramp',
                                         S_churn=2, S_min=40, S_max=80)
            _, base = build_edm_schedule(56, S_churn=2, S_min=40, S_max=80)
            self.assertEqual((ramp['ramp_start_index'], ramp['ramp_end_index']), (33, 47))
            self.assertEqual(ramp['scale_factors'][:34], [1.0]*34)
            self.assertAlmostEqual(ramp['scale_factors'][47], alpha, places=6)
            self.assertEqual(ramp['churn_noise_std'], base['churn_noise_std'])
            self.assertEqual(ramp['initial_std'], 80)
        _, threshold = build_edm_schedule(56, sigma_star_mode='late_ramp',
                                           ramp_start_sigma=1.0, ramp_end_sigma=0.1)
        self.assertLess(threshold['ramp_start_index'], threshold['ramp_end_index'])

    def test_actual_churn_matches_manifest(self):
        model = ToyDenoiser()
        kwargs = dict(score_model=model, batch_size=2, num_steps=8, device='cpu', img_size=2,
                      sigma_star=0.9, S_churn=2, S_min=0, S_max=float('inf'))
        state = torch.get_rng_state().clone()
        manifest = effective_sampler_settings(edm_sampler, kwargs)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with patch('torch.randn', return_value=torch.zeros(2, 1, 2, 2)), \
             patch('torch.randn_like', side_effect=lambda x: torch.ones_like(x)):
            edm_sampler(**kwargs)
        self.assertAlmostEqual(float(model.calls[0][0][0, 0, 0, 0]), manifest['schedule']['churn_noise_std'][0], places=5)
        self.assertAlmostEqual(float(model.calls[0][1][0]), manifest['schedule']['sigma_hat'][0], places=5)

    def test_invalid_settings_fail_before_rng_or_network(self):
        cases = [dict(sigma_star=0), dict(sigma_star=float('nan')), dict(sigma_star_mode='typo'),
                 dict(sigma_min=80), dict(rho=0), dict(S_noise=-1),
                 dict(sigma_star_initial_state='typo'),
                 dict(sigma_star_mode='late_ramp', ramp_start_frac=0.9, ramp_end_frac=0.2),
                 dict(sigma_star_mode='late_ramp', ramp_start_sigma=1),
                 dict(sigma_star_mode='late_ramp', ramp_start_sigma=0.1, ramp_end_sigma=1),
                 dict(sigma_star_mode='late_ramp', ramp_start_frac=0.60, ramp_end_frac=0.61),
                 dict(sigma_star_mode='late_ramp', sigma_star=100000)]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                model = ToyDenoiser()
                before = torch.get_rng_state().clone()
                with self.assertRaises(ValueError):
                    edm_sampler(model, 1, 8, 'cpu', 2, **kwargs)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertFalse(model.calls)

    def test_runner_forwards_all_controls(self):
        from sbgm.generate.generation import GenerationConfig, GenerationRunner
        root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(root / 'repro/01_small_test/small_test_config.yaml')
        cfg.edm.update(sigma_star=0.95, sigma_star_mode='late_ramp', sampling_steps=8,
                       ramp_start_frac=0.2, ramp_end_frac=0.8,
                       sigma_star_initial_state='legacy_sigma_max')
        cfg.paths.stats_load_dir = 'unused'
        batch = {'date': ['20190101'], 'prcp_hr': torch.ones(1, 1, 4, 4),
                 'prcp_lr': torch.ones(1, 1, 4, 4), 'lsm_hr': torch.ones(1, 1, 4, 4)}
        with tempfile.TemporaryDirectory() as tmp, \
             patch('sbgm.generate.generation._build_back_transforms', return_value={}), \
             patch('sbgm.generate.generation.edm_sampler', side_effect=RuntimeError('sampler reached')) as sampler:
            runner = GenerationRunner(ToyDenoiser(), cfg, 'cpu', Path(tmp), GenerationConfig(tmp))
            with self.assertRaisesRegex(RuntimeError, 'sampler reached'):
                runner.run([batch], save=False)
            for key, value in sigma_star_kwargs(cfg.edm).items():
                self.assertEqual(sampler.call_args.kwargs[key], value)

    def test_grid_overrides_and_collision_guard(self):
        import importlib
        module = importlib.import_module('sbgm.generate.generation_sigma_grid_main')
        cfg = OmegaConf.create({
            'paths': {}, 'training': {'device': 'cpu'}, 'data_handling': {},
            'edm': {'sampling_steps': 8, 'sigma_star_mode': 'global'},
            'full_gen_eval': {'ensemble_size': 2, 'max_dates': 1, 'split': 'valid',
                             'sigma_star_grid': [0.9, 1.1],
                             'sigma_control': {'sigma_star_mode': 'late_ramp',
                                               'sigma_star_initial_state': 'legacy_sigma_max'}}})
        received = []
        class Runner:
            def __init__(self, **kwargs):
                received.append(dict(kwargs['cfg'].edm))
            def run(self, loader):
                pass
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(module, '_resolve_base_out_dir', return_value=Path(tmp)), \
             patch.object(module, 'get_model', return_value=(ToyDenoiser(), tmp, 'unused')) as get_model, \
             patch.object(module, 'checkpoint_info', return_value={}), \
             patch.object(module.torch, 'load', return_value={'network_params': {'weight': torch.tensor(0.25)}}), \
             patch.object(module, 'get_final_gen_dataloader', return_value=[]), \
             patch.object(module, 'GenerationRunner', Runner):
            module.generation_sigma_grid_main(cfg)
            self.assertEqual([c['sigma_star'] for c in received], [0.9, 1.1])
            self.assertTrue(all(c['sigma_star_mode'] == 'late_ramp' for c in received))
            self.assertTrue(all(c['sigma_star_initial_state'] == 'legacy_sigma_max' for c in received))
            get_model.reset_mock()
            with self.assertRaises(FileExistsError):
                module.generation_sigma_grid_main(cfg)
            get_model.assert_not_called()
            cfg.full_gen_eval.sigma_star_grid = [0.901, 0.902]
            with self.assertRaisesRegex(ValueError, 'two-decimal'):
                module.generation_sigma_grid_main(cfg)
            get_model.assert_not_called()

    def test_strict_metrics_reject_missing_arrays(self):
        import importlib
        module = importlib.import_module('sbgm.evaluate.evaluate_prcp.eval_sigma_star.metrics_sigma_control')
        cfg = OmegaConf.create({'full_gen_eval': {'sigma_control': {'require_generation_manifest': True}}})
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, 'EvalDataResolver') as resolver:
            root = Path(tmp)
            (root / 'sigma_star=1.00').mkdir()
            resolver.return_value.list_dates.return_value = ['20190101']
            resolver.return_value.load_obs.return_value = None
            with self.assertRaisesRegex(ValueError, 'Missing physical arrays'):
                module.evaluate_sigma_control(cfg, [1.0], root, root / 'eval')

    def test_evaluation_requires_observed_matching_sampler(self):
        cfg = {'edm': {'sampling_steps': 8}, 'full_gen_eval': {'sigma_control': {'require_generation_manifest': True}}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                sigma_generation_metadata(tmp, [1.0], cfg)
            sampler = effective_sampler_settings(edm_sampler, dict(score_model=None, batch_size=2, num_steps=8, device='cpu', img_size=2))
            write_provenance(Path(tmp) / 'sigma_star=1.00/meta', cfg, stage='generation', sampler=sampler)
            with self.assertRaisesRegex(ValueError, 'did not complete'):
                sigma_generation_metadata(tmp, [1.0], cfg)
            (Path(tmp) / 'sigma_star=1.00/meta/manifest.json').write_text('{"n_days": 1}')
            self.assertEqual(sigma_generation_metadata(tmp, [1.0], cfg)[0]['sampler'], sampler)
            cfg['edm']['sigma_star_mode'] = 'late_ramp'
            with self.assertRaisesRegex(ValueError, 'sigma_star_mode'):
                sigma_generation_metadata(tmp, [1.0], cfg)


if __name__ == '__main__':
    unittest.main()
