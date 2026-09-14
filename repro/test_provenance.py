"""Metadata must describe the call, preserve RNG state, and never overwrite."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from sbgm.provenance import checkpoint_info, effective_sampler_settings, git_info, write_provenance
from sbgm.score_sampling import edm_sampler


class ProvenanceTests(unittest.TestCase):
    def test_effective_defaults_and_explicit_overrides(self):
        kwargs = dict(score_model=None, batch_size=2, num_steps=2, device="cpu", img_size=32, sigma_star=0.95)
        actual = effective_sampler_settings(edm_sampler, kwargs)
        self.assertEqual(actual['sigma_star_mode'], 'global')
        self.assertEqual(actual['sigma_star'], 0.95)
        self.assertNotIn('score_model', actual)
        kwargs['sigma_star_mode'] = 'late_ramp'
        self.assertEqual(effective_sampler_settings(edm_sampler, kwargs)['sigma_star_mode'], 'late_ramp')

    def test_manifest_hash_rng_and_unique_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'weights'
            checkpoint.write_bytes(b'known checkpoint bytes')
            info = checkpoint_info(checkpoint)
            self.assertEqual(info['sha256'], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            state = torch.get_rng_state().clone()
            cfg = {'paths': {'data_dir': '/example/data'}, 'full_gen_eval': {'split': 'test', 'seed': 504}}
            paths = [write_provenance(tmp, cfg, stage='test', device='cpu', checkpoint=info) for _ in range(2)]
            self.assertNotEqual(*paths)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            manifest = yaml.safe_load(paths[0].read_text())
            self.assertEqual(manifest['config'], cfg)
            self.assertEqual(manifest['checkpoint'], info)
            self.assertEqual(manifest['git'], git_info())
            self.assertEqual(manifest['torch']['device'], 'cpu')


if __name__ == '__main__':
    unittest.main()
