"""Regression checks for the repro launchers; no real data or checkpoint needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_real_wrapper_passes_explicit_artifacts_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / 'capture-python'
            captured = root / 'args.json'
            python.write_text(f'#!{sys.executable}\nimport json, sys\n'
                              f'from pathlib import Path\nPath({str(captured)!r}).write_text(json.dumps(sys.argv[1:]))\n'
                              'sys.exit(17)\n')
            python.chmod(0o755)
            env = {**os.environ, 'PYTHON': str(python), 'DATA_DIR': str(root / 'data with spaces'),
                   'PUBLISHED_CHECKPOINT': str(root / 'arbitrary weight filename.pth.tar'),
                   'STATS_LOAD_DIR': str(root / 'stats'), 'DEVICE': 'cpu'}
            result = subprocess.run(['bash', str(ROOT / 'repro/02_real_artifact_smoke/run_real_artifact_smoke.sh')],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 17)
            self.assertNotIn('[PASS]', result.stdout)
            args = captured.read_text()
            args = json.loads(args)
            self.assertEqual(args[:2], ['-m', 'repro.smoke'])
            self.assertIn('--require-real', args)
            for flag, key in [('--checkpoint', 'PUBLISHED_CHECKPOINT'), ('--data-root', 'DATA_DIR'),
                              ('--stats-root', 'STATS_LOAD_DIR')]:
                self.assertEqual(args[args.index(flag) + 1], env[key])

    def test_real_mode_cannot_fall_back_to_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'must-not-exist'
            result = subprocess.run([sys.executable, '-m', 'repro.smoke', '--require-real', '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('--require-real needs', result.stderr)
            self.assertFalse(output.exists())

    def test_unimplemented_level_does_not_pass(self):
        result = subprocess.run(['bash', str(ROOT / 'repro/03_end_to_end_smoke/run_end_to_end_smoke.sh')],
                                cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('not implemented', result.stderr)

    def test_missing_checkpoint_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'must-not-exist'
            result = subprocess.run([sys.executable, '-m', 'repro.smoke', '--require-real',
                                     '--data-root', tmp, '--stats-root', tmp,
                                     '--checkpoint', str(Path(tmp) / 'missing.pth.tar'), '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Missing file:', result.stderr)
            self.assertFalse(output.exists())

    def test_renamed_workflow_references_exist(self):
        for name in ('run_reduced_local.sh', 'run_reduced_lumi.sh'):
            source = (ROOT / 'repro/04_reduced_run' / name).read_text()
            self.assertNotIn('02_reduced_run', source)
            self.assertIn('repro/04_reduced_run/reduced_run_config.yaml', source)


if __name__ == '__main__':
    unittest.main()
