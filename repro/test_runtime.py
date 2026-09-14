"""Path safety checks; no model dependencies or dataset required."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sbgm.runtime import SOURCE_ROOT, external_output, setup_environment, validate_output_paths


class RuntimeTests(unittest.TestCase):
    def test_external_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            os.environ.update(CEDDAR_RUNS=tmp, SAMPLE_DIR=str(Path(tmp, "custom").resolve()))
            setup_environment()
            self.assertEqual(os.environ["SAMPLE_DIR"], str(Path(tmp, "custom").resolve()))
            self.assertTrue(Path(os.environ["TMPDIR"]).is_dir())
            self.assertEqual(os.environ["EVAL_DIR"], str(Path(tmp, "custom/evaluation").resolve()))

    def test_repository_and_symlink_are_rejected(self):
        with self.assertRaises(ValueError):
            external_output(SOURCE_ROOT / "samples")
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "source"
            link.symlink_to(SOURCE_ROOT, target_is_directory=True)
            with self.assertRaises(ValueError):
                external_output(link / "samples")

    def test_resolved_config_is_checked(self):
        with self.assertRaises(ValueError):
            validate_output_paths({"paths": {"sample_dir": str(SOURCE_ROOT / "outputs")}})


if __name__ == "__main__":
    unittest.main()
