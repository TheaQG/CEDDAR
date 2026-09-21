import unittest

import numpy as np

from revision_evaluation.dry_bias_decomposition import (
    field_components,
    summarize_components,
    summarize_wet_values,
)


class DryBiasTests(unittest.TestCase):
    def test_exact_decomposition_and_quantiles(self):
        field = np.array([
            [0.0, 0.5],
            [1.0, 4.0],
        ])

        valid = np.ones_like(field, dtype=bool)

        counts, wet = field_components(field, valid, threshold=1.0)
        summary = summarize_components(counts)
        quantiles = summarize_wet_values([wet])

        self.assertEqual(counts["n_pixel_days"], 4,)
        self.assertEqual(counts["n_wet_pixel_days"], 2,)
        self.assertAlmostEqual(summary["wet_frequency"], 0.5,)
        self.assertAlmostEqual(summary["conditional_mean_wet"], 2.5,)
        self.assertAlmostEqual(summary["mean_precip"], 1.375,)
        self.assertAlmostEqual(summary["wet_contribution"], 1.25,)
        self.assertAlmostEqual(summary["dry_contribution"], 0.125,)
        self.assertAlmostEqual(summary["reconstruction_error"], 0.0,)
        self.assertAlmostEqual(quantiles["p50"], 2.5,)


    def test_each_method_has_its_own_wet_mask(self):
        valid = np.ones(3, dtype=bool)

        obs = np.array(
            [0.0, 2.0, 4.0]
        )
        pred = np.array(
            [2.0, 0.0, 8.0]
        )

        _, wet_obs = field_components(obs, valid, threshold=1.0)
        _, wet_pred = field_components(pred, valid, threshold=1.0)

        self.assertTrue(np.array_equal(wet_obs, [2.0,4.0]))
        self.assertTrue(np.array_equal(wet_pred, [2.0,8.0]))


    def test_no_wet_cases_return_nan_statistics(self):
        field = np.array(
            [0.0, 0.2, 0.9]
        )
        valid = np.ones(3, dtype=bool)

        counts, wet = field_components(field, valid, threshold=1.0)

        summary = summarize_components(counts)
        quantiles = summarize_wet_values([wet])

        self.assertEqual(counts["n_wet_pixel_days"], 0)
        self.assertTrue(np.isnan(summary["conditional_mean_wet"]))
        self.assertTrue(np.isnan(quantiles["p50"]))
        self.assertTrue(np.isnan(quantiles["p90"]))
        self.assertTrue(np.isnan(quantiles["p99"]))


if __name__ == "__main__":
    unittest.main()