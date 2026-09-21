import unittest
import numpy as np

from revision_evaluation.object_metrics import object_metrics, equal_area_mask
from revision_evaluation.sal_analysis import domain_diameter, field_sal, sal_components


class ObjectTests(unittest.TestCase):
    def test_components_and_sal_scaled_volume(self):
        field = np.zeros((6, 6))
        field[0, 0] = field[1, 1] = field[5, 5] = 2.
        valid = np.ones_like(field, dtype=bool)
        result = object_metrics(field, valid, field >= 1)
        self.assertEqual(result['n_objects'], 2)  # Diagonal pair is connected.
        self.assertEqual(result['wet_pixel_count'], 3)
        self.assertEqual(result['mean_object_area'], 1.5)
        self.assertEqual(result['median_object_area'], 1.5)
        self.assertAlmostEqual(result['largest_object_fraction'], 2/3)
        valid[1, 1] = False
        self.assertEqual(object_metrics(field, valid, field >= 1)['wet_pixel_count'], 2)
        valid[:] = True
        ref, pred = np.zeros((6, 6)), np.zeros((6, 6))
        ref[2:4, 2:4] = 2.
        pred[1:5, 1:5] = 2.
        d = domain_diameter(valid)
        reference = field_sal(ref, valid, 1)
        sal = sal_components(reference, field_sal(pred, valid, 1), d)
        self.assertAlmostEqual(d, np.sqrt(50))
        self.assertAlmostEqual(sal['S'], 1.2)  # Areas 4 vs 16; mass-fraction proxy would give zero.
        self.assertAlmostEqual(sal['A'], 1.2)
        self.assertAlmostEqual(sal['L'], 0)
        sal = sal_components(reference, field_sal(2*ref, valid, 1), d)
        self.assertAlmostEqual(sal['S'], 0)
        self.assertAlmostEqual(sal['A'], 2/3)
        self.assertAlmostEqual(sal['L'], 0)
        shifted = np.roll(ref, 1, axis=1)
        self.assertAlmostEqual(sal_components(reference, field_sal(shifted, valid, 1), d)['L'], 1/d)

    def test_equal_area_and_predictable_ties(self):
        obs = np.array([[0., 0., 2., 3.]])
        field = np.array([[10., 20., 30., 40.]])
        valid = np.ones_like(obs, dtype=bool)
        mask, info = equal_area_mask(field, obs, valid, 1)
        self.assertEqual(info['target_wet_fraction'], .5)
        self.assertEqual(info['effective_threshold'], 25.)
        self.assertEqual(info['achieved_wet_fraction'], .5)
        np.testing.assert_array_equal(mask, [[False, False, True, True]])
        mask, info = equal_area_mask(np.zeros_like(field), obs, valid, 1)
        self.assertTrue(mask.all())  # Inclusive zero ties, no arbitrary pixel selection.
        self.assertEqual(info['wet_fraction_difference'], .5)
        mask, info = equal_area_mask(field, obs, valid, 10)
        self.assertFalse(mask.any())
        self.assertTrue(np.isnan(info['effective_threshold']))
        mask, info = equal_area_mask(field, np.ones_like(obs), valid, 1)
        self.assertTrue(mask.all())

    def test_empty_events_and_undefined_components(self):
        field = np.zeros((3, 3))
        valid = np.ones_like(field, dtype=bool)
        result = object_metrics(field, valid, field >= 1)
        self.assertEqual(result['n_objects'], 0)
        self.assertEqual(result['wet_pixel_count'], 0)
        self.assertEqual(result['wet_fraction'], 0)
        for key in ('mean_object_area', 'median_object_area', 'largest_object_fraction',
                    'mean_exceedance_intensity', 'max_exceedance_intensity'):
            self.assertTrue(np.isnan(result[key]))
        empty = field_sal(field, valid, 1)
        observed = field_sal(np.ones_like(field)*2, valid, 1)
        sal = sal_components(observed, empty, domain_diameter(valid))
        self.assertEqual(sal['A'], -2)
        self.assertTrue(np.isnan(sal['S']) and np.isnan(sal['L']))
        self.assertTrue(np.isnan(sal_components(empty, empty, domain_diameter(valid))['A']))
        self.assertTrue(np.isnan(object_metrics(field, valid & False, field >= 1)['n_objects']))


if __name__ == '__main__':
    unittest.main()
