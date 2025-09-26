"""
This is the unittest of the cokriging module.
"""

import unittest

import numpy as np

import gstools as gs
from gstools.cokriging import SCCK


class TestCokriging(unittest.TestCase):
    def setUp(self):
        self.cov_models = [gs.Gaussian, gs.Exponential, gs.Spherical]
        # test data
        self.data = np.array(
            [
                [0.3, 1.2, 0.5, 0.47],
                [1.9, 0.6, 1.0, 0.56],
                [1.1, 3.2, 1.5, 0.74],
                [3.3, 4.4, 2.0, 1.47],
                [4.7, 3.8, 2.5, 1.74],
            ]
        )
        # condition positions and values
        self.cond_pos = (self.data[:, 0], self.data[:, 1], self.data[:, 2])
        self.cond_val = self.data[:, 3]
        # test positions and secondary data
        self.pos = np.array([0.5, 1.5, 2.5, 3.5])
        self.sec_data = np.array([2.8, 2.2, 3.1, 2.9])

    def test_scck_basic(self):
        """Test basic SCCK functionality."""
        for Model in self.cov_models:
            model = Model(dim=1, var=2, len_scale=2)
            scck = SCCK(
                model,
                self.cond_pos[:1],
                self.cond_val,
                cross_corr=0.7,
                secondary_var=1.5,
            )

            # test field estimation (default returns field + variance)
            field, var = scck(self.pos, secondary_data=self.sec_data)
            self.assertEqual(field.shape, (4,))
            self.assertEqual(var.shape, (4,))

            # test field only
            field_only = scck(
                self.pos, secondary_data=self.sec_data, return_var=False)
            self.assertEqual(field_only.shape, (4,))

            # test field + variance
            field, var = scck(
                self.pos, secondary_data=self.sec_data, return_var=True)
            self.assertEqual(field.shape, (4,))
            self.assertEqual(var.shape, (4,))
            # variance should be positive
            self.assertTrue(np.all(var > 0))

    def test_scck_vs_simple_kriging(self):
        """Test SCCK reduces to Simple Kriging with zero cross-correlation."""
        model = gs.Exponential(dim=1, var=2, len_scale=2)

        # Simple Kriging with mean=0 (to match SCCK which uses unbiased=False)
        sk = gs.krige.Simple(model, self.cond_pos[:1], self.cond_val, mean=0.0)
        sk_field, sk_var = sk(self.pos, return_var=True)

        # SCCK with zero cross-correlation
        scck = SCCK(
            model,
            self.cond_pos[:1],
            self.cond_val,
            cross_corr=0.0,
            secondary_var=1.5,
        )
        scck_field, scck_var = scck(
            self.pos, secondary_data=self.sec_data, return_var=True)

        # should be identical (allowing small numerical differences)
        np.testing.assert_allclose(sk_field, scck_field, rtol=1e-10)
        np.testing.assert_allclose(sk_var, scck_var, rtol=1e-10)

    def test_variance_behavior(self):
        """Test SCCK variance behavior (MM1 can show inflation)."""
        model = gs.Exponential(dim=1, var=2, len_scale=2)

        # Simple Kriging with mean=0
        sk = gs.krige.Simple(model, self.cond_pos[:1], self.cond_val, mean=0.0)
        __, sk_var = sk(self.pos, return_var=True)

        # SCCK with moderate cross-correlation
        scck = SCCK(
            model,
            self.cond_pos[:1],
            self.cond_val,
            cross_corr=0.6,
            secondary_var=1.5,
        )
        __, scck_var = scck(
            self.pos, secondary_data=self.sec_data, return_var=True)

        # SCCK variance should be non-negative (MM1 can inflate variance)
        self.assertTrue(np.all(scck_var >= 0))
        # Variance should be finite
        self.assertTrue(np.all(np.isfinite(scck_var)))

    def test_theoretical_consistency(self):
        """Test MM1 theoretical formulas and consistency."""
        model = gs.Exponential(dim=1, var=2, len_scale=2)

        scck = SCCK(
            model,
            self.cond_pos[:1],
            self.cond_val,
            cross_corr=0.6,
            secondary_var=1.5,
        )

        # Test cross-covariance ratio computation
        k = scck._compute_k()
        expected_k = scck.cross_corr * \
            np.sqrt(model.sill * scck.secondary_var) / model.sill
        self.assertAlmostEqual(k, expected_k, places=10)

        # Test collocated weight computation
        test_variance = np.array([0.5, 1.0, 1.5])
        weights = scck._compute_collocated_weight(test_variance, k)

        # Weights should be finite
        self.assertTrue(np.all(np.isfinite(weights)))

        # Test MM1 variance formula consistency
        scck_var = scck._compute_scck_variance(test_variance, k)
        expected_var = test_variance * (1 - weights * k)
        expected_var = np.maximum(0.0, expected_var)

        np.testing.assert_allclose(scck_var, expected_var, rtol=1e-12)

    def test_numerical_stability(self):
        """Test numerical stability in edge cases."""
        model = gs.Exponential(dim=1, var=2, len_scale=2)

        # Test with very small cross-correlation
        scck_small = SCCK(
            model,
            self.cond_pos[:1],
            self.cond_val,
            cross_corr=1e-15,
            secondary_var=1.5,
        )
        field_small, var_small = scck_small(
            self.pos, secondary_data=self.sec_data, return_var=True)

        self.assertTrue(np.all(np.isfinite(field_small)))
        self.assertTrue(np.all(np.isfinite(var_small)))
        self.assertTrue(np.all(var_small >= 0))

        # Test with high cross-correlation
        scck_high = SCCK(
            model,
            self.cond_pos[:1],
            self.cond_val,
            cross_corr=0.99,
            secondary_var=model.sill,
        )
        field_high, var_high = scck_high(
            self.pos, secondary_data=self.sec_data, return_var=True)

        self.assertTrue(np.all(np.isfinite(field_high)))
        self.assertTrue(np.all(np.isfinite(var_high)))
        self.assertTrue(np.all(var_high >= 0))

    def test_input_validation(self):
        """Test input validation."""
        model = gs.Exponential(dim=1, var=2, len_scale=2)

        # invalid cross-correlation
        with self.assertRaises(ValueError):
            SCCK(model, self.cond_pos[:1], self.cond_val,
                 cross_corr=1.5, secondary_var=1.0)

        # invalid secondary variance
        with self.assertRaises(ValueError):
            SCCK(model, self.cond_pos[:1], self.cond_val,
                 cross_corr=0.5, secondary_var=-1.0)

        # missing secondary data
        scck = SCCK(model, self.cond_pos[:1], self.cond_val,
                    cross_corr=0.5, secondary_var=1.0)
        with self.assertRaises(ValueError):
            scck(self.pos)

    def test_edge_cases(self):
        """Test edge cases."""
        model = gs.Exponential(dim=1, var=2, len_scale=2)

        # perfect cross-correlation
        scck = SCCK(
            model,
            self.cond_pos[:1],
            self.cond_val,
            cross_corr=1.0,
            secondary_var=model.sill,
        )
        field, var = scck(
            self.pos, secondary_data=self.sec_data, return_var=True)
        self.assertTrue(np.all(var >= 0))

        # very small cross-correlation (should behave like zero)
        scck = SCCK(
            model,
            self.cond_pos[:1],
            self.cond_val,
            cross_corr=1e-16,
            secondary_var=1.5,
        )
        field, var = scck(
            self.pos, secondary_data=self.sec_data, return_var=True)
        self.assertTrue(np.all(var >= 0))


if __name__ == "__main__":
    unittest.main()
