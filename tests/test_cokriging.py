"""
This is the unittest of the cokriging module.
"""

import unittest

import numpy as np

import gstools as gs
from gstools.cokriging import SimpleCollocated, IntrinsicCollocated


class TestCokriging(unittest.TestCase):
    def setUp(self):
        self.cov_models = [gs.Gaussian, gs.Exponential, gs.Spherical]
        self.dims = range(1, 4)
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
        # secondary data at conditioning locations (5 values to match cond_val)
        self.sec_cond_data = np.array([1.8, 1.2, 2.1, 2.9, 2.4])
        # grids for structured testing
        self.x = np.linspace(0, 5, 51)
        self.y = np.linspace(0, 6, 61)
        self.z = np.linspace(0, 7, 71)
        self.grids = (self.x, self.y, self.z)

    def test_simple(self):
        """Test Simple Collocated across models and dimensions."""
        for Model in self.cov_models:
            for dim in self.dims:
                model = Model(dim=dim, var=2, len_scale=2)

                # secondary data
                if dim == 1:
                    sec_data = np.linspace(0.5, 2.0, 51)
                elif dim == 2:
                    sec_data = np.random.RandomState(42).rand(51, 61)
                else:
                    sec_data = np.random.RandomState(42).rand(51, 61, 71)

                scck = SimpleCollocated(
                    model,
                    self.cond_pos[:dim],
                    self.cond_val,
                    cross_corr=0.7,
                    secondary_var=1.5,
                )

                field, var = scck.structured(self.grids[:dim], secondary_data=sec_data)
                self.assertTrue(np.all(np.isfinite(field)))
                self.assertTrue(np.all(np.isfinite(var)))
                self.assertTrue(np.all(var >= -1e-6))

    def test_scck_vs_simple_kriging(self):
        """Test SCCK reduces to Simple Kriging with zero cross-correlation."""
        model = gs.Exponential(dim=1, var=2, len_scale=2)

        # Simple Kriging with mean=0 (to match SCCK which uses unbiased=False)
        sk = gs.krige.Simple(model, self.cond_pos[:1], self.cond_val, mean=0.0)
        sk_field, sk_var = sk(self.pos, return_var=True)

        # SCCK with zero cross-correlation
        scck = SimpleCollocated(
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

    def test_zero_cross_correlation(self):
        """Test zero cross-correlation equals Simple Kriging."""
        model = gs.Gaussian(dim=1, var=2, len_scale=2)
        pos = np.array([2.5])
        sec_data = np.array([999.0])

        sk = gs.krige.Simple(model, self.cond_pos[:1], self.cond_val, mean=0.0)
        sk_field, sk_var = sk(pos, return_var=True)

        # SCCK
        scck = SimpleCollocated(
            model, self.cond_pos[:1], self.cond_val,
            cross_corr=0.0, secondary_var=1.5,
            mean=0.0, secondary_mean=0.0
        )
        scck_field, scck_var = scck(pos, secondary_data=sec_data, return_var=True)
        self.assertAlmostEqual(scck_field[0], sk_field[0], places=2)
        self.assertAlmostEqual(scck_var[0], sk_var[0], places=2)

        # ICCK
        icck = IntrinsicCollocated(
            model, self.cond_pos[:1], self.cond_val,
            self.cond_pos[:1], self.sec_cond_data,
            cross_corr=0.0, secondary_var=1.5,
            mean=0.0, secondary_mean=0.0
        )
        icck_field, icck_var = icck(pos, secondary_data=sec_data, return_var=True)
        self.assertAlmostEqual(icck_field[0], sk_field[0], places=2)
        self.assertAlmostEqual(icck_var[0], sk_var[0], places=2)

    def test_perfect_correlation(self):
        """Test perfect correlation edge case."""
        model = gs.Gaussian(dim=1, var=2, len_scale=2)
        pos = np.array([2.0])

        icck = IntrinsicCollocated(
            model, self.cond_pos[:1], self.cond_val,
            self.cond_pos[:1], self.sec_cond_data,
            cross_corr=1.0, secondary_var=2.0,
            mean=0.0, secondary_mean=0.0
        )
        _, icck_var = icck(pos, secondary_data=np.array([1.0]), return_var=True)

        self.assertAlmostEqual(icck_var[0], 0.0, places=5)

    def test_intrinsic(self):
        """Test Intrinsic Collocated across models and dimensions."""
        for Model in self.cov_models:
            for dim in self.dims:
                model = Model(dim=dim, var=2, len_scale=2)

                # secondary data
                if dim == 1:
                    sec_data = np.linspace(0.5, 2.0, 51)
                elif dim == 2:
                    sec_data = np.random.RandomState(42).rand(51, 61)
                else:
                    sec_data = np.random.RandomState(42).rand(51, 61, 71)

                icck = IntrinsicCollocated(
                    model,
                    self.cond_pos[:dim],
                    self.cond_val,
                    self.cond_pos[:dim],
                    self.sec_cond_data,
                    cross_corr=0.7,
                    secondary_var=1.5,
                )

                field, var = icck.structured(self.grids[:dim], secondary_data=sec_data)
                self.assertTrue(np.all(np.isfinite(field)))
                self.assertTrue(np.all(np.isfinite(var)))
                self.assertTrue(np.all(var >= -1e-6))



    def test_icck_variance_formula(self):
        """Test ICCK variance: var = (1 - rho^2) * var_sk."""
        model = gs.Gaussian(dim=1, var=2, len_scale=3)
        pos = np.array([2.0])

        for cross_corr in [0.3, 0.6, 0.9]:
            sk = gs.krige.Simple(model, self.cond_pos[:1], self.cond_val, mean=0.0)
            _, sk_var = sk(pos, return_var=True)

            icck = IntrinsicCollocated(
                model, self.cond_pos[:1], self.cond_val,
                self.cond_pos[:1], self.sec_cond_data,
                cross_corr=cross_corr, secondary_var=1.5,
                mean=0.0, secondary_mean=0.0
            )
            _, icck_var = icck(pos, secondary_data=np.array([1.0]), return_var=True)

            expected = (1 - cross_corr**2) * sk_var[0]
            self.assertAlmostEqual(icck_var[0], expected, places=2)

    def test_raise(self):
        """Test error handling."""
        model = gs.Exponential(dim=1, var=2, len_scale=2)

        # SCCK: invalid cross-correlation
        with self.assertRaises(ValueError):
            SimpleCollocated(model, self.cond_pos[:1], self.cond_val,
                 cross_corr=1.5, secondary_var=1.0)

        # SCCK: invalid secondary variance
        with self.assertRaises(ValueError):
            SimpleCollocated(model, self.cond_pos[:1], self.cond_val,
                 cross_corr=0.5, secondary_var=-1.0)

        # SCCK: missing secondary data
        scck = SimpleCollocated(model, self.cond_pos[:1], self.cond_val,
                    cross_corr=0.5, secondary_var=1.0)
        with self.assertRaises(ValueError):
            scck(self.pos)

        # ICCK: invalid cross-correlation
        with self.assertRaises(ValueError):
            IntrinsicCollocated(model, self.cond_pos[:1], self.cond_val,
                 self.cond_pos[:1], self.sec_cond_data,
                 cross_corr=1.5, secondary_var=1.0)

        # ICCK: invalid secondary variance
        with self.assertRaises(ValueError):
            IntrinsicCollocated(model, self.cond_pos[:1], self.cond_val,
                 self.cond_pos[:1], self.sec_cond_data,
                 cross_corr=0.5, secondary_var=-1.0)

        # ICCK: mismatched secondary data length
        with self.assertRaises(ValueError):
            IntrinsicCollocated(model, self.cond_pos[:1], self.cond_val,
                 self.cond_pos[:1], self.sec_data[:2],
                 cross_corr=0.5, secondary_var=1.0)

        # ICCK: missing secondary data
        icck = IntrinsicCollocated(model, self.cond_pos[:1], self.cond_val,
                    self.cond_pos[:1], self.sec_cond_data,
                    cross_corr=0.5, secondary_var=1.0)
        with self.assertRaises(ValueError):
            icck(self.pos)




if __name__ == "__main__":
    unittest.main()
