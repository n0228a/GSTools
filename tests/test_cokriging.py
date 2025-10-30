"""
This is the unittest of the cokriging module.
"""

import unittest

import numpy as np

import gstools as gs


class TestCokriging(unittest.TestCase):
    def setUp(self):
        # Simple 1D test case
        self.model = gs.Gaussian(dim=1, var=2, len_scale=2)
        self.cond_pos = ([0.3, 1.9, 1.1, 3.3, 4.7],)
        self.cond_val = np.array([0.47, 0.56, 0.74, 1.47, 1.74])
        self.sec_cond_val = np.array([1.8, 1.2, 2.1, 2.9, 2.4])
        self.pos = np.linspace(0, 5, 51)
        # Dummy secondary data
        self.sec_data = np.random.RandomState(42).rand(len(self.pos))

    def test_secondary_data_required(self):
        """Test that secondary_data is required on call."""
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=0.5, secondary_var=1.0
        )
        scck = gs.cokriging.SimpleCollocated(
            correlogram, self.cond_pos, self.cond_val
        )
        with self.assertRaises(ValueError):
            scck(self.pos)

    def test_correlogram_type_required(self):
        """Test that first argument must be a Correlogram."""
        with self.assertRaises(TypeError):
            gs.cokriging.SimpleCollocated(
                self.model, self.cond_pos, self.cond_val
            )

    def test_icck_secondary_cond_required(self):
        """Test ICCK requires secondary conditioning data."""
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=0.5, secondary_var=1.0
        )
        with self.assertRaises(ValueError):
            gs.cokriging.IntrinsicCollocated(
                correlogram, self.cond_pos, self.cond_val,
                secondary_cond_pos=None, secondary_cond_val=None
            )

    def test_icck_secondary_cond_length(self):
        """Test ICCK secondary conditioning data length validation."""
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=0.5, secondary_var=1.0
        )
        with self.assertRaises(ValueError):
            gs.cokriging.IntrinsicCollocated(
                correlogram, self.cond_pos, self.cond_val,
                self.cond_pos, self.sec_cond_val[:3]  # Wrong length
            )

    def test_zero_correlation_equals_sk(self):
        """Test that ρ=0 gives Simple Kriging results."""
        # Reference: Simple Kriging
        sk = gs.krige.Simple(self.model, self.cond_pos, self.cond_val, mean=0.0)
        sk_field, sk_var = sk(self.pos, return_var=True)

        # SCCK with ρ=0
        correlogram_scck = gs.MarkovModel1(
            self.model, cross_corr=0.0, secondary_var=1.5
        )
        scck = gs.cokriging.SimpleCollocated(
            correlogram_scck, self.cond_pos, self.cond_val
        )
        scck_field, scck_var = scck(self.pos, secondary_data=self.sec_data, return_var=True)
        np.testing.assert_allclose(scck_field, sk_field, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(scck_var, sk_var, rtol=1e-6, atol=1e-9)

        # ICCK with ρ=0
        correlogram_icck = gs.MarkovModel1(
            self.model, cross_corr=0.0, secondary_var=1.5
        )
        icck = gs.cokriging.IntrinsicCollocated(
            correlogram_icck, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val
        )
        icck_field, icck_var = icck(self.pos, secondary_data=self.sec_data, return_var=True)
        np.testing.assert_allclose(icck_field, sk_field, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(icck_var, sk_var, rtol=1e-6, atol=1e-9)

    def test_scck_variance_formula(self):
        """Test SCCK variance formula."""
        cross_corr = 0.7
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(self.model, self.cond_pos, self.cond_val, mean=0.0)
        _, sk_var = sk(self.pos, return_var=True)

        # Calculate expected SCCK variance
        C_Z0 = self.model.sill
        C_Y0 = secondary_var
        C_YZ0 = cross_corr * np.sqrt(C_Z0 * C_Y0)
        k = C_YZ0 / C_Z0

        numerator = k * sk_var
        denominator = C_Y0 - (k**2) * (C_Z0 - sk_var)
        lambda_Y0 = np.where(np.abs(denominator) < 1e-15, 0.0, numerator / denominator)
        expected_var = sk_var * (1.0 - lambda_Y0 * k)
        expected_var = np.maximum(0.0, expected_var)

        # Actual SCCK variance
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=cross_corr, secondary_var=secondary_var
        )
        scck = gs.cokriging.SimpleCollocated(
            correlogram, self.cond_pos, self.cond_val
        )
        _, actual_var = scck(self.pos, secondary_data=self.sec_data, return_var=True)
        np.testing.assert_allclose(actual_var, expected_var, rtol=1e-6, atol=1e-9)

    def test_icck_variance_formula(self):
        """Test ICCK variance formula."""
        cross_corr = 0.7
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(self.model, self.cond_pos, self.cond_val, mean=0.0)
        _, sk_var = sk(self.pos, return_var=True)

        # Expected ICCK variance
        C_Z0 = self.model.sill
        C_Y0 = secondary_var
        C_YZ0 = cross_corr * np.sqrt(C_Z0 * C_Y0)
        rho_squared = (C_YZ0**2) / (C_Y0 * C_Z0)
        expected_var = (1.0 - rho_squared) * sk_var

        # Actual ICCK variance
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=cross_corr, secondary_var=secondary_var
        )
        icck = gs.cokriging.IntrinsicCollocated(
            correlogram, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val
        )
        _, actual_var = icck(self.pos, secondary_data=self.sec_data, return_var=True)
        np.testing.assert_allclose(actual_var, expected_var, rtol=1e-6, atol=1e-9)

    def test_perfect_correlation_variance(self):
        """Test that ρ=±1 gives near-zero variance for ICCK."""
        for rho in [-1.0, 1.0]:
            correlogram = gs.MarkovModel1(
                self.model, cross_corr=rho, secondary_var=1.5
            )
            icck = gs.cokriging.IntrinsicCollocated(
                correlogram, self.cond_pos, self.cond_val,
                self.cond_pos, self.sec_cond_val
            )
            _, icck_var = icck(self.pos, secondary_data=self.sec_data, return_var=True)
            self.assertTrue(np.allclose(icck_var, 0.0, atol=1e-12))

    def test_variance_reduction(self):
        """Test that cokriging reduces variance compared to simple kriging."""
        cross_corr = 0.8
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(self.model, self.cond_pos, self.cond_val, mean=0.0)
        _, sk_var = sk(self.pos, return_var=True)

        # Get ICCK variance
        correlogram = gs.MarkovModel1(
            self.model, cross_corr=cross_corr, secondary_var=secondary_var
        )
        icck = gs.cokriging.IntrinsicCollocated(
            correlogram, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val
        )
        _, icck_var = icck(self.pos, secondary_data=self.sec_data, return_var=True)

        # ICCK variance ≤ SK variance
        self.assertTrue(np.all(icck_var <= sk_var + 1e-8))
        self.assertTrue(np.mean(icck_var) < np.mean(sk_var))


if __name__ == "__main__":
    unittest.main()
