"""
This is the unittest of the cokriging module.

Tests only the NEW logic added by CollocatedCokriging on top of Krige.
Inherited functionality (grids, models, dimensions, anisotropy) is tested in test_krige.py.
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
        scck = gs.cokriging.SimpleCollocated(
            self.model, self.cond_pos, self.cond_val,
            cross_corr=0.5, secondary_var=1.0
        )
        with self.assertRaises(ValueError):
            scck(self.pos)

    def test_cross_corr_validation(self):
        """Test cross_corr must be in [-1, 1]."""
        with self.assertRaises(ValueError):
            gs.cokriging.SimpleCollocated(
                self.model, self.cond_pos, self.cond_val,
                cross_corr=1.5, secondary_var=1.0
            )
        with self.assertRaises(ValueError):
            gs.cokriging.SimpleCollocated(
                self.model, self.cond_pos, self.cond_val,
                cross_corr=-1.5, secondary_var=1.0
            )

    def test_secondary_var_validation(self):
        """Test secondary_var must be positive."""
        with self.assertRaises(ValueError):
            gs.cokriging.SimpleCollocated(
                self.model, self.cond_pos, self.cond_val,
                cross_corr=0.5, secondary_var=-1.0
            )
        with self.assertRaises(ValueError):
            gs.cokriging.SimpleCollocated(
                self.model, self.cond_pos, self.cond_val,
                cross_corr=0.5, secondary_var=0.0
            )

    def test_icck_secondary_cond_length(self):
        """Test ICCK secondary conditioning data length validation."""
        with self.assertRaises(ValueError):
            gs.cokriging.IntrinsicCollocated(
                self.model, self.cond_pos, self.cond_val,
                self.cond_pos, self.sec_cond_val[:3],  # Wrong length
                cross_corr=0.5, secondary_var=1.0
            )

    def test_zero_correlation_equals_sk(self):
        """Test that ρ=0 gives Simple Kriging results."""
        # Reference: Simple Kriging
        sk = gs.krige.Simple(self.model, self.cond_pos, self.cond_val, mean=0.0)
        sk_field, sk_var = sk(self.pos, return_var=True)

        # SCCK with ρ=0
        scck = gs.cokriging.SimpleCollocated(
            self.model, self.cond_pos, self.cond_val,
            cross_corr=0.0, secondary_var=1.5
        )
        scck_field, scck_var = scck(self.pos, secondary_data=self.sec_data, return_var=True)
        np.testing.assert_allclose(scck_field, sk_field, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(scck_var, sk_var, rtol=1e-6, atol=1e-9)

        # ICCK with ρ=0
        icck = gs.cokriging.IntrinsicCollocated(
            self.model, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val,
            cross_corr=0.0, secondary_var=1.5
        )
        icck_field, icck_var = icck(self.pos, secondary_data=self.sec_data, return_var=True)
        np.testing.assert_allclose(icck_field, sk_field, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(icck_var, sk_var, rtol=1e-6, atol=1e-9)

    def test_scck_variance_formula(self):
        """Test SCCK variance: σ²_SCCK = σ²_SK * (1 - λ_Y0 * k)."""
        cross_corr = 0.7
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(self.model, self.cond_pos, self.cond_val, mean=0.0)
        _, sk_var = sk(self.pos, return_var=True)

        # Calculate expected SCCK variance components
        C_Z0 = self.model.sill
        C_Y0 = secondary_var
        C_YZ0 = cross_corr * np.sqrt(C_Z0 * C_Y0)
        k = C_YZ0 / C_Z0

        # Collocated weight λ_Y0 = k*σ²_SK / (C_Y0 - k²(C_Z0 - σ²_SK))
        numerator = k * sk_var
        denominator = C_Y0 - (k**2) * (C_Z0 - sk_var)
        lambda_Y0 = np.where(np.abs(denominator) < 1e-15, 0.0, numerator / denominator)
        expected_var = sk_var * (1.0 - lambda_Y0 * k)
        expected_var = np.maximum(0.0, expected_var)

        # Actual SCCK variance
        scck = gs.cokriging.SimpleCollocated(
            self.model, self.cond_pos, self.cond_val,
            cross_corr=cross_corr, secondary_var=secondary_var
        )
        _, actual_var = scck(self.pos, secondary_data=self.sec_data, return_var=True)
        np.testing.assert_allclose(actual_var, expected_var, rtol=1e-6, atol=1e-9)

    def test_icck_variance_formula(self):
        """Test ICCK variance: σ²_ICCK = (1-ρ₀²)·σ²_SK."""
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
        icck = gs.cokriging.IntrinsicCollocated(
            self.model, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val,
            cross_corr=cross_corr, secondary_var=secondary_var
        )
        _, actual_var = icck(self.pos, secondary_data=self.sec_data, return_var=True)
        np.testing.assert_allclose(actual_var, expected_var, rtol=1e-6, atol=1e-9)

    def test_perfect_correlation_variance(self):
        """Test that ρ=±1 gives near-zero variance for ICCK."""
        for rho in [-1.0, 1.0]:
            icck = gs.cokriging.IntrinsicCollocated(
                self.model, self.cond_pos, self.cond_val,
                self.cond_pos, self.sec_cond_val,
                cross_corr=rho, secondary_var=1.5
            )
            _, icck_var = icck(self.pos, secondary_data=self.sec_data, return_var=True)
            self.assertTrue(np.allclose(icck_var, 0.0, atol=1e-12))

    def test_scck_variance_inflation(self):
        """Test SCCK variance behavior in unstable region (small denominator)."""
        # Setup: high cross-correlation with secondary_var chosen to make
        # denominator D = C_Y0 - k²(C_Z0 - σ²_SK) small, demonstrating
        # SCCK instability region where variance reduction is minimal
        cross_corr = 0.9
        C_Z0 = self.model.sill
        C_Y0 = C_Z0 * (cross_corr**2) * 1.05  # slightly above k²·C_Z0
        secondary_var = C_Y0

        # Get SK variance
        sk = gs.krige.Simple(self.model, self.cond_pos, self.cond_val, mean=0.0)
        _, sk_var = sk(self.pos, return_var=True)

        # Get SCCK variance in unstable configuration
        scck = gs.cokriging.SimpleCollocated(
            self.model, self.cond_pos, self.cond_val,
            cross_corr=cross_corr, secondary_var=secondary_var
        )
        _, scck_var = scck(self.pos, secondary_data=self.sec_data, return_var=True)

        # In unstable region: variance reduction is minimal
        mask = sk_var > 1e-10
        variance_reduction = 1.0 - np.divide(scck_var, sk_var, where=mask, out=np.zeros_like(scck_var))
        # At some points, reduction should be less than 10%
        self.assertTrue(np.any(variance_reduction < 0.1))

        # Ensure values are finite and non-negative (implementation clamping)
        self.assertTrue(np.all(np.isfinite(scck_var)))
        self.assertTrue(np.all(scck_var >= -1e-12))
        # Check not exploding
        self.assertTrue(np.max(scck_var) < 1e6 * C_Z0)

    def test_variance_reduction(self):
        """Test that cokriging methods reduce variance compared to simple kriging."""
        cross_corr = 0.8
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(self.model, self.cond_pos, self.cond_val, mean=0.0)
        _, sk_var = sk(self.pos, return_var=True)

        # Get ICCK variance
        icck = gs.cokriging.IntrinsicCollocated(
            self.model, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val,
            cross_corr=cross_corr, secondary_var=secondary_var
        )
        _, icck_var = icck(self.pos, secondary_data=self.sec_data, return_var=True)

        # Get SCCK variance
        scck = gs.cokriging.SimpleCollocated(
            self.model, self.cond_pos, self.cond_val,
            cross_corr=cross_corr, secondary_var=secondary_var
        )
        _, scck_var = scck(self.pos, secondary_data=self.sec_data, return_var=True)

        # ICCK variance ≤ SK variance (guaranteed by formula σ²_ICCK = (1-ρ₀²)·σ²_SK)
        self.assertTrue(np.all(icck_var <= sk_var + 1e-8))

        # Both methods should be finite and non-negative
        self.assertTrue(np.all(np.isfinite(icck_var)))
        self.assertTrue(np.all(np.isfinite(scck_var)))
        self.assertTrue(np.all(icck_var >= -1e-12))
        self.assertTrue(np.all(scck_var >= -1e-12))

        # On average, both methods should reduce variance compared to SK
        self.assertTrue(np.mean(icck_var) < np.mean(sk_var))
        self.assertTrue(np.mean(scck_var) < np.mean(sk_var))

    def test_exact_interpolation_at_conditioning_point(self):
        """Test exact interpolation: field equals observed value at conditioning point."""
        cross_corr = 0.7
        secondary_var = 1.5

        # Create secondary data at conditioning locations
        sec_at_cond = np.interp(self.cond_pos[0], self.pos, self.sec_data)

        # SCCK: predict at first conditioning point
        scck = gs.cokriging.SimpleCollocated(
            self.model, self.cond_pos, self.cond_val,
            cross_corr=cross_corr, secondary_var=secondary_var, mean=0.0
        )
        pos_test = np.array([self.cond_pos[0][0]])
        sec_test = np.array([sec_at_cond[0]])
        scck_field, scck_var = scck(pos_test, secondary_data=sec_test, return_var=True)

        # Should recover the conditioning value
        np.testing.assert_allclose(scck_field[0], self.cond_val[0], rtol=1e-6, atol=1e-9)
        # Variance should be very small (near zero for exact interpolation)
        self.assertTrue(scck_var[0] < 1e-6)

        # ICCK: predict at first conditioning point
        icck = gs.cokriging.IntrinsicCollocated(
            self.model, self.cond_pos, self.cond_val,
            self.cond_pos, self.sec_cond_val,
            cross_corr=cross_corr, secondary_var=secondary_var, mean=0.0
        )
        # For ICCK, use the actual secondary value at conditioning point
        sec_test_icck = np.array([self.sec_cond_val[0]])
        icck_field, icck_var = icck(pos_test, secondary_data=sec_test_icck, return_var=True)

        # Should recover the conditioning value
        np.testing.assert_allclose(icck_field[0], self.cond_val[0], rtol=1e-6, atol=1e-9)
        # Variance should be very small
        self.assertTrue(icck_var[0] < 1e-6)


if __name__ == "__main__":
    unittest.main()
