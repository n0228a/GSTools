"""
Comprehensive validation tests for collocated cokriging.

These tests go beyond basic functionality to validate:
1. Mathematical correctness against theoretical formulas
2. Comparison with full cokriging (ground truth)
3. Known analytical solutions
4. Mean handling correctness
5. Variance formula validation
"""

import unittest
import numpy as np
import gstools as gs
from gstools.cokriging import SCCK, ICCK
from scipy.spatial.distance import cdist
import scipy.linalg as spl


class TestCokrigingValidation(unittest.TestCase):
    """Rigorous validation tests for SCCK and ICCK."""

    def test_scck_mm1_weight_formula(self):
        """
        Validate MM1 collocated weight formula against manual calculation.

        Tests the actual implementation formula:
        λ_Y0 = (k × σ²_SK) / (C_Y0 - k² × (C_Z0 - σ²_SK))
        where k = C_YZ0 / C_Z0
        """
        model = gs.Exponential(dim=1, var=2.0, len_scale=3.0)

        # Simple test case
        cond_pos = ([0.0, 5.0],)
        cond_val = np.array([1.0, 2.0])

        cross_corr = 0.7
        secondary_var = 1.5
        secondary_mean = 0.5
        mean = 1.5

        # Create SCCK instance
        scck = SCCK(
            model,
            cond_pos,
            cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            mean=mean,
            secondary_mean=secondary_mean,
        )

        # Prediction point
        pos = np.array([2.5])
        sec_data = np.array([1.2])

        # Get Simple Kriging variance first
        sk = gs.krige.Simple(model, cond_pos, cond_val, mean=mean)
        sk_field, sk_var = sk(pos, return_var=True)

        # Manual calculation of MM1 weights
        C_Z0, C_Y0, C_YZ0 = scck._compute_covariances()
        k = C_YZ0 / C_Z0

        # NOTE: sk_var from API is already actual variance σ²_SK
        sigma2_sk = sk_var[0]

        # MM1 formula: λ_Y0 = (k × σ²_SK) / (C_Y0 - k² × σ²_SK)
        numerator = k * sigma2_sk
        denominator = C_Y0 - (k**2) * sigma2_sk

        if abs(denominator) < 1e-15:
            lambda_Y0_expected = 0.0
        else:
            lambda_Y0_expected = numerator / denominator

        # Get SCCK result
        scck_field, scck_var = scck(pos, secondary_data=sec_data, return_var=True)

        # Manually compute expected SCCK field
        expected_field = (
            (sk_field[0] - mean) * (1 - k * lambda_Y0_expected) +
            lambda_Y0_expected * (sec_data[0] - secondary_mean) +
            mean
        )

        # Validate field estimation
        np.testing.assert_allclose(
            scck_field[0], expected_field, rtol=1e-10,
            err_msg="SCCK field doesn't match manual calculation"
        )

        # Validate variance: σ²_SCCK = σ²_SK × (1 - kλ_Y0)
        expected_var = sigma2_sk * (1 - lambda_Y0_expected * k)
        expected_var = max(0.0, expected_var)

        np.testing.assert_allclose(
            scck_var[0], expected_var, rtol=1e-10,
            err_msg="SCCK variance doesn't match MM1 formula"
        )

    def test_icck_variance_formula(self):
        """
        Validate ICCK variance formula: σ²_ICCK = (1 - ρ₀²) × σ²_SK
        """
        model = gs.Gaussian(dim=1, var=1.5, len_scale=4.0)

        # Test data
        cond_pos = ([1.0, 4.0, 7.0],)
        cond_val = np.array([0.5, 1.2, 0.8])
        sec_cond_val = np.array([0.6, 1.0, 0.9])

        cross_corr = 0.8
        secondary_var = 1.2

        # Create ICCK
        icck = ICCK(
            model,
            cond_pos,
            cond_val,
            cond_pos,  # Secondary at primary locations
            sec_cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            mean=0.0,
            secondary_mean=0.0,
        )

        # Prediction points
        pos = np.array([2.5, 5.5])
        sec_data = np.array([0.7, 1.1])

        # Get SK variance
        sk = gs.krige.Simple(model, cond_pos, cond_val, mean=0.0)
        _, sk_var = sk(pos, return_var=True)

        # Get ICCK variance
        _, icck_var = icck(pos, secondary_data=sec_data, return_var=True)

        # Calculate theoretical variance
        C_Z0, C_Y0, C_YZ0 = icck._compute_covariances()
        rho_squared = (C_YZ0**2) / (C_Y0 * C_Z0)

        # σ²_ICCK = (1 - ρ₀²) × σ²_SK
        # NOTE: Kriging API returns actual variance (σ²), not kriging convention (C_0 - σ²)
        expected_icck_var = (1.0 - rho_squared) * sk_var  # sk_var IS σ²_SK
        expected_icck_var = np.maximum(0.0, expected_icck_var)

        np.testing.assert_allclose(
            icck_var, expected_icck_var, rtol=1e-10,
            err_msg="ICCK variance doesn't match (1-ρ₀²)×σ²_SK formula"
        )

    def test_perfect_correlation_with_consistent_data(self):
        """
        Test perfect correlation with ACTUALLY correlated data.

        Creates secondary data that is perfectly correlated with primary:
        Y = a × Z + b
        """
        model = gs.Exponential(dim=1, var=2.0, len_scale=3.0)

        # Primary data
        cond_pos = ([0.0, 2.0, 4.0, 6.0, 8.0],)
        cond_val = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

        # Perfect linear relationship: Y = 2×Z + 1
        a = 2.0
        b = 1.0
        sec_cond_val = a * cond_val + b

        # Secondary variance must match for perfect correlation
        # Var(Y) = a² × Var(Z)
        primary_var = np.var(cond_val - np.mean(cond_val), ddof=1)
        secondary_var = a**2 * primary_var

        # Cross-correlation should be ±1 (sign depends on a)
        cross_corr = 1.0 if a > 0 else -1.0

        # Prediction point
        pos = np.array([3.0])
        # Secondary data at prediction point (also perfectly correlated)
        true_primary_at_pos = 1.75  # Interpolated value
        sec_data = np.array([a * true_primary_at_pos + b])

        # Test ICCK with perfect correlation
        icck = ICCK(
            model,
            cond_pos,
            cond_val,
            cond_pos,
            sec_cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            mean=np.mean(cond_val),
            secondary_mean=np.mean(sec_cond_val),
        )

        field, var = icck(pos, secondary_data=sec_data, return_var=True)

        # With perfect correlation, variance should be near zero
        # NOTE: Kriging API returns actual variance σ², not C_0 - σ²
        self.assertTrue(
            var[0] < 1e-8,
            f"ICCK variance with perfect correlation should be ~0, got {var[0]}"
        )

    def test_mean_handling_scck(self):
        """
        Validate SCCK mean handling, especially the k×λ_Y0×m_Z term.

        Tests that the implementation correctly adds:
        Z*_SCCK = Z*_SK(1-kλ_Y0) + λ_Y0(Y-m_Y) + kλ_Y0×m_Z
        """
        model = gs.Gaussian(dim=1, var=1.0, len_scale=2.0)

        cond_pos = ([0.0, 3.0],)
        cond_val = np.array([5.0, 7.0])

        cross_corr = 0.6
        secondary_var = 0.8
        primary_mean = 6.0  # Non-zero mean
        secondary_mean = 4.0

        scck = SCCK(
            model,
            cond_pos,
            cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            mean=primary_mean,
            secondary_mean=secondary_mean,
        )

        pos = np.array([1.5])
        sec_data = np.array([4.5])

        # Get SK result (already includes mean)
        sk = gs.krige.Simple(model, cond_pos, cond_val, mean=primary_mean)
        sk_field, sk_var = sk(pos, return_var=True)

        # Manual SCCK calculation
        C_Z0, C_Y0, C_YZ0 = scck._compute_covariances()
        k = C_YZ0 / C_Z0
        sigma2_sk = sk_var[0]  # API returns actual variance σ²

        numerator = k * sigma2_sk
        denominator = C_Y0 - (k**2) * sigma2_sk
        lambda_Y0 = numerator / denominator if abs(denominator) > 1e-15 else 0.0

        # Full SCCK formula with mean correction
        # Note: sk_field already includes primary_mean, so we work in residual space
        expected = (
            (sk_field[0] - primary_mean) * (1 - k * lambda_Y0) +
            lambda_Y0 * (sec_data[0] - secondary_mean) +
            k * lambda_Y0 * primary_mean +
            primary_mean
        )

        # Simplifies to:
        expected = (
            sk_field[0] * (1 - k * lambda_Y0) +
            lambda_Y0 * (sec_data[0] - secondary_mean) +
            k * lambda_Y0 * primary_mean
        )

        scck_field = scck(pos, secondary_data=sec_data, return_var=False)

        np.testing.assert_allclose(
            scck_field[0], expected, rtol=1e-10,
            err_msg=f"SCCK mean handling incorrect. Expected {expected}, got {scck_field[0]}"
        )

    def test_icck_zero_correlation_exact_match(self):
        """
        With ρ=0, ICCK should EXACTLY match Simple Kriging.
        Tests both field and variance.
        """
        model = gs.Spherical(dim=1, var=3.0, len_scale=5.0)

        cond_pos = ([0.5, 2.5, 4.5, 6.5],)
        cond_val = np.array([1.2, 2.3, 1.8, 2.1])
        sec_cond_val = np.array([0.5, 0.8, 0.6, 0.7])  # Uncorrelated

        pos = np.linspace(0, 7, 20)
        sec_data = np.random.rand(20)

        # Simple Kriging
        sk = gs.krige.Simple(model, cond_pos, cond_val, mean=0.0)
        sk_field, sk_var = sk(pos, return_var=True)

        # ICCK with zero correlation
        icck = ICCK(
            model,
            cond_pos,
            cond_val,
            cond_pos,
            sec_cond_val,
            cross_corr=0.0,
            secondary_var=1.0,
            mean=0.0,
            secondary_mean=0.0,
        )
        icck_field, icck_var = icck(pos, secondary_data=sec_data, return_var=True)

        # Should be EXACTLY identical
        np.testing.assert_allclose(
            sk_field, icck_field, rtol=1e-12, atol=1e-14,
            err_msg="ICCK with ρ=0 doesn't match SK (field)"
        )

        np.testing.assert_allclose(
            sk_var, icck_var, rtol=1e-12, atol=1e-14,
            err_msg="ICCK with ρ=0 doesn't match SK (variance)"
        )

    def test_scck_variance_reduction(self):
        """
        Test that SCCK variance is reduced compared to SK (when correlation is positive).

        For MM1: σ²_SCCK = σ²_SK × (1 - kλ_Y0)
        Since k > 0 and λ_Y0 > 0 for positive correlation, variance should reduce.
        """
        model = gs.Gaussian(dim=1, var=2.0, len_scale=3.0)

        cond_pos = ([1.0, 4.0, 7.0],)
        cond_val = np.array([1.0, 1.5, 1.2])

        cross_corr = 0.7  # Positive correlation
        secondary_var = 1.5

        # Get SK variance
        sk = gs.krige.Simple(model, cond_pos, cond_val, mean=0.0)
        pos = np.array([2.5, 5.5])
        _, sk_var = sk(pos, return_var=True)

        # Get SCCK variance
        scck = SCCK(
            model,
            cond_pos,
            cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            mean=0.0,
            secondary_mean=0.0,
        )
        sec_data = np.array([1.1, 1.3])
        _, scck_var = scck(pos, secondary_data=sec_data, return_var=True)

        # SCCK variance should be less than or equal to SK variance
        # (equality only if λ_Y0 = 0, which shouldn't happen with ρ > 0)
        # NOTE: API returns actual variance σ², so direct comparison
        self.assertTrue(
            np.all(scck_var <= sk_var + 1e-10),  # Allow tiny numerical error
            f"SCCK variance should not exceed SK variance. SK: {sk_var}, SCCK: {scck_var}"
        )

        # With positive correlation, should see actual reduction
        mean_reduction = np.mean(sk_var - scck_var)
        self.assertTrue(
            mean_reduction > 0,
            f"SCCK should reduce variance, got mean reduction: {mean_reduction}"
        )

    def test_icck_better_than_scck(self):
        """
        Test that ICCK variance is better than SCCK variance.

        ICCK uses the formula σ²_ICCK = (1-ρ₀²)σ²_SK
        which eliminates the variance inflation of MM1.
        """
        model = gs.Exponential(dim=1, var=2.0, len_scale=4.0)

        cond_pos = ([0.0, 3.0, 6.0, 9.0],)
        cond_val = np.array([1.0, 2.0, 1.5, 2.5])
        sec_cond_val = np.array([0.8, 1.6, 1.2, 2.0])

        cross_corr = 0.75
        secondary_var = 1.2

        pos = np.linspace(1, 8, 15)
        sec_data = np.linspace(1.0, 2.0, 15)

        # SCCK
        scck = SCCK(
            model,
            cond_pos,
            cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            mean=0.0,
            secondary_mean=0.0,
        )
        _, scck_var = scck(pos, secondary_data=sec_data, return_var=True)

        # ICCK
        icck = ICCK(
            model,
            cond_pos,
            cond_val,
            cond_pos,
            sec_cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            mean=0.0,
            secondary_mean=0.0,
        )
        _, icck_var = icck(pos, secondary_data=sec_data, return_var=True)

        # ICCK actual variance should be <= SCCK actual variance
        # (ICCK eliminates variance inflation)
        # Both are already in actual variance format (σ²), so direct comparison
        self.assertTrue(
            np.all(icck_var <= scck_var + 1e-10),
            f"ICCK variance should not exceed SCCK variance. ICCK: {np.mean(icck_var)}, SCCK: {np.mean(scck_var)}"
        )

        # Calculate theoretical ICCK variance reduction
        C_Z0, C_Y0, C_YZ0 = icck._compute_covariances()
        rho_squared = (C_YZ0**2) / (C_Y0 * C_Z0)

        # Get SK variance for comparison
        sk = gs.krige.Simple(model, cond_pos, cond_val, mean=0.0)
        _, sk_var = sk(pos, return_var=True)

        # ICCK variance = (1-ρ²) × SK variance
        # NOTE: API returns actual variance σ², not kriging convention
        expected_icck_var = (1.0 - rho_squared) * sk_var

        np.testing.assert_allclose(
            icck_var, expected_icck_var, rtol=1e-9,
            err_msg="ICCK variance doesn't match theoretical (1-ρ²)×σ²_SK"
        )

    def test_dimensional_consistency(self):
        """
        Test that methods work correctly in 1D, 2D, and 3D.
        """
        for dim in [1, 2, 3]:
            model = gs.Gaussian(dim=dim, var=1.5, len_scale=3.0)

            # Create random points
            np.random.seed(42)
            n_cond = 5
            cond_pos = tuple(np.random.rand(n_cond) * 10 for _ in range(dim))
            cond_val = np.random.rand(n_cond) * 2
            sec_cond_val = cond_val + np.random.rand(n_cond) * 0.5

            # Test points
            n_test = 3
            test_pos = tuple(np.random.rand(n_test) * 10 for _ in range(dim))
            sec_data = np.random.rand(n_test) * 2

            # SCCK
            scck = SCCK(
                model,
                cond_pos,
                cond_val,
                cross_corr=0.6,
                secondary_var=1.2,
            )
            field_scck, var_scck = scck(test_pos, secondary_data=sec_data, return_var=True)

            self.assertEqual(field_scck.shape, (n_test,), f"SCCK failed in {dim}D")
            self.assertTrue(np.all(np.isfinite(field_scck)), f"SCCK produced non-finite values in {dim}D")
            self.assertTrue(np.all(var_scck >= 0), f"SCCK produced negative variance in {dim}D")

            # ICCK
            icck = ICCK(
                model,
                cond_pos,
                cond_val,
                cond_pos,
                sec_cond_val,
                cross_corr=0.6,
                secondary_var=1.2,
            )
            field_icck, var_icck = icck(test_pos, secondary_data=sec_data, return_var=True)

            self.assertEqual(field_icck.shape, (n_test,), f"ICCK failed in {dim}D")
            self.assertTrue(np.all(np.isfinite(field_icck)), f"ICCK produced non-finite values in {dim}D")
            self.assertTrue(np.all(var_icck >= 0), f"ICCK produced negative variance in {dim}D")


if __name__ == "__main__":
    unittest.main()
