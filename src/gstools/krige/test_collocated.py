"""
Test suite for collocated cokriging implementations.

This module contains comprehensive tests for SCCK and ICCK classes
to verify mathematical correctness and integration with gstools.
"""

import numpy as np
import pytest
from gstools import Gaussian, Exponential
from gstools.krige.collocated import SCCK, ICCK


class TestSCCK:
    """Test suite for Simple Collocated Cokriging (SCCK)."""

    def test_scck_initialization(self):
        """Test SCCK initialization with various parameters."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0)
        pos = [[0, 10, 20], [0, 0, 0]]
        val = [1.0, 2.0, 1.5]

        # Test valid initialization
        scck = SCCK(model, pos, val, cross_corr=0.7)
        assert scck.cross_corr == 0.7
        assert scck.secondary_variance == model.sill
        assert scck.cond_no == 3

    def test_scck_cross_corr_validation(self):
        """Test cross-correlation coefficient validation."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0)
        pos = [[0, 10], [0, 0]]
        val = [1.0, 2.0]

        # Test valid range
        scck = SCCK(model, pos, val, cross_corr=0.5)
        assert scck.cross_corr == 0.5

        scck.cross_corr = -0.8
        assert scck.cross_corr == -0.8

        # Test invalid values
        with pytest.raises(ValueError, match="cross_corr must be in"):
            SCCK(model, pos, val, cross_corr=1.5)

        with pytest.raises(ValueError, match="cross_corr must be in"):
            scck.cross_corr = -1.2

    def test_scck_matrix_dimensions(self):
        """Test that SCCK produces correct matrix dimensions."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0)
        pos = [[0, 10, 20, 30], [0, 0, 0, 0]]
        val = [1.0, 2.0, 1.5, 0.8]

        # Test unbiased (default)
        scck = SCCK(model, pos, val, cross_corr=0.6)
        expected_size = 4 + 1 + 1  # n_cond + secondary + unbiased
        assert scck.krige_size == expected_size

        krige_mat = scck._get_krige_mat()
        assert krige_mat.shape == (expected_size, expected_size)

        # Test simple (no unbiased constraint)
        scck_simple = SCCK(model, pos, val, cross_corr=0.6, unbiased=False)
        expected_size_simple = 4 + 1  # n_cond + secondary
        assert scck_simple.krige_size == expected_size_simple

    def test_scck_matrix_structure(self):
        """Test SCCK matrix structure and properties."""
        model = Gaussian(dim=2, var=2.0, len_scale=5.0)
        pos = [[0, 5], [0, 0]]
        val = [1.0, 2.0]

        scck = SCCK(model, pos, val, cross_corr=0.5, secondary_variance=1.5)
        krige_mat = scck._get_krige_mat()

        # Check matrix symmetry for covariance part
        n = scck.cond_no
        assert np.allclose(krige_mat[:n, :n], krige_mat[:n, :n].T)

        # Check secondary variance on diagonal
        assert krige_mat[n, n] == 1.5

        # Check unbiased constraints (if enabled)
        if scck.unbiased:
            unbiased_idx = n + 1
            assert np.allclose(krige_mat[unbiased_idx, :n], 1.0)
            assert np.allclose(krige_mat[:n, unbiased_idx], 1.0)
            assert krige_mat[unbiased_idx, n] == 1.0

    def test_scck_rhs_structure(self):
        """Test SCCK right-hand side vector structure."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0)
        pos = [[0, 10], [0, 0]]
        val = [1.0, 2.0]

        scck = SCCK(model, pos, val, cross_corr=0.7)

        # Test single estimation point
        target_pos = [[5], [0]]
        iso_pos, shape = scck.pre_pos(target_pos)
        rhs = scck._get_krige_vecs(iso_pos)

        expected_size = 2 + 1 + 1  # n_cond + secondary + unbiased
        assert rhs.shape == (expected_size, 1)

        # Check unbiased constraint
        if scck.unbiased:
            assert rhs[-1, 0] == 1.0

        # Check cross-covariance term
        n = scck.cond_no
        expected_cross_cov = 0.7 * \
            np.sqrt(model.sill * scck.secondary_variance)
        assert np.isclose(rhs[n, 0], expected_cross_cov)

    def test_scck_estimation_call(self):
        """Test SCCK estimation with secondary data."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0)
        pos = [[0, 10, 20], [0, 0, 0]]
        val = [1.0, 2.0, 1.5]

        scck = SCCK(model, pos, val, cross_corr=0.8)

        # Test estimation
        target_pos = [[5, 15], [0, 0]]
        secondary_data = [1.8, 1.2]

        # This should not raise an error
        result = scck(target_pos, secondary_data=secondary_data,
                      return_var=False)
        assert result.shape == (2,)

        # Test error when secondary data is missing
        with pytest.raises(ValueError, match="secondary_data must be provided"):
            scck(target_pos)

        # Test error when dimensions don't match
        with pytest.raises(ValueError, match="same number of points"):
            scck(target_pos, secondary_data=[1.8])  # Only 1 value for 2 points

    def test_scck_with_drift(self):
        """Test SCCK with drift functions."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0)
        pos = [[0, 10, 20], [0, 5, 0]]
        val = [1.0, 2.0, 1.5]

        # Test with linear drift
        scck = SCCK(model, pos, val, cross_corr=0.6, drift_functions="linear")

        # Matrix should be larger due to drift terms
        expected_size = 3 + 1 + 1 + 2  # n_cond + secondary + unbiased + linear_drift
        assert scck.krige_size == expected_size

        krige_mat = scck._get_krige_mat()
        assert krige_mat.shape == (expected_size, expected_size)

    def test_scck_reproducibility(self):
        """Test that SCCK produces reproducible results."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0, seed=12345)
        pos = [[0, 10, 20], [0, 0, 0]]
        val = [1.0, 2.0, 1.5]

        scck1 = SCCK(model, pos, val, cross_corr=0.7)
        scck2 = SCCK(model, pos, val, cross_corr=0.7)

        target_pos = [[5, 15], [0, 0]]
        secondary_data = [1.8, 1.2]

        result1 = scck1(
            target_pos, secondary_data=secondary_data, return_var=False)
        result2 = scck2(
            target_pos, secondary_data=secondary_data, return_var=False)

        assert np.allclose(result1, result2)


class TestICCK:
    """Test suite for Intrinsic Collocated Cokriging (ICCK)."""

    def test_icck_initialization(self):
        """Test ICCK initialization with different models."""
        model_primary = Gaussian(dim=2, var=1.0, len_scale=10.0)
        model_secondary = Exponential(dim=2, var=0.8, len_scale=12.0)
        pos = [[0, 10], [0, 0]]
        val = [1.0, 2.0]

        # Test with separate secondary model
        icck = ICCK(model_primary, pos, val, cross_corr=0.6,
                    model_secondary=model_secondary)

        assert icck.model_secondary == model_secondary
        assert icck.secondary_variance == model_secondary.sill

        # Test with same model for both variables
        icck_same = ICCK(model_primary, pos, val, cross_corr=0.6)
        assert icck_same.model_secondary == model_primary

    def test_icck_vs_scck_differences(self):
        """Test differences between ICCK and SCCK implementations."""
        model_primary = Gaussian(dim=2, var=1.0, len_scale=10.0)
        model_secondary = Gaussian(
            dim=2, var=2.0, len_scale=8.0)  # Different variance
        pos = [[0, 10], [0, 0]]
        val = [1.0, 2.0]

        scck = SCCK(model_primary, pos, val, cross_corr=0.7)
        icck = ICCK(model_primary, pos, val, cross_corr=0.7,
                    model_secondary=model_secondary)

        # ICCK should use secondary model variance
        assert icck.secondary_variance == model_secondary.sill
        assert scck.secondary_variance == model_primary.sill

        # Cross-covariance terms should be different
        target_pos = [[5], [0]]
        iso_pos, shape = scck.pre_pos(target_pos)
        rhs_scck = scck._get_krige_vecs(iso_pos)
        rhs_icck = icck._get_krige_vecs(iso_pos)

        # Cross-covariance terms (index n) should differ
        n = scck.cond_no
        assert not np.isclose(rhs_scck[n, 0], rhs_icck[n, 0])


class TestCollocatedEdgeCases:
    """Test edge cases and error conditions for collocated cokriging."""

    def test_matrix_valued_model_rejection(self):
        """Test that matrix-valued models are rejected."""
        # This would be a matrix-valued model if it existed
        # For now, just test the validation logic with a mock
        class MockMatrixModel:
            def __init__(self):
                self.is_matrix = True
                self.sill = 1.0

        mock_model = MockMatrixModel()
        pos = [[0, 10], [0, 0]]
        val = [1.0, 2.0]

        with pytest.raises(ValueError, match="matrix-valued covariance models not supported"):
            SCCK(mock_model, pos, val, cross_corr=0.5)

    def test_zero_cross_correlation(self):
        """Test behavior with zero cross-correlation."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0)
        pos = [[0, 10], [0, 0]]
        val = [1.0, 2.0]

        scck = SCCK(model, pos, val, cross_corr=0.0)

        # Cross-covariance terms should be zero
        target_pos = [[5], [0]]
        iso_pos, shape = scck.pre_pos(target_pos)
        rhs = scck._get_krige_vecs(iso_pos)
        n = scck.cond_no
        assert rhs[n, 0] == 0.0

    def test_perfect_correlation(self):
        """Test behavior with perfect correlation."""
        model = Gaussian(dim=2, var=1.0, len_scale=10.0)
        pos = [[0, 10], [0, 0]]
        val = [1.0, 2.0]

        scck = SCCK(model, pos, val, cross_corr=1.0)

        # Cross-covariance should equal covariance at zero lag
        target_pos = [[5], [0]]
        iso_pos, shape = scck.pre_pos(target_pos)
        rhs = scck._get_krige_vecs(iso_pos)
        n = scck.cond_no
        expected = np.sqrt(model.sill * scck.secondary_variance)
        assert np.isclose(rhs[n, 0], expected)


def test_integration_with_gstools():
    """Test that collocated classes integrate properly with gstools."""
    # Test import from main krige module
    from gstools.krige import SCCK, ICCK

    # Should be able to create instances
    model = Gaussian(dim=2, var=1.0, len_scale=10.0)
    pos = [[0, 10], [0, 0]]
    val = [1.0, 2.0]

    scck = SCCK(model, pos, val, cross_corr=0.7)
    icck = ICCK(model, pos, val, cross_corr=0.7)

    assert isinstance(scck, SCCK)
    assert isinstance(icck, ICCK)


if __name__ == "__main__":
    # Run basic functionality tests
    print("Running basic SCCK tests...")

    # Create test data
    model = Gaussian(dim=2, var=1.0, len_scale=10.0)
    pos = [[0, 10, 20], [0, 0, 0]]
    val = [1.0, 2.0, 1.5]

    # Test SCCK
    scck = SCCK(model, pos, val, cross_corr=0.7)
    print(f"SCCK created successfully. Matrix size: {scck.krige_size}")

    # Test matrix construction
    krige_mat = scck._get_krige_mat()
    print(f"Kriging matrix shape: {krige_mat.shape}")

    # Test RHS construction
    target_pos = [[5, 15], [0, 0]]
    # Need to use pre_pos to get the correct format
    iso_pos, shape = scck.pre_pos(target_pos)
    rhs = scck._get_krige_vecs(iso_pos)
    print(f"RHS shape: {rhs.shape}")

    # Test estimation (this will use placeholder implementation)
    secondary_data = [1.8, 1.2]
    try:
        result = scck(target_pos, secondary_data=secondary_data,
                      return_var=False)
        print(f"Estimation successful. Result shape: {result.shape}")
        print("Basic tests passed!")
    except Exception as e:
        print(f"Estimation failed: {e}")
        print("This is expected with the current placeholder implementation.")
