"""
Test correlogram classes for collocated cokriging.

This tests the new Correlogram architecture including:
- MarkovModel1 implementation
- Numerical equivalence with old API via from_parameters()
- Cross-covariance computations
"""

import numpy as np
import pytest

from gstools import Gaussian, MarkovModel1
from gstools.cokriging import SimpleCollocated, IntrinsicCollocated
from gstools.cokriging.correlogram import Correlogram


class TestMarkovModel1:
    """Test MarkovModel1 correlogram implementation."""

    def setup_method(self):
        """Setup common test data."""
        self.model = Gaussian(dim=1, var=0.5, len_scale=2.0)
        self.cross_corr = 0.8
        self.secondary_var = 1.5
        self.primary_mean = 1.0
        self.secondary_mean = 0.5

    def test_initialization(self):
        """Test MarkovModel1 initialization."""
        mm1 = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
            primary_mean=self.primary_mean,
            secondary_mean=self.secondary_mean,
        )

        assert mm1.primary_model == self.model
        assert mm1.cross_corr == self.cross_corr
        assert mm1.secondary_var == self.secondary_var
        assert mm1.primary_mean == self.primary_mean
        assert mm1.secondary_mean == self.secondary_mean

    def test_is_correlogram(self):
        """Test that MarkovModel1 is a Correlogram instance."""
        mm1 = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
        )
        assert isinstance(mm1, Correlogram)

    def test_validation(self):
        """Test parameter validation."""
        # Invalid cross_corr (outside [-1, 1])
        with pytest.raises(ValueError, match="cross_corr must be in"):
            MarkovModel1(
                primary_model=self.model,
                cross_corr=1.5,  # Invalid
                secondary_var=self.secondary_var,
            )

        # Invalid secondary_var (negative)
        with pytest.raises(ValueError, match="secondary_var must be positive"):
            MarkovModel1(
                primary_model=self.model,
                cross_corr=self.cross_corr,
                secondary_var=-1.0,  # Invalid
            )

    def test_compute_covariances(self):
        """Test covariance computation at zero lag."""
        mm1 = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
        )

        C_Z0, C_Y0, C_YZ0 = mm1.compute_covariances()

        # Check values
        assert C_Z0 == self.model.sill  # Primary variance
        assert C_Y0 == self.secondary_var  # Secondary variance

        # Check MM1 formula: C_YZ(0) = rho * sqrt(C_Z(0) * C_Y(0))
        expected_C_YZ0 = self.cross_corr * np.sqrt(C_Z0 * C_Y0)
        assert np.isclose(C_YZ0, expected_C_YZ0)

    def test_cross_covariance(self):
        """Test cross-covariance computation at distance h."""
        mm1 = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
        )

        # Test at h=0
        C_YZ_0 = mm1.cross_covariance(0.0)
        _, _, C_YZ0_expected = mm1.compute_covariances()
        assert np.isclose(C_YZ_0, C_YZ0_expected)

        # Test at h=1.0
        h = 1.0
        C_YZ_h = mm1.cross_covariance(h)

        # MM1 formula: C_YZ(h) = (C_YZ(0) / C_Z(0)) * C_Z(h)
        C_Z0, _, C_YZ0 = mm1.compute_covariances()
        C_Z_h = self.model.covariance(h)
        expected = (C_YZ0 / C_Z0) * C_Z_h
        assert np.isclose(C_YZ_h, expected)

    def test_cross_covariance_array(self):
        """Test cross-covariance computation with array input."""
        mm1 = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
        )

        h_array = np.array([0.0, 0.5, 1.0, 2.0])
        C_YZ_array = mm1.cross_covariance(h_array)

        assert C_YZ_array.shape == h_array.shape

        # Verify each element
        for i, h in enumerate(h_array):
            C_YZ_single = mm1.cross_covariance(h)
            assert np.isclose(C_YZ_array[i], C_YZ_single)


class TestSimpleCollocatedNewAPI:
    """Test SimpleCollocated with new correlogram API."""

    def setup_method(self):
        """Setup common test data."""
        np.random.seed(42)
        self.model = Gaussian(dim=1, var=0.5, len_scale=2.0)
        self.cond_pos = [0.5, 2.1, 3.8]
        self.cond_val = np.array([0.8, 1.2, 1.8])
        self.cross_corr = 0.8
        self.secondary_var = 1.5
        self.primary_mean = 1.0
        self.secondary_mean = 0.5

    def test_new_api(self):
        """Test SimpleCollocated with new correlogram API."""
        correlogram = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
            primary_mean=self.primary_mean,
            secondary_mean=self.secondary_mean,
        )

        scck = SimpleCollocated(
            correlogram,
            cond_pos=self.cond_pos,
            cond_val=self.cond_val,
        )

        # Should initialize without error
        assert scck.correlogram == correlogram
        assert scck.algorithm == "simple"

    def test_requires_correlogram(self):
        """Test that SimpleCollocated requires a Correlogram object."""
        with pytest.raises(TypeError, match="must be a Correlogram instance"):
            SimpleCollocated(
                self.model,  # Wrong: should be a Correlogram
                cond_pos=self.cond_pos,
                cond_val=self.cond_val,
            )

    def test_backward_compatibility(self):
        """Test backward compatibility via from_parameters()."""
        # New API
        correlogram = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
            primary_mean=self.primary_mean,
            secondary_mean=self.secondary_mean,
        )
        scck_new = SimpleCollocated(
            correlogram,
            cond_pos=self.cond_pos,
            cond_val=self.cond_val,
        )

        # Old API (via from_parameters)
        with pytest.warns(DeprecationWarning):
            scck_old = SimpleCollocated.from_parameters(
                model=self.model,
                cond_pos=self.cond_pos,
                cond_val=self.cond_val,
                cross_corr=self.cross_corr,
                secondary_var=self.secondary_var,
                mean=self.primary_mean,
                secondary_mean=self.secondary_mean,
            )

        # Both should produce same covariances
        C_new = scck_new.correlogram.compute_covariances()
        C_old = scck_old.correlogram.compute_covariances()

        assert np.allclose(C_new, C_old)

    def test_numerical_equivalence(self):
        """Test numerical equivalence between new and old API."""
        # Setup interpolation grid
        gridx = np.linspace(0.0, 5.0, 11)
        secondary_data = np.ones(11) * self.secondary_mean

        # New API
        correlogram = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
            primary_mean=self.primary_mean,
            secondary_mean=self.secondary_mean,
        )
        scck_new = SimpleCollocated(correlogram, self.cond_pos, self.cond_val)
        field_new, var_new = scck_new(gridx, secondary_data=secondary_data, return_var=True)

        # Old API
        with pytest.warns(DeprecationWarning):
            scck_old = SimpleCollocated.from_parameters(
                self.model, self.cond_pos, self.cond_val,
                cross_corr=self.cross_corr,
                secondary_var=self.secondary_var,
                mean=self.primary_mean,
                secondary_mean=self.secondary_mean,
            )
        field_old, var_old = scck_old(gridx, secondary_data=secondary_data, return_var=True)

        # Results should be numerically equivalent
        assert np.allclose(field_new, field_old, rtol=1e-10)
        assert np.allclose(var_new, var_old, rtol=1e-10)


class TestIntrinsicCollocatedNewAPI:
    """Test IntrinsicCollocated with new correlogram API."""

    def setup_method(self):
        """Setup common test data."""
        np.random.seed(42)
        self.model = Gaussian(dim=1, var=0.5, len_scale=2.0)
        self.cond_pos = [0.5, 2.1, 3.8]
        self.cond_val = np.array([0.8, 1.2, 1.8])
        self.sec_at_primary = np.array([0.4, 0.6, 0.7])
        self.cross_corr = 0.8
        self.secondary_var = 1.5
        self.primary_mean = 1.0
        self.secondary_mean = 0.5

    def test_new_api(self):
        """Test IntrinsicCollocated with new correlogram API."""
        correlogram = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
            primary_mean=self.primary_mean,
            secondary_mean=self.secondary_mean,
        )

        icck = IntrinsicCollocated(
            correlogram,
            cond_pos=self.cond_pos,
            cond_val=self.cond_val,
            secondary_cond_pos=self.cond_pos,
            secondary_cond_val=self.sec_at_primary,
        )

        # Should initialize without error
        assert icck.correlogram == correlogram
        assert icck.algorithm == "intrinsic"

    def test_numerical_equivalence(self):
        """Test numerical equivalence between new and old API."""
        # Setup interpolation grid
        gridx = np.linspace(0.0, 5.0, 11)
        secondary_data = np.ones(11) * self.secondary_mean

        # New API
        correlogram = MarkovModel1(
            primary_model=self.model,
            cross_corr=self.cross_corr,
            secondary_var=self.secondary_var,
            primary_mean=self.primary_mean,
            secondary_mean=self.secondary_mean,
        )
        icck_new = IntrinsicCollocated(
            correlogram,
            self.cond_pos,
            self.cond_val,
            self.cond_pos,
            self.sec_at_primary,
        )
        field_new, var_new = icck_new(gridx, secondary_data=secondary_data, return_var=True)

        # Old API
        with pytest.warns(DeprecationWarning):
            icck_old = IntrinsicCollocated.from_parameters(
                self.model,
                self.cond_pos,
                self.cond_val,
                self.cond_pos,
                self.sec_at_primary,
                cross_corr=self.cross_corr,
                secondary_var=self.secondary_var,
                mean=self.primary_mean,
                secondary_mean=self.secondary_mean,
            )
        field_old, var_old = icck_old(gridx, secondary_data=secondary_data, return_var=True)

        # Results should be numerically equivalent
        assert np.allclose(field_new, field_old, rtol=1e-10)
        assert np.allclose(var_new, var_old, rtol=1e-10)
