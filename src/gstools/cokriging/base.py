"""
GStools subpackage providing base collocated cokriging functionality.

.. currentmodule:: gstools.cokriging.base

The following base classes are provided

.. autosummary::
   CollocatedCokriging
"""

import numpy as np
from gstools.krige.base import Krige

__all__ = ["CollocatedCokriging"]


class CollocatedCokriging(Krige):
    """
    Base class for collocated cokriging methods.

    This class provides unified functionality for both Simple Collocated Cokriging (SCCK)
    and Intrinsic Collocated Cokriging (ICCK), following the same pattern as the kriging
    module where different methods are parameter variations of a common base.

    The class handles all common functionality:
    - Input validation for cross-correlation and secondary variance
    - Covariance calculations (C_Z0, C_Y0, C_YZ0)
    - Secondary data management
    - Edge case handling (zero correlation, perfect correlation)
    - Variance post-processing for proper ICCK variance estimation

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance model for the primary variable.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the primary variable conditions (nan values will be ignored)
    cross_corr : :class:`float`
        Cross-correlation coefficient between primary and secondary variables
        at zero lag. Must be in [-1, 1].
    secondary_var : :class:`float`
        Variance of the secondary variable. Must be positive.
    algorithm : :class:`str`
        Cokriging algorithm to use. Either "MM1" (SCCK) or "intrinsic" (ICCK).
    secondary_cond_pos : :class:`list`, optional
        tuple, containing secondary variable condition positions (only for ICCK)
    secondary_cond_val : :class:`numpy.ndarray`, optional
        values of secondary variable at primary locations (only for ICCK)
    mean : :class:`float`, optional
        Mean value for simple kriging. Default: 0.0
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signature: f(x, [y, z, ...])
        This is used for detrended kriging, where the trend is subtracted
        from the conditions before kriging is applied.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class:`float` or :class:`list`, optional
        The measurement error at the conditioning points.
        Either "nugget" to apply the model-nugget, a single value applied to
        all points or an array with individual values for each point.
        The measurement error has to be <= nugget.
        The "exact=True" variant only works with "cond_err='nugget'".
        Default: "nugget"
    pseudo_inv : :class:`bool`, optional
        Whether the kriging system is solved with the pseudo inverted
        kriging matrix. If `True`, this leads to more numerical stability
        and redundant points are averaged. But it can take more time.
        Default: True
    pseudo_inv_type : :class:`str` or :any:`callable`, optional
        Here you can select the algorithm to compute the pseudo-inverse matrix:

            * `"pinv"`: use `pinv` from `scipy` which uses `SVD`
            * `"pinvh"`: use `pinvh` from `scipy` which uses eigen-values

        If you want to use another routine to invert the kriging matrix,
        you can pass a callable which takes a matrix and returns the inverse.
        Default: `"pinv"`
    fit_normalizer : :class:`bool`, optional
        Whether to fit the data-normalizer to the given conditioning data.
        Default: False
    fit_variogram : :class:`bool`, optional
        Whether to fit the given variogram model to the data.
        Default: False

    References
    ----------
    .. [Samson2020] Samson, M., & Deutsch, C. V. (2020). Collocated Cokriging.
       In J. L. Deutsch (Ed.), Geostatistics Lessons. Retrieved from
       http://geostatisticslessons.com/lessons/collocatedcokriging
    .. [Wackernagel2003] Wackernagel, H. Multivariate Geostatistics,
       Springer, Berlin, 2003.
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        cross_corr,
        secondary_var,
        algorithm,
        secondary_cond_pos=None,
        secondary_cond_val=None,
        mean=0.0,
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type="pinv",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        # Validate algorithm parameter
        if algorithm not in ["MM1", "intrinsic"]:
            raise ValueError(
                "algorithm must be 'MM1' (SCCK) or 'intrinsic' (ICCK)")
        self.algorithm = algorithm

        # Validate cross-correlation and secondary variance
        self.cross_corr = float(cross_corr)
        if not -1.0 <= self.cross_corr <= 1.0:
            raise ValueError("cross_corr must be in [-1, 1]")

        self.secondary_var = float(secondary_var)
        if self.secondary_var <= 0:
            raise ValueError("secondary_var must be positive")

        # Handle secondary conditioning data (required for ICCK)
        if algorithm == "intrinsic":
            if secondary_cond_pos is None or secondary_cond_val is None:
                raise ValueError(
                    "secondary_cond_pos and secondary_cond_val required for ICCK"
                )
            self.secondary_cond_pos = secondary_cond_pos
            self.secondary_cond_val = np.asarray(
                secondary_cond_val, dtype=np.double)

            # Validate that secondary data matches primary locations
            if len(self.secondary_cond_val) != len(cond_val):
                raise ValueError(
                    "secondary_cond_val must have same length as primary cond_val"
                )
        else:
            # MM1 (SCCK) doesn't require secondary conditioning data
            self.secondary_cond_pos = None
            self.secondary_cond_val = None

        # Initialize as Simple Kriging (unbiased=False)
        super().__init__(
            model=model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            mean=mean,
            unbiased=False,  # Simple kriging base
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )

    def __call__(self, pos=None, secondary_data=None, **kwargs):
        """
        Estimate using collocated cokriging.

        Parameters
        ----------
        pos : :class:`list`
            tuple, containing the given positions (x, [y, z])
        secondary_data : :class:`numpy.ndarray`
            Secondary variable values at estimation positions.
        **kwargs
            Standard Krige parameters (return_var, chunk_size, only_mean, etc.)

        Returns
        -------
        field : :class:`numpy.ndarray`
            Collocated cokriging estimated field values.
        krige_var : :class:`numpy.ndarray`, optional
            Collocated cokriging estimation variance (if return_var=True).
        """
        if secondary_data is None:
            raise ValueError(
                "secondary_data required for collocated cokriging")

        # Store secondary data for use in _summate
        self._secondary_data = np.asarray(secondary_data, dtype=np.double)

        try:
            # Call parent class with variance fix for ICCK
            result = super().__call__(pos=pos, **kwargs)

            # Fix variance post-processing for ICCK: restore stored variance if computed
            if (self.algorithm == "intrinsic" and
                isinstance(result, tuple) and len(result) == 2 and
                    hasattr(self, '_icck_stored_variance')):
                field, _ = result  # Ignore the base class modified variance
                variance = self._icck_stored_variance
                delattr(self, '_icck_stored_variance')
                return field, variance
            else:
                return result
        finally:
            # Clean up temporary attribute
            if hasattr(self, '_secondary_data'):
                delattr(self, '_secondary_data')

    def _compute_covariances(self):
        """
        Compute the three scalar covariances: C_Z0, C_Y0, C_YZ0.

        Returns
        -------
        tuple
            (C_Z0, C_Y0, C_YZ0) covariances at zero lag
        """
        # C_Z0: primary variable variance at zero lag
        C_Z0 = self.model.sill

        # C_Y0: secondary variable variance at zero lag
        C_Y0 = self.secondary_var

        # C_YZ0: cross-covariance at zero lag
        C_YZ0 = self.cross_corr * np.sqrt(C_Z0 * C_Y0)

        return C_Z0, C_Y0, C_YZ0

    def _compute_correlation_coeff_squared(self, C_Z0, C_Y0, C_YZ0):
        """
        Compute squared correlation coefficient ρ₀² = C²_YZ0/(C_Y0×C_Z0).

        Parameters
        ----------
        C_Z0, C_Y0, C_YZ0 : float
            Covariances at zero lag

        Returns
        -------
        float
            Squared correlation coefficient
        """
        # Handle edge case where variances are zero
        if C_Y0 * C_Z0 <= 1e-15:
            return 0.0

        return (C_YZ0**2) / (C_Y0 * C_Z0)

    def _summate(self, field, krige_var, c_slice, k_vec, return_var):
        """Override to implement algorithm-specific collocated cokriging estimators."""
        # Get covariances at zero lag
        C_Z0, C_Y0, C_YZ0 = self._compute_covariances()

        # Handle trivial case where cross-correlation is zero (both algorithms)
        if abs(C_YZ0) < 1e-15:
            # Reduces to SK when C_YZ0 = 0
            return super()._summate(field, krige_var, c_slice, k_vec, return_var)

        # Import at function level to avoid circular imports
        from gstools.krige.base import _calc_field_krige_and_variance

        # Always compute both SK field and variance (required for both algorithms)
        sk_field_chunk, sk_var_chunk = _calc_field_krige_and_variance(
            self._krige_mat, k_vec, self._krige_cond
        )

        # Get secondary data at estimation positions
        secondary_chunk = self._secondary_data[c_slice]

        # Algorithm-specific implementations
        if self.algorithm == "MM1":
            self._summate_mm1(field, krige_var, c_slice, sk_field_chunk,
                              sk_var_chunk, secondary_chunk, C_Z0, C_Y0, C_YZ0, return_var)
        elif self.algorithm == "intrinsic":
            self._summate_intrinsic(field, krige_var, c_slice, k_vec, sk_field_chunk,
                                    sk_var_chunk, secondary_chunk, C_Z0, C_Y0, C_YZ0, return_var)

    def _summate_mm1(self, field, krige_var, c_slice, sk_field_chunk, sk_var_chunk,
                     secondary_chunk, C_Z0, C_Y0, C_YZ0, return_var):
        """Implement MM1 (SCCK) algorithm."""
        # Compute MM1 parameters
        k = C_YZ0 / C_Z0  # Cross-covariance ratio

        # Compute collocated weight using MM1 formula
        numerator = k * (C_Z0 - sk_var_chunk)
        denominator = C_Y0 - k**2 * (C_Z0 - sk_var_chunk)

        # Handle numerical issues
        collocated_weights = np.where(
            np.abs(denominator) < 1e-15,
            0.0,
            numerator / denominator
        )

        # MM1 Estimator: Z_SCCK = Z_SK * (1 - k*λ_Y0) + λ_Y0 * Y
        field[c_slice] = (
            sk_field_chunk * (1 - k * collocated_weights) +
            collocated_weights * secondary_chunk
        )

        # Handle variance if requested
        if return_var:
            scck_variance = sk_var_chunk * (1 - collocated_weights * k)
            # Note: Due to MM1 limitations, variance may actually be larger than SK
            krige_var[c_slice] = np.maximum(0.0, scck_variance)

    def _summate_intrinsic(self, field, krige_var, c_slice, k_vec, sk_field_chunk,
                           sk_var_chunk, secondary_chunk, C_Z0, C_Y0, C_YZ0, return_var):
        """Implement Intrinsic (ICCK) algorithm."""
        # Compute SK weights by solving kriging system: λ_SK = A^{-1} × b
        krige_mat_inv = self._inv(self._krige_mat)
        # Shape: (n_cond, n_estimation_points)
        sk_weights = krige_mat_inv @ k_vec

        # Compute ICCK weights based on SK weights
        lambda_weights, mu_weights, lambda_Y0 = self._compute_icck_weights(
            sk_weights, C_Y0, C_YZ0
        )

        # Apply ICCK estimator reformulation
        # Since λ = λ_SK and μ = -(C_YZ0/C_Y0) × λ_SK, we can write:
        # Z_ICCK = Z_SK + μ^T × Y_conditioning + λ_Y0 × Y(x0)

        # Handle both single point and multiple points estimation
        if sk_weights.ndim == 1:
            # Single estimation point
            secondary_contribution = np.sum(
                mu_weights * self.secondary_cond_val)
        else:
            # Multiple estimation points (sk_weights is n_cond x n_points)
            secondary_contribution = np.sum(
                mu_weights * self.secondary_cond_val[:, None], axis=0
            )

        # Collocated contribution
        collocated_contribution = lambda_Y0 * secondary_chunk

        # Final ICCK estimate
        field[c_slice] = (
            sk_field_chunk + secondary_contribution + collocated_contribution
        )

        # Handle variance if requested
        if return_var:
            rho_squared = self._compute_correlation_coeff_squared(
                C_Z0, C_Y0, C_YZ0)
            icck_variance = self._compute_icck_variance(
                sk_var_chunk, rho_squared)

            # Store the ICCK variance for later restoration (base class will modify it)
            if not hasattr(self, '_icck_stored_variance'):
                self._icck_stored_variance = np.empty_like(krige_var)
            self._icck_stored_variance[c_slice] = icck_variance

            # Set the krige_var to match the base class expectation
            # Base class will do: final_var = max(sill - krige_var, 0)
            # We want: final_var = icck_variance
            # So: icck_variance = max(sill - krige_var, 0)
            # Therefore: krige_var = sill - icck_variance (when icck_variance <= sill)
            krige_var[c_slice] = self.model.sill - icck_variance

    def _compute_icck_weights(self, sk_weights, C_Y0, C_YZ0):
        """
        Compute ICCK weights based on SK solution.

        Parameters
        ----------
        sk_weights : numpy.ndarray
            Simple kriging weights (λ_SK)
        C_Y0, C_YZ0 : float
            Secondary and cross covariances at zero lag

        Returns
        -------
        tuple
            (λ, μ, λ_Y0) - ICCK weights
        """
        # λ = λ_SK (keep SK weights for primary)
        lambda_weights = sk_weights

        # Handle edge case where C_Y0 is zero
        if abs(C_Y0) < 1e-15:
            # If secondary variance is zero, no contribution from secondary
            mu_weights = np.zeros_like(sk_weights)
            lambda_Y0 = 0.0
        else:
            # μ = -(C_YZ0/C_Y0) × λ_SK
            mu_weights = -(C_YZ0 / C_Y0) * sk_weights

            # λ_Y0 = C_YZ0/C_Y0
            lambda_Y0 = C_YZ0 / C_Y0

        return lambda_weights, mu_weights, lambda_Y0

    def _compute_icck_variance(self, sk_variance, rho_squared):
        """
        Compute ICCK variance: σ²_ICCK = (1-ρ₀²) × σ²_SK.

        Parameters
        ----------
        sk_variance : float or numpy.ndarray
            Simple kriging variance
        rho_squared : float
            Squared correlation coefficient ρ₀²

        Returns
        -------
        float or numpy.ndarray
            ICCK variance (same shape as sk_variance)
        """
        # Edge case: perfect correlation |ρ₀|=1 (ρ₀² ≈ 1)
        if abs(rho_squared - 1.0) < 1e-15:
            # With perfect correlation, effective dimension drops and variance → 0
            # This is the degenerate case mentioned in the theory
            return np.zeros_like(sk_variance)

        # Edge case: SK variance is zero (σ²_SK = 0)
        # This means estimation location is perfectly interpolated by primaries
        # In this case, adding secondaries doesn't change the zero variance
        sk_var_zero = np.abs(sk_variance) < 1e-15
        if np.any(sk_var_zero):
            result = (1.0 - rho_squared) * sk_variance
            result = np.where(sk_var_zero, 0.0, result)
            return np.maximum(0.0, result)

        # Standard ICCK variance formula
        icck_variance = (1.0 - rho_squared) * sk_variance

        # Ensure non-negative variance
        return np.maximum(0.0, icck_variance)
