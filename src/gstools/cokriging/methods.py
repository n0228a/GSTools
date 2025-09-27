"""
GStools subpackage providing cokriging methods.

.. currentmodule:: gstools.cokriging.methods

The following classes are provided

.. autosummary::
   SCCK
   ICCK
"""

import numpy as np
from gstools.krige.base import Krige

__all__ = ["SCCK", "ICCK"]


class SCCK(Krige):
    """
    Simple Collocated Cokriging using Markov Model I (MM1) algorithm.

    SCCK extends simple kriging by incorporating secondary variable information
    at estimation locations. The MM1 algorithm assumes a Markov-type
    coregionalization model where ρ_yz(h) = ρ_yz(0)ρ_z(h), enabling efficient
    reuse of simple kriging computations with collocated adjustments.

    The estimator follows the elegant form:
    Z_SCCK(x) = Z_SK(x) × (1 - k×λ_Y0) + λ_Y0 × Y(x)

    where k is the cross-covariance ratio and λ_Y0 is the collocated weight.

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance model for the primary variable.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
    cross_corr : :class:`float`
        Cross-correlation coefficient between primary and secondary variables
        at zero lag. Must be in [-1, 1].
    secondary_var : :class:`float`
        Variance of the secondary variable. Must be positive.
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
        self.cross_corr = float(cross_corr)
        if not -1.0 <= self.cross_corr <= 1.0:
            raise ValueError("cross_corr must be in [-1, 1]")

        self.secondary_var = float(secondary_var)
        if self.secondary_var <= 0:
            raise ValueError("secondary_var must be positive")

        # Initialize as Simple Kriging (unbiased=False)
        super().__init__(
            model=model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            mean=mean,
            unbiased=False,  # Simple kriging
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
        Estimate using SCCK with MM1 algorithm.

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
            SCCK estimated field values.
        krige_var : :class:`numpy.ndarray`, optional
            SCCK estimation variance (if return_var=True).
        """
        if secondary_data is None:
            raise ValueError("secondary_data required for SCCK")

        # Store secondary data for use in _summateed
        self._secondary_data = np.asarray(secondary_data, dtype=np.double)

        try:
            return super().__call__(pos=pos, **kwargs)
        finally:
            # Clean up temporary attribute
            if hasattr(self, '_secondary_data'):
                delattr(self, '_secondary_data')

    def _summate(self, field, krige_var, c_slice, k_vec, return_var):
        """Override to implement MM1 SCCK estimator."""
        # Handle trivial case where cross-correlation is zero
        if abs(self.cross_corr) < 1e-15:
            return super()._summate(field, krige_var, c_slice, k_vec, return_var)

        # Import at function level to avoid circular imports
        from gstools.krige.base import _calc_field_krige_and_variance

        # ALWAYS compute both SK field and variance (SCCK mathematical requirement)
        sk_field_chunk, sk_var_chunk = _calc_field_krige_and_variance(
            self._krige_mat, k_vec, self._krige_cond
        )
        # Apply MM1 transformation (single, consistent algorithm)
        secondary_chunk = self._secondary_data[c_slice]
        k = self._compute_k()
        collocated_weights = self._compute_collocated_weight(sk_var_chunk, k)
        # MM1 Estimator: Z_SCCK = Z_SK * (1 - k*λ_Y0) + λ_Y0 * Y
        field[c_slice] = (
            sk_field_chunk * (1 - k * collocated_weights) +
            collocated_weights * secondary_chunk
        )

        # Handle variance based on user request (harmonious with base class)
        if return_var:
            scck_variances = self._compute_scck_variance(sk_var_chunk, k)
            krige_var[c_slice] = scck_variances
        # If return_var=False, krige_var is None and we don't touch it

    def _compute_k(self):
        """Compute cross-covariance ratio k = C_YZ(0)/C_Z(0)."""
        cross_cov_zero = self.cross_corr * np.sqrt(
            self.model.sill * self.secondary_var
        )
        return cross_cov_zero / self.model.sill

    def _compute_collocated_weight(self, sk_variance, k):
        """
        Compute collocated weight using MM1 formula.

        Parameters
        ----------
        sk_variance : :class:`float` or :class:`numpy.ndarray`
            Simple kriging variance.
        k : :class:`float`
            Cross-covariance ratio.

        Returns
        -------
        :class:`float` or :class:`numpy.ndarray`
            Collocated weight (same shape as sk_variance).
        """
        numerator = k * (self.model.sill - sk_variance)
        denominator = (
            self.secondary_var - k**2 * (self.model.sill - sk_variance)
        )
        # Handle numerical issues
        return np.where(
            np.abs(denominator) < 1e-15,
            0.0,
            numerator / denominator
        )

    def _compute_scck_variance(self, sk_variance, k):
        """
        Compute SCCK variance using MM1 formula.

        Note: MM1 SCCK is known to suffer from variance inflation issues
        in geostatistics literature. The variance may be larger than
        simple kriging variance due to the simplified covariance structure.
        For better variance estimation, consider Intrinsic Collocated
        Cokriging (ICCK) with MM2 model.

        Parameters
        ----------
        sk_variance : :class:`float` or :class:`numpy.ndarray`
            Simple kriging variance.
        k : :class:`float`
            Cross-covariance ratio.

        Returns
        -------
        :class:`float` or :class:`numpy.ndarray`
            SCCK variance (same shape as sk_variance).
        """
        collocated_weights = self._compute_collocated_weight(sk_variance, k)
        scck_variance = sk_variance * (1 - collocated_weights * k)

        # Note: Due to MM1 limitations, variance may actually be larger than SK
        return np.maximum(0.0, scck_variance)


class ICCK(Krige):
    """
    Intrinsic Collocated Cokriging using improved variance estimation.

    ICCK builds on Simple Kriging (or Ordinary Kriging) solutions and provides
    improved variance estimation compared to SCCK. Unlike SCCK's MM1 approach,
    ICCK requires secondary data at all primary conditioning locations and uses
    the more accurate variance formula σ²_ICCK = (1-ρ₀²)σ²_SK.

    The ICCK weights are:
    - λ = λ_SK (keep Simple Kriging weights for primary variable)
    - μ = -(C_YZ0/C_Y0) × λ_SK (adjustment weights for secondary at primary locations)
    - λ_Y0 = C_YZ0/C_Y0 (collocated weight for secondary at estimation point)

    The ICCK variance eliminates the inflation issues seen in MM1:
    σ²_ICCK = (1-ρ₀²) × σ²_SK, where ρ₀² = C²_YZ0/(C_Y0×C_Z0)

    Parameters
    ----------
    model : :any:`CovModel`
        Covariance model for the primary variable.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the primary variable conditions (nan values will be ignored)
    secondary_cond_pos : :class:`list`
        tuple, containing the secondary variable condition positions (x, [y, z])
    secondary_cond_val : :class:`numpy.ndarray`
        the values of the secondary variable conditions at primary locations
    cross_corr : :class:`float`
        Cross-correlation coefficient between primary and secondary variables
        at zero lag. Must be in [-1, 1].
    secondary_var : :class:`float`
        Variance of the secondary variable. Must be positive.
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
        secondary_cond_pos,
        secondary_cond_val,
        cross_corr,
        secondary_var,
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
        self.cross_corr = float(cross_corr)
        if not -1.0 <= self.cross_corr <= 1.0:
            raise ValueError("cross_corr must be in [-1, 1]")

        self.secondary_var = float(secondary_var)
        if self.secondary_var <= 0:
            raise ValueError("secondary_var must be positive")

        # Store secondary conditioning data
        self.secondary_cond_pos = secondary_cond_pos
        self.secondary_cond_val = np.asarray(
            secondary_cond_val, dtype=np.double)

        # Validate that secondary data matches primary locations
        if len(self.secondary_cond_val) != len(cond_val):
            raise ValueError(
                "secondary_cond_val must have same length as primary cond_val"
            )

        # Initialize as Simple Kriging (unbiased=False)
        super().__init__(
            model=model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            mean=mean,
            unbiased=False,  # Simple kriging
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
        Estimate using ICCK.

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
            ICCK estimated field values.
        krige_var : :class:`numpy.ndarray`, optional
            ICCK estimation variance (if return_var=True).
        """
        if secondary_data is None:
            raise ValueError("secondary_data required for ICCK")

        # Store secondary data for use in _summate
        self._secondary_data = np.asarray(secondary_data, dtype=np.double)

        try:
            # Call parent class but handle variance post-processing differently
            result = super().__call__(pos=pos, **kwargs)

            # Fix variance post-processing: restore ICCK variance if computed
            if isinstance(result, tuple) and len(result) == 2 and hasattr(self, '_icck_stored_variance'):
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

    def _summate(self, field, krige_var, c_slice, k_vec, return_var):
        """Override to implement ICCK estimator."""
        # Import at function level to avoid circular imports
        from gstools.krige.base import _calc_field_krige_and_variance

        # Get covariances at zero lag
        C_Z0, C_Y0, C_YZ0 = self._compute_covariances()

        # Handle trivial case where cross-correlation is zero
        if abs(C_YZ0) < 1e-15:
            # ICCK reduces to SK when C_YZ0 = 0
            return super()._summate(field, krige_var, c_slice, k_vec, return_var)

        # Always compute both SK field and variance (required for ICCK)
        sk_field_chunk, sk_var_chunk = _calc_field_krige_and_variance(
            self._krige_mat, k_vec, self._krige_cond
        )

        # Get secondary data at estimation positions
        secondary_chunk = self._secondary_data[c_slice]

        # Compute SK weights by solving kriging system: λ_SK = A^{-1} × b
        # k_vec contains the RHS vector b (covariances from estimation to conditioning points)
        # _krige_mat contains the LHS matrix A (conditioning covariances)
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
        # The secondary contribution is: -(C_YZ0/C_Y0) × λ_SK^T × Y_conditioning

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
