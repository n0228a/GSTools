"""
GStools subpackage providing collocated cokriging.

.. currentmodule:: gstools.cokriging.base

The following classes are provided

.. autosummary::
   CollocatedCokriging
"""

import numpy as np
from gstools.krige.base import Krige

__all__ = ["CollocatedCokriging"]


class CollocatedCokriging(Krige):
    """
    Collocated cokriging.

    Collocated cokriging uses secondary data at the estimation location
    to improve the primary variable estimate. This implementation supports
    both Simple Collocated Cokriging (SCCK) using the MM1 algorithm
    and Intrinsic Collocated Cokriging (ICCK) using proportional covariances.

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
    secondary_mean : :class:`float`, optional
        Mean value of the secondary variable. Default: 0.0
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
        secondary_mean=0.0,
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

        self.secondary_mean = float(secondary_mean)

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
        Generate the collocated cokriging field.

        The field is saved as `self.field` and is also returned.
        The error variance is saved as `self.krige_var` and is also returned.

        Parameters
        ----------
        pos : :class:`list`
            the position tuple, containing main direction and transversal
            directions (x, [y, z])
        secondary_data : :class:`numpy.ndarray`
            Secondary variable values at the given positions.
        **kwargs
            Keyword arguments passed to Krige.__call__.

        Returns
        -------
        field : :class:`numpy.ndarray`
            the collocated cokriging field
        krige_var : :class:`numpy.ndarray`, optional
            the collocated cokriging error variance
            (if return_var is True)
        """
        if secondary_data is None:
            raise ValueError(
                "secondary_data required for collocated cokriging")

        user_return_var = kwargs.get('return_var', True)
        # always get variance for weight calculation
        kwargs_with_var = kwargs.copy()
        kwargs_with_var['return_var'] = True
        # get simple kriging results
        sk_field, sk_var = super().__call__(pos=pos, **kwargs_with_var)
        secondary_data = np.asarray(secondary_data, dtype=np.double)

        if self.algorithm == "MM1":
            cokriging_field, cokriging_var = self._apply_mm1_cokriging(
                sk_field, sk_var, secondary_data, user_return_var)
        elif self.algorithm == "intrinsic":
            # ICCK: secondary-at-primary correction applied in _summate
            collocated_contribution = self._lambda_Y0 * (
                secondary_data - self.secondary_mean)
            cokriging_field = sk_field + collocated_contribution

            # ICCK variance: σ²_ICCK = (1-ρ₀²) × σ²_SK
            if user_return_var:
                C_Z0, C_Y0, C_YZ0 = self._compute_covariances()
                # Compute ρ₀² with division-by-zero protection
                if C_Y0 * C_Z0 < 1e-15:
                    rho_squared = 0.0
                else:
                    rho_squared = (C_YZ0**2) / (C_Y0 * C_Z0)
                # sk_var is already in actual variance format (σ²)
                icck_var = (1.0 - rho_squared) * sk_var
                icck_var = np.maximum(0.0, icck_var)
                cokriging_var = icck_var
            else:
                cokriging_var = None
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        if user_return_var:
            return cokriging_field, cokriging_var
        return cokriging_field

    def _apply_mm1_cokriging(self, sk_field, sk_var, secondary_data, return_var):
        """Apply simple collocated cokriging (MM1 algorithm)."""
        C_Z0, C_Y0, C_YZ0 = self._compute_covariances()
        k = C_YZ0 / C_Z0

        # NOTE: sk_var from super().__call__() is already actual variance σ²
        # MM1 collocated weights: λ_Y0 = (k × σ²_SK) / (C_Y0 - k² × σ²_SK)
        numerator = k * sk_var
        denominator = C_Y0 - (k**2) * (C_Z0 - sk_var)
        collocated_weights = np.where(
            np.abs(denominator) < 1e-15,
            0.0,
            numerator / denominator
        )

        # MM1 estimator with mean correction
        scck_field = (
            sk_field * (1 - k * collocated_weights) +
            collocated_weights * (secondary_data - self.secondary_mean) +
            k * collocated_weights * self.mean
        )

        if return_var:
            # MM1 variance: σ²_SCCK = σ²_SK × (1 - k × λ_Y0)
            scck_variance = sk_var * (1 - collocated_weights * k)
            scck_variance = np.maximum(0.0, scck_variance)
        else:
            scck_variance = None
        return scck_field, scck_variance

    def _summate(self, field, krige_var, c_slice, k_vec, return_var):
        """Apply intrinsic collocated cokriging during kriging solve."""
        if self.algorithm == "MM1":
            super()._summate(field, krige_var, c_slice, k_vec, return_var)
            return

        elif self.algorithm == "intrinsic":
            # extract SK weights
            sk_weights = self._krige_mat @ k_vec
            C_Z0, C_Y0, C_YZ0 = self._compute_covariances()

            if abs(C_YZ0) < 1e-15:
                self._lambda_Y0 = 0.0
                self._secondary_at_primary = 0.0
                super()._summate(field, krige_var, c_slice, k_vec, return_var)
                return

            # ICCK weights (proportional assumption)
            lambda_weights = sk_weights[:self.cond_no]
            mu_weights = -(C_YZ0 / C_Y0) * lambda_weights
            lambda_Y0 = C_YZ0 / C_Y0

            # secondary-at-primary contribution
            secondary_residuals = self.secondary_cond_val - self.secondary_mean
            if sk_weights.ndim == 1:
                secondary_at_primary = np.sum(mu_weights * secondary_residuals)
            else:
                secondary_at_primary = np.sum(
                    mu_weights * secondary_residuals[:, None], axis=0)

            # store weights for __call__ method
            self._lambda_Y0 = lambda_Y0
            self._secondary_at_primary = secondary_at_primary

            # compute base SK field and apply secondary-at-primary correction
            super()._summate(field, krige_var, c_slice, k_vec, return_var)
            field[c_slice] += secondary_at_primary
            # NOTE: Variance is handled in __call__(), not here
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _compute_covariances(self):
        """Compute covariances at zero lag."""
        C_Z0 = self.model.sill
        C_Y0 = self.secondary_var
        C_YZ0 = self.cross_corr * np.sqrt(C_Z0 * C_Y0)
        return C_Z0, C_Y0, C_YZ0
