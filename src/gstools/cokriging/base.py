"""
GStools subpackage providing collocated cokriging.

.. currentmodule:: gstools.cokriging.base

The following classes are provided

.. autosummary::
   CollocatedCokriging
"""

import numpy as np

from gstools.cokriging.correlogram import Correlogram
from gstools.krige.base import Krige

__all__ = ["CollocatedCokriging"]


class CollocatedCokriging(Krige):
    """
    Collocated cokriging base class using Correlogram models.

    Collocated cokriging uses secondary data at the estimation location
    to improve the primary variable estimate. This implementation supports
    both Simple Collocated Cokriging and Intrinsic Collocated Cokriging.

    **Cross-Covariance Modeling:**

    This class uses a :any:`Correlogram` object to define the spatial
    relationship between primary and secondary variables. Different correlogram
    models (MM1, MM2, etc.) make different assumptions about cross-covariance.

    **Algorithm Selection:**

    - **Simple Collocated** ("simple"):
      Uses only collocated secondary at estimation point. Simpler but
      may show variance inflation :math:`\\sigma^2_{\\text{SCCK}} > \\sigma^2_{\\text{SK}}`.

    - **Intrinsic Collocated** ("intrinsic"):
      Uses collocated secondary plus secondary at all primary locations.
      Provides accurate variance: :math:`\\sigma^2_{\\text{ICCK}} = (1-\\rho_0^2) \\cdot \\sigma^2_{\\text{SK}} \\leq \\sigma^2_{\\text{SK}}`.

    Parameters
    ----------
    correlogram : :any:`Correlogram`
        Correlogram object defining the cross-covariance structure between
        primary and secondary variables (e.g., :any:`MarkovModel1`).
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the primary variable conditions (nan values will be ignored)
    algorithm : :class:`str`
        Cokriging algorithm to use. Either "simple" (SCCK) or "intrinsic" (ICCK).
    secondary_cond_pos : :class:`list`, optional
        tuple, containing secondary variable condition positions (only for ICCK)
    secondary_cond_val : :class:`numpy.ndarray`, optional
        values of secondary variable at primary locations (only for ICCK)
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
        Directional variogram fitting is triggered by setting
        any anisotropy factor of the model to anything unequal 1
        but the main axes of correlation are taken from the model
        rotation angles. If the model is a spatio-temporal latlon
        model, this will raise an error.
        This assumes the sill to be the data variance and with
        standard bins provided by the :any:`standard_bins` routine.
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
        correlogram,
        cond_pos,
        cond_val,
        algorithm,
        secondary_cond_pos=None,
        secondary_cond_val=None,
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type="pinv",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        # Validate correlogram
        if not isinstance(correlogram, Correlogram):
            raise TypeError(
                f"correlogram must be a Correlogram instance, got {type(correlogram)}"
            )
        self.correlogram = correlogram

        # validate algorithm parameter
        if algorithm not in ["simple", "intrinsic"]:
            raise ValueError(
                "algorithm must be 'simple' or 'intrinsic'")
        self.algorithm = algorithm

        # handle secondary conditioning data (required for intrinsic)
        if algorithm == "intrinsic":
            if secondary_cond_pos is None or secondary_cond_val is None:
                raise ValueError(
                    "secondary_cond_pos and secondary_cond_val required for ICCK"
                )
            self.secondary_cond_pos = secondary_cond_pos
            self.secondary_cond_val = np.asarray(
                secondary_cond_val, dtype=np.double)

            if len(self.secondary_cond_val) != len(cond_val):
                raise ValueError(
                    "secondary_cond_val must have same length as primary cond_val"
                )
        else:
            self.secondary_cond_pos = None
            self.secondary_cond_val = None

        # initialize as simple kriging (unbiased=False)
        super().__init__(
            model=correlogram.primary_model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            mean=correlogram.primary_mean,
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

        # apply algorithm-specific post-processing
        if self.algorithm == "simple":
            cokriging_field, cokriging_var = self._apply_simple_collocated(
                sk_field, sk_var, secondary_data, user_return_var)
        elif self.algorithm == "intrinsic":
            cokriging_field, cokriging_var = self._apply_intrinsic_collocated(
                sk_field, sk_var, secondary_data, user_return_var)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        if user_return_var:
            return cokriging_field, cokriging_var
        return cokriging_field

    def _apply_simple_collocated(self, sk_field, sk_var, secondary_data, return_var):
        """Apply simple collocated cokriging."""
        C_Z0, C_Y0, C_YZ0 = self._compute_covariances()
        k = C_YZ0 / C_Z0

        # compute collocated weight
        numerator = k * sk_var
        denominator = C_Y0 - (k**2) * (C_Z0 - sk_var)
        collocated_weights = np.where(
            np.abs(denominator) < 1e-15,
            0.0,
            numerator / denominator
        )

        # apply collocated cokriging estimator
        scck_field = (
            sk_field * (1 - k * collocated_weights) +
            collocated_weights * (secondary_data - self.correlogram.secondary_mean) +
            k * collocated_weights * self.mean
        )

        if return_var:
            # simple collocated variance
            scck_variance = sk_var * (1 - collocated_weights * k)
            scck_variance = np.maximum(0.0, scck_variance)
        else:
            scck_variance = None
        return scck_field, scck_variance

    def _apply_intrinsic_collocated(self, sk_field, sk_var, secondary_data, return_var):
        """
        Apply intrinsic collocated cokriging.

        Adds the collocated secondary contribution at estimation locations
        and computes ICCK variance.

        Note: The secondary-at-primary contribution is already added during
        the kriging solve in _summate().
        """
        # apply collocated secondary contribution
        collocated_contribution = self._lambda_Y0 * (
            secondary_data - self.correlogram.secondary_mean)
        icck_field = sk_field + collocated_contribution

        # compute intrinsic variance
        if return_var:
            C_Z0, C_Y0, C_YZ0 = self._compute_covariances()
            if C_Y0 * C_Z0 < 1e-15:
                rho_squared = 0.0
            else:
                rho_squared = (C_YZ0**2) / (C_Y0 * C_Z0)
            icck_var = (1.0 - rho_squared) * sk_var
            icck_var = np.maximum(0.0, icck_var)
        else:
            icck_var = None
        return icck_field, icck_var

    def _summate(self, field, krige_var, c_slice, k_vec, return_var):
        """Apply intrinsic collocated cokriging during kriging solve."""
        if self.algorithm == "simple":
            super()._summate(field, krige_var, c_slice, k_vec, return_var)
            return

        elif self.algorithm == "intrinsic":
            sk_weights = self._krige_mat @ k_vec
            C_Z0, C_Y0, C_YZ0 = self._compute_covariances()

            if abs(C_YZ0) < 1e-15:
                self._lambda_Y0 = 0.0
                self._secondary_at_primary = 0.0
                super()._summate(field, krige_var, c_slice, k_vec, return_var)
                return

            lambda_weights = sk_weights[:self.cond_no]
            mu_weights = -(C_YZ0 / C_Y0) * lambda_weights
            lambda_Y0 = C_YZ0 / C_Y0

            secondary_residuals = self.secondary_cond_val - self.correlogram.secondary_mean
            if sk_weights.ndim == 1:
                secondary_at_primary = np.sum(mu_weights * secondary_residuals)
            else:
                secondary_at_primary = np.sum(
                    mu_weights * secondary_residuals[:, None], axis=0)

            self._lambda_Y0 = lambda_Y0
            self._secondary_at_primary = secondary_at_primary

            super()._summate(field, krige_var, c_slice, k_vec, return_var)
            field[c_slice] += secondary_at_primary
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _compute_covariances(self):
        """
        Compute covariances at zero lag.

        Delegates to the correlogram object.
        """
        return self.correlogram.compute_covariances()
