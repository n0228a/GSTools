"""
GStools subpackage providing cokriging methods.

.. currentmodule:: gstools.cokriging.methods

The following classes are provided

.. autosummary::
   SimpleCollocated
   IntrinsicCollocated
"""

import numpy as np
from gstools.cokriging.base import CollocatedCokriging

__all__ = ["SimpleCollocated", "IntrinsicCollocated"]


class SimpleCollocated(CollocatedCokriging):
    """
    Simple collocated cokriging.

    Simple collocated cokriging extends simple kriging by incorporating
    secondary variable data at the estimation location only.

    **Markov Model I (MM1) Assumption:**

    Assumes C_YZ(h) = ρ_YZ(0)·√(C_Z(h)·C_Y(h)) under MM1 where ρ_Y(h) = ρ_Z(h),
    meaning both variables share the same spatial correlation structure. This
    requires similar spatial correlation patterns between primary and secondary variables.

    **Known Limitation:**

    MM1 can produce variance inflation where σ²_SCCK > σ²_SK in some cases.
    For accurate variance estimation, use IntrinsicCollocated instead.

    **Estimator:**

    Z*_SCCK = Z*_SK·(1-k·λ_Y0) + λ_Y0·(Y(u0)-m_Y) + k·λ_Y0·m_Z

    where k = C_YZ(0)/C_Z(0) and λ_Y0 is computed from the MM1 formula.

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
        Mean value for simple kriging (primary variable mean m_Z). Default: 0.0
    secondary_mean : :class:`float`, optional
        Mean value of the secondary variable (m_Y).
        Required for simple collocated cokriging to properly handle
        the anomaly-space formulation: Y(u) - m_Y.
        Default: 0.0
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signature: f(x, [y, z, ...])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
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
        # Initialize using base class with simple collocated algorithm
        super().__init__(
            model=model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            algorithm="simple",
            mean=mean,
            secondary_mean=secondary_mean,
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )


class IntrinsicCollocated(CollocatedCokriging):
    """
    Intrinsic collocated cokriging.

    Intrinsic collocated cokriging extends simple kriging by incorporating
    secondary variable data at both the estimation location AND at all
    primary conditioning locations.

    **Markov Model I (MM1) Assumption:**

    Like SimpleCollocated, assumes C_YZ(h) = ρ_YZ(0)·√(C_Z(h)·C_Y(h)).

    **Advantage over SimpleCollocated:**

    Uses improved variance formula that eliminates MM1 variance inflation:
    σ²_ICCK = (1-ρ₀²)·σ²_SK ≤ σ²_SK

    where ρ₀² = C²_YZ(0)/(C_Y(0)·C_Z(0)) is the squared correlation at zero lag.

    **Trade-off:**

    Requires secondary data at all primary locations (not just at estimation point).
    Matrix size nearly doubles compared to SimpleCollocated.

    **ICCK Weights:**

    - λ = λ_SK (Simple Kriging weights for primaries)
    - μ = -(C_YZ(0)/C_Y(0))·λ_SK (secondary-at-primary adjustment)
    - λ_Y0 = C_YZ(0)/C_Y(0) (collocated weight)

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
        Mean value for simple kriging (primary variable mean m_Z). Default: 0.0
    secondary_mean : :class:`float`, optional
        Mean value of the secondary variable (m_Y).
        Required for intrinsic collocated cokriging to properly handle
        the anomaly-space formulation: Y(u) - m_Y.
        Default: 0.0
    normalizer : :any:`None` or :any:`Normalizer`, optional
        Normalizer to be applied to the input data to gain normality.
        The default is None.
    trend : :any:`None` or :class:`float` or :any:`callable`, optional
        A callable trend function. Should have the signature: f(x, [y, z, ...])
        This is used for detrended kriging, where the trended is subtracted
        from the conditions before kriging is applied.
        This can be used for regression kriging, where the trend function
        is determined by an external regression algorithm.
        If no normalizer is applied, this behaves equal to 'mean'.
        The default is None.
    exact : :class:`bool`, optional
        Whether the interpolator should reproduce the exact input values.
        If `False`, `cond_err` is interpreted as measurement error
        at the conditioning points and the result will be more smooth.
        Default: False
    cond_err : :class:`str`, :class :class:`float` or :class:`list`, optional
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
        # Initialize using base class with intrinsic algorithm
        super().__init__(
            model=model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            algorithm="intrinsic",
            secondary_cond_pos=secondary_cond_pos,
            secondary_cond_val=secondary_cond_val,
            mean=mean,
            secondary_mean=secondary_mean,
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )
