"""
GStools subpackage providing cokriging methods.

.. currentmodule:: gstools.cokriging.methods

The following classes are provided

.. autosummary::
   SimpleCollocated
   IntrinsicCollocated
"""

import warnings
from gstools.cokriging.base import CollocatedCokriging
from gstools.cokriging.correlogram import Correlogram, MarkovModel1

__all__ = ["SimpleCollocated", "IntrinsicCollocated"]


class SimpleCollocated(CollocatedCokriging):
    """
    Simple collocated cokriging.

    Simple collocated cokriging extends simple kriging by incorporating
    secondary variable data at the estimation location only.

    **Cross-Covariance Model:**

    This class uses a :any:`Correlogram` object (typically :any:`MarkovModel1`)
    to define the spatial relationship between primary and secondary variables.

    **Known Limitation:**

    Simple collocated cokriging can produce variance inflation :math:`\\sigma^2_{\\text{SCCK}} > \\sigma^2_{\\text{SK}}`
    in some cases. For accurate variance estimation, use :any:`IntrinsicCollocated` instead.

    **Estimator:**

    The SCCK estimator is:

    .. math::
       Z^*_{\\text{SCCK}}(u_0) = Z^*_{\\text{SK}}(u_0) \\cdot (1 - k \\cdot \\lambda_{Y0})
       + \\lambda_{Y0} \\cdot (Y(u_0) - m_Y) + k \\cdot \\lambda_{Y0} \\cdot m_Z

    where:

    .. math::
       k = \\frac{C_{YZ}(0)}{C_Z(0)}

    and the collocated weight :math:`\\lambda_{Y0}` is location-dependent:

    .. math::
       \\lambda_{Y0}(u_0) = \\frac{k \\cdot \\sigma^2_{\\text{SK}}(u_0)}
       {C_Y(0) - k^2(C_Z(0) - \\sigma^2_{\\text{SK}}(u_0))}

    **Variance:**

    .. math::
       \\sigma^2_{\\text{SCCK}}(u_0) = \\sigma^2_{\\text{SK}}(u_0) \\cdot (1 - \\lambda_{Y0}(u_0) \\cdot k)

    Parameters
    ----------
    correlogram : :any:`Correlogram`
        Correlogram object defining the cross-covariance structure.
        Typically a :any:`MarkovModel1` instance.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the conditions (nan values will be ignored)
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

    Examples
    --------
    >>> import gstools as gs
    >>> import numpy as np
    >>>
    >>> # Define primary model and correlogram
    >>> model = gs.Gaussian(dim=1, var=0.5, len_scale=2.0)
    >>> correlogram = gs.MarkovModel1(
    ...     primary_model=model,
    ...     cross_corr=0.8,
    ...     secondary_var=1.5,
    ...     primary_mean=1.0,
    ...     secondary_mean=0.5
    ... )
    >>>
    >>> # Setup cokriging
    >>> cond_pos = [0.5, 2.1, 3.8]
    >>> cond_val = [0.8, 1.2, 1.8]
    >>> scck = gs.SimpleCollocated(correlogram, cond_pos, cond_val)
    >>>
    >>> # Interpolate
    >>> gridx = np.linspace(0.0, 5.0, 51)
    >>> secondary_data = np.ones(51) * 0.5  # secondary values at gridx
    >>> field = scck(gridx, secondary_data=secondary_data)

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
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type="pinv",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        # Check if correlogram is actually a Correlogram object
        if not isinstance(correlogram, Correlogram):
            raise TypeError(
                f"First argument must be a Correlogram instance. "
                f"Got {type(correlogram).__name__}. "
                f"For backward compatibility, use SimpleCollocated.from_parameters() instead."
            )

        # Initialize using base class with simple collocated algorithm
        super().__init__(
            correlogram=correlogram,
            cond_pos=cond_pos,
            cond_val=cond_val,
            algorithm="simple",
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )

    @classmethod
    def from_parameters(
        cls,
        model,
        cond_pos,
        cond_val,
        cross_corr,
        secondary_var,
        mean=0.0,
        secondary_mean=0.0,
        **kwargs
    ):
        """
        Create SimpleCollocated from individual parameters (backward compatible).

        .. deprecated:: 1.6
           Use :any:`MarkovModel1` directly instead. This method exists for
           backward compatibility and will be removed in a future version.

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
            Mean value for simple kriging (primary variable mean). Default: 0.0
        secondary_mean : :class:`float`, optional
            Mean value of the secondary variable. Default: 0.0
        **kwargs
            Additional keyword arguments passed to SimpleCollocated.

        Returns
        -------
        SimpleCollocated
            Instance of SimpleCollocated with MarkovModel1 correlogram.

        Examples
        --------
        >>> import gstools as gs
        >>> model = gs.Gaussian(dim=1, var=0.5, len_scale=2.0)
        >>> scck = gs.SimpleCollocated.from_parameters(
        ...     model, cond_pos=[0.5, 2.1], cond_val=[0.8, 1.2],
        ...     cross_corr=0.8, secondary_var=1.5,
        ...     mean=1.0, secondary_mean=0.5
        ... )
        """
        warnings.warn(
            "SimpleCollocated.from_parameters() is deprecated. "
            "Use MarkovModel1 directly:\n"
            "  correlogram = gs.MarkovModel1(primary_model=model, cross_corr=..., ...)\n"
            "  scck = gs.SimpleCollocated(correlogram, cond_pos, cond_val)",
            DeprecationWarning,
            stacklevel=2
        )

        correlogram = MarkovModel1(
            primary_model=model,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            primary_mean=mean,
            secondary_mean=secondary_mean
        )

        return cls(correlogram, cond_pos, cond_val, **kwargs)


class IntrinsicCollocated(CollocatedCokriging):
    """
    Intrinsic collocated cokriging.

    Intrinsic collocated cokriging extends simple kriging by incorporating
    secondary variable data at both the estimation location AND at all
    primary conditioning locations.

    **Cross-Covariance Model:**

    This class uses a :any:`Correlogram` object (typically :any:`MarkovModel1`)
    to define the spatial relationship between primary and secondary variables.

    **Advantage over SimpleCollocated:**

    Uses improved variance formula that eliminates variance inflation:

    .. math::
       \\sigma^2_{\\text{ICCK}}(u_0) = (1 - \\rho_0^2) \\cdot \\sigma^2_{\\text{SK}}(u_0)
       \\leq \\sigma^2_{\\text{SK}}(u_0)

    where:

    .. math::
       \\rho_0^2 = \\frac{C_{YZ}^2(0)}{C_Y(0) \\cdot C_Z(0)}

    is the squared correlation at zero lag.

    **Trade-off:**

    Requires secondary data at all primary locations (not just at estimation point).
    The kriging system is effectively doubled in size compared to :any:`SimpleCollocated`.

    **Estimator:**

    The ICCK estimator combines primary and secondary data:

    .. math::
       Z^*_{\\text{ICCK}}(u_0) = \\sum_{i=1}^{n} \\lambda_i Z(u_i)
       + \\sum_{i=1}^{n} \\mu_i Y(u_i) + \\lambda_{Y0} Y(u_0) + \\text{(mean terms)}

    **ICCK Weights:**

    .. math::
       \\lambda_i &= \\lambda^{\\text{SK}}_i \\quad \\text{(Simple Kriging weights for primaries)} \\\\
       \\mu_i &= -\\frac{C_{YZ}(0)}{C_Y(0)} \\cdot \\lambda^{\\text{SK}}_i \\quad \\text{(secondary-at-primary adjustment)} \\\\
       \\lambda_{Y0} &= \\frac{C_{YZ}(0)}{C_Y(0)} \\quad \\text{(collocated weight)}

    Parameters
    ----------
    correlogram : :any:`Correlogram`
        Correlogram object defining the cross-covariance structure.
        Typically a :any:`MarkovModel1` instance.
    cond_pos : :class:`list`
        tuple, containing the given condition positions (x, [y, z])
    cond_val : :class:`numpy.ndarray`
        the values of the primary variable conditions (nan values will be ignored)
    secondary_cond_pos : :class:`list`
        tuple, containing the secondary variable condition positions (x, [y, z])
    secondary_cond_val : :class:`numpy.ndarray`
        the values of the secondary variable conditions at primary locations
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

    Examples
    --------
    >>> import gstools as gs
    >>> import numpy as np
    >>>
    >>> # Define primary model and correlogram
    >>> model = gs.Gaussian(dim=1, var=0.5, len_scale=2.0)
    >>> correlogram = gs.MarkovModel1(
    ...     primary_model=model,
    ...     cross_corr=0.8,
    ...     secondary_var=1.5,
    ...     primary_mean=1.0,
    ...     secondary_mean=0.5
    ... )
    >>>
    >>> # Setup cokriging
    >>> cond_pos = [0.5, 2.1, 3.8]
    >>> cond_val = [0.8, 1.2, 1.8]
    >>> sec_at_primary = [0.4, 0.6, 0.7]
    >>> icck = gs.IntrinsicCollocated(
    ...     correlogram, cond_pos, cond_val,
    ...     secondary_cond_pos=cond_pos,
    ...     secondary_cond_val=sec_at_primary
    ... )
    >>>
    >>> # Interpolate
    >>> gridx = np.linspace(0.0, 5.0, 51)
    >>> secondary_data = np.ones(51) * 0.5
    >>> field = icck(gridx, secondary_data=secondary_data)

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
        secondary_cond_pos,
        secondary_cond_val,
        normalizer=None,
        trend=None,
        exact=False,
        cond_err="nugget",
        pseudo_inv=True,
        pseudo_inv_type="pinv",
        fit_normalizer=False,
        fit_variogram=False,
    ):
        # Check if correlogram is actually a Correlogram object
        if not isinstance(correlogram, Correlogram):
            raise TypeError(
                f"First argument must be a Correlogram instance. "
                f"Got {type(correlogram).__name__}. "
                f"For backward compatibility, use IntrinsicCollocated.from_parameters() instead."
            )

        # Initialize using base class with intrinsic algorithm
        super().__init__(
            correlogram=correlogram,
            cond_pos=cond_pos,
            cond_val=cond_val,
            algorithm="intrinsic",
            secondary_cond_pos=secondary_cond_pos,
            secondary_cond_val=secondary_cond_val,
            normalizer=normalizer,
            trend=trend,
            exact=exact,
            cond_err=cond_err,
            pseudo_inv=pseudo_inv,
            pseudo_inv_type=pseudo_inv_type,
            fit_normalizer=fit_normalizer,
            fit_variogram=fit_variogram,
        )

    @classmethod
    def from_parameters(
        cls,
        model,
        cond_pos,
        cond_val,
        secondary_cond_pos,
        secondary_cond_val,
        cross_corr,
        secondary_var,
        mean=0.0,
        secondary_mean=0.0,
        **kwargs
    ):
        """
        Create IntrinsicCollocated from individual parameters (backward compatible).

        .. deprecated:: 1.6
           Use :any:`MarkovModel1` directly instead. This method exists for
           backward compatibility and will be removed in a future version.

        Parameters
        ----------
        model : :any:`CovModel`
            Covariance model for the primary variable.
        cond_pos : :class:`list`
            tuple, containing the given condition positions (x, [y, z])
        cond_val : :class:`numpy.ndarray`
            the values of the primary variable conditions
        secondary_cond_pos : :class:`list`
            tuple, containing the secondary variable condition positions
        secondary_cond_val : :class:`numpy.ndarray`
            the values of the secondary variable conditions at primary locations
        cross_corr : :class:`float`
            Cross-correlation coefficient between primary and secondary variables
            at zero lag. Must be in [-1, 1].
        secondary_var : :class:`float`
            Variance of the secondary variable. Must be positive.
        mean : :class:`float`, optional
            Mean value for simple kriging (primary variable mean). Default: 0.0
        secondary_mean : :class:`float`, optional
            Mean value of the secondary variable. Default: 0.0
        **kwargs
            Additional keyword arguments passed to IntrinsicCollocated.

        Returns
        -------
        IntrinsicCollocated
            Instance of IntrinsicCollocated with MarkovModel1 correlogram.

        Examples
        --------
        >>> import gstools as gs
        >>> model = gs.Gaussian(dim=1, var=0.5, len_scale=2.0)
        >>> icck = gs.IntrinsicCollocated.from_parameters(
        ...     model, cond_pos=[0.5, 2.1], cond_val=[0.8, 1.2],
        ...     secondary_cond_pos=[0.5, 2.1], secondary_cond_val=[0.4, 0.6],
        ...     cross_corr=0.8, secondary_var=1.5,
        ...     mean=1.0, secondary_mean=0.5
        ... )
        """
        warnings.warn(
            "IntrinsicCollocated.from_parameters() is deprecated. "
            "Use MarkovModel1 directly:\n"
            "  correlogram = gs.MarkovModel1(primary_model=model, cross_corr=..., ...)\n"
            "  icck = gs.IntrinsicCollocated(correlogram, cond_pos, cond_val, "
            "secondary_cond_pos, secondary_cond_val)",
            DeprecationWarning,
            stacklevel=2
        )

        correlogram = MarkovModel1(
            primary_model=model,
            cross_corr=cross_corr,
            secondary_var=secondary_var,
            primary_mean=mean,
            secondary_mean=secondary_mean
        )

        return cls(
            correlogram, cond_pos, cond_val,
            secondary_cond_pos, secondary_cond_val,
            **kwargs
        )
