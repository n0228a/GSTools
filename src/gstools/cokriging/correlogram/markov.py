"""
GStools subpackage providing Markov model correlograms.

.. currentmodule:: gstools.cokriging.correlogram.markov

The following classes are provided

.. autosummary::
   MarkovModel1
"""

import numpy as np

from gstools.cokriging.correlogram.base import Correlogram

__all__ = ["MarkovModel1"]


class MarkovModel1(Correlogram):
    """
    Markov Model I (MM1) correlogram for collocated cokriging.

    The Markov Model I assumes that the cross-covariance between primary
    and secondary variables follows the primary variable's spatial structure:

    .. math::
        C_{YZ}(h) = \\frac{C_{YZ}(0)}{C_Z(0)} \\cdot C_Z(h)

    where:
        - :math:`C_{YZ}(h)` is the cross-covariance at distance h
        - :math:`C_{YZ}(0)` is the cross-covariance at zero lag
        - :math:`C_Z(h)` is the primary variable's covariance at distance h
        - :math:`C_Z(0)` is the primary variable's variance

    **Key Assumption**: This implies that both variables share the same
    spatial correlation structure: :math:`\\rho_Y(h) = \\rho_Z(h)`.

    **When to Use**:
        - Primary variable has well-defined spatial structure
        - Secondary variable tracks primary's spatial patterns
        - Most common choice for collocated cokriging

    **Limitations**:
        - Assumes identical spatial ranges for both variables
        - May be suboptimal if secondary has different range/structure
        - For those cases, consider MM2 (future implementation)

    Parameters
    ----------
    primary_model : :any:`CovModel`
        Covariance model for the primary variable (Z). This defines the
        spatial structure that both variables are assumed to share.
    cross_corr : :class:`float`
        Cross-correlation coefficient :math:`\\rho_{YZ}(0)` at zero lag.
        Must be in [-1, 1]. Computed as:
        :math:`\\rho_{YZ}(0) = C_{YZ}(0) / \\sqrt{C_Y(0) \\cdot C_Z(0)}`
    secondary_var : :class:`float`
        Variance of the secondary variable :math:`C_Y(0)`. Must be positive.
    primary_mean : :class:`float`, optional
        Mean value of the primary variable :math:`m_Z`. Default: 0.0
    secondary_mean : :class:`float`, optional
        Mean value of the secondary variable :math:`m_Y`. Default: 0.0

    Attributes
    ----------
    primary_model : :any:`CovModel`
        The primary variable's covariance model.
    cross_corr : :class:`float`
        Cross-correlation at zero lag.
    secondary_var : :class:`float`
        Secondary variable variance.
    primary_mean : :class:`float`
        Primary variable mean.
    secondary_mean : :class:`float`
        Secondary variable mean.

    References
    ----------
    .. [Samson2020] Samson, M., & Deutsch, C. V. (2020). Collocated Cokriging.
       In J. L. Deutsch (Ed.), Geostatistics Lessons. Retrieved from
       http://geostatisticslessons.com/lessons/collocatedcokriging
    .. [Wackernagel2003] Wackernagel, H. Multivariate Geostatistics,
       Springer, Berlin, 2003.

    Examples
    --------
    >>> import gstools as gs
    >>> import numpy as np
    >>>
    >>> # Define primary model and MM1 correlogram
    >>> model = gs.Gaussian(dim=1, var=0.5, len_scale=2.0)
    >>> mm1 = gs.MarkovModel1(
    ...     primary_model=model,
    ...     cross_corr=0.8,
    ...     secondary_var=1.5,
    ...     primary_mean=1.0,
    ...     secondary_mean=0.5
    ... )
    >>>
    >>> # Compute covariances at zero lag
    >>> C_Z0, C_Y0, C_YZ0 = mm1.compute_covariances()
    >>> print(f"Primary variance: {C_Z0:.3f}")
    Primary variance: 0.500
    >>> print(f"Secondary variance: {C_Y0:.3f}")
    Secondary variance: 1.500
    >>> print(f"Cross-covariance at zero lag: {C_YZ0:.3f}")
    Cross-covariance at zero lag: 0.693
    >>>
    >>> # Compute cross-covariance at distance h=1.0
    >>> h = 1.0
    >>> C_YZ_h = mm1.cross_covariance(h)
    >>> print(f"Cross-covariance at h={h}: {C_YZ_h:.3f}")
    Cross-covariance at h=1.0: 0.531
    >>>
    >>> # Use with Simple Collocated Cokriging
    >>> cond_pos = [0.5, 2.1, 3.8]
    >>> cond_val = [0.8, 1.2, 1.8]
    >>> scck = gs.SimpleCollocated(mm1, cond_pos, cond_val)
    """

    def compute_covariances(self):
        """
        Compute covariances at zero lag using MM1 formula.

        Returns
        -------
        C_Z0 : :class:`float`
            Primary variable variance (sill of primary model).
        C_Y0 : :class:`float`
            Secondary variable variance (as specified).
        C_YZ0 : :class:`float`
            Cross-covariance at zero lag, computed as:
            :math:`C_{YZ}(0) = \\rho_{YZ}(0) \\cdot \\sqrt{C_Y(0) \\cdot C_Z(0)}`

        Notes
        -----
        The cross-covariance at zero lag is derived from the cross-correlation
        and the variances of both variables. This ensures consistency with
        the correlation coefficient definition.
        """
        C_Z0 = self.primary_model.sill
        C_Y0 = self.secondary_var
        C_YZ0 = self.cross_corr * np.sqrt(C_Z0 * C_Y0)
        return C_Z0, C_Y0, C_YZ0

    def cross_covariance(self, h):
        """
        Compute cross-covariance at distance h using MM1 formula.

        Parameters
        ----------
        h : :class:`float` or :class:`numpy.ndarray`
            Distance(s) at which to compute cross-covariance.

        Returns
        -------
        C_YZ_h : :class:`float` or :class:`numpy.ndarray`
            Cross-covariance at distance h, computed using MM1:
            :math:`C_{YZ}(h) = \\frac{C_{YZ}(0)}{C_Z(0)} \\cdot C_Z(h)`

        Notes
        -----
        The MM1 formula uses the primary variable's covariance function
        to model the cross-covariance. This assumes both variables have
        the same spatial correlation structure (same range, same shape).

        The ratio :math:`k = C_{YZ}(0) / C_Z(0)` acts as a scaling factor
        that relates the primary covariance to the cross-covariance.
        """
        C_Z0, C_Y0, C_YZ0 = self.compute_covariances()

        # Handle edge case: zero primary variance
        if C_Z0 < 1e-15:
            return np.zeros_like(h) if isinstance(h, np.ndarray) else 0.0

        # MM1 formula: C_YZ(h) = (C_YZ(0) / C_Z(0)) * C_Z(h)
        k = C_YZ0 / C_Z0
        C_Z_h = self.primary_model.covariance(h)
        return k * C_Z_h


# TODO: Future implementation
# class MarkovModel2(Correlogram):
#     """
#     Markov Model II (MM2) correlogram for collocated cokriging.
#
#     MM2 uses the secondary variable's spatial structure:
#         C_YZ(h) = (C_YZ(0) / C_Y(0)) * C_Y(h)
#
#     This is useful when the secondary variable has a more stable
#     or better-defined spatial structure than the primary variable.
#
#     Requires:
#         - secondary_model: CovModel for secondary variable
#
#     References:
#         - Samson & Deutsch (2020), Geostatistics Lessons
#     """
#     pass
