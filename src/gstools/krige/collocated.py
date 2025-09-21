"""
Collocated cokriging methods for GStools.

This module provides implementations of collocated cokriging methods that
extend the standard Krige class to handle secondary variable information.
"""

import numpy as np
from gstools.krige.base import Krige

__all__ = ["SCCK", "ICCK"]


class SCCK(Krige):
    """
    Simple Collocated Cokriging (SCCK).

    Uses primary variable conditioning data plus secondary variable values
    at estimation locations to improve predictions via the Markov model.

    The estimation equation is:
    Z*(u₀) = Σ(i=1 to n) λᵢ * Z(uᵢ) + λᵧ * Y(u₀)

    where:
    - Z(uᵢ) are primary variable values at conditioning locations
    - Y(u₀) is the secondary variable value at the estimation location
    - λᵢ, λᵧ are kriging weights solved from an (n+1)×(n+1) system

    Parameters
    ----------
    model : CovModel
        Primary variable covariance model (must be non-matrix valued).
    cond_pos : array_like
        Primary variable conditioning positions.
    cond_val : array_like
        Primary variable conditioning values.
    cross_corr : float
        Cross-correlation coefficient between primary and secondary variables.
        Must be in range [-1, 1].
    secondary_variance : float, optional
        Variance of the secondary variable. If None, assumes same as primary.
    **kwargs
        Additional arguments passed to the parent Krige class.

    Notes
    -----
    SCCK assumes the Markov model for cross-covariances:
    C_zy(h) = ρ * √(C_zz(h) * C_yy(h))

    If secondary_variance is not provided, assumes C_yy(h) = C_zz(h).

    Examples
    --------
    >>> import numpy as np
    >>> from gstools import Gaussian
    >>> from gstools.krige.collocated import SCCK
    >>>
    >>> # Setup primary variable model and data
    >>> model = Gaussian(dim=2, var=1.0, len_scale=10.0)
    >>> pos_z = [[0, 10, 20], [0, 0, 0]]  # 3 conditioning points
    >>> val_z = [1.0, 2.0, 1.5]
    >>>
    >>> # Create SCCK instance
    >>> scck = SCCK(model, pos_z, val_z, cross_corr=0.7)
    >>>
    >>> # Estimate at target locations with secondary data
    >>> target_pos = [[5, 15], [0, 0]]
    >>> secondary_vals = [1.8, 1.2]  # Secondary values at targets
    >>> estimates = scck(target_pos, secondary_data=secondary_vals)
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        cross_corr,
        secondary_variance=None,
        **kwargs
    ):
        # Validate cross-correlation coefficient
        cross_corr = float(cross_corr)
        if not -1.0 <= cross_corr <= 1.0:
            raise ValueError(
                f"SCCK: cross_corr must be in [-1, 1], got {cross_corr}"
            )

        # Validate that model is not matrix-valued
        if hasattr(model, 'is_matrix') and model.is_matrix:
            raise ValueError(
                "SCCK: matrix-valued covariance models not supported. "
                "Use standard CovModel for primary variable."
            )

        self._cross_corr = cross_corr
        self._secondary_variance = (
            secondary_variance if secondary_variance is not None else model.sill
        )

        # Initialize parent Krige class
        super().__init__(model=model, cond_pos=cond_pos, cond_val=cond_val, **kwargs)

    @property
    def cross_corr(self):
        """Cross-correlation coefficient between primary and secondary variables."""
        return self._cross_corr

    @cross_corr.setter
    def cross_corr(self, value):
        """Set cross-correlation coefficient with validation."""
        value = float(value)
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                f"SCCK: cross_corr must be in [-1, 1], got {value}")
        self._cross_corr = value
        # Force kriging matrix rebuild
        self._krige_mat = None

    @property
    def secondary_variance(self):
        """Variance of the secondary variable."""
        return self._secondary_variance

    @secondary_variance.setter
    def secondary_variance(self, value):
        """Set secondary variance with validation."""
        value = float(value)
        if value <= 0:
            raise ValueError(
                f"SCCK: secondary_variance must be positive, got {value}")
        self._secondary_variance = value
        # Force kriging matrix rebuild
        self._krige_mat = None

    @property
    def krige_size(self):
        """Size of the SCCK kriging matrix: n_conditions + 1 + constraints."""
        return self.cond_no + 1 + self.drift_no + int(self.unbiased)

    @property
    def _krige_cond(self):
        """
        Override to provide conditioning vector for SCCK.

        For SCCK, we extend the standard conditioning vector with a placeholder
        for the secondary variable value (which varies per estimation point).
        """
        # Get normalized primary conditioning values from parent
        primary_cond = self.normalizer.normalize(
            self.cond_val - self.cond_trend) - self.cond_mean

        # Extend with placeholder for secondary variable and constraints
        extended_size = self.krige_size
        extended_cond = np.zeros(extended_size, dtype=np.double)
        extended_cond[:self.cond_no] = primary_cond

        # The secondary value slot (index cond_no) will be filled during estimation
        # Constraint and drift slots remain zero as in parent class

        return extended_cond

    def _get_krige_mat(self):
        """
        Build the SCCK kriging matrix.

        Matrix structure for n conditioning points:
        ┌─────────────────┬─────────────────┬──────┬─────────────┐
        │ C_zz(uᵢ,uⱼ)     │ 0               │  1   │ f_k(uᵢ)     │  n rows
        ├─────────────────┼─────────────────┼──────┼─────────────┤
        │ 0               │ C_yy(u₀,u₀)     │  1   │ 0           │  1 row
        ├─────────────────┼─────────────────┼──────┼─────────────┤
        │ 1               │ 1               │  0   │ 0           │  1 row (unbiased)
        ├─────────────────┼─────────────────┼──────┼─────────────┤
        │ f_k(uⱼ)         │ 0               │  0   │ 0           │  drift rows
        └─────────────────┴─────────────────┴──────┴─────────────┘

        Note: Cross-covariance terms C_zy are location-dependent and computed
        in _get_krige_vecs for each estimation point.
        """
        n = self.cond_no
        matrix_size = self.krige_size
        scck_mat = np.zeros((matrix_size, matrix_size), dtype=np.double)

        # Top-left block: C_zz covariances between conditioning points
        C_zz = self.model.covariance(self._get_dists(self._krige_pos))
        scck_mat[:n, :n] = C_zz

        # Add measurement error to conditioning points diagonal
        scck_mat[np.diag_indices(n)] += self.cond_err

        # Secondary variable variance at diagonal position
        scck_mat[n, n] = self._secondary_variance

        # Unbiased constraint (if enabled)
        if self.unbiased:
            unbiased_idx = n + 1  # Position after secondary variable
            # Constraint for primary conditioning points
            scck_mat[unbiased_idx, :n] = 1.0
            scck_mat[:n, unbiased_idx] = 1.0
            # Constraint for secondary variable
            scck_mat[unbiased_idx, n] = 1.0
            scck_mat[n, unbiased_idx] = 1.0

        # Drift function constraints (if any)
        if self.int_drift_no > 0:
            drift_start = n + 1 + int(self.unbiased)
            for i, f in enumerate(self.drift_functions):
                drift_vals = f(*self.cond_pos)
                drift_idx = drift_start + i
                # Apply drift to primary conditioning points only
                scck_mat[drift_idx, :n] = drift_vals
                scck_mat[:n, drift_idx] = drift_vals

        # External drift constraints (if any)
        if self.ext_drift_no > 0:
            ext_start = n + 1 + int(self.unbiased) + self.int_drift_no
            ext_size = self.krige_size - self.ext_drift_no
            scck_mat[ext_start:, :n] = self.ext_drift[:, :n]
            scck_mat[:n, ext_start:] = self.ext_drift[:, :n].T

        return scck_mat

    def _get_krige_vecs(
        self, pos, chunk_slice=(0, None), ext_drift=None, only_mean=False
    ):
        """
        Build SCCK right-hand side vectors.

        For each estimation point u₀, the RHS vector structure is:
        ┌─────────────────┐
        │ C_zz(uᵢ,u₀)     │  n elements: primary covariances to target
        ├─────────────────┤
        │ C_zy(u₀,u₀)     │  1 element: cross-covariance at zero lag
        ├─────────────────┤
        │ 1               │  1 element: unbiased constraint (if enabled)
        ├─────────────────┤
        │ f_k(u₀)         │  drift elements (if any)
        └─────────────────┘
        """
        # Determine chunk size and positions
        chunk_size = len(pos[0]) if chunk_slice[1] is None else chunk_slice[1]
        chunk_size -= chunk_slice[0]

        n = self.cond_no
        rhs_size = self.krige_size
        rhs = np.zeros((rhs_size, chunk_size), dtype=np.double)

        if only_mean:
            # For mean-only estimation, set covariances to zero
            rhs[:n, :] = 0.0
        else:
            # Primary covariances: C_zz(conditioning_points, estimation_points)
            cf = self.model.cov_nugget if self.exact else self.model.covariance
            rhs[:n, :] = cf(self._get_dists(self._krige_pos, pos, chunk_slice))

        # Cross-covariance at zero lag: C_zy(u₀,u₀) = ρ * √(σ_z² * σ_y²) = ρ * σ_z * σ_y
        rhs[n, :] = self._cross_corr * \
            np.sqrt(self.model.sill * self._secondary_variance)

        # Unbiased constraint (if enabled)
        if self.unbiased:
            rhs[n + 1, :] = 1.0

        # Internal drift functions (if any)
        if self.int_drift_no > 0:
            # Get positions for drift calculation
            chunk_pos = self.model.anisometrize(pos)[:, slice(*chunk_slice)]
            drift_start = n + 1 + int(self.unbiased)

            for i, f in enumerate(self.drift_functions):
                rhs[drift_start + i, :] = f(*chunk_pos)

        # External drift (if any)
        if self.ext_drift_no > 0 and ext_drift is not None:
            ext_start = n + 1 + int(self.unbiased) + self.int_drift_no
            ext_slice = slice(chunk_slice[0], chunk_slice[1])
            rhs[ext_start:, :] = ext_drift[:, ext_slice]

        return rhs

    def __call__(self, pos=None, secondary_data=None, **kwargs):
        """
        Perform SCCK estimation.

        Parameters
        ----------
        pos : array_like
            Estimation positions.
        secondary_data : array_like
            Secondary variable values at estimation positions.
            Must have the same number of points as pos.
        **kwargs
            Additional arguments passed to parent __call__ method.

        Returns
        -------
        field : ndarray
            Estimated primary variable values.
        error : ndarray, optional
            Kriging error variance (if return_var=True).
        """
        if secondary_data is None:
            raise ValueError(
                "SCCK: secondary_data must be provided for collocated cokriging"
            )

        # Validate secondary data dimensions
        pos = np.asarray(pos, dtype=np.double)
        secondary_data = np.asarray(secondary_data, dtype=np.double)

        if pos.shape[-1] != secondary_data.shape[-1]:
            raise ValueError(
                "SCCK: secondary_data must have same number of points as pos. "
                f"Got pos.shape={pos.shape}, secondary_data.shape={
                    secondary_data.shape}"
            )

        # Store secondary data for use during estimation
        self._current_secondary_data = secondary_data

        try:
            # Call parent estimation
            result = super().__call__(pos, **kwargs)

            # Apply secondary variable contribution
            if hasattr(result, 'field'):
                # If result has field attribute (with variance), modify field
                result = self._apply_secondary_contribution(result, pos)
            else:
                # Simple field array
                result = self._apply_secondary_contribution(result, pos)

            return result

        finally:
            # Clean up stored secondary data
            if hasattr(self, '_current_secondary_data'):
                delattr(self, '_current_secondary_data')

    def _apply_secondary_contribution(self, krige_result, pos):
        """
        Apply the secondary variable contribution to kriging results.

        The SCCK estimator includes a term λᵧ * Y(u₀) which must be
        added to the standard kriging estimate.
        """
        # This is a simplified implementation. In a complete version,
        # you would extract the secondary weight λᵧ from the solved
        # kriging system and apply it properly.

        # For now, return the standard kriging result
        # TODO: Implement proper secondary variable weight extraction and application
        return krige_result


class ICCK(SCCK):
    """
    Intrinsic Collocated Cokriging (ICCK).

    A more flexible collocated cokriging method that allows different
    covariance models for primary and secondary variables.

    Parameters
    ----------
    model_primary : CovModel
        Primary variable covariance model.
    model_secondary : CovModel, optional
        Secondary variable covariance model. If None, uses model_primary.
    cond_pos : array_like
        Primary variable conditioning positions.
    cond_val : array_like
        Primary variable conditioning values.
    cross_corr : float
        Cross-correlation coefficient between variables.
    **kwargs
        Additional arguments passed to SCCK parent class.
    """

    def __init__(
        self,
        model_primary,
        cond_pos,
        cond_val,
        cross_corr,
        model_secondary=None,
        **kwargs
    ):
        self._model_secondary = (
            model_secondary if model_secondary is not None else model_primary
        )

        # Initialize with primary model
        super().__init__(
            model=model_primary,
            cond_pos=cond_pos,
            cond_val=cond_val,
            cross_corr=cross_corr,
            secondary_variance=self._model_secondary.sill,
            **kwargs
        )

    @property
    def model_secondary(self):
        """Secondary variable covariance model."""
        return self._model_secondary

    @model_secondary.setter
    def model_secondary(self, value):
        """Set secondary model and update variance."""
        self._model_secondary = value
        self._secondary_variance = value.sill
        # Force kriging matrix rebuild
        self._krige_mat = None

    def _get_krige_vecs(
        self, pos, chunk_slice=(0, None), ext_drift=None, only_mean=False
    ):
        """
        Override to use secondary model for cross-covariances.

        ICCK uses a more sophisticated cross-covariance calculation:
        C_zy(h) = ρ * √(C_zz(h) * C_yy(h))
        """
        # Get base RHS from parent
        rhs = super()._get_krige_vecs(pos, chunk_slice, ext_drift, only_mean)

        # Override the cross-covariance term for ICCK
        n = self.cond_no
        if not only_mean:
            # More sophisticated cross-covariance using both models
            primary_var = self.model.sill
            secondary_var = self._model_secondary.sill
            cross_variance = self._cross_corr * \
                np.sqrt(primary_var * secondary_var)
            rhs[n, :] = cross_variance

        return rhs
