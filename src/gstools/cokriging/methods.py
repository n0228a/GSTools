"""
GStools subpackage providing cokriging methods.

.. currentmodule:: gstools.cokriging.methods

Cokriging Classes
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   SCCK
"""

import numpy as np
from gstools.krige.base import Krige

__all__ = ["SCCK"]


class SCCK(Krige):
    """
    Simple Collocated Cokriging (SCCK).

    SCCK extends simple kriging by using secondary variable information
    at estimation locations to improve predictions.

    Parameters
    ----------
    model : CovModel
        Primary variable covariance model.
    cond_pos : array_like
        Primary variable conditioning positions.
    cond_val : array_like
        Primary variable conditioning values.
    cross_corr : float
        Cross-correlation coefficient between primary and secondary variables.
    **kwargs
        Additional arguments passed to Krige base class.
    """

    def __init__(
        self,
        model,
        cond_pos,
        cond_val,
        cross_corr,
        **kwargs
    ):
        # Store cross-correlation
        self.cross_corr = float(cross_corr)
        if not -1.0 <= self.cross_corr <= 1.0:
            raise ValueError("cross_corr must be in [-1, 1]")

        # Initialize as Simple Kriging (unbiased=False)
        super().__init__(
            model=model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            unbiased=False,  # Simple kriging
            **kwargs
        )

    def __call__(self, pos=None, secondary_data=None, **kwargs):
        """
        Estimate using SCCK.

        Parameters
        ----------
        pos : array_like
            Estimation positions.
        secondary_data : array_like
            Secondary variable values at estimation positions.
        **kwargs
            Standard Krige parameters (return_var, chunk_size, etc.)
        """
        if secondary_data is None:
            raise ValueError("secondary_data required for SCCK")

        # Store data for _summate to access
        self._secondary_data = np.asarray(secondary_data)

        # Store preprocessed positions for _summate
        iso_pos, shape = self.pre_pos(
            pos, kwargs.get('mesh_type', 'unstructured'))
        self._current_positions = iso_pos

        try:
            # Call parent with standard Krige functionality
            return super().__call__(pos=pos, **kwargs)
        finally:
            # Clean up
            if hasattr(self, '_secondary_data'):
                delattr(self, '_secondary_data')
            if hasattr(self, '_current_positions'):
                delattr(self, '_current_positions')

    def _summate(self, field, krige_var, c_slice, k_vec, return_var):
        """
        Override the solving process for SCCK.

        This is where SCCK differs from standard kriging - we solve
        a (n+1) x (n+1) system for each point individually.
        """
        # Get indices for this chunk
        start_idx = c_slice.start if c_slice.start is not None else 0
        stop_idx = c_slice.stop if c_slice.stop is not None else len(
            self._secondary_data)

        # Solve for each point in chunk
        for i in range(start_idx, stop_idx):
            target_pos = self._current_positions[:, i]
            secondary_val = self._secondary_data[i]
            est, var = self._solve_scck_point(
                target_pos, secondary_val, return_var)
            field[i] = est
            if return_var:
                krige_var[i] = var

    def _solve_scck_point(self, target_pos, secondary_value, return_var=True):
        """
        Solve SCCK system for a single estimation point.

        Parameters
        ----------
        target_pos : array_like
            Target position for estimation.
        secondary_value : float
            Secondary variable value at target position.
        return_var : bool
            Whether to compute variance.

        Returns
        -------
        estimate : float
            SCCK estimate at target position.
        variance : float
            Kriging variance (if return_var=True).
        """
        n = self.cond_no

        # Build (n+1) × (n+1) SCCK matrix
        A = np.zeros((n + 1, n + 1))

        # Top-left: C_zz covariances between conditioning points
        C_zz = self.model.covariance(self._get_dists(self._krige_pos))
        A[:n, :n] = C_zz

        # Add measurement error to diagonal
        A[np.diag_indices(n)] += self.cond_err

        # Cross-covariances: C_zy from conditioning points to target
        target_dists = self._get_dists(
            self._krige_pos, target_pos.reshape(-1, 1))
        C_zy = self.cross_corr * self.model.covariance(target_dists.flatten())
        A[:n, n] = C_zy  # Right column
        A[n, :n] = C_zy  # Bottom row

        # Secondary variance at (n,n)
        A[n, n] = self.model.sill

        # Build RHS vector
        b = np.zeros(n + 1)
        b[:n] = self.model.covariance(target_dists.flatten())  # C_zz to target
        # Cross-covariance at zero lag
        b[n] = self.cross_corr * self.model.sill

        # Solve system
        weights = np.linalg.solve(A, b)

        # SCCK estimate: λ_z @ Z + λ_y * Y
        estimate = weights[:n] @ self.cond_val + weights[n] * secondary_value

        # Compute variance if requested
        variance = 0.0
        if return_var:
            variance = max(0.0, self.model.var * (1.0 - weights @ b))

        return estimate, variance
