"""
GStools cokriging methods.

.. currentmodule:: gstools.cokriging.methods

Cokriging Methods
^^^^^^^^^^^^^^^^^
Methods for multivariate spatial interpolation

.. autosummary::
   SimpleCollocatedCoKrige

----
"""

import numpy as np
from gstools.cokriging.base import CoKrige


class SimpleCollocatedCoKrige(CoKrige):
    """
    Simple Collocated CoKriging (SCCK).

    SCCK extends simple kriging by incorporating a secondary variable that is
    exhaustively sampled. It uses Markov Model I to simplify the cross-covariance
    structure, assuming the secondary variable has the same spatial structure
    as the primary variable.

    Parameters
    ----------
    pos_z : :class:`list`
        tuple, containing the given primary variable positions (x, [y, z])
    val_z : :class:`numpy.ndarray`
        the values of the primary variable conditions
    pos_y : :class:`list`
        tuple, containing the given secondary variable positions (x, [y, z])
    val_y : :class:`numpy.ndarray`
        the values of the secondary variable conditions
    model : :any:`MarkovModel1`
        Markov Model 1 for cross-covariance modeling
    mm_type : :class:`str`, optional
        Markov model type. Currently only "MM1" supported. Default: "MM1"
    mean : :class:`float`, optional
        mean value used for simple kriging. Default: 0.0
    **kwargs
        Additional arguments passed to CoKrige base class

    Notes
    -----
    The SCCK system of equations is:
    Sum_alpha lambda_Z,alpha rho_z(u_alpha - u_beta) + lambda_Y0 rho_yz(u_beta - u_0) = rho_z(u_beta - u_0)
    Sum_alpha lambda_Z,alpha rho_yz(u_alpha - u_0) + lambda_Y0 = rho_yz(0)
    """

    def __init__(
        self,
        pos_z,
        val_z,
        pos_y,
        val_y,
        model,
        mm_type="MM1",
        mean=0.0,
        **kwargs
    ):
        from gstools.covmodel.models import MarkovModel1

        # validate model type
        if not isinstance(model, MarkovModel1):
            raise TypeError("model must be a MarkovModel1 instance")

        if mm_type != "MM1":
            raise ValueError("Currently only MM1 supported")

        # validate and store secondary data
        pos_y, val_y = self._validate_secondary_data(pos_y, val_y)
        self._pos_y = pos_y
        self._val_y = val_y
        self._mm_model = model
        self._mm_type = mm_type

        # initialize CoKrige base class with primary data
        super().__init__(
            model=model.base_model,  # use base model for primary variable
            cond_pos=pos_z,
            cond_val=val_z,
            mean=mean,
            **kwargs
        )

    @property
    def pos_y(self):
        """:class:`list`: The position tuple of the secondary conditions."""
        return self._pos_y

    @property
    def val_y(self):
        """:class:`list`: The values of the secondary conditions."""
        return self._val_y

    @property
    def mm_model(self):
        """:any:`MarkovModel1`: The Markov Model 1 for cross-covariance."""
        return self._mm_model

    @property
    def krige_size(self):
        """:class:`int`: Size of the SCCK kriging system."""
        # For compatibility with base class, report standard size
        # SCCK uses a custom solving approach
        return self.cond_no

    def _get_krige_mat(self):
        """
        SCCK requires position-dependent matrices, so this method
        returns the base primary-primary correlation matrix that will
        be used as a building block in the per-target system construction.
        """
        n = self.cond_no
        primary_dists = self._get_dists(self._krige_pos)
        res = self.mm_model.base_model.correlation(primary_dists)
        res[np.diag_indices(n)] += self.cond_err
        return self._inv(res)

    def _get_krige_vecs(self, pos, chunk_slice=(0, None), ext_drift=None, only_mean=False):
        """
        SCCK uses custom matrix solving, so this method is not used
        in the standard way. Kept for interface compatibility.
        """
        chunk_size = len(range(*chunk_slice))
        n = self.cond_no
        res = np.empty((n, chunk_size), dtype=np.double)
        
        if only_mean:
            res[:n, :] = 0.0
        else:
            target_dists = self._get_dists(self._krige_pos, pos, chunk_slice)
            res[:n, :] = self.mm_model.base_model.correlation(target_dists)
        
        return res

    def __call__(self, pos, secondary_values=None, chunk_size=None, only_mean=False, return_var=False, mesh_type="unstructured"):
        """
        Evaluate SCCK at given positions.
        
        Parameters
        ----------
        pos : array-like
            Target positions for estimation
        secondary_values : array-like
            Values of secondary variable at target positions. 
            Required for collocated cokriging.
        
        Returns
        -------
        estimates : numpy.ndarray
            SCCK estimates at target locations
        variances : numpy.ndarray, optional
            Kriging variances (if return_var=True)
        
        Notes
        -----
        SCCK requires solving a position-dependent system for each target
        location due to cross-correlation terms in the system matrix.
        """
        pos = self.model.isometrize(pos)
        if pos.ndim == 1:
            pos = pos.reshape(-1, 1)
        
        n_targets = pos.shape[1]
        n_cond = self.cond_no
        
        # Validate secondary values
        if secondary_values is None:
            raise ValueError("secondary_values must be provided for collocated cokriging. "
                           "These are the secondary variable values at target locations.")
        
        secondary_values = np.asarray(secondary_values)
        if len(secondary_values) != n_targets:
            raise ValueError(f"secondary_values length ({len(secondary_values)}) must match "
                           f"number of target positions ({n_targets})")
        
        # Get base correlation matrix components
        K_zz_inv = self._get_krige_mat()
        K_zz = np.linalg.inv(K_zz_inv)
        
        # Prepare result arrays
        result = np.zeros(n_targets)
        variance = np.zeros(n_targets) if return_var else None
        
        # Solve SCCK system for each target position
        for i in range(n_targets):
            target_pos = pos[:, i:i+1]
            y_at_target = secondary_values[i]
            
            # Build SCCK system matrix for this target
            A = np.zeros((n_cond + 1, n_cond + 1))
            
            # Upper-left: primary-primary correlations
            A[:n_cond, :n_cond] = K_zz
            
            # Cross-correlation terms (position-dependent)
            dists_to_target = self._get_dists(self._krige_pos, target_pos, (0, 1))
            cross_corr_to_target = self.mm_model.cross_correlogram(dists_to_target.flatten())
            
            A[:n_cond, n_cond] = cross_corr_to_target  # Right column
            A[n_cond, :n_cond] = cross_corr_to_target  # Bottom row
            A[n_cond, n_cond] = 1.0  # Constraint coefficient
            
            # Build RHS vector
            b = np.zeros(n_cond + 1)
            b[:n_cond] = self.mm_model.base_model.correlation(dists_to_target.flatten())
            b[n_cond] = self.mm_model.cross_corr
            
            # Solve SCCK system
            try:
                weights = np.linalg.solve(A, b)
                
                # SCCK estimate: Z*(u_0) = Sum lambda_alpha Z(u_alpha) + lambda_Y0 Y(u_0)
                estimate = (weights[:n_cond] @ (self.cond_val - self.mean) + 
                           weights[n_cond] * (y_at_target - np.mean(self.val_y)) + 
                           self.mean)
                
                result[i] = estimate
                
                # Kriging variance
                if return_var:
                    var_reduction = weights @ b
                    variance[i] = max(0.0, self.model.var * (1.0 - var_reduction))
                    
            except np.linalg.LinAlgError:
                # Fallback to mean if system is singular
                result[i] = self.mean
                if return_var:
                    variance[i] = self.model.var
        
        if return_var:
            return result, variance
        return result