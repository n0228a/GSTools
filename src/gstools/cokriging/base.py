"""
GStools cokriging base classes.

.. currentmodule:: gstools.cokriging.base

Base Classes
^^^^^^^^^^^^
Base classes for multivariate spatial interpolation

.. autosummary::
   CoKrige

----
"""

import numpy as np
from gstools.krige.base import Krige


class CoKrige(Krige):
    """
    Base class for cokriging methods.
    
    Cokriging extends kriging to handle multiple spatially correlated variables.
    This base class provides common functionality for multivariate interpolation.
    
    Parameters
    ----------
    model : :any:`CovModel`
        Cross-covariance model for multivariate interpolation
    cond_pos : :class:`list`
        Primary variable condition positions
    cond_val : :class:`numpy.ndarray`
        Primary variable condition values
    **kwargs
        Additional arguments passed to Krige base class
    """
    
    def __init__(self, model, cond_pos, cond_val, **kwargs):
        super().__init__(
            model=model,
            cond_pos=cond_pos, 
            cond_val=cond_val,
            **kwargs
        )
    
    def _validate_secondary_data(self, pos_secondary, val_secondary):
        """Validate secondary variable data."""
        pos_secondary = np.asarray(pos_secondary)
        val_secondary = np.asarray(val_secondary)
        
        if pos_secondary.shape[-1] != len(val_secondary):
            raise ValueError("Secondary positions and values must have same number of points")
            
        return pos_secondary, val_secondary