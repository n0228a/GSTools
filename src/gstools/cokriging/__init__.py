"""
GStools subpackage providing cokriging functionality.

.. currentmodule:: gstools.cokriging

Cokriging Classes
^^^^^^^^^^^^^^^^^
Classes for multivariate spatial interpolation

.. autosummary::
   SimpleCollocatedCoKrige

----
"""

from gstools.cokriging.methods import SimpleCollocatedCoKrige

__all__ = [
    "SimpleCollocatedCoKrige",
]
