"""
GStools subpackage providing cokriging.

.. currentmodule:: gstools.cokriging

Cokriging Classes
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   CollocatedCokriging
   SimpleCollocated
   IntrinsicCollocated
"""

from gstools.cokriging.base import CollocatedCokriging
from gstools.cokriging.methods import SimpleCollocated, IntrinsicCollocated

__all__ = ["CollocatedCokriging", "SimpleCollocated", "IntrinsicCollocated"]
