"""
GStools subpackage providing cokriging.

.. currentmodule:: gstools.cokriging

Cokriging Classes
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   CollocatedCokriging
   SCCK
   ICCK
"""

from gstools.cokriging.base import CollocatedCokriging
from gstools.cokriging.methods import SCCK, ICCK

__all__ = ["CollocatedCokriging", "SCCK", "ICCK"]
