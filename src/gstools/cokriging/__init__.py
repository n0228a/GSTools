"""
GStools subpackage providing cokriging.

.. currentmodule:: gstools.cokriging

Cokriging Classes
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   SCCK
   ICCK
"""

from gstools.cokriging.methods import SCCK, ICCK

__all__ = ["SCCK", "ICCK"]
