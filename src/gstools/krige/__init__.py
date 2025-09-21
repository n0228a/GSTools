"""
GStools subpackage providing kriging.

.. currentmodule:: gstools.krige

Kriging Classes
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   Krige
   Simple
   Ordinary
   Universal
   ExtDrift
   Detrended

Collocated Cokriging Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   SCCK
   ICCK
"""

from gstools.krige.base import Krige
from gstools.krige.methods import (
    Detrended,
    ExtDrift,
    Ordinary,
    Simple,
    Universal,
)
from gstools.krige.collocated import SCCK, ICCK

__all__ = ["Krige", "Simple", "Ordinary", "Universal",
           "ExtDrift", "Detrended", "SCCK", "ICCK"]
