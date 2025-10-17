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

Correlogram Models
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree:

   Correlogram
   MarkovModel1
"""

from gstools.cokriging.base import CollocatedCokriging
from gstools.cokriging.methods import IntrinsicCollocated, SimpleCollocated
from gstools.cokriging.correlogram import Correlogram, MarkovModel1

__all__ = [
    "CollocatedCokriging",
    "SimpleCollocated",
    "IntrinsicCollocated",
    "Correlogram",
    "MarkovModel1",
]
