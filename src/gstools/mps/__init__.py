"""
GStools subpackage for Multiple Point Statistics (MPS).

.. currentmodule:: gstools.mps

Multiple Point Statistics
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
   :toctree:

   DirectSampling
   TrainingImage
"""

from gstools.mps.DS import DirectSampling
from gstools.mps.TI import TrainingImage

__all__ = ["DirectSampling", "TrainingImage"]
