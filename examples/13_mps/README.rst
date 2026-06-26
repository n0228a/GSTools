Multiple Point Statistics
=========================

Two-point geostatistics describes spatial structure with pairs of points:
covariance models, variograms, kriging, and spatial random fields. That is
often enough, but it is not a natural language for connected or curvilinear
patterns such as channels, fractures, lenses, or object-like facies bodies.

**Multiple Point Statistics (MPS)** uses a **training image (TI)** instead: a
2D or 3D structured array containing patterns considered representative of the
field to simulate. The TI may be categorical, such as integer facies codes, or
continuous, such as permeability, porosity, elevation, or another property.

Use MPS when the geometry of the pattern matters more than a variogram can
express. If a covariance model already describes the field well, :any:`SRF`
is usually simpler and faster; for categorical fields generated from
continuous Gaussian fields, :any:`PGS` may also be a better first choice.

Direct Sampling in GSTools
--------------------------

GSTools implements the **Direct Sampling (DS)** algorithm
(`Mariethoz et al., 2010 <https://doi.org/10.1029/2008WR007621>`_) with the
**Direct Sampling Best Candidate (DSBC)** parametrization recommended by
`Juda et al. (2022) <https://doi.org/10.1016/j.acags.2022.100091>`_.
The two main classes mirror the usual GSTools model/generator pattern:

* :any:`TrainingImage` stores the TI and the distance used to compare patterns.
* :any:`DirectSampling` generates a realization on a structured grid.

At each unsimulated grid node, DS gathers the already known neighboring values
around that node. This neighborhood is the *data event*. DS then searches the
TI for candidate neighborhoods with the same relative cell positions and
similar values; when a good candidate is found, the value at the candidate
center is copied into the simulation grid. In an unconditional simulation,
these known neighbors are values simulated earlier in the random path; hard
data simply add measured values that must be honored.

Three parameters control the quality/runtime tradeoff:

* ``n_neighbors`` (``n``): the maximum number of known neighbors in each data
  event. Larger values preserve richer patterns but cost more.
* ``scan_fraction`` (``f``): the fraction of the TI search window examined for
  each simulated node. Larger values search harder but run slower.
* ``threshold`` (``t``): an early-acceptance distance. Start with
  ``threshold=0.0`` (DSBC), which scans the requested fraction and takes the
  best candidate found. Tune positive thresholds only when this is not
  sufficient.

The examples form a short learning path:

* ``00``: minimal unconditional categorical simulation: use only the TI to
  generate a new field with similar patterns.
* ``01``: add hard conditioning data: keep the TI-based pattern reproduction,
  but force measured values to be honored at their grid locations.
* ``02``: switch to continuous variables and compare the ``"l1"``, ``"l2"``,
  and ``"variation"`` distance metrics.
* ``03``: apply the same workflow to the Strebelle channelized fluvial TI.

Examples
--------
