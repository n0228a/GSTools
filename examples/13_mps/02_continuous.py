r"""
Continuous variables and distance metrics
-----------------------------------------

Direct Sampling is not limited to categorical facies: with ``categorical=False``
it simulates continuous variables (permeability, porosity, elevation, among other continuous properties).
The distance metric then selects how two continuous neighborhoods are compared:

* ``"l1"``: Manhattan distance on the raw values.
* ``"l2"``: Euclidean distance on the raw values.
* ``"variation"``: compares relative variations, so similar local shapes can
  match even when the local mean shifts.

We keep ``threshold=0.0`` for all three runs. This is the DSBC setting: DS
scans the requested fraction of the training image and takes the best candidate
found. That keeps this first continuous example focused on the metric choice
instead of threshold tuning.

No conditioning data are used here. Example ``01`` already introduced hard
conditioning, so this page isolates one new concept: how continuous data events
are compared. Example ``03`` combines the workflow with conditioning again on a
real training image.
"""

import matplotlib.pyplot as plt
import numpy as np

import gstools as gs

###############################################################################
# A smooth, continuous synthetic training image with connected high-value bands.

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = np.sin(gx / 6.0) * np.cos(gy / 8.0)

###############################################################################
# Simulate the same grid three times, changing only the distance metric.

grid = [np.arange(32, dtype=float), np.arange(32, dtype=float)]
metrics = [
    ("l1", "Manhattan (L1)"),
    ("l2", "Euclidean (L2)"),
    ("variation", "Variation"),
]
fields = {}
for distance, label in metrics:
    ti = gs.TrainingImage(ti_data, categorical=False, distance=distance)
    ds = gs.DirectSampling(
        ti, n_neighbors=12, scan_fraction=0.3, threshold=0.0
    )
    fields[label] = ds(grid, seed=3)

###############################################################################
# Plot. A shared colour scale keeps the comparison visually fair.

arrays = [ti_data, *fields.values()]
vmin = min(arr.min() for arr in arrays)
vmax = max(arr.max() for arr in arrays)

fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes = axes.ravel()
im = axes[0].imshow(
    ti_data,
    cmap="RdBu_r",
    origin="lower",
    vmin=vmin,
    vmax=vmax,
)
axes[0].set_title("Training image")
for ax, (label, field) in zip(axes[1:], fields.items()):
    ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(label)
fig.colorbar(im, ax=axes, shrink=0.8)
