r"""
A real training image: the Strebelle channels
----------------------------------------------

The previous examples used tiny synthetic training images. Here we use the
classic **Strebelle (2002) channelized fluvial training image**, a widely used
reference example for MPS, and condition the simulation on random hard data.

.. note::

    **Data source / license.** The training image is downloaded from
    `GeoDataSets <https://github.com/GeostatsGuy/GeoDataSets>`_ by Michael
    Pyrcz (GeostatsGuy), distributed under the **MIT license**. The underlying
    channel TI is due to Strebelle, S. (2002), *Conditional simulation of
    complex geological structures using multiple-point statistics*,
    Mathematical Geology, 34(1), 1-21.
"""

import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# Download the Strebelle TI once and keep it next to this example. The cached
# ``.npz`` file is ignored by Git.

TI_URL = (
    "https://raw.githubusercontent.com/GeostatsGuy/"
    "GeoDataSets/master/MPS_Training_image_and_Realizations_500.npz"
)
if "__file__" in globals():
    CACHE = Path(__file__).resolve().with_name("mps_strebelle.npz")
else:
    cwd = Path.cwd()
    CACHE = Path("mps_strebelle.npz")
    for candidate in (
        cwd / "mps_strebelle.npz",
        cwd / "examples" / "13_mps" / "mps_strebelle.npz",
        cwd.parent / "examples" / "13_mps" / "mps_strebelle.npz",
    ):
        if candidate.exists():
            CACHE = candidate
            break
if not CACHE.exists():
    urllib.request.urlretrieve(TI_URL, CACHE)

with np.load(CACHE) as data:
    ti_arr = data["array1"].astype(int)

ti = gs.TrainingImage(ti_arr, categorical=True)

###############################################################################
# Take unique conditioning points from the training image patterns. The grid is
# intentionally modest so the example remains suitable for the documentation
# gallery while still using the real 256x256 TI.

sg_size = 56
n_cond = 60
rng = np.random.default_rng(0)
cond_idx = rng.choice(sg_size * sg_size, size=n_cond, replace=False)
cond_x_idx, cond_y_idx = np.unravel_index(cond_idx, (sg_size, sg_size))
cond_x = cond_x_idx.astype(float)
cond_y = cond_y_idx.astype(float)
cond_val = ti_arr[cond_x_idx, cond_y_idx]

###############################################################################
# Simulate with DSBC-style parameters (best-candidate + partial scan).

ds = gs.DirectSampling(ti, n_neighbors=20, scan_fraction=0.08, threshold=0.0)
ds.set_condition([cond_x, cond_y], cond_val)
grid = [np.arange(sg_size, dtype=float), np.arange(sg_size, dtype=float)]
field = ds(grid, seed=42)

assert np.all(field[cond_x_idx, cond_y_idx] == cond_val)

###############################################################################
# Plot the training image crop next to the conditional realization.

cmap = ListedColormap(["#c9a96e", "#2b6cb0"])  # shale / sand
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.5))
ax0.imshow(
    ti_arr[:sg_size, :sg_size],
    cmap=cmap,
    origin="lower",
    vmin=0,
    vmax=1,
)
ax0.set_title("Training image (crop)")
ax1.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax1.scatter(
    cond_y,
    cond_x,
    c=cond_val,
    cmap=cmap,
    edgecolors="k",
    linewidths=0.5,
    s=18,
    vmin=0,
    vmax=1,
)
ax1.set_title("Conditional DS realization")
fig.tight_layout()
