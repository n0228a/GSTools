r"""
Conditioning to hard data
-------------------------

Real simulations must honour measured data (boreholes, samples). Direct
Sampling supports this through :meth:`~gstools.DirectSampling.set_condition`:
conditioning values are pinned into the grid before simulation and preserved
exactly, while the rest of the field is filled in around them.

This example adds one idea to the previous one: keep the same TI-based pattern
reproduction, but now force selected grid cells to match hard data exactly.
We start by loading the same packages as before.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import gstools as gs

###############################################################################
# We reuse the synthetic channel training image from the first example.

gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
ti_data = ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(int)
ti = gs.TrainingImage(ti_data, categorical=True)

###############################################################################
# Draw 40 unique "hard data" points on the simulation grid and read their
# facies from the TI. In a real study these would be field measurements; here
# we sample the TI so the conditioning data are consistent with the patterns.

rng = np.random.default_rng(0)
cond_idx = rng.choice(40 * 40, size=40, replace=False)
cond_x_idx, cond_y_idx = np.unravel_index(cond_idx, (40, 40))
cond_x = cond_x_idx.astype(float)
cond_y = cond_y_idx.astype(float)
cond_val = ti_data[cond_x_idx, cond_y_idx]

###############################################################################
# Set the conditioning data and simulate. ``set_condition`` snaps each point to
# the nearest grid node; because our points already lie on grid nodes, the
# values are honoured exactly at those cells.

ds = gs.DirectSampling(ti, n_neighbors=12, scan_fraction=0.3, threshold=0.0)
ds.set_condition([cond_x, cond_y], cond_val)
grid = [np.arange(40, dtype=float), np.arange(40, dtype=float)]
field = ds(grid, seed=7)

assert np.all(field[cond_x_idx, cond_y_idx] == cond_val)

###############################################################################
# Plot the training image next to the conditional realization. The markers show
# the hard data locations; every marker sits on a cell whose simulated facies
# matches the datum.

cmap = ListedColormap(["#d8c690", "#2f6f9f"])
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.imshow(ti_data, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax0.set_title("Training image")
ax1.imshow(field, cmap=cmap, origin="lower", vmin=0, vmax=1)
ax1.scatter(
    cond_y,
    cond_x,
    c=cond_val,
    cmap=cmap,
    edgecolors="white",
    linewidths=0.8,
    s=34,
    vmin=0,
    vmax=1,
)
ax1.set_title("Conditional DS realization (markers = hard data)")
fig.tight_layout()
