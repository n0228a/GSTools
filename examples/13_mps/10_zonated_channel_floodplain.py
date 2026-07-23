r"""
Zonated Direct Sampling: channel belt in a floodplain matrix
--------------------------------------------------------------

Demonstrates :any:`Zone` / ``MPSModel(zones=...)`` together with one DS
post-processing pass: a meandering channel belt (coarse "stone" cobble
facies) cuts across a floodplain matrix (fine "mud_cracks" facies). Both
TIs are kept at native resolution and the channel is sized to their native
grain scale.
"""

import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import gstools as gs

# 1. Training images: channel (stone) and floodplain (mud_cracks)
STONE_URL = (
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/"
    "master/stone.tiff"
)
STONE_CACHE = "stone.tiff"
if not os.path.exists(STONE_CACHE):
    urllib.request.urlretrieve(STONE_URL, STONE_CACHE)
stone_data = np.array(Image.open(STONE_CACHE)).astype(float)[:200, :200]

MUD_URL = (
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/"
    "master/mud_cracks.tiff"
)
MUD_CACHE = "mud_cracks.tiff"
if not os.path.exists(MUD_CACHE):
    urllib.request.urlretrieve(MUD_URL, MUD_CACHE)
mud_data = np.array(Image.open(MUD_CACHE)).astype(float)[:200, :200]

stone_ti = gs.TrainingImage(
    stone_data, categorical=False, distance="l2", n_neighbors=75
)
mud_ti = gs.TrainingImage(
    mud_data, categorical=False, distance="l2", n_neighbors=75
)

# 2. Zone geometry: a meandering channel belt across the floodplain
# Both TIs stay at native 200x200 resolution -- no resampling.
nx, ny = 700, 350
gx_idx, gy_idx = np.meshgrid(
    np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij"
)

WAVELENGTH = 350.0  # 2 full meander cycles across nx=700
AMPLITUDE = 80.0
HALF_WIDTH = 45.0  # 90px total width, ~3.5x the measured 25px median grain diameter

y_center = 175.0 + AMPLITUDE * np.sin(2.0 * np.pi * gx_idx / WAVELENGTH)
channel_mask = np.abs(gy_idx - y_center) <= HALF_WIDTH

# 3. Model + simulation (no rotation, no scale -- nonstationarity removed
# entirely: it collapsed neighbour lags in the channel and degraded the
# floodplain fidelity even at scale=1, see design doc)
# threshold/scan_fraction kept tight -- an empirical A/B test (loose 0.1 vs
# tight 0.01 threshold on a small zoned grid) showed 0.1 visibly degrades
# reproduction fidelity on BOTH sides of the zone boundary, not just one.
model = gs.MPSModel(
    mud_ti,
    zones=[gs.Zone(stone_ti, where=channel_mask)],
    scan_fraction=0.4,
    threshold=0.01,
    post_processing=1,
    post_processing_factor=2.0,
)
ds = gs.DirectSampling(model)

x = np.arange(nx, dtype=float)
y = np.arange(ny, dtype=float)
print(f"Simulating zonated channel/floodplain field ({nx}x{ny})...")
field = ds([x, y], seed=1, num_threads=200)

# 4. Verification: output values are a subset of each zone's TI values
# (CLAUDE.md MPS validity criterion, checked per zone)
stone_values = set(np.unique(stone_data).tolist())
mud_values = set(np.unique(mud_data).tolist())
channel_out = set(np.unique(field[channel_mask]).tolist())
floodplain_out = set(np.unique(field[~channel_mask]).tolist())
assert channel_out <= stone_values, "channel output leaked non-stone values"
assert floodplain_out <= mud_values, "floodplain output leaked non-mud values"
print("Verified: channel output is a subset of the stone TI's value set.")
print("Verified: floodplain output is a subset of the mud_cracks TI's value set.")

# 5. Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axes.ravel()

ax1.imshow(stone_data.T, cmap="gray", origin="lower", vmin=0, vmax=1.0)
ax1.set_title("a) Channel TI (stone)")
ax1.axis("off")

ax2.imshow(mud_data.T, cmap="gray", origin="lower")
ax2.set_title("b) Floodplain TI (mud_cracks)")
ax2.axis("off")

ax3.imshow(channel_mask.T, cmap="gray", origin="lower")
x_line = np.arange(nx, dtype=float)
ax3.plot(
    x_line,
    175.0 + AMPLITUDE * np.sin(2.0 * np.pi * x_line / WAVELENGTH),
    "r--",
    lw=1,
)
ax3.set_title("c) Zone geometry (channel belt + centerline)")
ax3.axis("off")

im4 = ax4.imshow(field.T, cmap="gray", origin="lower")
ax4.set_title("d) Simulation")
ax4.axis("off")
plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

fig.tight_layout()
plt.savefig("zonated_channel_floodplain.png")
print("Saved zonated_channel_floodplain.png")
