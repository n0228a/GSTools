r"""
3D categorical simulation with voxel and slice plots
----------------------------------------------------

Direct Sampling is not limited to 2D images. A training image can be any
structured array, including a 3D categorical volume. This example visualizes a
saved high-resolution 3D Direct Sampling result with voxel views and
orthogonal slice plots. The training image is synthetic rather than a published
3D benchmark, so the example stays self-contained while still using true 3D
data events.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import gstools as gs

example_dir = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
)
data_path = example_dir / "input" / "synthetic_3d_channel_ti.npz"
output_dir = example_dir / "output"
output_path = output_dir / "three_dimensional_channel_volume.png"


###############################################################################
# **Small display and plotting helpers.**
#
# The plotting helper is used only when regenerating the saved figure. Normal
# execution displays the bundled PNG at the end of the example.
def show_saved_figure(path):
    """Display a precomputed PNG in the gallery."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing precomputed figure: {path}. "
            "Set generate_output = True to create it."
        )
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image)
    ax.axis("off")
    fig.tight_layout(pad=0)


def plot_voxels(ax, volume, title):
    """Plot the active facies of a small 3D categorical volume."""
    active = volume.astype(bool)
    colors = np.empty(active.shape, dtype=object)
    colors[:] = "#00000000"
    colors[active] = "#1f8fbfcc"
    ax.voxels(
        active, facecolors=colors, edgecolor="#0c425c55", linewidth=0.025
    )
    ax.set_title(title, pad=8)
    ax.set_box_aspect(volume.shape)
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=24, azim=-55)
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])


def center_slices(volume):
    """Return orthogonal centre slices from a 3D array."""
    cx, cy, cz = (size // 2 for size in volume.shape)
    return [
        (volume[cx, :, :].T, f"x={cx}"),
        (volume[:, cy, :].T, f"y={cy}"),
        (volume[:, :, cz].T, f"z={cz}"),
    ]


def plot_slices(row_axes, volume, prefix, cmap):
    """Plot the three centre slices of a categorical volume."""
    for ax, (slc, title) in zip(row_axes, center_slices(volume)):
        ax.imshow(
            slc,
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{prefix} {title}")
        ax.set_xticks([])
        ax.set_yticks([])


def plot_3d_result(ti_data, field):
    """Plot voxel renderings and matching orthogonal slices."""
    cmap = ListedColormap(["#efe1b7", "#1f8fbf"])

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    layout = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.45, 1, 1],
        hspace=0.08,
        wspace=0.06,
    )

    ax_ti_3d = fig.add_subplot(layout[0, 0:3], projection="3d")
    ax_sim_3d = fig.add_subplot(layout[0, 3:6], projection="3d")
    plot_voxels(ax_ti_3d, ti_data, "Training image, facies 1")
    plot_voxels(ax_sim_3d, field, "Simulation, facies 1")

    ti_slice_axes = [
        fig.add_subplot(layout[1, 2 * i : 2 * i + 2]) for i in range(3)
    ]
    sim_slice_axes = [
        fig.add_subplot(layout[2, 2 * i : 2 * i + 2]) for i in range(3)
    ]
    plot_slices(ti_slice_axes, ti_data, "TI", cmap)
    plot_slices(sim_slice_axes, field, "Simulation", cmap)

    legend_handles = [
        Patch(facecolor="#efe1b7", label="0"),
        Patch(facecolor="#1f8fbf", label="1"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        title="Facies",
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    return fig


###############################################################################
# **Load the generated 3D training image.**
#
# The channel facies was generated once from two smooth surfaces so the TI
# contains connected structures in all three directions.
if not data_path.exists():
    raise FileNotFoundError(
        f"Missing bundled training image: {data_path}. "
        "Run this example from examples/13_mps or restore the input assets."
    )

with np.load(data_path) as data:
    ti_data = data["ti"].astype(int)

###############################################################################
# **Configure the 3D data event.**
#
# A larger neighborhood and partial TI scan improve the 3D continuity compared
# with a minimal gallery run. ``max_radius`` limits neighbor selection to a
# local 3D search radius. ``boundary="partial"`` allows data events near volume
# boundaries to use the available in-domain neighbors.
ti = gs.TrainingImage(
    ti_data,
    categorical=True,
    n_neighbors=100,
    max_radius=12,
)
model = gs.MPSModel(ti, scan_fraction=0.35, threshold=0.0, boundary="partial")
ds = gs.DirectSampling(model)

grid_size = 28
grid = [np.arange(grid_size, dtype=float) for _ in range(3)]

###############################################################################
# **Run Direct Sampling, or reuse the saved result.**
#
# The setup above always runs so the example remains readable. The simulation
# runs only when ``generate_output`` is set to ``True``. The default is
# ``False`` so normal execution displays the saved figure.
generate_output = False
if generate_output:
    output_dir.mkdir(exist_ok=True)
    print(f"Simulating 3D categorical field ({grid_size}^3 nodes)...")
    field = ds(grid, seed=2, num_threads=4)
    fig = plot_3d_result(ti_data, field)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved {output_path}.")
    plt.close(fig)

###############################################################################
# **Display the saved 3D result.**
#
# In normal use this loads the bundled PNG. When ``generate_output=True``, it
# displays the PNG that was just refreshed.
show_saved_figure(output_path)
