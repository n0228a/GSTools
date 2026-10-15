#!/usr/bin/env python
"""Generate DSBC tuning candidates for the 3D MPS gallery examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import gstools as gs

EXAMPLE_DIR = ROOT / "examples" / "13_mps"
INPUT_DIR = EXAMPLE_DIR / "input"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "output" / "tuning_3d_dsbc"

FOLD_COLORS = ["#d8c69f", "#1d76a8"]
FOLD_LABELS = ["0 background facies", "1 folded facies"]
FOLD_CMAP = ListedColormap(FOLD_COLORS)

FLUVSIM_COLORS = ["#d8c69f", "#1f78b4", "#74a661", "#b85c38", "#7b61a8"]
FLUVSIM_LABELS = [
    "0 background",
    "1 channel facies",
    "2 margin facies",
    "3 secondary facies",
    "4 rare facies",
]
FLUVSIM_CMAP = ListedColormap(FLUVSIM_COLORS)

FOLD_SHAPE = (48, 48, 40)
FLUVSIM_SHAPE = (48, 48, 32)
THRESHOLD = 0.0

FOLD_SEEDS = (10, 17, 24)
FLUVSIM_SEEDS = (11, 18, 25)

FOLD_CONFIGS = (
    (48, 14, 0.00075),
    (64, 14, 0.001),
    (80, 18, 0.001),
    (96, 18, 0.001),
    (64, 22, 0.0015),
    (80, 22, 0.0015),
    (96, 22, 0.0015),
    (80, 26, 0.002),
    (96, 26, 0.002),
    (128, 28, 0.0025),
)

FLUVSIM_CONFIGS = (
    (
        "light_channel",
        48,
        14,
        0.00075,
        1.5,
        ((0, 40), (1, 110), (2, 30), (3, 20), (4, 4)),
    ),
    (
        "light_balanced",
        64,
        14,
        0.001,
        1.5,
        ((0, 45), (1, 120), (2, 35), (3, 25), (4, 5)),
    ),
    (
        "channel_margin",
        80,
        18,
        0.001,
        2.0,
        ((0, 50), (1, 150), (2, 50), (3, 30), (4, 6)),
    ),
    (
        "channel_strong",
        96,
        18,
        0.001,
        2.0,
        ((0, 50), (1, 170), (2, 55), (3, 35), (4, 8)),
    ),
    (
        "moderate_channel",
        64,
        22,
        0.0015,
        2.5,
        ((0, 60), (1, 180), (2, 60), (3, 40), (4, 8)),
    ),
    (
        "moderate_balanced",
        80,
        22,
        0.0015,
        2.5,
        ((0, 70), (1, 190), (2, 70), (3, 50), (4, 10)),
    ),
    (
        "detailed_channel",
        96,
        22,
        0.0015,
        3.0,
        ((0, 70), (1, 210), (2, 75), (3, 55), (4, 10)),
    ),
    (
        "detailed_margin",
        80,
        26,
        0.002,
        3.0,
        ((0, 80), (1, 220), (2, 90), (3, 60), (4, 12)),
    ),
    (
        "high_search_channel",
        96,
        26,
        0.002,
        3.0,
        ((0, 80), (1, 240), (2, 90), (3, 70), (4, 12)),
    ),
    (
        "high_search_detailed",
        128,
        28,
        0.0025,
        3.5,
        ((0, 90), (1, 260), (2, 100), (3, 80), (4, 15)),
    ),
)

MANIFEST_FIELDS = (
    "timestamp",
    "example",
    "run_id",
    "profile",
    "seed",
    "shape",
    "n_neighbors",
    "max_radius",
    "scan_fraction",
    "threshold",
    "cond_weight",
    "conditioning_counts",
    "conditioning_total",
    "runtime_seconds",
    "facies_counts",
    "image_3d",
    "image_slices",
    "image_combined",
)


@dataclass(frozen=True)
class Candidate:
    """One tuning candidate."""

    example: str
    run_id: str
    profile: str
    seed: int
    shape: Tuple[int, int, int]
    n_neighbors: int
    max_radius: int
    scan_fraction: float
    cond_weight: float = 1.0
    cond_counts: Tuple[Tuple[int, int], ...] = ()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate DSBC tuning images for the 3D Fold and Fluvsim MPS "
            "examples."
        )
    )
    parser.add_argument(
        "--examples",
        choices=("fold", "fluvsim", "both"),
        default="both",
        help="Which example sweep to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where candidate images and manifests are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of candidates per selected example.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of candidate simulations to run concurrently.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="GSTools DirectSampling threads per candidate.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
        help="DPI used for saved PNG review images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run candidates even when all expected PNG files exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected candidates without running simulations.",
    )
    parser.add_argument(
        "--smoke-shape",
        default=None,
        help=(
            "Override simulation shape as XxYxZ for quick script validation. "
            "Do not use this for final tuning figures."
        ),
    )
    parser.add_argument(
        "--smoke-scan-fraction",
        type=float,
        default=None,
        help=(
            "Override scan_fraction for quick script validation. "
            "Do not use this for final tuning figures."
        ),
    )
    parser.add_argument(
        "--fast-plot",
        action="store_true",
        help="Use coarser 3D voxel decimation for faster review rendering.",
    )
    return parser.parse_args(argv)


def build_candidates(example_choice: str, limit: int = None) -> List[Candidate]:
    """Build the selected candidate list."""
    groups = []
    if example_choice in {"fold", "both"}:
        groups.append(("fold", build_fold_candidates()))
    if example_choice in {"fluvsim", "both"}:
        groups.append(("fluvsim", build_fluvsim_candidates()))

    candidates = []
    for _, group in groups:
        candidates.extend(group if limit is None else group[:limit])
    return candidates


def parse_shape_override(value: str = None):
    """Parse an optional XxYxZ shape override."""
    if value is None:
        return None
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError("--smoke-shape must use integer XxYxZ form") from exc
    if len(shape) != 3 or any(size < 4 for size in shape):
        raise ValueError("--smoke-shape must contain three sizes >= 4")
    return shape


def build_fold_candidates() -> List[Candidate]:
    """Build Fold candidates from documented DSBC tuning ranges."""
    candidates = []
    for cfg_idx, (n_neighbors, max_radius, scan_fraction) in enumerate(
        FOLD_CONFIGS, start=1
    ):
        for seed in FOLD_SEEDS:
            candidates.append(
                Candidate(
                    example="fold",
                    run_id=f"fold_c{cfg_idx:02d}_s{seed}",
                    profile=f"fold_c{cfg_idx:02d}",
                    seed=seed,
                    shape=FOLD_SHAPE,
                    n_neighbors=n_neighbors,
                    max_radius=max_radius,
                    scan_fraction=scan_fraction,
                )
            )
    return candidates


def build_fluvsim_candidates() -> List[Candidate]:
    """Build Fluvsim candidates with channel-weighted conditioning profiles."""
    candidates = []
    for cfg_idx, cfg in enumerate(FLUVSIM_CONFIGS, start=1):
        profile, n_neighbors, max_radius, scan_fraction, cond_weight, counts = cfg
        for seed in FLUVSIM_SEEDS:
            candidates.append(
                Candidate(
                    example="fluvsim",
                    run_id=f"fluvsim_c{cfg_idx:02d}_s{seed}",
                    profile=profile,
                    seed=seed,
                    shape=FLUVSIM_SHAPE,
                    n_neighbors=n_neighbors,
                    max_radius=max_radius,
                    scan_fraction=scan_fraction,
                    cond_weight=cond_weight,
                    cond_counts=counts,
                )
            )
    return candidates


def candidate_paths(
    output_dir: Path, candidate: Candidate
) -> Tuple[Path, Path, Path]:
    """Return the 3D, slice, and combined output paths."""
    example_dir = output_dir / candidate.example
    return (
        example_dir / f"{candidate.run_id}_3d.png",
        example_dir / f"{candidate.run_id}_slices.png",
        example_dir / f"{candidate.run_id}_combined.png",
    )


def is_finished(output_dir: Path, candidate: Candidate) -> bool:
    """Return True if all candidate PNGs exist and are non-empty."""
    return all(
        path.exists() and path.stat().st_size > 0
        for path in candidate_paths(output_dir, candidate)
    )


def load_training_image(example: str) -> np.ndarray:
    """Load the bundled training image for one example."""
    if example == "fold":
        path = INPUT_DIR / "gaia_unil_fold_categorical_3d.npz"
    elif example == "fluvsim":
        path = INPUT_DIR / "gaia_unil_fluvsim_5facies_crop.npz"
    else:
        raise ValueError(f"Unknown example: {example!r}")

    with np.load(path) as data:
        return data["facies"].astype(int)


def rescale_ti_nearest(ti_data: np.ndarray, shape: Tuple[int, int, int]):
    """Nearest-neighbor resampling used only to choose conditioning data."""
    idx = [
        np.linspace(0, ti_data.shape[axis] - 1, shape[axis])
        .round()
        .astype(int)
        for axis in range(3)
    ]
    return ti_data[np.ix_(*idx)]


def sample_conditioning_from_ti(
    ti_data: np.ndarray,
    shape: Tuple[int, int, int],
    counts: Tuple[Tuple[int, int], ...],
    seed: int,
):
    """Sample deterministic hard data from all requested facies."""
    target = rescale_ti_nearest(ti_data, shape)
    rng = np.random.default_rng(seed)
    cond_idx = []
    cond_val = []
    actual_counts = {}

    for facies, count in counts:
        coords = np.argwhere(target == facies)
        if len(coords) == 0:
            actual_counts[int(facies)] = 0
            continue
        actual_count = min(int(count), len(coords))
        selected = coords[
            rng.choice(len(coords), size=actual_count, replace=False)
        ]
        cond_idx.append(selected)
        cond_val.append(np.full(actual_count, facies, dtype=float))
        actual_counts[int(facies)] = actual_count

    if not cond_idx:
        return None, None, actual_counts

    return np.vstack(cond_idx), np.concatenate(cond_val), actual_counts


def run_candidate_task(task):
    """Run one candidate. This function is process-pool friendly."""
    candidate, output_dir, num_threads, dpi, overwrite, fast_plot = task
    paths = candidate_paths(output_dir, candidate)
    if not overwrite and all(path.exists() for path in paths):
        return None

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ti_data = load_training_image(candidate.example)

    ti = gs.TrainingImage(
        ti_data,
        categorical=True,
        n_neighbors=candidate.n_neighbors,
        max_radius=candidate.max_radius,
    )
    model = gs.MPSModel(
        ti,
        scan_fraction=candidate.scan_fraction,
        threshold=THRESHOLD,
        boundary="partial",
        cond_weight=candidate.cond_weight,
    )
    ds = gs.DirectSampling(model)

    actual_cond_counts = {}
    cond_idx = cond_val = None
    if candidate.cond_counts:
        cond_idx, cond_val, actual_cond_counts = sample_conditioning_from_ti(
            ti_data,
            candidate.shape,
            candidate.cond_counts,
            seed=candidate.seed,
        )
        if cond_idx is not None:
            ds.set_condition(
                [
                    cond_idx[:, 0].astype(float),
                    cond_idx[:, 1].astype(float),
                    cond_idx[:, 2].astype(float),
                ],
                cond_val,
            )

    grid = [np.arange(size, dtype=float) for size in candidate.shape]
    field = ds(grid, seed=candidate.seed, num_threads=num_threads).astype(int)
    runtime = time.perf_counter() - started

    assert field.shape == candidate.shape
    if candidate.example == "fold":
        assert set(np.unique(field)) <= {0, 1}
    else:
        unique_values = set(np.unique(field))
        expected_values = {0, 1, 2, 3, 4}
        assert unique_values <= expected_values
        if candidate.shape == FLUVSIM_SHAPE:
            assert unique_values == expected_values
        else:
            conditioned = {
                facies for facies, count in actual_cond_counts.items() if count
            }
            assert conditioned <= unique_values
    if cond_idx is not None:
        assert np.all(
            field[cond_idx[:, 0], cond_idx[:, 1], cond_idx[:, 2]] == cond_val
        )

    metadata = metadata_title(candidate, runtime, actual_cond_counts)
    image_3d, image_slices, image_combined = paths
    plot_3d_only(ti_data, field, candidate, metadata, fast_plot).savefig(
        image_3d, dpi=dpi, bbox_inches="tight"
    )
    plt.close("all")
    plot_slices_only(ti_data, field, candidate, metadata).savefig(
        image_slices, dpi=dpi, bbox_inches="tight"
    )
    plt.close("all")
    plot_combined(ti_data, field, candidate, metadata, fast_plot).savefig(
        image_combined, dpi=dpi, bbox_inches="tight"
    )
    plt.close("all")

    facies_counts = count_values(field)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "example": candidate.example,
        "run_id": candidate.run_id,
        "profile": candidate.profile,
        "seed": candidate.seed,
        "shape": shape_text(candidate.shape),
        "n_neighbors": candidate.n_neighbors,
        "max_radius": candidate.max_radius,
        "scan_fraction": candidate.scan_fraction,
        "threshold": THRESHOLD,
        "cond_weight": candidate.cond_weight,
        "conditioning_counts": json.dumps(
            actual_cond_counts or dict(candidate.cond_counts),
            sort_keys=True,
        ),
        "conditioning_total": sum(
            (actual_cond_counts or dict(candidate.cond_counts)).values()
        ),
        "runtime_seconds": round(runtime, 3),
        "facies_counts": json.dumps(facies_counts, sort_keys=True),
        "image_3d": repo_path(image_3d),
        "image_slices": repo_path(image_slices),
        "image_combined": repo_path(image_combined),
    }
    return row


def metadata_title(
    candidate: Candidate,
    runtime: float = None,
    cond_counts: Dict[int, int] = None,
) -> str:
    """Return compact metadata text for plot titles."""
    parts = [
        candidate.run_id,
        f"seed={candidate.seed}",
        f"shape={shape_text(candidate.shape)}",
        f"n={candidate.n_neighbors}",
        f"r={candidate.max_radius}",
        f"f={candidate.scan_fraction:g}",
        "t=0",
    ]
    if candidate.cond_counts:
        total = sum((cond_counts or dict(candidate.cond_counts)).values())
        parts.extend([f"cw={candidate.cond_weight:g}", f"cond={total}"])
    if runtime is not None:
        parts.append(f"{runtime:.1f}s")
    return " | ".join(parts)


def count_values(values: np.ndarray) -> Dict[int, int]:
    """Return facies counts as a plain dict."""
    unique, counts = np.unique(values, return_counts=True)
    return {int(value): int(count) for value, count in zip(unique, counts)}


def shape_text(shape: Tuple[int, int, int]) -> str:
    """Format a shape compactly for manifests and plot titles."""
    return "x".join(str(size) for size in shape)


def repo_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def decimate_volume(volume: np.ndarray, step: int) -> np.ndarray:
    """Return a smaller volume for Matplotlib voxel plotting."""
    return volume[::step, ::step, ::step]


def voxel_steps(example: str, fast_plot: bool) -> Tuple[int, int]:
    """Return TI and simulation voxel decimation steps."""
    if fast_plot:
        return (8, 4) if example == "fold" else (5, 3)
    return (5, 2) if example == "fold" else (3, 2)


def cutaway_mask(shape: Tuple[int, int, int]) -> np.ndarray:
    """Hide one corner so internal facies geometry stays visible."""
    ix, iy, iz = np.indices(shape)
    return (
        (ix > 0.52 * shape[0])
        & (iy < 0.48 * shape[1])
        & (iz > 0.42 * shape[2])
    )


def plot_binary_voxels(ax, volume: np.ndarray, title: str) -> None:
    """Plot a cutaway binary-facies voxel view."""
    active = (volume == 1) & ~cutaway_mask(volume.shape)
    colors = np.empty(volume.shape, dtype=object)
    colors[:] = "#00000000"
    colors[active] = FOLD_COLORS[1] + "cc"
    ax.voxels(
        active,
        facecolors=colors,
        edgecolor="#0b3f5755",
        linewidth=0.02,
    )
    style_3d_axis(ax, volume.shape, title)


def plot_multifacies_voxels(ax, volume: np.ndarray, title: str) -> None:
    """Plot a cutaway voxel view for all non-background facies."""
    active = (volume > 0) & ~cutaway_mask(volume.shape)
    colors = np.empty(volume.shape, dtype=object)
    colors[:] = "#00000000"
    for facies, color in enumerate(FLUVSIM_COLORS):
        if facies == 0:
            continue
        colors[(volume == facies) & active] = color + "d0"
    ax.voxels(
        active,
        facecolors=colors,
        edgecolor="#24313a44",
        linewidth=0.015,
    )
    style_3d_axis(ax, volume.shape, title)


def style_3d_axis(ax, shape: Tuple[int, int, int], title: str) -> None:
    """Apply the shared 3D axis style."""
    ax.set_title(title, pad=8)
    ax.set_box_aspect(shape)
    ax.set_axis_off()
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-48)
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.set_zlim(0, shape[2])


def center_slices(volume: np.ndarray):
    """Return orthogonal centre slices from a 3D array."""
    cx, cy, cz = (size // 2 for size in volume.shape)
    return [
        (volume[cx, :, :].T, f"x={cx}"),
        (volume[:, cy, :].T, f"y={cy}"),
        (volume[:, :, cz].T, f"z={cz}"),
    ]


def plot_slice_row(row_axes, volume, prefix, cmap, vmax):
    """Plot the three centre slices of a categorical volume."""
    for ax, (slc, title) in zip(row_axes, center_slices(volume)):
        ax.imshow(
            slc,
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(f"{prefix} {title}")
        ax.set_xticks([])
        ax.set_yticks([])


def plot_3d_only(
    ti_data: np.ndarray,
    field: np.ndarray,
    candidate: Candidate,
    metadata: str,
    fast_plot: bool,
):
    """Build a 3D-only review figure."""
    ti_step, sim_step = voxel_steps(candidate.example, fast_plot)
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    if candidate.example == "fold":
        plot_binary_voxels(
            axes[0],
            decimate_volume(ti_data, step=ti_step),
            "Training image, facies 1",
        )
        plot_binary_voxels(
            axes[1],
            decimate_volume(field, step=sim_step),
            "Simulation, facies 1",
        )
        add_legend(fig, FOLD_COLORS, FOLD_LABELS, ncol=2)
    else:
        plot_multifacies_voxels(
            axes[0],
            decimate_volume(ti_data, step=ti_step),
            "Training image, non-background facies",
        )
        plot_multifacies_voxels(
            axes[1],
            decimate_volume(field, step=sim_step),
            "Simulation, non-background facies",
        )
        add_legend(fig, FLUVSIM_COLORS, FLUVSIM_LABELS, ncol=5)
    fig.suptitle(metadata, fontsize=11)
    return fig


def plot_slices_only(
    ti_data: np.ndarray,
    field: np.ndarray,
    candidate: Candidate,
    metadata: str,
):
    """Build a slice-only review figure."""
    cmap, labels, colors, vmax = style_for_example(candidate.example)
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    layout = fig.add_gridspec(2, 6, hspace=0.08, wspace=0.06)
    ti_axes = [fig.add_subplot(layout[0, 2 * i : 2 * i + 2]) for i in range(3)]
    sim_axes = [
        fig.add_subplot(layout[1, 2 * i : 2 * i + 2]) for i in range(3)
    ]
    plot_slice_row(ti_axes, ti_data, "TI", cmap, vmax)
    plot_slice_row(sim_axes, field, "Simulation", cmap, vmax)
    add_legend(fig, colors, labels, ncol=len(labels))
    fig.suptitle(metadata, fontsize=11)
    return fig


def plot_combined(
    ti_data: np.ndarray,
    field: np.ndarray,
    candidate: Candidate,
    metadata: str,
    fast_plot: bool,
):
    """Build a combined 3D and slice review sheet."""
    ti_step, sim_step = voxel_steps(candidate.example, fast_plot)
    cmap, labels, colors, vmax = style_for_example(candidate.example)
    fig = plt.figure(figsize=(13, 10.5), constrained_layout=True)
    layout = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.35, 1, 1],
        hspace=0.08,
        wspace=0.06,
    )
    ax_ti_3d = fig.add_subplot(layout[0, 0:3], projection="3d")
    ax_sim_3d = fig.add_subplot(layout[0, 3:6], projection="3d")
    if candidate.example == "fold":
        plot_binary_voxels(
            ax_ti_3d,
            decimate_volume(ti_data, step=ti_step),
            "Training image, facies 1",
        )
        plot_binary_voxels(
            ax_sim_3d,
            decimate_volume(field, step=sim_step),
            "Simulation, facies 1",
        )
    else:
        plot_multifacies_voxels(
            ax_ti_3d,
            decimate_volume(ti_data, step=ti_step),
            "Training image, non-background facies",
        )
        plot_multifacies_voxels(
            ax_sim_3d,
            decimate_volume(field, step=sim_step),
            "Simulation, non-background facies",
        )

    ti_axes = [fig.add_subplot(layout[1, 2 * i : 2 * i + 2]) for i in range(3)]
    sim_axes = [
        fig.add_subplot(layout[2, 2 * i : 2 * i + 2]) for i in range(3)
    ]
    plot_slice_row(ti_axes, ti_data, "TI", cmap, vmax)
    plot_slice_row(sim_axes, field, "Simulation", cmap, vmax)
    add_legend(fig, colors, labels, ncol=len(labels))
    fig.suptitle(metadata, fontsize=11)
    return fig


def style_for_example(example: str):
    """Return cmap, labels, colors, and vmax for one example."""
    if example == "fold":
        return FOLD_CMAP, FOLD_LABELS, FOLD_COLORS, 1
    return FLUVSIM_CMAP, FLUVSIM_LABELS, FLUVSIM_COLORS, 4


def add_legend(
    fig,
    colors: Sequence[str],
    labels: Sequence[str],
    ncol: int,
) -> None:
    """Add a bottom legend to a figure."""
    fig.legend(
        handles=[
            Patch(facecolor=color, label=label)
            for color, label in zip(colors, labels)
        ],
        loc="lower center",
        ncol=ncol,
        title="Facies code",
        frameon=False,
        bbox_to_anchor=(0.5, -0.055),
    )


def update_manifests(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    """Upsert rows into CSV and JSONL manifests."""
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "manifest.csv"
    jsonl_path = output_dir / "manifest.jsonl"

    existing = read_existing_manifest(csv_path)
    new_keys = {(row["example"], row["run_id"]) for row in rows}
    merged = [
        row
        for row in existing
        if (row.get("example"), row.get("run_id")) not in new_keys
    ]
    merged.extend(rows)
    merged.sort(key=lambda row: (str(row["example"]), str(row["run_id"])))

    csv_tmp = csv_path.with_suffix(".csv.tmp")
    jsonl_tmp = jsonl_path.with_suffix(".jsonl.tmp")

    with csv_tmp.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in merged:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})

    with jsonl_tmp.open("w") as file_handle:
        for row in merged:
            file_handle.write(json.dumps(row, sort_keys=True) + "\n")

    csv_tmp.replace(csv_path)
    jsonl_tmp.replace(jsonl_path)


def read_existing_manifest(csv_path: Path) -> List[Dict[str, object]]:
    """Read existing CSV manifest rows, if present."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def read_manifest_keys(output_dir: Path):
    """Return candidate keys already present in the manifest."""
    return {
        (row.get("example"), row.get("run_id"))
        for row in read_existing_manifest(output_dir / "manifest.csv")
    }


def manifest_row_for_existing(
    output_dir: Path, candidate: Candidate
) -> Dict[str, object]:
    """Build a recovery manifest row for an already-rendered candidate."""
    image_3d, image_slices, image_combined = candidate_paths(
        output_dir, candidate
    )
    conditioning_counts = dict(candidate.cond_counts)
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "example": candidate.example,
        "run_id": candidate.run_id,
        "profile": candidate.profile,
        "seed": candidate.seed,
        "shape": shape_text(candidate.shape),
        "n_neighbors": candidate.n_neighbors,
        "max_radius": candidate.max_radius,
        "scan_fraction": candidate.scan_fraction,
        "threshold": THRESHOLD,
        "cond_weight": candidate.cond_weight,
        "conditioning_counts": json.dumps(
            conditioning_counts,
            sort_keys=True,
        ),
        "conditioning_total": sum(conditioning_counts.values()),
        "runtime_seconds": "",
        "facies_counts": "",
        "image_3d": repo_path(image_3d),
        "image_slices": repo_path(image_slices),
        "image_combined": repo_path(image_combined),
    }


def selected_examples(candidates: Iterable[Candidate]) -> str:
    """Return a compact selected-example summary."""
    counts = {}
    for candidate in candidates:
        counts[candidate.example] = counts.get(candidate.example, 0) + 1
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def main(argv: Sequence[str] = None) -> int:
    """Command-line entry point."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be >= 1")
    if (
        args.smoke_scan_fraction is not None
        and not 0 < args.smoke_scan_fraction <= 1
    ):
        raise ValueError("--smoke-scan-fraction must be in (0, 1]")

    output_dir = args.output_dir.resolve()
    candidates = build_candidates(args.examples, limit=args.limit)
    shape_override = parse_shape_override(args.smoke_shape)
    smoke_mode = shape_override is not None or args.smoke_scan_fraction is not None
    if shape_override is not None:
        candidates = [
            replace(candidate, shape=shape_override) for candidate in candidates
        ]
    if args.smoke_scan_fraction is not None:
        candidates = [
            replace(candidate, scan_fraction=args.smoke_scan_fraction)
            for candidate in candidates
        ]
    if smoke_mode:
        output_dir = output_dir / "_smoke"
    fast_plot = args.fast_plot or smoke_mode

    print(f"Selected candidates: {selected_examples(candidates)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    if shape_override is not None:
        print(f"Smoke shape override: {shape_text(shape_override)}", flush=True)
    if args.smoke_scan_fraction is not None:
        print(
            f"Smoke scan fraction override: {args.smoke_scan_fraction:g}",
            flush=True,
        )
    if fast_plot:
        print("Fast voxel plotting: enabled", flush=True)

    manifest_keys = read_manifest_keys(output_dir)
    pending = []
    recovered_rows = []
    task_overwrite = {}
    for candidate in candidates:
        key = (candidate.example, candidate.run_id)
        has_images = is_finished(output_dir, candidate)
        if not args.overwrite and has_images:
            if key not in manifest_keys:
                recovered_rows.append(
                    manifest_row_for_existing(output_dir, candidate)
                )
            continue
        if args.overwrite or not has_images:
            pending.append(candidate)
            task_overwrite[key] = args.overwrite

    skipped = len(candidates) - len(pending)
    if skipped:
        print(f"Skipping {skipped} finished candidate(s).", flush=True)
    if recovered_rows:
        update_manifests(output_dir, recovered_rows)
        print(
            f"Recovered {len(recovered_rows)} missing manifest row(s).",
            flush=True,
        )
    if args.dry_run:
        for candidate in candidates:
            print(metadata_title(candidate), flush=True)
        return 0
    if not pending:
        print("Nothing to do.", flush=True)
        return 0

    tasks = [
        (
            candidate,
            output_dir,
            args.num_threads,
            args.dpi,
            task_overwrite[(candidate.example, candidate.run_id)],
            fast_plot,
        )
        for candidate in pending
    ]
    rows = []
    if args.workers == 1:
        for task in tasks:
            row = run_candidate_task(task)
            if row is not None:
                rows.append(row)
                update_manifests(output_dir, [row])
                print(
                    f"Finished {row['run_id']} "
                    f"in {row['runtime_seconds']}s",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_candidate_task, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                if row is not None:
                    rows.append(row)
                    update_manifests(output_dir, [row])
                    print(
                        f"Finished {row['run_id']} "
                        f"in {row['runtime_seconds']}s",
                        flush=True,
                    )

    print(f"Wrote {len(rows)} manifest row(s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
