"""
GStools subpackage providing the Direct Sampling MPS simulation class.

.. currentmodule:: gstools.mps

The following classes and functions are provided

.. autosummary::
   DirectSampling
"""

import numpy as np

from gstools.field.base import Field
from gstools.random.rng import RNG

__all__ = ["DirectSampling"]


def _precompute_offsets(shape, max_offset=None):
    """Neighbour offsets from the origin, sorted by Euclidean distance.

    Parameters
    ----------
    shape : tuple
        Simulation grid shape.
    max_offset : int, optional
        Maximum offset in any dimension.
        Default: ``max(shape)``.

    Returns
    -------
    numpy.ndarray, shape (N, dim)
    """
    dim = len(shape)
    if max_offset is None:
        max_offset = max(shape)
    rng_vals = np.arange(-max_offset, max_offset + 1)
    grid = np.array(np.meshgrid(*[rng_vals] * dim, indexing="ij"))
    offsets = grid.reshape(dim, -1).T
    offsets = offsets[np.any(offsets != 0, axis=1)]
    idx = np.argsort(np.sum(offsets**2, axis=1))
    return offsets[idx]


def ds_simulate(
    training_image,
    sim_shape,
    n_neighbors,
    threshold,
    scan_fraction,
    seed,
    conditions=None,
    cond_weight=1.0,
    max_offset=None,
):
    """Direct Sampling univariate simulation (Mariethoz2010, Juda2022).

    Parameters
    ----------
    training_image : TrainingImage
        Training image; provides ``training_image.distance()`` and
        ``training_image.adjust_value()``.
    sim_shape : tuple
        Simulation grid shape.
    n_neighbors : int
        Maximum number of neighbours in the data event (Juda2022 §2).
    threshold : float
        Distance threshold for early acceptance (Juda2022 §2).
        ``0.0`` → DSBC mode.
    scan_fraction : float
        Maximum TI scan fraction per node (Mariethoz2010 §3 ¶24).
    seed : int
        RNG seed.
    conditions : dict, optional
        ``{tuple_index: value}`` mapping of conditioning data.
    cond_weight : float, optional
        Weight δ for conditioning nodes (Mariethoz2010 §3 ¶26).
    max_offset : int, optional
        Maximum neighbour search radius in grid units.

    Returns
    -------
    numpy.ndarray
    """
    rng = np.random.default_rng(seed)
    ti_data = training_image.data
    ti_shape = np.array(ti_data.shape)
    sim_shape_arr = np.array(sim_shape)
    ti_size = int(ti_shape.prod())

    sg = np.full(sim_shape, np.nan)
    is_cond = np.zeros(sim_shape, dtype=bool)
    informed = np.zeros(sim_shape, dtype=bool)

    if conditions:
        for idx, val in conditions.items():
            sg[idx] = val
            is_cond[idx] = True
            informed[idx] = True

    offset_arr = _precompute_offsets(sim_shape, max_offset)
    max_scan_ti = max(1, int(scan_fraction * ti_size))

    def _rand_ti():
        return ti_data[tuple(rng.integers(0, s) for s in ti_shape)]

    def _get_neighbors(x_i):
        cands = x_i + offset_arr
        valid = cands[np.all((cands >= 0) & (cands < sim_shape_arr), axis=1)]
        return valid[informed[tuple(valid.T)]][:n_neighbors]

    def _simulate_node(x_i):
        nbrs = _get_neighbors(x_i)
        if len(nbrs) == 0:
            return _rand_ti()

        lags = (nbrs - x_i).astype(np.float64)  # (k, dim)
        data_event_sim = sg[tuple(nbrs.T)]  # (k,)
        cond_mask = is_cond[tuple(nbrs.T)]  # (k,)
        lag_norms = np.linalg.norm(lags, axis=1)  # (k,)

        # Search window Y(L_i) — Juda2022 Eq. 5, Mariethoz2010 §3 ¶19
        win_lo = np.maximum(0, np.ceil(-lags.min(axis=0))).astype(int)
        win_hi = np.minimum(
            ti_shape - 1, np.floor(ti_shape - 1 - lags.max(axis=0))
        ).astype(int)
        if np.any(win_lo > win_hi):
            return _rand_ti()

        win_shape = tuple(win_hi - win_lo + 1)
        win_size = int(np.prod(win_shape))
        max_scan = min(max_scan_ti, win_size)
        start = int(rng.integers(0, win_size))

        best_d, best_v, best_data_event_ti = np.inf, None, None

        for k in range(max_scan):
            y = win_lo + np.array(
                np.unravel_index((start + k) % win_size, win_shape)
            )
            ti_coords = np.round(y + lags).astype(int)
            # Residual validity — Mariethoz2010 §3 ¶21
            data_event_ti = ti_data[tuple(ti_coords.T)]
            dist_val = training_image.distance(
                data_event_sim,
                data_event_ti,
                cond_mask,
                cond_weight,
                lag_norms,
            )
            if dist_val < best_d:
                best_d, best_v, best_data_event_ti = (
                    dist_val,
                    ti_data[tuple(y)],
                    data_event_ti,
                )
            if dist_val <= threshold:
                break

        if best_v is None:
            return _rand_ti()
        return training_image.adjust_value(
            best_v, data_event_sim, best_data_event_ti
        )

    path = np.argwhere(np.isnan(sg))
    path = path[rng.permutation(len(path))]

    for x_i in path:
        x_i_t = tuple(x_i)
        val = _simulate_node(x_i)
        if np.isnan(val):
            raise ValueError(
                f"Simulation produced NaN at {x_i}. Check TI data."
            )
        sg[x_i_t] = val
        informed[x_i_t] = True

    return sg


class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses :class:`gstools.field.base.Field`. Takes a :class:`TrainingImage`
    (analogous to :class:`CovModel`) and produces fields on structured grids.

    Parameters
    ----------
    ti : TrainingImage
        The training image (the MPS model).
    n_neighbors : int, optional
        Maximum neighbors in data event. Default: 32.
    scan_fraction : float, optional
        Maximum fraction of TI to scan per node. Default: 1.
    threshold : float, optional
        Distance threshold. 0.0 -> DSBC mode. Default: 0.0.
    cond_weight : float, optional
        Weight for conditioning nodes in distance. Default: 1.0.
    max_offset : int, optional
        Maximum neighbor search radius in grid units.
    seed : int or nan, optional
        Master RNG seed. Default: nan.
    """

    default_field_names = ["field"]

    def __init__(
        self,
        ti,
        n_neighbors=32,
        scan_fraction=1,
        threshold=0.0,
        cond_weight=1.0,
        max_offset=None,
        seed=np.nan,
    ):
        super().__init__(model=None, dim=ti.ndim, value_type="scalar")
        self._ti = ti
        self._n_neighbors = int(n_neighbors)
        self._scan_fraction = float(scan_fraction)
        self._threshold = float(threshold)
        self._cond_weight = float(cond_weight)
        self._max_offset = max_offset
        self._cond_pos = None
        self._cond_val = None
        self.rng = RNG(None if np.isnan(seed) else int(seed))

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        mesh_type="structured",
        post_process=True,
        store=True,
    ):
        """Generate the spatial random field via Direct Sampling.

        The field is saved as ``self.field`` and is also returned.

        Parameters
        ----------
        pos : :class:`list`, optional
            The position tuple, containing main direction and transversal
            directions. Only structured grids are supported.
        seed : :class:`int`, optional
            Seed for the RNG. If ``np.nan``, the current seed is kept.
            Default: ``np.nan``
        mesh_type : :class:`str`, optional
            Grid type. Must be ``"structured"``.
            Default: ``"structured"``
        post_process : :class:`bool`, optional
            Whether to apply post-processing transformations (mean,
            normalizer, trend) to the field. Default: :any:`True`
        store : :class:`bool` or :class:`str`, optional
            Whether to store the field (``True``), not store it (``False``),
            or store it under a custom name (string).
            Default: :any:`True`

        Returns
        -------
        field : :class:`numpy.ndarray`
            The simulated field.
        """
        if mesh_type != "structured":
            raise ValueError(
                "DirectSampling: only structured grids are supported."
            )
        name, save = self.get_store_config(store)
        pos, shape = self.pre_pos(pos, mesh_type)
        conditions = self._conditions_to_grid(self.pos)
        if not np.isnan(seed):
            self.rng.seed = int(seed)
        iseed = int(self.rng.random.randint(0, 2**31))
        field = ds_simulate(
            training_image=self._ti,
            sim_shape=shape,
            n_neighbors=self._n_neighbors,
            threshold=self._threshold,
            scan_fraction=self._scan_fraction,
            seed=iseed,
            conditions=conditions,
            cond_weight=self._cond_weight,
            max_offset=self._max_offset,
        )
        return self.post_field(field, name, post_process, save)

    def _conditions_to_grid(self, axes):
        """Smart snapping: Mariethoz 2010 collision rule."""
        if self._cond_pos is None:
            return {}
        candidates = {}  # idx -> (val, dist_sq)
        for k in range(self._cond_val.shape[0]):
            idx = tuple(
                int(np.argmin(np.abs(axes[d] - self._cond_pos[d][k])))
                for d in range(self.dim)
            )
            dist_sq = sum(
                (axes[d][idx[d]] - self._cond_pos[d][k]) ** 2
                for d in range(self.dim)
            )
            if idx not in candidates or dist_sq < candidates[idx][1]:
                candidates[idx] = (self._cond_val[k], dist_sq)
        return {idx: val for idx, (val, _) in candidates.items()}

    def set_condition(self, cond_pos, cond_val, cond_weight=None):
        """Set the conditioning data for the simulation.

        Parameters
        ----------
        cond_pos : :class:`list`
            The position tuple of the conditioning data ``(x, [y, z])``.
        cond_val : :class:`numpy.ndarray`
            The values at the conditioning positions.
        cond_weight : :class:`float`, optional
            Conditioning weight δ. If given, overrides the ``cond_weight``
            set at construction. Default: :any:`None` (keep existing weight)
        """
        from gstools.krige.tools import set_condition as _gs_set_condition

        self._cond_pos, self._cond_val = _gs_set_condition(
            cond_pos, cond_val, self.dim
        )
        if cond_weight is not None:
            self._cond_weight = float(cond_weight)

    @property
    def ti(self):
        """TrainingImage: The training image model."""
        return self._ti

    @property
    def n_neighbors(self):
        """:class:`int`: Maximum neighbours in the data event."""
        return self._n_neighbors

    @n_neighbors.setter
    def n_neighbors(self, value):
        self._n_neighbors = int(value)

    @property
    def scan_fraction(self):
        """:class:`float`: Maximum fraction of TI to scan per node."""
        return self._scan_fraction

    @scan_fraction.setter
    def scan_fraction(self, value):
        self._scan_fraction = float(value)

    @property
    def threshold(self):
        """:class:`float`: Distance threshold (0.0 → DSBC mode)."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = float(value)

    @property
    def cond_weight(self):
        """:class:`float`: Weight for conditioning nodes in distance."""
        return self._cond_weight

    @cond_weight.setter
    def cond_weight(self, value):
        self._cond_weight = float(value)

    @property
    def max_offset(self):
        """:class:`int` or :any:`None`: Maximum neighbour search radius."""
        return self._max_offset

    @max_offset.setter
    def max_offset(self, value):
        self._max_offset = None if value is None else int(value)

    def __repr__(self):
        return (
            f"DirectSampling(dim={self.dim}, "
            f"n_neighbors={self.n_neighbors}, "
            f"scan_fraction={self.scan_fraction}, "
            f"threshold={self.threshold})"
        )
