import numpy as np

from gstools.field.base import Field



def _precompute_offsets(shape, max_offset=None):
    """Precompute offsets from the origin sorted by Euclidean distance.

    Parameters
    ----------
    shape : tuple
        Grid shape.
    max_offset : int, optional
        Maximum offset in any dimension. Default: max(shape).

    Returns
    -------
    offset_arr : ndarray, shape (N, dim)
    """
    dim = len(shape)
    if max_offset is None:
        max_offset = min(max(shape), 20)
    rng_vals = np.arange(-max_offset, max_offset + 1)
    grid = np.array(np.meshgrid(*[rng_vals] * dim, indexing="ij"))
    offsets = grid.reshape(dim, -1).T
    offsets = offsets[np.any(offsets != 0, axis=1)]
    idx = np.argsort(np.sum(offsets**2, axis=1))
    return offsets[idx]


def ds_simulate(
    ti_data,
    sg_shape,
    n,
    t,
    f,
    seed,
    conditions=None,
    cond_weight=1.0,
    categorical=True,
    d_max=None,
    postprocess=0,
    boundary="strict",
    max_offset=None,
):
    """Direct Sampling simulation (Mariethoz et al. 2010).

    Parameters
    ----------
    ti_data : ndarray or dict of ndarray
        Training image. Dict triggers multivariate mode (Mariethoz2010 §5).
    sg_shape : tuple, simulation grid shape
    n : int, max neighbors
    t : float, distance threshold (0.0 for DSBC)
    f : float, max scan fraction
    seed : int, random seed
    conditions : dict or None
        {tuple_index: scalar_value}
    cond_weight : float, delta for conditioning weight
    categorical : bool
    d_max : float or None
    weights : reserved for multivariate
    postprocess : int, number of post-processing passes
    boundary : str, "strict" (Juda2022 Eq. 5) or "partial" (Mariethoz2010 §3 ¶21)
    max_offset : int or None, passed to _precompute_offsets

    Returns
    -------
    sg : ndarray
    """
    rng = np.random.default_rng(seed)
    dim = len(sg_shape)
    ti_shape = ti_data.shape
    n_neighbors = int(n)

    if not categorical and d_max is None:
        d_max = float(ti_data.max() - ti_data.min())
        if d_max == 0:
            d_max = 1.0

    sg = np.full(sg_shape, np.nan)
    is_cond = np.zeros(sg_shape, dtype=bool)
    sg_informed = np.zeros(sg_shape, dtype=bool)

    if conditions:
        for idx, val in conditions.items():
            sg[idx] = val
            is_cond[idx] = True
            sg_informed[idx] = True

    def _get_neighbors(x_i):
        candidates = x_i + offset_list
        in_bounds = np.all((candidates >= 0) & (candidates < sg_shape), axis=1)
        valid = candidates[in_bounds]
        return valid[sg_informed[tuple(valid.T)]][:n_neighbors]

    def _rand_ti_val():
        return ti_data[tuple(rng.integers(0, s) for s in ti_shape)]

    def _dist(de_sg, de_ti, cond_mask):
        n = de_sg.shape[0]
        if n == 0:
            return 0.0
        mismatches = (de_sg != de_ti).astype(np.float64)
        if cond_mask is None or not np.any(cond_mask):
            return float(np.mean(mismatches))
        w = np.ones(n, dtype=np.float64)
        w[cond_mask] = cond_weight
        return float(np.dot(w, mismatches) / w.sum())

    offset_list = _precompute_offsets(sg_shape, max_offset)
    ti_shape_arr = np.array(ti_shape)
    max_scan_ti = max(1, int(f * ti_shape_arr.prod()))

    uninformed = np.argwhere(np.isnan(sg))
    path = uninformed[rng.permutation(len(uninformed))]

    def _simulate_node(x_i):
        neighbor_coords = _get_neighbors(x_i)
        if len(neighbor_coords) == 0:
            return _rand_ti_val()

        lags = (neighbor_coords - x_i).astype(np.float64)
        cond_mask = is_cond[tuple(neighbor_coords.T)]
        de_sg = sg[tuple(neighbor_coords.T)]

        rounded = np.round(lags).astype(int)
        sw_lo = np.maximum(0, -rounded.min(axis=0))
        sw_hi = np.minimum(
            ti_shape_arr - 1, ti_shape_arr - 1 - rounded.max(axis=0)
        )
        if np.any(sw_lo > sw_hi):
            return _rand_ti_val()

        sw_shape = tuple(sw_hi - sw_lo + 1)
        sw_size = int(np.prod(sw_shape))
        max_scan = min(max_scan_ti, sw_size)
        start = rng.integers(0, sw_size)

        best_d = np.inf
        best_v = None
        for count in range(max_scan):
            y = sw_lo + np.array(
                np.unravel_index(int((start + count) % sw_size), sw_shape)
            )
            ti_coords = np.round(y + lags).astype(int)
            dv = _dist(de_sg, ti_data[tuple(ti_coords.T)], cond_mask)
            if dv < best_d:
                best_d = dv
                best_v = ti_data[tuple(y)]
            if dv <= t:
                break

        return best_v if best_v is not None else _rand_ti_val()

    for x_i in path:
        x_i_t = tuple(x_i)
        sg[x_i_t] = _simulate_node(x_i)
        sg_informed[x_i_t] = True

    return sg


class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses gstools.field.base.Field. Takes a TrainingImage
    (analogous to CovModel) and produces fields on structured grids.

    Parameters
    ----------
    ti : TrainingImage
        Training image (the MPS model).
    n_neighbors : int
        Maximum number of neighbors in data event.
    scan_fraction : float, optional
        Maximum fraction of TI to scan per node. Default: 1.0.
    threshold : float, optional
        Distance threshold for accepting a pattern. Default: 0.0 (DSBC).
    cond_weight : float, optional
        Weight delta for conditioning data in distance. Default: 1.0.
    postprocess : int, optional
        Number of post-processing passes. Default: 0.
    boundary : str, optional
        How to handle TI boundaries when lag vectors extend outside the TI.
        ``"strict"`` (default): only scan nodes y where ALL lags fit inside
        the TI (Juda2022 Eq. 5).
    max_offset : int, optional
        Maximum offset component (in grid units) for neighbor precomputation.
        None (default) uses the full extent of the simulation grid.
    """

    default_field_names = ["field"]

    def __init__(
        self,
        ti,
        n_neighbors=32,
        scan_fraction: float = 1,
        threshold: float = 0.0,
        cond_weight: float = 1.0,
        postprocess: int = 0,
        boundary: str = "strict",
        max_offset=None,
    ):
        super().__init__(model=None, dim=ti.ndim, value_type="scalar")
        self._ti = ti
        self._n_neighbors = n_neighbors
        self._scan_fraction = scan_fraction
        self._threshold = threshold
        self._cond_weight = cond_weight
        self._postprocess = postprocess
        self._boundary = boundary
        self._max_offset = max_offset
        self._cond_pos = None
        self._cond_val = None

    def __call__(
        self,
        pos=None,
        seed=np.nan,
        mesh_type: str = "structured",
        post_process: bool = True,
        store: bool = True,
    ) -> np.ndarray:
        """Generate the MPS field.

        Parameters
        ----------
        pos : list of arrays, optional
            Position tuple for structured grid.
        seed : int, optional
            Seed for RNG.
        mesh_type : str, optional
            Must be "structured". Default: "structured".
        post_process : bool, optional
            Whether to apply mean/normalizer/trend. Default: True.
        store : str or bool, optional
            Whether to store field. Default: True.

        Returns
        -------
        field : numpy.ndarray
        """
        if mesh_type != "structured":
            raise ValueError("DirectSampling only supports structured grids.")
        name, save = self.get_store_config(store)
        pos, shape = self.pre_pos(pos, mesh_type)
        conditions = self._conditions_to_grid(self.pos, shape)
        iseed = int(seed) if not np.isnan(seed) else 42
        field = self._simulate(shape, conditions, iseed, self._boundary)
        return self.post_field(field, name, post_process, save)

    def _simulate(self, shape, conditions, seed, boundary=None):
        if boundary is None:
            boundary = self._boundary
        return ds_simulate(
            ti_data=self._ti.data,
            sg_shape=shape,
            n=self._n_neighbors,
            t=self._threshold,
            f=self._scan_fraction,
            seed=seed,
            conditions=conditions,
            cond_weight=self._cond_weight,
            categorical=self._ti.categorical,
            postprocess=self._postprocess,
            boundary=boundary,
            max_offset=self._max_offset,
        )

    def _conditions_to_grid(self, pos, shape) -> dict:
        if self._cond_pos is None:
            return {}
        n_pts = self._cond_val.shape[0]
        idx = np.empty((self.dim, n_pts), dtype=int)
        for d in range(self.dim):
            idx[d] = np.argmin(
                np.abs(pos[d][:, None] - self._cond_pos[d][None, :]), axis=0
            )
        return {
            tuple(int(idx[d, k]) for d in range(self.dim)): self._cond_val[k]
            for k in range(n_pts)
        }

    def set_condition(self, cond_pos, cond_val, weight: float = None):
        """Set conditioning data.

        Same convention as gstools.Krige: cond_pos is a list of coordinate
        arrays [x, y, ...], each of length N (i.e. shape dim × N).
        NaN values in cond_val are silently dropped.

        Parameters
        ----------
        cond_pos : list of array-like, length dim
            Coordinate arrays, one per dimension, e.g. ``[x_arr, y_arr]``.
        cond_val : array-like, shape (N,)
            Values at conditioning points.
        weight : float, optional
            Conditioning weight delta. Overrides the init value.
        """
        from gstools.krige.tools import set_condition as _gs_set_condition

        self._cond_pos, self._cond_val = _gs_set_condition(
            cond_pos, cond_val, self.dim
        )
        if weight is not None:
            self._cond_weight = weight

    @property
    def ti(self):
        """TrainingImage: The training image model."""
        return self._ti

    def __repr__(self):
        return (
            f"DirectSampling(dim={self.dim}, n={self._n_neighbors}, "
            f"f={self._scan_fraction}, t={self._threshold})"
        )
