# MPS GSTools Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the full MPSTools implementation into `gstools/mps/` as clean Python, following the same conventions as `gstools/krige/base.py` and `gstools/field/srf.py`.

**Architecture:** `TI.py` holds the full `TrainingImage` class (data + distance logic). `DS.py` holds private simulation helpers at module level plus the `DirectSampling` class — mirroring how `krige/base.py` puts private wrappers above `Krige`. No new files, no Rust, no tests, no quality metrics.

**Tech Stack:** Python, NumPy, `gstools.field.base.Field`, `gstools.random.rng.RNG`, `gstools.krige.tools.set_condition`, `gstools.tools.geometric.matrix_rotate` / `matrix_anisotropify`

---

## File Map

| File | Role |
|---|---|
| `src/gstools/mps/TI.py` | `TrainingImage`: data storage, distance functions, value access |
| `src/gstools/mps/DS.py` | `_precompute_offsets`, `_ds_simulate`, `DirectSampling` class |
| `src/gstools/mps/__init__.py` | **Untouched** — already exports both classes |
| `src/gstools/__init__.py` | **Untouched** — already exports both classes |

Source reference: `/home/niklas/dev/MPSTools/src/mpstools/` and `/home/niklas/dev/MPSTools/tests/baseline/oracle.py`

---

## Task 1: TI.py — univariate

**Files:**
- Replace: `src/gstools/mps/TI.py`

Adds `d_max`, `_distance_single()` (l1 only), `distance()`, `value_at()`, `categories()`, `size`.
Multivariate (`distance="l2"/"variation"`, dict data) intentionally absent — comes in later tasks.
`_multivariate = False` is set now so DS.py can test for it without crashing later.

- [ ] **Step 1: Replace TI.py**

```python
"""Training image: the MPS statistical model."""
import numpy as np

__all__ = ["TrainingImage"]


class TrainingImage:
    """Training image for multiple point statistics simulation.

    The MPS analogue of a covariance model. Encapsulates training data
    and the distance function used to compare data events.

    Parameters
    ----------
    data : numpy.ndarray
        Training image data (n-d array).
    categorical : bool, optional
        Whether the variable is categorical. Default: True.
    """

    def __init__(self, data, categorical=True):
        self._data = np.asarray(data)
        self._categorical = bool(categorical)
        self._multivariate = False

        # d_max for continuous variables (normalises distance to [0, 1])
        self._d_max = {}
        if not self._categorical:
            dmax = float(self._data.max() - self._data.min())
            self._d_max["_default"] = dmax if dmax > 0 else 1.0

        # unique category values
        self._categories = {}
        if self._categorical:
            self._categories["_default"] = np.unique(self._data)

    # --- Properties ---

    @property
    def data(self):
        """ndarray: Raw training image data."""
        return self._data

    @property
    def ndim(self):
        """int: Number of spatial dimensions."""
        return self._data.ndim

    @property
    def shape(self):
        """tuple: Shape of the training image."""
        return self._data.shape

    @property
    def size(self):
        """int: Total number of nodes."""
        return int(np.prod(self._data.shape))

    @property
    def categorical(self):
        """bool: Whether the variable is categorical."""
        return self._categorical

    def categories(self):
        """ndarray: Unique category values, or None for continuous."""
        return self._categories.get("_default")

    def value_at(self, index):
        """Value at a flat or nd index."""
        return self._data[index]

    # --- Distance ---

    def _distance_single(self, de_sg, de_ti, cond_mask=None,
                         cond_weight=1.0, lag_norms=None):
        """Distance between two univariate data events.

        Categorical: fraction of non-matching (Juda2022 Eq. 6).
        Continuous L1: normalised mean absolute difference (Juda2022 Eq. 7).
        lag_norms is accepted for API compatibility (used by l2 in a later task).
        """
        n = len(de_sg)
        if n == 0:
            return 0.0
        w = np.ones(n, dtype=np.float64)
        if cond_mask is not None:
            w[cond_mask] = cond_weight
        if self._categorical:
            mismatches = (de_sg != de_ti).astype(np.float64)
            return float(np.dot(w, mismatches) / w.sum())
        diffs = np.abs(de_sg - de_ti) / self._d_max["_default"]
        return float(np.dot(w, diffs) / w.sum())

    def distance(self, de_sg, de_ti, cond_mask=None,
                 cond_weight=1.0, lag_norms=None):
        """Distance between two data events. Returns float in [0, 1]."""
        return self._distance_single(de_sg, de_ti, cond_mask, cond_weight,
                                     lag_norms)

    def __repr__(self):
        return (
            f"TrainingImage(shape={self.shape}, categorical={self._categorical})"
        )
```

- [ ] **Step 2: Smoke-test**

```bash
cd /home/niklas/dev/GSTools
python -c "
import numpy as np
import gstools as gs

# categorical
ti = gs.TrainingImage(np.array([[0,1],[1,0]]))
assert ti.shape == (2, 2)
assert ti.ndim == 2
assert ti.size == 4
assert ti.categorical is True
assert list(ti.categories()) == [0, 1]
assert ti.value_at((0, 1)) == 1

# distance categorical: full mismatch
de_sg = np.array([0, 0])
de_ti = np.array([1, 1])
assert ti.distance(de_sg, de_ti) == 1.0

# continuous
rng = np.random.default_rng(0)
ti_c = gs.TrainingImage(rng.uniform(0, 10, (5, 5)), categorical=False)
de_sg = np.array([0.0, 5.0])
de_ti = np.array([0.0, 5.0])
assert ti_c.distance(de_sg, de_ti) == 0.0
print('Task 1 OK')
"
```

Expected: `Task 1 OK`

- [ ] **Step 3: Commit**

```bash
cd /home/niklas/dev/GSTools
git add src/gstools/mps/TI.py
git commit -m "feat(mps): extend TrainingImage with distance, d_max, value_at (univariate)"
```

---

## Task 2: DS.py — clean simulation + continuous

**Files:**
- Replace: `src/gstools/mps/DS.py`

Key changes vs. the current stub:
1. `_precompute_offsets` uses oracle's distance-cap approach (handles 3D without memory explosion).
2. `_ds_simulate` (renamed from `ds_simulate`, now private) takes a `TrainingImage` object and calls `ti.distance()` — no hardcoded categorical logic.
3. `_conditions_to_grid` gets the Mariethoz2010 §3 ¶12 tie-breaking fix: when two conditioning points snap to the same grid node, keep the one closest to the node centre.
4. The `DirectSampling` class keeps the existing GSTools RNG/seed management (`update`, `reset_seed`, `seed`, `ti` setter). Only `__call__` changes to call `_ds_simulate` instead of `ds_simulate`.
5. `max_offset` removed (replaced by the oracle's automatic cap logic inside `_precompute_offsets`).

- [ ] **Step 1: Replace DS.py**

```python
"""Direct Sampling field generator."""
import warnings

import numpy as np

from gstools.field.base import Field
from gstools.krige.tools import set_condition as _gs_set_condition
from gstools.random.rng import RNG

__all__ = ["DirectSampling"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _precompute_offsets(shape):
    """All offsets in an n-d grid sorted by Euclidean distance.

    Excludes the origin. Caps the search radius at max(shape)-1 for 1D/2D
    and max(shape)//2 for 3D+ to avoid memory explosion.
    """
    dim = len(shape)
    cap = (max(shape) - 1) if dim <= 2 else max(shape) // 2
    ranges = [range(-min(s - 1, cap), min(s, cap + 1)) for s in shape]
    offsets = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(dim, -1).T
    dists = np.linalg.norm(offsets, axis=1)
    mask = dists > 0
    offsets = offsets[mask]
    order = np.argsort(dists[mask])
    return [tuple(int(v) for v in offsets[i]) for i in order]


def _ds_simulate(ti, sg_shape, n, t, f, seed, conditions=None,
                 cond_weight=1.0):
    """Direct Sampling simulation (Mariethoz et al. 2010).

    Parameters
    ----------
    ti : TrainingImage
    sg_shape : tuple
    n : int, max neighbors
    t : float, distance threshold (0.0 for DSBC)
    f : float, max scan fraction (0 < f <= 1)
    seed : int
    conditions : dict or None, {nd_tuple_index: value}
    cond_weight : float, delta for conditioning nodes

    Returns
    -------
    sg : ndarray
    """
    rng = np.random.default_rng(seed)
    dim = len(sg_shape)
    ti_data = ti.data
    ti_shape = ti_data.shape
    n_neighbors = int(n)

    sg = np.full(sg_shape, np.nan)
    is_cond = np.zeros(sg_shape, dtype=bool)

    if conditions:
        for idx, val in conditions.items():
            sg[idx] = val
            is_cond[idx] = True

    offset_list = _precompute_offsets(sg_shape)
    ti_size = int(np.prod(ti_shape))
    max_scan_ti = max(1, int(f * ti_size))

    uninformed = np.argwhere(np.isnan(sg))
    path = uninformed[rng.permutation(len(uninformed))]

    def _rand_ti_val():
        return ti_data[tuple(rng.integers(0, s) for s in ti_shape)]

    def _simulate_node(x_i):
        x_i = tuple(x_i)

        # 1. Find n closest informed neighbours (Juda2022 Eq. 1)
        lags = []
        cond_mask = []
        for offset in offset_list:
            nb = tuple(x_i[d] + offset[d] for d in range(dim))
            if all(0 <= nb[d] < sg_shape[d] for d in range(dim)):
                if not np.isnan(sg[nb]):
                    lags.append(np.array(offset, dtype=np.float64))
                    cond_mask.append(bool(is_cond[nb]))
                    if len(lags) >= n_neighbors:
                        break

        if not lags:
            return _rand_ti_val()

        lags_arr = np.array(lags)          # (n_found, dim)
        cond_mask_arr = np.array(cond_mask)  # (n_found,) bool

        # 2. SG data event (Juda2022 Eq. 4)
        de_sg = np.array([sg[tuple(int(x_i[d] + lags[j][d])
                                   for d in range(dim))]
                          for j in range(len(lags))])

        # 3. Search window Y(L_i) — bounding box of lag vectors (Juda2022 Eq. 5)
        sw_lo = [0] * dim
        sw_hi = [ti_shape[d] - 1 for d in range(dim)]
        for lag in lags_arr:
            for d in range(dim):
                sw_lo[d] = max(sw_lo[d], int(np.ceil(-lag[d])))
                sw_hi[d] = min(sw_hi[d],
                               int(np.floor(ti_shape[d] - 1 - lag[d])))
        if any(sw_lo[d] > sw_hi[d] for d in range(dim)):
            return _rand_ti_val()

        sw_shape = tuple(sw_hi[d] - sw_lo[d] + 1 for d in range(dim))
        sw_size = int(np.prod(sw_shape))
        max_scan = min(max_scan_ti, sw_size)
        start = rng.integers(0, sw_size)

        best_d = np.inf
        best_v = None

        for count in range(max_scan):
            sw_flat = int((start + count) % sw_size)
            sw_nd = np.unravel_index(sw_flat, sw_shape)
            y = tuple(sw_lo[d] + sw_nd[d] for d in range(dim))

            # Residual validity check (Mariethoz2010 §3 ¶21)
            valid = True
            for lag in lags_arr:
                nb_ti = tuple(int(round(y[d] + lag[d])) for d in range(dim))
                if not all(0 <= nb_ti[d] < ti_shape[d] for d in range(dim)):
                    valid = False
                    break
            if not valid:
                continue

            # 4. TI data event
            de_ti = np.array([
                ti_data[tuple(int(round(y[d] + lags_arr[j][d]))
                               for d in range(dim))]
                for j in range(len(lags))
            ])

            # 5. Distance (delegated to TrainingImage)
            dv = ti.distance(de_sg, de_ti, cond_mask_arr, cond_weight)

            if dv < best_d:
                best_d = dv
                best_v = ti_data[y]
            if dv <= t:
                break

        return best_v if best_v is not None else _rand_ti_val()

    for x_i in path:
        sg[tuple(x_i)] = _simulate_node(x_i)

    return sg


# ---------------------------------------------------------------------------
# DirectSampling
# ---------------------------------------------------------------------------

class DirectSampling(Field):
    """Multiple Point Statistics simulation using Direct Sampling.

    Subclasses gstools.field.base.Field. Takes a TrainingImage
    (analogous to CovModel) and produces fields on structured grids.

    Parameters
    ----------
    ti : TrainingImage
        Training image (the MPS model).
    n_neighbors : int, optional
        Maximum number of neighbors in data event. Default: 32.
    scan_fraction : float, optional
        Maximum fraction of TI to scan per node. Default: 1.0.
    threshold : float, optional
        Distance threshold for accepting a pattern. Default: 0.0 (DSBC).
    cond_weight : float, optional
        Weight delta for conditioning nodes in distance. Default: 1.0.
    seed : int or None, optional
        Master RNG seed. Default: numpy.nan (random).
    """

    default_field_names = ["field"]

    def __init__(self, ti, n_neighbors=32, scan_fraction=1.0,
                 threshold=0.0, cond_weight=1.0, seed=np.nan):
        super().__init__(model=None, dim=ti.ndim, value_type="scalar")
        self._ti = ti
        self._n_neighbors = n_neighbors
        self._scan_fraction = scan_fraction
        self._threshold = threshold
        self._cond_weight = cond_weight
        self._cond_pos = None
        self._cond_val = None
        # RNG management (GSTools convention)
        self._seed = np.nan
        self._rng = None
        self._ds_seed = None
        self.update(seed=seed)

    def __call__(self, pos=None, seed=np.nan, mesh_type="structured",
                 post_process=True, store=True):
        """Generate the MPS field.

        Parameters
        ----------
        pos : list of arrays, optional
            Position tuple for structured grid.
        seed : int, optional
            Seed for RNG. Default: numpy.nan (keep current).
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
        self.update(seed=seed)
        pos, shape = self.pre_pos(pos, mesh_type)
        conditions = self._conditions_to_grid(self.pos, shape)
        field = _ds_simulate(
            ti=self._ti,
            sg_shape=shape,
            n=self._n_neighbors,
            t=self._threshold,
            f=self._scan_fraction,
            seed=self._ds_seed,
            conditions=conditions,
            cond_weight=self._cond_weight,
        )
        return self.post_field(field, name, post_process, save)

    def update(self, ti=None, seed=np.nan):
        """Update the TI and/or seed.

        Parameters
        ----------
        ti : TrainingImage or None, optional
        seed : int or None or numpy.nan, optional
            numpy.nan keeps the current seed.
        """
        if ti is not None:
            if self.ti != ti:
                self._ti = ti
                if seed is None or not np.isnan(seed):
                    self.reset_seed(seed)
                else:
                    self.reset_seed(self._seed)
            elif seed is None or not np.isnan(seed):
                self.seed = seed
        elif seed is None or not np.isnan(seed):
            self.seed = seed
        elif np.isnan(seed) and self._rng is None:
            self.reset_seed(self._seed)

    def reset_seed(self, seed=np.nan):
        """Reset the master RNG and derive a fresh simulation seed.

        Parameters
        ----------
        seed : int or None or numpy.nan, optional
        """
        if seed is None or not np.isnan(seed):
            self._seed = seed
        self._rng = RNG(self._seed)
        self._ds_seed = self._rng._master_rng()

    def set_condition(self, cond_pos, cond_val, weight=None):
        """Set conditioning data.

        Same convention as gstools.Krige: cond_pos is a list of coordinate
        arrays [x, y, ...], each of length N.
        NaN values in cond_val are silently dropped.

        Parameters
        ----------
        cond_pos : list of array-like, length dim
        cond_val : array-like, shape (N,)
        weight : float, optional
            Conditioning weight delta. Overrides init value.
        """
        self._cond_pos, self._cond_val = _gs_set_condition(
            cond_pos, cond_val, self.dim
        )
        if weight is not None:
            self._cond_weight = weight

    def _conditions_to_grid(self, axes, shape):
        """Convert physical conditioning positions to grid index dict.

        When two points snap to the same node, keep the one closest to the
        node centre (Mariethoz2010 §3 ¶12).
        """
        if self._cond_pos is None:
            return None
        conditions = {}
        best_dist = {}
        for i in range(len(self._cond_val)):
            idx = tuple(
                int(np.argmin(np.abs(axes[d] - self._cond_pos[d, i])))
                for d in range(self.dim)
            )
            delta = np.array([
                self._cond_pos[d, i] - axes[d][idx[d]]
                for d in range(self.dim)
            ])
            dist = float(np.linalg.norm(delta))
            if idx not in best_dist or dist < best_dist[idx]:
                conditions[idx] = self._cond_val[i]
                best_dist[idx] = dist
        return conditions

    @property
    def seed(self):
        """:class:`int`: Seed of the master RNG."""
        return self._seed

    @seed.setter
    def seed(self, new_seed):
        if new_seed is not self._seed:
            self.reset_seed(new_seed)

    @property
    def ti(self):
        """TrainingImage: The training image model."""
        return self._ti

    @ti.setter
    def ti(self, ti):
        self.update(ti=ti)

    def __repr__(self):
        return (
            f"DirectSampling(dim={self.dim}, n={self._n_neighbors}, "
            f"f={self._scan_fraction}, t={self._threshold}, "
            f"seed={self.seed})"
        )
```

- [ ] **Step 2: Smoke-test**

```bash
cd /home/niklas/dev/GSTools
python -c "
import numpy as np
import gstools as gs

rng = np.random.default_rng(0)

# categorical
ti_data = rng.integers(0, 3, (20, 20))
ti = gs.TrainingImage(ti_data, categorical=True)
ds = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5, seed=42)
pos = [np.arange(10, dtype=float), np.arange(10, dtype=float)]
field = ds(pos)
assert field.shape == (10, 10)
assert not np.any(np.isnan(field))
assert set(np.unique(field)).issubset({0, 1, 2})

# continuous
ti_c = gs.TrainingImage(rng.uniform(0, 10, (20, 20)), categorical=False)
ds_c = gs.DirectSampling(ti_c, n_neighbors=8, scan_fraction=0.5, seed=42)
field_c = ds_c(pos)
assert field_c.shape == (10, 10)
assert float(field_c.min()) >= 0.0
assert float(field_c.max()) <= 10.0

# seed reproducibility
ds2 = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5, seed=42)
field2 = ds2(pos)
assert np.array_equal(field, field2)

# conditioning preserved
ds3 = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5, seed=42)
ds3.set_condition([np.array([2.0]), np.array([3.0])], np.array([0]))
field3 = ds3(pos)
assert field3[2, 3] == 0

print('Task 2 OK')
"
```

Expected: `Task 2 OK`

- [ ] **Step 3: Commit**

```bash
cd /home/niklas/dev/GSTools
git add src/gstools/mps/DS.py
git commit -m "feat(mps): rewrite DS simulation to use ti.distance(), fix conditioning tie-breaking"
```

---

## Task 3: DS.py — parallel

**Files:**
- Modify: `src/gstools/mps/DS.py`

Adds `parallel` parameter to `DirectSampling`. The parameter is fully wired (warnings fire, `_resolve_parallel_mode` returns the correct code), but `_ds_simulate` ignores the mode for now — pure Python is single-threaded. Effect comes when a Rust backend is added later.

- [ ] **Step 1: Add `_available_ram_bytes` before `_ds_simulate`**

Insert this function between `_precompute_offsets` and `_ds_simulate`:

```python
def _available_ram_bytes():
    """Best-effort estimate of available RAM in bytes."""
    import os
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    return 2 * 1024**3
```

- [ ] **Step 2: Add `parallel` parameter to `DirectSampling.__init__`**

Add `parallel="auto"` to the parameter list and validation block. In `__init__`, after the `threshold` / `cond_weight` assignments:

```python
    def __init__(self, ti, n_neighbors=32, scan_fraction=1.0,
                 threshold=0.0, cond_weight=1.0, parallel="auto",
                 seed=np.nan):
        if parallel not in ("auto", "inner", "outer", True, False, None):
            raise ValueError(
                f"parallel must be 'auto', 'inner', 'outer', True, False, "
                f"or None, got {parallel!r}"
            )
        super().__init__(model=None, dim=ti.ndim, value_type="scalar")
        self._ti = ti
        self._n_neighbors = n_neighbors
        self._scan_fraction = scan_fraction
        self._threshold = threshold
        self._cond_weight = cond_weight
        self._parallel = parallel
        self._cond_pos = None
        self._cond_val = None
        self._seed = np.nan
        self._rng = None
        self._ds_seed = None
        self.update(seed=seed)
```

- [ ] **Step 3: Add `_resolve_parallel_mode` method to `DirectSampling`**

Add after `_conditions_to_grid`:

```python
    def _resolve_parallel_mode(self, shape):
        """Return 0=sequential, 1=CpuInner, 2=CpuOuter."""
        import math
        p = self._parallel
        if p is False or p is None:
            return 0

        N = math.prod(shape)
        n = (max(self._n_neighbors.values())
             if isinstance(self._n_neighbors, dict)
             else self._n_neighbors)
        dag_bytes = n * N * 4
        available = _available_ram_bytes()

        if p == "outer" and dag_bytes >= 0.5 * available:
            warnings.warn(
                f"parallel='outer' requested but DAG requires "
                f"~{dag_bytes // 2**20} MB "
                f"(>{available // 2**21} MB available). "
                f"Falling back to inner/sequential.",
                RuntimeWarning,
                stacklevel=3,
            )
            p = "auto"

        if p == "inner" and self._threshold != 0.0:
            warnings.warn(
                "parallel='inner' with threshold>0 uses 'best-in-chunk' "
                "semantics, which differs from sequential early-stop. "
                "Use parallel='auto' or parallel=False for exact output.",
                RuntimeWarning,
                stacklevel=3,
            )
            return 1

        want_outer = (p == "outer") or (
            p in ("auto", True) and dag_bytes < 0.5 * available
        )
        want_inner = (p == "inner") or (
            p in ("auto", True) and self._threshold == 0.0
        )
        if want_outer:
            return 2
        if want_inner:
            return 1
        return 0
```

- [ ] **Step 4: Wire `_resolve_parallel_mode` into `__call__`**

In `__call__`, add one line after `conditions = self._conditions_to_grid(...)` (the result is computed but not yet passed to `_ds_simulate` — pure Python ignores it):

```python
        conditions = self._conditions_to_grid(self.pos, shape)
        _parallel_mode = self._resolve_parallel_mode(shape)  # noqa: F841
        field = _ds_simulate(
            ...
        )
```

- [ ] **Step 5: Update `__repr__`**

```python
    def __repr__(self):
        return (
            f"DirectSampling(dim={self.dim}, n={self._n_neighbors}, "
            f"f={self._scan_fraction}, t={self._threshold}, "
            f"parallel={self._parallel!r}, seed={self.seed})"
        )
```

- [ ] **Step 6: Smoke-test**

```bash
cd /home/niklas/dev/GSTools
python -c "
import warnings
import numpy as np
import gstools as gs

ti = gs.TrainingImage(np.random.default_rng(0).integers(0, 2, (20, 20)))
pos = [np.arange(10, dtype=float)] * 2

# all parallel values accepted without error
for p in ('auto', 'inner', 'outer', True, False, None):
    ds = gs.DirectSampling(ti, parallel=p, seed=1)
    ds(pos)

# invalid parallel raises
try:
    gs.DirectSampling(ti, parallel='bad')
    assert False
except ValueError:
    pass

# inner + threshold warns
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    ds = gs.DirectSampling(ti, parallel='inner', threshold=0.1, seed=1)
    ds(pos)
    assert any('inner' in str(x.message) for x in w)

print('Task 3 OK')
"
```

Expected: `Task 3 OK`

- [ ] **Step 7: Commit**

```bash
cd /home/niklas/dev/GSTools
git add src/gstools/mps/DS.py
git commit -m "feat(mps): add parallel parameter to DirectSampling"
```

---

## Task 4: DS.py — boundary

**Files:**
- Modify: `src/gstools/mps/DS.py`

Adds `boundary="strict"/"partial"` to both `DirectSampling` and `_ds_simulate`.
`"strict"` (default): search window computed once from bounding box of lag vectors.
`"partial"`: scan the full TI flat; per candidate, drop lags whose transformed offset
falls outside the TI and normalise distance over the remaining lags.
Warning fires when `boundary="partial"` is used without rotation/affinity maps
(those come in Task 7; the warning is correct from day one).

- [ ] **Step 1: Add `boundary` to `_ds_simulate` signature and implement**

Replace the current `_ds_simulate` function with the version below. The main loop
gains a `boundary` branch; the `"strict"` path is identical to Task 2's code.
The `"partial"` path scans `ti_size` flat indices and drops out-of-bounds lags
per candidate.

```python
def _ds_simulate(ti, sg_shape, n, t, f, seed, conditions=None,
                 cond_weight=1.0, boundary="strict"):
    """Direct Sampling simulation (Mariethoz et al. 2010).

    Parameters
    ----------
    ti : TrainingImage
    sg_shape : tuple
    n : int, max neighbors
    t : float, distance threshold (0.0 for DSBC)
    f : float, max scan fraction
    seed : int
    conditions : dict or None
    cond_weight : float
    boundary : str, "strict" or "partial"
    """
    rng = np.random.default_rng(seed)
    dim = len(sg_shape)
    ti_data = ti.data
    ti_shape = ti_data.shape
    n_neighbors = int(n)

    sg = np.full(sg_shape, np.nan)
    is_cond = np.zeros(sg_shape, dtype=bool)

    if conditions:
        for idx, val in conditions.items():
            sg[idx] = val
            is_cond[idx] = True

    offset_list = _precompute_offsets(sg_shape)
    ti_size = int(np.prod(ti_shape))
    max_scan_ti = max(1, int(f * ti_size))

    uninformed = np.argwhere(np.isnan(sg))
    path = uninformed[rng.permutation(len(uninformed))]

    def _rand_ti_val():
        return ti_data[tuple(rng.integers(0, s) for s in ti_shape)]

    def _simulate_node(x_i):
        x_i = tuple(x_i)

        lags = []
        cond_mask = []
        for offset in offset_list:
            nb = tuple(x_i[d] + offset[d] for d in range(dim))
            if all(0 <= nb[d] < sg_shape[d] for d in range(dim)):
                if not np.isnan(sg[nb]):
                    lags.append(np.array(offset, dtype=np.float64))
                    cond_mask.append(bool(is_cond[nb]))
                    if len(lags) >= n_neighbors:
                        break

        if not lags:
            return _rand_ti_val()

        lags_arr = np.array(lags)
        cond_mask_arr = np.array(cond_mask)
        de_sg = np.array([
            sg[tuple(int(x_i[d] + lags[j][d]) for d in range(dim))]
            for j in range(len(lags))
        ])

        best_d = np.inf
        best_v = None

        if boundary == "strict":
            sw_lo = [0] * dim
            sw_hi = [ti_shape[d] - 1 for d in range(dim)]
            for lag in lags_arr:
                for d in range(dim):
                    sw_lo[d] = max(sw_lo[d], int(np.ceil(-lag[d])))
                    sw_hi[d] = min(sw_hi[d],
                                   int(np.floor(ti_shape[d] - 1 - lag[d])))
            if any(sw_lo[d] > sw_hi[d] for d in range(dim)):
                return _rand_ti_val()

            sw_shape = tuple(sw_hi[d] - sw_lo[d] + 1 for d in range(dim))
            sw_size = int(np.prod(sw_shape))
            max_scan = min(max_scan_ti, sw_size)
            start = rng.integers(0, sw_size)

            for count in range(max_scan):
                sw_flat = int((start + count) % sw_size)
                sw_nd = np.unravel_index(sw_flat, sw_shape)
                y = tuple(sw_lo[d] + sw_nd[d] for d in range(dim))

                valid = True
                for lag in lags_arr:
                    nb_ti = tuple(int(round(y[d] + lag[d])) for d in range(dim))
                    if not all(0 <= nb_ti[d] < ti_shape[d] for d in range(dim)):
                        valid = False
                        break
                if not valid:
                    continue

                de_ti = np.array([
                    ti_data[tuple(int(round(y[d] + lags_arr[j][d]))
                                  for d in range(dim))]
                    for j in range(len(lags))
                ])
                dv = ti.distance(de_sg, de_ti, cond_mask_arr, cond_weight)
                if dv < best_d:
                    best_d = dv
                    best_v = ti_data[y]
                if dv <= t:
                    break

        else:  # "partial": scan TI flat, drop out-of-bounds lags per candidate
            max_scan = min(max_scan_ti, ti_size)
            start = rng.integers(0, ti_size)

            for count in range(max_scan):
                y = np.unravel_index(int((start + count) % ti_size), ti_shape)

                filt_sg, filt_ti, filt_cond = [], [], []
                for j, lag in enumerate(lags_arr):
                    nb_ti = tuple(int(round(y[d] + lag[d])) for d in range(dim))
                    if all(0 <= nb_ti[d] < ti_shape[d] for d in range(dim)):
                        filt_sg.append(de_sg[j])
                        filt_ti.append(ti_data[nb_ti])
                        filt_cond.append(cond_mask_arr[j])

                if not filt_sg:
                    continue

                dv = ti.distance(
                    np.array(filt_sg),
                    np.array(filt_ti),
                    np.array(filt_cond, dtype=bool),
                    cond_weight,
                )
                if dv < best_d:
                    best_d = dv
                    best_v = ti_data[y]
                if dv <= t:
                    break

        return best_v if best_v is not None else _rand_ti_val()

    for x_i in path:
        sg[tuple(x_i)] = _simulate_node(x_i)

    return sg
```

- [ ] **Step 2: Add `boundary` to `DirectSampling.__init__`**

```python
    def __init__(self, ti, n_neighbors=32, scan_fraction=1.0,
                 threshold=0.0, cond_weight=1.0, boundary="strict",
                 parallel="auto", seed=np.nan):
        if boundary not in ("strict", "partial"):
            raise ValueError(
                f"boundary must be 'strict' or 'partial', got {boundary!r}"
            )
        if parallel not in ("auto", "inner", "outer", True, False, None):
            raise ValueError(
                f"parallel must be 'auto', 'inner', 'outer', True, False, "
                f"or None, got {parallel!r}"
            )
        super().__init__(model=None, dim=ti.ndim, value_type="scalar")
        self._ti = ti
        self._n_neighbors = n_neighbors
        self._scan_fraction = scan_fraction
        self._threshold = threshold
        self._cond_weight = cond_weight
        self._boundary = boundary
        self._parallel = parallel
        self._cond_pos = None
        self._cond_val = None
        self._seed = np.nan
        self._rng = None
        self._ds_seed = None
        self.update(seed=seed)
```

- [ ] **Step 3: Add warning for `partial` without rotation/affinity in `__call__`**

In `__call__`, after `self.update(seed=seed)` and before `pre_pos`:

```python
        if (self._boundary == "partial"
                and not hasattr(self, "_rotation_map")
                and not hasattr(self, "_affinity_map")):
            warnings.warn(
                "boundary='partial' is designed for non-stationary simulations "
                "with rotation_map or affinity_map (Mariethoz2010 §6.2). "
                "Without these, boundary='strict' avoids edge-biased distances.",
                UserWarning,
                stacklevel=2,
            )
```

Note: `_rotation_map` / `_affinity_map` are added in Task 7. Until then this
warning fires whenever `boundary="partial"` is used — which is correct.

- [ ] **Step 4: Pass `boundary` to `_ds_simulate` in `__call__`**

```python
        field = _ds_simulate(
            ti=self._ti,
            sg_shape=shape,
            n=self._n_neighbors,
            t=self._threshold,
            f=self._scan_fraction,
            seed=self._ds_seed,
            conditions=conditions,
            cond_weight=self._cond_weight,
            boundary=self._boundary,
        )
```

- [ ] **Step 5: Smoke-test**

```bash
cd /home/niklas/dev/GSTools
python -c "
import warnings
import numpy as np
import gstools as gs

ti = gs.TrainingImage(np.random.default_rng(0).integers(0, 2, (20, 20)))
pos = [np.arange(10, dtype=float)] * 2

# strict (default) works
ds = gs.DirectSampling(ti, boundary='strict', seed=1)
f1 = ds(pos)
assert not np.any(np.isnan(f1))

# partial warns (no rotation map yet)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    ds2 = gs.DirectSampling(ti, boundary='partial', seed=1)
    f2 = ds2(pos)
    assert any('partial' in str(x.message) for x in w)
assert not np.any(np.isnan(f2))

# invalid boundary raises
try:
    gs.DirectSampling(ti, boundary='bad')
    assert False
except ValueError:
    pass

print('Task 4 OK')
"
```

Expected: `Task 4 OK`

- [ ] **Step 6: Commit**

```bash
cd /home/niklas/dev/GSTools
git add src/gstools/mps/DS.py
git commit -m "feat(mps): add boundary parameter (strict/partial) to DirectSampling"
```

---

## Task 5: TI.py — l2/variation distance metrics

**Files:**
- Modify: `src/gstools/mps/TI.py`
- Modify: `src/gstools/mps/DS.py` (pass `lag_norms` through to `ti.distance()`)

Adds `distance="l1"/"l2"/"variation"` and `distance_power` to `TrainingImage`.
Adds `adjust_value()` for the variation mean-shift (Mariethoz2010 Eq. 9).
DS passes `lag_norms` (Euclidean norms of lag vectors) to `ti.distance()` so
l2 spatial weighting works.

- [ ] **Step 1: Extend `TrainingImage.__init__`**

Add `distance="l1"` and `distance_power=1.0` parameters:

```python
    def __init__(self, data, categorical=True, distance="l1",
                 distance_power=1.0):
        _valid = ("l1", "l2", "variation")
        if distance not in _valid:
            raise ValueError(
                f"distance must be one of {_valid}, got {distance!r}"
            )
        self._data = np.asarray(data)
        self._categorical = bool(categorical)
        self._multivariate = False
        self._distance_type = distance
        self._distance_power = float(distance_power)

        self._d_max = {}
        if not self._categorical:
            dmax = float(self._data.max() - self._data.min())
            self._d_max["_default"] = dmax if dmax > 0 else 1.0

        self._categories = {}
        if self._categorical:
            self._categories["_default"] = np.unique(self._data)
```

- [ ] **Step 2: Add `distance_type` property**

```python
    @property
    def distance_type(self):
        """str: Distance metric ('l1', 'l2', 'variation')."""
        return self._distance_type
```

- [ ] **Step 3: Replace `_distance_single` with full three-path version**

```python
    def _distance_single(self, de_sg, de_ti, cond_mask=None,
                         cond_weight=1.0, lag_norms=None):
        """Distance between two univariate data events.

        Categorical: Juda2022 Eq. 6 (all distance_type values use this).
        Continuous L1: Juda2022 Eq. 7.
        Continuous L2: Mariethoz2010 Eq. 4-5.
        Continuous variation: Mariethoz2010 Eq. 9.
        """
        n = len(de_sg)
        if n == 0:
            return 0.0

        if self._categorical:
            w = np.ones(n, dtype=np.float64)
            if cond_mask is not None:
                w[cond_mask] = cond_weight
            mismatches = (de_sg != de_ti).astype(np.float64)
            return float(np.dot(w, mismatches) / w.sum())

        d_max = self._d_max["_default"]

        if self._distance_type == "l1":
            w = np.ones(n, dtype=np.float64)
            if cond_mask is not None:
                w[cond_mask] = cond_weight
            diffs = np.abs(de_sg - de_ti) / d_max
            return float(np.dot(w, diffs) / w.sum())

        elif self._distance_type == "l2":
            norms = (np.asarray(lag_norms, dtype=np.float64)
                     if lag_norms is not None
                     else np.ones(n, dtype=np.float64))
            norms = np.where(norms == 0.0, 1e-10, norms)
            raw_w = norms ** (-self._distance_power)
            if cond_mask is not None:
                raw_w[cond_mask] *= cond_weight
            alpha = raw_w / (d_max**2 * raw_w.sum())
            return float(np.sqrt(np.dot(alpha, (de_sg - de_ti) ** 2)))

        else:  # "variation"
            sg_mean = float(np.mean(de_sg))
            ti_mean = float(np.mean(de_ti))
            diffs = (de_sg - sg_mean) - (de_ti - ti_mean)
            w = np.ones(n, dtype=np.float64)
            if cond_mask is not None:
                w[cond_mask] = cond_weight
            alpha = w / (d_max**2 * w.sum())
            return float(np.sqrt(np.dot(alpha, diffs**2)))
```

- [ ] **Step 4: Add `adjust_value` method**

```python
    def adjust_value(self, ti_val, de_sg, de_ti):
        """Adjust TI value before assignment (Mariethoz2010 Eq. 9).

        For variation distance on continuous variables:
            Z(x_i) = Z(y) − Z̄(y) + Z̄(x_i)
        For l1 and l2, returns ti_val unchanged.
        """
        if self._distance_type != "variation" or self._categorical:
            return ti_val
        return float(ti_val - float(np.mean(de_ti)) + float(np.mean(de_sg)))
```

- [ ] **Step 5: Update `_ds_simulate` in DS.py to compute `lag_norms` and call `adjust_value`**

In the `"strict"` branch of `_simulate_node`, after building `lags_arr`:
```python
        lag_norms = np.linalg.norm(lags_arr, axis=1)
```

Then in the inner scan loop, replace `best_v = ti_data[y]` with:
```python
                    best_v = ti.adjust_value(ti_data[y], de_sg, de_ti)
```

And pass `lag_norms` to `ti.distance`:
```python
                dv = ti.distance(de_sg, de_ti, cond_mask_arr, cond_weight,
                                 lag_norms)
```

Apply the same change in the `"partial"` branch (pass `lag_norms[filt_idx]`
where `filt_idx` tracks which lags survived the bounds check).

Full updated `_simulate_node` for the strict branch (partial follows the same
pattern — apply `lag_norms[filt_idx]` and `adjust_value`):

```python
    def _simulate_node(x_i):
        x_i = tuple(x_i)

        lags = []
        cond_mask = []
        for offset in offset_list:
            nb = tuple(x_i[d] + offset[d] for d in range(dim))
            if all(0 <= nb[d] < sg_shape[d] for d in range(dim)):
                if not np.isnan(sg[nb]):
                    lags.append(np.array(offset, dtype=np.float64))
                    cond_mask.append(bool(is_cond[nb]))
                    if len(lags) >= n_neighbors:
                        break

        if not lags:
            return _rand_ti_val()

        lags_arr = np.array(lags)
        cond_mask_arr = np.array(cond_mask)
        lag_norms = np.linalg.norm(lags_arr, axis=1)
        de_sg = np.array([
            sg[tuple(int(x_i[d] + lags[j][d]) for d in range(dim))]
            for j in range(len(lags))
        ])

        best_d = np.inf
        best_v = None

        if boundary == "strict":
            sw_lo = [0] * dim
            sw_hi = [ti_shape[d] - 1 for d in range(dim)]
            for lag in lags_arr:
                for d in range(dim):
                    sw_lo[d] = max(sw_lo[d], int(np.ceil(-lag[d])))
                    sw_hi[d] = min(sw_hi[d],
                                   int(np.floor(ti_shape[d] - 1 - lag[d])))
            if any(sw_lo[d] > sw_hi[d] for d in range(dim)):
                return _rand_ti_val()

            sw_shape = tuple(sw_hi[d] - sw_lo[d] + 1 for d in range(dim))
            sw_size = int(np.prod(sw_shape))
            max_scan = min(max_scan_ti, sw_size)
            start = rng.integers(0, sw_size)

            for count in range(max_scan):
                sw_flat = int((start + count) % sw_size)
                sw_nd = np.unravel_index(sw_flat, sw_shape)
                y = tuple(sw_lo[d] + sw_nd[d] for d in range(dim))

                valid = True
                for lag in lags_arr:
                    nb_ti = tuple(int(round(y[d] + lag[d])) for d in range(dim))
                    if not all(0 <= nb_ti[d] < ti_shape[d] for d in range(dim)):
                        valid = False
                        break
                if not valid:
                    continue

                de_ti = np.array([
                    ti_data[tuple(int(round(y[d] + lags_arr[j][d]))
                                  for d in range(dim))]
                    for j in range(len(lags))
                ])
                dv = ti.distance(de_sg, de_ti, cond_mask_arr, cond_weight,
                                 lag_norms)
                if dv < best_d:
                    best_d = dv
                    best_v = ti.adjust_value(ti_data[y], de_sg, de_ti)
                if dv <= t:
                    break

        else:  # "partial"
            max_scan = min(max_scan_ti, ti_size)
            start = rng.integers(0, ti_size)

            for count in range(max_scan):
                y = np.unravel_index(int((start + count) % ti_size), ti_shape)

                filt_sg, filt_ti, filt_cond, filt_norms = [], [], [], []
                filt_idx = []
                for j, lag in enumerate(lags_arr):
                    nb_ti = tuple(int(round(y[d] + lag[d])) for d in range(dim))
                    if all(0 <= nb_ti[d] < ti_shape[d] for d in range(dim)):
                        filt_idx.append(j)
                        filt_sg.append(de_sg[j])
                        filt_ti.append(ti_data[nb_ti])
                        filt_cond.append(cond_mask_arr[j])
                        filt_norms.append(lag_norms[j])

                if not filt_sg:
                    continue

                filt_sg_arr = np.array(filt_sg)
                filt_ti_arr = np.array(filt_ti)
                dv = ti.distance(
                    filt_sg_arr,
                    filt_ti_arr,
                    np.array(filt_cond, dtype=bool),
                    cond_weight,
                    np.array(filt_norms),
                )
                if dv < best_d:
                    best_d = dv
                    best_v = ti.adjust_value(ti_data[y], filt_sg_arr,
                                             filt_ti_arr)
                if dv <= t:
                    break

        return best_v if best_v is not None else _rand_ti_val()
```

- [ ] **Step 6: Smoke-test**

```bash
cd /home/niklas/dev/GSTools
python -c "
import numpy as np
import gstools as gs

rng = np.random.default_rng(0)
data = rng.uniform(0, 10, (20, 20))

# l1 (default)
ti_l1 = gs.TrainingImage(data, categorical=False, distance='l1')
ds = gs.DirectSampling(ti_l1, n_neighbors=8, scan_fraction=0.5, seed=42)
f1 = ds([np.arange(10, dtype=float)] * 2)
assert 0.0 <= float(f1.min()) and float(f1.max()) <= 10.0

# l2
ti_l2 = gs.TrainingImage(data, categorical=False, distance='l2')
ds2 = gs.DirectSampling(ti_l2, n_neighbors=8, scan_fraction=0.5, seed=42)
f2 = ds2([np.arange(10, dtype=float)] * 2)
assert 0.0 <= float(f2.min()) and float(f2.max()) <= 10.0

# variation
ti_v = gs.TrainingImage(data, categorical=False, distance='variation')
ds3 = gs.DirectSampling(ti_v, n_neighbors=8, scan_fraction=0.5, seed=42)
f3 = ds3([np.arange(10, dtype=float)] * 2)
assert f3.shape == (10, 10)

# adjust_value: variation shifts by mean difference
de_sg = np.array([2.0, 4.0])
de_ti = np.array([6.0, 8.0])
adj = ti_v.adjust_value(7.0, de_sg, de_ti)
# Z(x) = 7.0 - mean([6,8]) + mean([2,4]) = 7 - 7 + 3 = 3.0
assert abs(adj - 3.0) < 1e-10

# invalid distance type raises
try:
    gs.TrainingImage(data, categorical=False, distance='bad')
    assert False
except ValueError:
    pass

print('Task 5 OK')
"
```

Expected: `Task 5 OK`

- [ ] **Step 7: Commit**

```bash
cd /home/niklas/dev/GSTools
git add src/gstools/mps/TI.py src/gstools/mps/DS.py
git commit -m "feat(mps): add l2/variation distance metrics and adjust_value to TrainingImage"
```

---

## Task 6: TI.py + DS.py — multivariate

**Files:**
- Modify: `src/gstools/mps/TI.py`
- Modify: `src/gstools/mps/DS.py`

`TrainingImage` now accepts `data` as a `dict` of same-shape arrays. Per-variable
`categorical` (dict or scalar bool) and `weights` (default: equal). All existing
methods work on multivariate TIs through internal dispatch on `self._multivariate`.

`_ds_simulate` dispatches on `ti._multivariate`. In multivariate mode, all variables
are assigned from the same TI location `y*`. The primary variable's field is returned
by `post_field`; secondary variables are stored in `DirectSampling._extra_fields`.

- [ ] **Step 1: Update `TrainingImage.__init__` to accept dict**

```python
    def __init__(self, data, categorical=True, weights=None,
                 distance="l1", distance_power=1.0):
        _valid = ("l1", "l2", "variation")
        if distance not in _valid:
            raise ValueError(
                f"distance must be one of {_valid}, got {distance!r}"
            )
        self._distance_type = distance
        self._distance_power = float(distance_power)

        if isinstance(data, dict):
            self._variables = {}
            shapes = set()
            for name, arr in data.items():
                arr = np.asarray(arr)
                self._variables[name] = arr
                shapes.add(arr.shape)
            if len(shapes) != 1:
                raise ValueError("All variables must have the same shape.")
            self._shape = shapes.pop()
            self._multivariate = True
            self._categorical = (categorical if isinstance(categorical, dict)
                                 else {n: bool(categorical)
                                       for n in self._variables})
            n_vars = len(self._variables)
            self._weights = (weights if weights is not None
                             else {n: 1.0 / n_vars for n in self._variables})
        else:
            self._variables = {"_default": np.asarray(data)}
            self._shape = self._variables["_default"].shape
            self._multivariate = False
            self._categorical = {"_default": bool(categorical)}
            self._weights = {"_default": 1.0}

        self._d_max = {}
        for name, arr in self._variables.items():
            if not self._categorical[name]:
                dmax = float(arr.max() - arr.min())
                self._d_max[name] = dmax if dmax > 0 else 1.0

        self._categories = {}
        for name, arr in self._variables.items():
            if self._categorical[name]:
                self._categories[name] = np.unique(arr)
```

- [ ] **Step 2: Update `data`, `categorical`, `ndim`, `shape` properties**

```python
    @property
    def data(self):
        """ndarray or dict of ndarray: Training image data."""
        if self._multivariate:
            return dict(self._variables)
        return self._variables["_default"]

    @property
    def ndim(self):
        """int: Number of spatial dimensions."""
        return len(self._shape)

    @property
    def shape(self):
        """tuple: Shape of the training image."""
        return self._shape

    @property
    def size(self):
        """int: Total number of nodes."""
        return int(np.prod(self._shape))

    @property
    def categorical(self):
        """bool or dict of bool: Whether variable(s) are categorical."""
        if self._multivariate:
            return dict(self._categorical)
        return self._categorical["_default"]

    @property
    def variables(self):
        """list or None: Variable names for multivariate TI."""
        return list(self._variables.keys()) if self._multivariate else None
```

- [ ] **Step 3: Update `_distance_single` to take `name` parameter; update `distance` to loop over variables**

```python
    def _distance_single(self, de_sg, de_ti, name="_default",
                         cond_mask=None, cond_weight=1.0, lag_norms=None):
        """Distance between two data events for one variable."""
        n = len(de_sg)
        if n == 0:
            return 0.0
        if self._categorical[name]:
            w = np.ones(n, dtype=np.float64)
            if cond_mask is not None:
                w[cond_mask] = cond_weight
            return float(np.dot(w, (de_sg != de_ti).astype(np.float64)) / w.sum())
        d_max = self._d_max[name]
        if self._distance_type == "l1":
            w = np.ones(n, dtype=np.float64)
            if cond_mask is not None:
                w[cond_mask] = cond_weight
            return float(np.dot(w, np.abs(de_sg - de_ti) / d_max) / w.sum())
        elif self._distance_type == "l2":
            norms = (np.asarray(lag_norms, dtype=np.float64)
                     if lag_norms is not None
                     else np.ones(n, dtype=np.float64))
            norms = np.where(norms == 0.0, 1e-10, norms)
            raw_w = norms ** (-self._distance_power)
            if cond_mask is not None:
                raw_w[cond_mask] *= cond_weight
            alpha = raw_w / (d_max**2 * raw_w.sum())
            return float(np.sqrt(np.dot(alpha, (de_sg - de_ti) ** 2)))
        else:  # "variation"
            sg_mean, ti_mean = float(np.mean(de_sg)), float(np.mean(de_ti))
            diffs = (de_sg - sg_mean) - (de_ti - ti_mean)
            w = np.ones(n, dtype=np.float64)
            if cond_mask is not None:
                w[cond_mask] = cond_weight
            alpha = w / (d_max**2 * w.sum())
            return float(np.sqrt(np.dot(alpha, diffs**2)))

    def distance(self, de_sg, de_ti, cond_mask=None,
                 cond_weight=1.0, lag_norms=None):
        """Distance between two data events. Returns float in [0, 1].

        For multivariate: weighted sum over variables (Mariethoz2010 Eq. 8).
        de_sg / de_ti are dicts of arrays in multivariate mode.
        """
        if not self._multivariate:
            return self._distance_single(de_sg, de_ti, "_default",
                                         cond_mask, cond_weight, lag_norms)
        d = 0.0
        for name in self._variables:
            d += self._weights[name] * self._distance_single(
                de_sg[name], de_ti[name], name, cond_mask, cond_weight,
                lag_norms,
            )
        return d
```

- [ ] **Step 4: Update `adjust_value` for multivariate**

```python
    def adjust_value(self, ti_val, de_sg, de_ti):
        """Adjust TI value before assignment (Mariethoz2010 Eq. 9).

        Handles both scalar (univariate) and dict (multivariate).
        """
        if self._distance_type != "variation":
            return ti_val
        if isinstance(ti_val, dict):
            result = {}
            for name, v in ti_val.items():
                if not self._categorical[name]:
                    sg_k = de_sg[name] if isinstance(de_sg, dict) else de_sg
                    ti_k = de_ti[name] if isinstance(de_ti, dict) else de_ti
                    result[name] = float(
                        v - float(np.mean(ti_k)) + float(np.mean(sg_k))
                    )
                else:
                    result[name] = v
            return result
        if self._categorical["_default"]:
            return ti_val
        return float(ti_val - float(np.mean(de_ti)) + float(np.mean(de_sg)))
```

- [ ] **Step 5: Update `value_at` and `categories` for multivariate**

```python
    def value_at(self, index):
        """Value(s) at a flat or nd index."""
        if self._multivariate:
            return {n: self._variables[n][index] for n in self._variables}
        return self._variables["_default"][index]

    def categories(self, variable=None):
        """Unique category values for a variable, or None for continuous."""
        key = variable if variable is not None else "_default"
        return self._categories.get(key)
```

- [ ] **Step 6: Add multivariate dispatch to `_ds_simulate` in DS.py**

Add `_extra_fields` storage at the top of `_ds_simulate` closure setup, then
add the multivariate branch. The full updated function is long; the key
additions are:

At the top of `_ds_simulate`, after `is_cond` setup, add the multivariate
dispatch block that defines `_is_informed`, `_get_sg_val`, `_set_sg_val`,
`_get_ti_val`, `_rand_ti_val`, `_build_de_sg`, `_build_de_ti`, and
`_uninformed_nodes` closures — identical to the oracle's multivariate branch
in `/home/niklas/dev/MPSTools/tests/baseline/oracle.py` lines 48–156.

The complete updated `_ds_simulate` is built from the oracle at
`/home/niklas/dev/MPSTools/tests/baseline/oracle.py` lines 6–493, with
these adaptations:
1. Signature: replace `ti_data, categorical, d_max, weights` with `ti: TrainingImage`.
2. Multivariate setup (lines 48–156): replace `_dist(de_sg_v, de_ti_v, ...)` calls
   with `ti.distance(de_sg_v, de_ti_v, cond_mask, cond_weight, lag_norms)`.
3. Univariate setup (lines 159–219): same — replace `_dist` with `ti.distance`.
4. Best-value assignment: replace `best_v = _adjust_value(...)` with
   `best_v = ti.adjust_value(ti.value_at(y), de_sg, de_ti)`.
5. Use the `_precompute_offsets` already in DS.py (oracle lines 578–595 replaced).
6. Keep `boundary` and `cond_weight` params from Tasks 2–4.
7. Remove `distance_type`, `distance_power`, `weights` params (now carried by `ti`).
8. Remove the standalone `_distance`, `_adjust_value` helpers (oracle lines 496–575) —
   no longer needed; logic lives in `TrainingImage`.

The oracle's closure helpers (`_is_informed`, `_get_sg_val`, `_set_sg_val`,
`_get_ti_val`, `_rand_ti_val`, `_build_de_sg`, `_build_de_ti`,
`_uninformed_nodes`, `_simulated_noncond_nodes`, `_set_informed_nan`) are
kept verbatim — they handle uni/multivariate dispatch cleanly.

The multivariate return value is a `dict` of arrays. `DirectSampling.__call__`
must handle this.

- [ ] **Step 7: Update `DirectSampling.__call__` to handle multivariate return**

```python
        raw = _ds_simulate(
            ti=self._ti,
            sg_shape=shape,
            n=self._n_neighbors,
            t=self._threshold,
            f=self._scan_fraction,
            seed=self._ds_seed,
            conditions=conditions,
            cond_weight=self._cond_weight,
            boundary=self._boundary,
        )
        if isinstance(raw, dict):
            var_names = list(raw.keys())
            self._extra_fields = {k: raw[k] for k in var_names[1:]}
            field = raw[var_names[0]]
        else:
            field = raw
        return self.post_field(field, name, post_process, save)
```

- [ ] **Step 8: Add `extra_fields` property to `DirectSampling`**

```python
    @property
    def extra_fields(self):
        """dict: Secondary-variable fields from multivariate simulation."""
        return getattr(self, "_extra_fields", {})
```

- [ ] **Step 9: Smoke-test**

```bash
cd /home/niklas/dev/GSTools
python -c "
import numpy as np
import gstools as gs

rng = np.random.default_rng(0)
a = rng.integers(0, 2, (20, 20))
b = a * 10  # B = A*10 in TI → should hold in SG too

ti = gs.TrainingImage({'a': a, 'b': b}, categorical={'a': True, 'b': False})
assert ti._multivariate is True
assert ti.ndim == 2
assert ti.shape == (20, 20)

ds = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5, seed=42)
pos = [np.arange(8, dtype=float)] * 2
field = ds(pos)
assert field.shape == (8, 8)

# secondary variable accessible
sec = ds.extra_fields
assert 'b' in sec
assert sec['b'].shape == (8, 8)

# same-location property: B = A*10 in TI → B_sim = A_sim*10
primary = field  # 'a' field
secondary = sec['b']
assert np.allclose(secondary, primary * 10.0)

print('Task 6 OK')
"
```

Expected: `Task 6 OK`

- [ ] **Step 10: Commit**

```bash
cd /home/niklas/dev/GSTools
git add src/gstools/mps/TI.py src/gstools/mps/DS.py
git commit -m "feat(mps): add multivariate support to TrainingImage and DirectSampling"
```

---

## Task 7: TI.py + DS.py — nonstationarity

**Files:**
- Modify: `src/gstools/mps/TI.py`
- Modify: `src/gstools/mps/DS.py`

Adds `angles` / `anis` to `TrainingImage` (broadcast to uniform maps in DS).
Adds `set_rotation(rotation_map)` / `set_affinity(affinity_map)` to `DirectSampling`.
Applies `L'_i = R(θ) · A(α) · L_i` before TI lookup (Mariethoz2010 §6.2).
Uses `gstools.tools.geometric.matrix_rotate` / `matrix_anisotropify`.

- [ ] **Step 1: Add `angles` / `anis` to `TrainingImage.__init__`**

Add `angles=None, anis=None` parameters and store them:

```python
    def __init__(self, data, categorical=True, weights=None,
                 angles=None, anis=None,
                 distance="l1", distance_power=1.0):
        # ... existing init code ...
        self._angles = angles
        self._anis = anis
```

Add properties:

```python
    @property
    def angles(self):
        """Stationary rotation angles, or None."""
        return self._angles

    @property
    def anis(self):
        """Stationary anisotropy ratios, or None."""
        return self._anis
```

- [ ] **Step 2: Add `set_rotation` / `set_affinity` to `DirectSampling`**

```python
    def set_rotation(self, rotation_map):
        """Set per-node rotation angles for nonstationarity.

        Parameters
        ----------
        rotation_map : array-like
            Shape (*sg_shape, n_angles): 1 angle for 2D, 3 for 3D.
        Consider using boundary='partial' so lags that fall outside the TI
        after rotation are dropped per candidate (Mariethoz2010 §6.2).
        """
        self._rotation_map = np.asarray(rotation_map)

    def set_affinity(self, affinity_map):
        """Set per-node affinity ratios for nonstationarity.

        Parameters
        ----------
        affinity_map : array-like
            Shape (*sg_shape, n_ratios): dim-1 ratios per node.
        """
        self._affinity_map = np.asarray(affinity_map)
```

Also add `self._rotation_map = None` and `self._affinity_map = None` in `__init__`.

- [ ] **Step 3: Update the partial-warning check in `__call__`**

Replace the `hasattr` check with direct attribute checks:

```python
        if (self._boundary == "partial"
                and self._rotation_map is None
                and self._affinity_map is None
                and self._ti.angles is None
                and self._ti.anis is None):
            warnings.warn(
                "boundary='partial' is designed for non-stationary simulations "
                "with rotation_map or affinity_map (Mariethoz2010 §6.2). "
                "Without these, boundary='strict' avoids edge-biased distances.",
                UserWarning,
                stacklevel=2,
            )
```

- [ ] **Step 4: Broadcast `ti.angles` / `ti.anis` to uniform maps in `__call__`**

After the warning block, before calling `_ds_simulate`:

```python
        effective_rotation = self._rotation_map
        effective_affinity = self._affinity_map

        if effective_rotation is None and self._ti.angles is not None:
            angles = np.asarray(self._ti.angles).ravel()
            effective_rotation = (
                np.tile(angles, int(np.prod(shape)))
                .reshape(list(shape) + [len(angles)])
                .astype(np.float64)
            )
        if effective_affinity is None and self._ti.anis is not None:
            anis = np.asarray(self._ti.anis).ravel()
            effective_affinity = (
                np.tile(anis, int(np.prod(shape)))
                .reshape(list(shape) + [len(anis)])
                .astype(np.float64)
            )
```

Pass them to `_ds_simulate`:

```python
        raw = _ds_simulate(
            ...,
            rotation_map=effective_rotation,
            affinity_map=effective_affinity,
        )
```

- [ ] **Step 5: Add `rotation_map` / `affinity_map` params to `_ds_simulate` and apply transforms**

Add `rotation_map=None, affinity_map=None` to `_ds_simulate` signature.

Add this import at the **top of DS.py** (not inside the function):

```python
from gstools.tools.geometric import matrix_anisotropify, matrix_rotate
```

Inside `_simulate_node`, after building `lags_arr`, apply transforms:

```python
        if rotation_map is not None:
            lags_arr = (
                matrix_rotate(dim, rotation_map[x_i]) @ lags_arr.T
            ).T
        if affinity_map is not None:
            lags_arr = (
                matrix_anisotropify(dim, affinity_map[x_i]) @ lags_arr.T
            ).T
```

Note: for 3D, `rotation_map[x_i]` is a length-3 array (3 Euler angles).
`matrix_rotate` and `matrix_anisotropify` accept the same `angles`/`anis`
formats as `CovModel`. See `gstools.tools.geometric` source for signature details.

After transforming `lags_arr`, `lag_norms` must be recomputed:

```python
        lag_norms = np.linalg.norm(lags_arr, axis=1)
```

- [ ] **Step 6: Smoke-test**

```bash
cd /home/niklas/dev/GSTools
python -c "
import numpy as np
import gstools as gs

rng = np.random.default_rng(0)
ti_data = rng.integers(0, 2, (30, 30))
ti = gs.TrainingImage(ti_data)
pos = [np.arange(10, dtype=float)] * 2
shape = (10, 10)

# zero rotation = no rotation
ds_no_rot = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5, seed=42)
f_no = ds_no_rot(pos)

ds_zero = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5, seed=42)
zero_map = np.zeros(list(shape) + [1])
ds_zero.set_rotation(zero_map)
f_zero = ds_zero(pos)
assert np.array_equal(f_no, f_zero), 'zero rotation should equal no rotation'

# nonzero rotation gives different result
ds_rot = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5, seed=42)
rot_map = np.full(list(shape) + [1], np.pi / 4)
ds_rot.set_rotation(rot_map)
f_rot = ds_rot(pos)
assert not np.array_equal(f_no, f_rot), 'nonzero rotation should differ'

# stationary via ti.angles
ti_ang = gs.TrainingImage(ti_data, angles=0.0)
ds_ang = gs.DirectSampling(ti_ang, n_neighbors=8, scan_fraction=0.5, seed=42)
f_ang = ds_ang(pos)
assert np.array_equal(f_no, f_ang), 'angles=0 should equal no rotation'

print('Task 7 OK')
"
```

Expected: `Task 7 OK`

- [ ] **Step 7: Commit**

```bash
cd /home/niklas/dev/GSTools
git add src/gstools/mps/TI.py src/gstools/mps/DS.py
git commit -m "feat(mps): add nonstationarity support (rotation/affinity maps) to DirectSampling"
```

---

## Task 8: DS.py — postprocess

**Files:**
- Modify: `src/gstools/mps/DS.py`

Adds `postprocess=0` to `DirectSampling`. After the main simulation loop,
runs `postprocess` sequential re-simulation passes through all non-conditioning
nodes (Strebelle & Remy 2005, cited in Mariethoz2010 §7 ¶48).

- [ ] **Step 1: Add `postprocess` to `_ds_simulate` signature and add the pass loop**

Add `postprocess=0` to `_ds_simulate` signature. After the main loop
`for x_i in path: sg[...] = _simulate_node(x_i)`, add:

```python
    for _ in range(postprocess):
        pp_nodes = np.argwhere(~is_cond & ~np.isnan(
            sg if not ti._multivariate else sg[list(ti._variables.keys())[0]]
        ))
        pp_path = pp_nodes[rng.permutation(len(pp_nodes))]
        for x_i in pp_path:
            x_i_t = tuple(x_i)
            if ti._multivariate:
                for k in ti._variables:
                    sg[k][x_i_t] = np.nan
            else:
                sg[x_i_t] = np.nan
            if ti._multivariate:
                val = _simulate_node(np.array(x_i_t))
                if isinstance(val, dict):
                    for k in ti._variables:
                        sg[k][x_i_t] = val[k]
                else:
                    sg[list(ti._variables.keys())[0]][x_i_t] = val
            else:
                sg[x_i_t] = _simulate_node(np.array(x_i_t))
```

- [ ] **Step 2: Add `postprocess` to `DirectSampling.__init__`**

```python
    def __init__(self, ti, n_neighbors=32, scan_fraction=1.0,
                 threshold=0.0, cond_weight=1.0, boundary="strict",
                 parallel="auto", postprocess=0, seed=np.nan):
        # ... existing validation ...
        self._postprocess = postprocess
        # ... rest of init ...
```

- [ ] **Step 3: Pass `postprocess` to `_ds_simulate`**

```python
        raw = _ds_simulate(
            ...,
            postprocess=self._postprocess,
        )
```

- [ ] **Step 4: Smoke-test**

```bash
cd /home/niklas/dev/GSTools
python -c "
import numpy as np
import gstools as gs

rng = np.random.default_rng(0)
ti_data = rng.integers(0, 2, (20, 20))
ti = gs.TrainingImage(ti_data)
pos = [np.arange(8, dtype=float)] * 2

# postprocess=0 and postprocess=1 on same seed give different output
ds0 = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5,
                         postprocess=0, seed=42)
f0 = ds0(pos)

ds1 = gs.DirectSampling(ti, n_neighbors=8, scan_fraction=0.5,
                         postprocess=1, seed=42)
f1 = ds1(pos)

assert not np.array_equal(f0, f1), 'postprocess=1 should differ from 0'
assert not np.any(np.isnan(f1)), 'no NaN after postprocess'

print('Task 8 OK')
"
```

Expected: `Task 8 OK`

- [ ] **Step 5: Commit**

```bash
cd /home/niklas/dev/GSTools
git add src/gstools/mps/DS.py
git commit -m "feat(mps): add postprocess parameter to DirectSampling"
```

---

## Spec Coverage Check

| Spec requirement | Task |
|---|---|
| TI: d_max, distance(), _distance_single() l1, value_at(), categories() | Task 1 |
| DS: clean sim calling ti.distance(), fix conditions tie-breaking | Task 2 |
| DS: parallel parameter + _resolve_parallel_mode + _available_ram_bytes | Task 3 |
| DS: boundary strict/partial | Task 4 |
| TI: l2/variation, adjust_value(), distance_power | Task 5 |
| DS: lag_norms passed to ti.distance, adjust_value called on best match | Task 5 |
| TI + DS: multivariate dict data, joint distance, extra_fields | Task 6 |
| TI + DS: angles/anis, set_rotation, set_affinity, lag transforms | Task 7 |
| DS: postprocess sequential re-simulation pass | Task 8 |
| No Rust dispatch, no config.py changes | Throughout (absent by design) |
| No quality.py | Throughout (absent by design) |
| No tests | Throughout (absent by design) |
