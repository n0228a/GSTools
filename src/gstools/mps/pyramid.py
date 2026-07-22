"""Multi-resolution pyramid for Direct Sampling.

Attribution: the pyramid is from Straubhaar, Renard & Chugunova (2020),
"Multiple-point statistics using multi-resolution images", applied in
Juda et al. 2022 §4.1 (L=2, r=2). It is NOT from Mariethoz 2010 — M10
para [25] argues against multigrids (see DS_Algorithm_Explanation.md §10).

Coarse-to-fine mechanics: the coarsest level is simulated first; each
level's output is injected into the next finer level as ``preset`` values
(informed but re-simulatable, not hard conditioning) at the origin-anchored
fine nodes ``j * r``. Post-processing passes (Me13 §4), when configured,
re-draw preset nodes from the finer level's own TI — this restores the
TI-subset property under ``"average"`` coarsening.
"""

import numpy as np

from gstools.mps.training_image import TrainingImage, Variable

__all__ = ["Pyramid"]

_METHODS = ("majority", "subsample", "average")


class Pyramid:
    """Multi-resolution pyramid configuration for Direct Sampling.

    Parameters
    ----------
    levels : :class:`int`
        Number of coarse levels above level 0 (the original resolution).
        Juda2022 Test Case 1 uses 2. Must be >= 1.
    reduction : :class:`int`, optional
        Per-level coarsening factor r (>= 2). Default: 2.
    method : :class:`str` or :class:`dict` or None, optional
        Coarsening method: ``"majority"`` (categorical only),
        ``"subsample"``, or ``"average"`` (continuous only, NaN-aware block
        mean). ``None`` (default) picks per kind: categorical ->
        ``"majority"``, continuous -> ``"subsample"``. A dict maps variable
        names to methods (missing names use the per-kind default).

        The continuous default is ``"subsample"`` because block averages
        are not TI values; with ``"average"`` the transferred nodes hold
        non-TI values at level 0 unless ``MPSModel(post_processing >= 1)``
        re-simulates them.
    var_levels : :class:`dict` or None, optional
        ``{variable: int in [0, levels]}`` — the coarsest pyramid level at
        which the variable exists; it enters the simulation at that level
        (Straubhaar 2020 per-variable level count). Missing variables are
        present at all levels. At least one variable must be present at the
        coarsest level. Default: ``None`` (all variables at all levels).

    Examples
    --------
    >>> import numpy as np
    >>> import gstools as gs
    >>> ti = gs.TrainingImage(np.zeros((32, 32)), categorical=True)
    >>> model = gs.MPSModel(ti, pyramid=gs.Pyramid(levels=2, reduction=2))
    """

    def __init__(self, levels, reduction=2, method=None, var_levels=None):
        self._levels = int(levels)
        if self._levels < 1:
            raise ValueError(f"Pyramid: levels must be >= 1, got {levels!r}")
        self._reduction = int(reduction)
        if self._reduction < 2:
            raise ValueError(
                f"Pyramid: reduction must be >= 2, got {reduction!r}"
            )
        if method is not None and not isinstance(method, (str, dict)):
            raise TypeError(
                f"Pyramid: method must be None, a str, or a dict, "
                f"got {type(method)!r}"
            )
        self._method = method
        self._var_levels = None if var_levels is None else dict(var_levels)

    @property
    def levels(self):
        """:class:`int`: Number of coarse levels above level 0."""
        return self._levels

    @property
    def reduction(self):
        """:class:`int`: Per-level coarsening factor r."""
        return self._reduction

    @property
    def method(self):
        """Coarsening method spec (str, dict, or None for per-kind defaults)."""
        return self._method

    @property
    def var_levels(self):
        """:class:`dict` or None: coarsest level per variable."""
        return None if self._var_levels is None else dict(self._var_levels)

    def resolve_methods(self, ti):
        """Resolve the per-variable coarsening method against a TI.

        Parameters
        ----------
        ti : :any:`TrainingImage`

        Returns
        -------
        dict
            ``{variable_name: method}`` with kind-validated methods.
        """
        out = {}
        for var in ti.variables:
            if isinstance(self._method, dict):
                m = self._method.get(var.name)
            else:
                m = self._method
            if m is None:
                m = "majority" if var.categorical else "subsample"
            if m not in _METHODS:
                raise ValueError(
                    f"Pyramid: unknown coarsening method {m!r} for variable "
                    f"{var.name!r}; valid: {_METHODS!r}"
                )
            if m == "majority" and not var.categorical:
                raise ValueError(
                    f"Pyramid: method 'majority' requires a categorical "
                    f"variable, but {var.name!r} is continuous."
                )
            if m == "average" and var.categorical:
                raise ValueError(
                    f"Pyramid: method 'average' requires a continuous "
                    f"variable, but {var.name!r} is categorical."
                )
            out[var.name] = m
        return out

    def resolve_var_levels(self, ti):
        """Resolve per-variable level counts against a TI.

        Returns
        -------
        dict
            ``{variable_name: int}``; missing variables get ``levels``.
        """
        names = [v.name for v in ti.variables]
        vl = self._var_levels or {}
        unknown = set(vl) - set(names)
        if unknown:
            raise ValueError(
                f"Pyramid: var_levels contains unknown variable(s) "
                f"{sorted(map(str, unknown))}; TI variables: "
                f"{sorted(map(str, names))}"
            )
        out = {n: int(vl.get(n, self._levels)) for n in names}
        for n, lev in out.items():
            if not 0 <= lev <= self._levels:
                raise ValueError(
                    f"Pyramid: var_levels[{n!r}] must be in "
                    f"[0, {self._levels}], got {lev}"
                )
        if max(out.values()) < self._levels:
            raise ValueError(
                f"Pyramid: no variable is present at the coarsest level "
                f"{self._levels}; reduce levels or raise a variable's "
                "level count."
            )
        return out

    def __repr__(self):
        args = [f"levels={self._levels}", f"reduction={self._reduction}"]
        if self._method is not None:
            args.append(f"method={self._method!r}")
        if self._var_levels:
            args.append(f"var_levels={self._var_levels!r}")
        return f"Pyramid({', '.join(args)})"


def _coarsen_array(data, r, method):
    """Coarsen an n-D array by one level step (origin-anchored blocks).

    ``"subsample"`` takes ``data[::r, ...]``; ``"majority"``/``"average"``
    aggregate each ``[j*r, min((j+1)*r, s))`` block per axis (edge blocks
    may be partial) over finite cells only. Majority ties break to the
    smallest value (``numpy.unique`` sorts; first argmax wins). An
    all-NaN block yields NaN.
    """
    if method == "subsample":
        return np.array(data[tuple(slice(None, None, r) for _ in data.shape)])
    shape = data.shape
    coarse_shape = tuple(-(-s // r) for s in shape)
    out = np.empty(coarse_shape, dtype=np.float64)
    for cidx in np.ndindex(*coarse_shape):
        block = data[
            tuple(
                slice(c * r, min((c + 1) * r, s)) for c, s in zip(cidx, shape)
            )
        ].ravel()
        vals = block[np.isfinite(block)]
        if vals.size == 0:
            out[cidx] = np.nan
        elif method == "average":
            out[cidx] = vals.mean()
        else:  # majority
            uniq, counts = np.unique(vals, return_counts=True)
            out[cidx] = uniq[np.argmax(counts)]
    return out


def _coarsen_ti(ti, r, methods, keep):
    """Coarsen every kept variable of ``ti`` by one level step.

    Variable metadata (kind, distance, weight, n_neighbors, max_radius,
    penalty_matrix) is preserved; ``n_neighbors``/``max_radius`` stay in
    index units — the coarser lattice IS the larger physical neighbourhood.
    """
    new_vars = [
        Variable(
            v.name,
            _coarsen_array(v.data, r, methods[v.name]),
            categorical=v.categorical,
            distance=v.distance,
            weight=v.weight,
            n_neighbors=v.n_neighbors,
            max_radius=v.max_radius,
            penalty_matrix=v.penalty_matrix,
        )
        for v in ti.variables
        if v.name in keep
    ]
    if ti.multivariate:
        return TrainingImage(new_vars, distance_power=ti.distance_power)
    return TrainingImage(new_vars[0], distance_power=ti.distance_power)


def _coarsen_conditions(conditions, r, fine_shape):
    """Down-sample ``{idx: {var: val}}`` hard data by one level step.

    Each fine node maps to its nearest coarse anchor ``round(i / r)``
    (integer arithmetic, clipped). Collisions are resolved per variable:
    the value whose fine node is closest to the coarse anchor wins; ties
    keep the first-seen entry (dict insertion order — deterministic).
    """
    coarse_shape = tuple(-(-s // r) for s in fine_shape)
    best = {}  # coarse idx -> {var: (val, dist_sq)}
    for idx, vd in conditions.items():
        cidx = tuple(
            min((i + r // 2) // r, cs - 1) for i, cs in zip(idx, coarse_shape)
        )
        dist_sq = sum((i - c * r) ** 2 for i, c in zip(idx, cidx))
        slot = best.setdefault(cidx, {})
        for v, val in vd.items():
            if v not in slot or dist_sq < slot[v][1]:
                slot[v] = (val, dist_sq)
    return {
        ci: {v: val for v, (val, _) in slot.items()}
        for ci, slot in best.items()
    }
