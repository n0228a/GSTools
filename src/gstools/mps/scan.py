"""TI search-window scanning for Direct Sampling.

Scans a candidate window for the best weighted-distance match (greedy
early-exit in DS mode, running-best argmin in DSBC mode). Depends only on
distance.py; receives TI arrays and the vectorized distance callable as
parameters (does not own a TrainingImage).
"""

from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng

from gstools import config as _mps_config
from gstools.mps.distance import compute_node_weights

if _mps_config._GSTOOLS_CORE_AVAIL:  # pragma: no cover
    import gstools_core as _gstools_core

    # MPS exports were added incrementally to gstools_core. Capability checks
    # keep an older installed core from breaking GSTools import: any missing
    # function simply leaves that particular MPS operation on the Python path.
    _mps_dist_block_cat_gsc = getattr(
        _gstools_core, "mps_dist_block_cat", None
    )
    _mps_dist_block_cat_rayon_gsc = getattr(
        _gstools_core, "mps_dist_block_cat_rayon", None
    )
    _mps_dist_block_cat_masked_gsc = getattr(
        _gstools_core, "mps_dist_block_cat_masked", None
    )
    _mps_dist_block_l1_gsc = getattr(_gstools_core, "mps_dist_block_l1", None)
    _mps_dist_block_l1_masked_gsc = getattr(
        _gstools_core, "mps_dist_block_l1_masked", None
    )
    _mps_dist_block_l2_gsc = getattr(_gstools_core, "mps_dist_block_l2", None)
    _mps_dist_block_l2_masked_gsc = getattr(
        _gstools_core, "mps_dist_block_l2_masked", None
    )
    _mps_dist_block_lp_gsc = getattr(_gstools_core, "mps_dist_block_lp", None)
    _mps_dist_block_lp_masked_gsc = getattr(
        _gstools_core, "mps_dist_block_lp_masked", None
    )
    _mps_dist_block_variation_gsc = getattr(
        _gstools_core, "mps_dist_block_variation", None
    )
    _mps_dist_block_variation_masked_gsc = getattr(
        _gstools_core, "mps_dist_block_variation_masked", None
    )
    _mps_scan_node_gsc = getattr(_gstools_core, "mps_scan_node", None)
    _mps_scan_node_cat_gsc = getattr(_gstools_core, "mps_scan_node_cat", None)
else:  # pragma: no cover
    _mps_dist_block_cat_gsc = None
    _mps_dist_block_cat_rayon_gsc = None
    _mps_dist_block_cat_masked_gsc = None
    _mps_dist_block_l1_gsc = None
    _mps_dist_block_l1_masked_gsc = None
    _mps_dist_block_l2_gsc = None
    _mps_dist_block_l2_masked_gsc = None
    _mps_dist_block_lp_gsc = None
    _mps_dist_block_lp_masked_gsc = None
    _mps_dist_block_variation_gsc = None
    _mps_dist_block_variation_masked_gsc = None
    _mps_scan_node_gsc = None
    _mps_scan_node_cat_gsc = None

# Action Plan 4 experiment. This remains private and disabled by default so
# existing backend behavior is unchanged outside the isolated scaling harness.
_MPS_RAYON_CANDIDATES = False

# Action Plan 6 measurement override. Normal dispatch uses the generic full
# scan only where the end-to-end harness found a benefit (positive-threshold
# DS). Tests and benchmarks can force it to exercise every supported metric.
_MPS_FULL_NODE_SCAN_FORCE = False

# Diagnostic hook, disabled by default and zero-cost when off: set to a list
# to record ``(best_d, accepted)`` once per scanned node, in simulation order.
# DSBC copies the running-best candidate, so ``best_d`` IS the distance the
# pasted value came from -- what a threshold would have to sit above to fire.
_MPS_SCAN_TRACE = None
# Node coordinates appended by simulate._simulate_node, index-aligned with
# _MPS_SCAN_TRACE; a length mismatch means a non-_scan_window path ran.
_MPS_SCAN_TRACE_NODES = []

# Criterion shows that Rayon overhead dominates below a full scan chunk.
# Keep the prototype serial for partial blocks; the scaling harness evaluates
# whether parallelizing 4096-candidate chunks pays off end to end.
_MPS_RAYON_MIN_CANDIDATES = 4096


@dataclass(frozen=True)
class _ScanConfig:
    """Domain-constant inputs to :func:`_scan_for_match`, built once per domain.

    Every field is invariant across the whole node path *and* across
    post-processing passes — it describes the domain (TI data, shape,
    per-variable ``d_max``) and engine-level search configuration, never
    anything that varies per pass.  ``scan_fraction`` is a per-pass value
    (Meerschman2013 §4 divides it by ``post_processing_factor``) and is passed
    to :func:`_scan_for_match` directly instead, from the caller's ``_Pass``.
    """

    variables: tuple
    weights: dict
    ti_vars: dict
    ti_flat: dict
    ti_strides: dict
    ti_shape: object
    threshold: float  # or None -> DSBC (see _scan_window's dsbc computation)
    cond_weight: dict  # {var: float} -- per-variable, from the kernel TI
    distance_power: dict  # {var: float} -- per-variable, from the kernel TI
    vec_distance_var: object
    d_max: dict  # {var: float or None} — normalization range per variable,
    # from the domain's (zone) TI; kernel type stays primary's.
    ti_has_nan: bool
    # New fields for Rust dispatch (populated by _DirectSamplingEngine)
    var_has_nan: dict  # {v: bool} — per-variable masked-distance dispatch
    var_categorical: dict  # {v: bool}
    var_p_norm: dict  # {v: float or None}  — None for categorical / variation
    var_d_max: dict  # {v: float or None}  — None for categorical
    ti_flat_f64: dict  # {v: ndarray float64} — pre-cast flat TI for Rust
    ti_matrix_f64: object  # float64, shape (n_variables, ti_size)
    var_index: dict  # {v: row in ti_matrix_f64}
    var_variation_p: dict  # {v: float or None}  — variation Lp exponent; None unless variation
    var_penalty_matrix: dict  # {v: ndarray or None} — per-category mismatch
    # penalties; the Rust categorical kernels only implement the plain binary
    # mismatch, so any variable with one set must stay on the Python path.
    scan_path: str = "sequential"  # window scan order; see module docstring


# DS-mode scan block size.  Large enough that per-call NumPy overhead is
# negligible (essentially full vectorization speed), small enough that the
# greedy DS scan does not overcompute far past the first accepted match.
# This is a call-overhead-amortization constant, not a cache-tuned one.
_SCAN_BLOCK = 4096

# Experimental DSBC selection-pool size (see _pool_scan). 1 keeps the running-
# best argmin exactly. k > 1 draws uniformly from the k closest candidates,
# restoring the conditional variance argmin removes. Every pooled candidate is
# a real TI anchor, so the subset property is unaffected.
_MPS_N_BEST = 1

# Softmax temperature for the k-best draw: p_i ~ exp(-d_i / tau) over the pooled
# candidates. None (default) draws uniformly. tau -> 0 recovers argmin, so tau is
# the continuous dial and _MPS_N_BEST only bounds the pool.
#
# RELATIVE to the pool's own distance range: tau_abs = _MPS_TAU * (dmax - dmin).
# Absolute temperatures would need recalibrating whenever the normalization,
# catalogue size or channel count moves the distance scale -- the same failure
# that made a literature threshold of t=0.08 meaningless here. A degenerate pool
# (zero range) falls back to the uniform draw.
_MPS_TAU = None

# Diagnostic hook, disabled by default and zero-cost when off: set to a dict
# to accumulate [sum, count] of each variable's WEIGHTED contribution to the
# joint distance, i.e. the effective (not nominal) per-variable weighting.
_MPS_DIST_TRACE = None

# Diagnostic hook, disabled by default and zero-cost when off: set to a list to
# record, once per scanned node, the DISTANCE FRONTIER as a function of the
# candidate's centre value -- the minimum distance attainable in each magnitude
# bin, plus that bin's candidate count. Answers whether argmin rejects a
# high-magnitude candidate because it is genuinely a worse pattern match, or
# because the metric prices magnitude.
_MPS_CORE_TRACE = None
_CORE_BINS = (0.0, 1.0, 2.0, 5.0, 10.0, np.inf)
_CORE_SCRATCH = []
_CORE_SPLIT = {}

# Spatially GATED acceptance threshold. Set to an array over the simulation
# grid to give each node its own t; None (default) uses cfg.threshold globally.
# Motivation: run101 shows a GLOBAL t degrades CRPS steeply (0.327 -> 0.586 at
# t=0.04) because it injects tolerance into the ~90% of nodes where argmin is
# already right. The frontier says extreme candidates sit ~2.6x out ONLY at
# core nodes, so the tolerance belongs only there.
# Forces the Python scan path, since the Rust dispatch keys off a single
# cfg.threshold for the whole domain.
_MPS_THRESHOLD_GRID = None
_MPS_CURRENT_NODE = None
# Current node coordinate, set by simulate._simulate_node right before the
# scan and read at the CORE_TRACE append site below -- appending it there
# (rather than via the separately-gated _MPS_SCAN_TRACE_NODES list) keeps the
# node 1:1 with its trace record even for nodes whose scan short-circuits
# (e.g. the random/degenerate-pool fallback) before this block ever runs,
# since such nodes simply never append to _MPS_CORE_TRACE either.
_MPS_CORE_NODE = None


def _random_scan_positions(u_start_i, win_size, max_scan):
    """Draw max_scan distinct window positions in random order.

    Keyed on the node's own u_start_i so no extra RNG draw is needed
    (see _scan_window). Shared by both the pure-Python and Rust-dispatch
    scan paths so their random orders can never drift apart.
    """
    return default_rng(int(u_start_i * 2**53)).choice(
        win_size, max_scan, replace=False
    )


def _pool_scan(
    lo,
    win_shape,
    u_start_i,
    max_scan,
    win_size,
    positions,
    start,
    dist_fn,
    n_best,
):
    """DSBC variant: draw uniformly from the n_best closest candidates.

    Mirrors _scan_window's block loop and its non-finite-block skip exactly,
    so n_best is the only difference between the two.
    """
    pool_d, pool_y = None, None
    for b0 in range(0, max_scan, _SCAN_BLOCK):
        b1 = min(b0 + _SCAN_BLOCK, max_scan)
        pos_blk = (
            (start + np.arange(b0, b1)) % win_size
            if positions is None
            else positions[b0:b1]
        )
        y_blk = lo + np.column_stack(np.unravel_index(pos_blk, win_shape))
        d_blk = dist_fn(y_blk)
        # Same gate as _scan_window: a block whose argmin is non-finite
        # contributes nothing (NaN wins argmin, so the block holds a NaN).
        if not np.isfinite(d_blk[int(np.argmin(d_blk))]):
            continue
        m = min(n_best, d_blk.size)
        idx = (
            np.argpartition(d_blk, m - 1)[:m]
            if m < d_blk.size
            else np.arange(d_blk.size)
        )
        cd, cy = d_blk[idx], y_blk[idx]
        if pool_d is not None:
            cd, cy = np.concatenate([pool_d, cd]), np.concatenate([pool_y, cy])
        if cd.size > n_best:
            keep = np.argpartition(cd, n_best - 1)[:n_best]
            cd, cy = cd[keep], cy[keep]
        pool_d, pool_y = cd, cy
    if pool_d is None:
        if _MPS_SCAN_TRACE is not None:
            _MPS_SCAN_TRACE.append((float("inf"), False))
        return None
    fin = np.isfinite(pool_d)
    pool_d, pool_y = pool_d[fin], pool_y[fin]
    if pool_d.size == 0:
        if _MPS_SCAN_TRACE is not None:
            _MPS_SCAN_TRACE.append((float("inf"), False))
        return None
    # Own RNG stream, keyed on the node like _random_scan_positions so the
    # draw is reproducible and uncorrelated with the scan order.
    rng = default_rng(int(u_start_i * 2**53) ^ 0x6B657374)
    if _MPS_TAU is None:
        j = int(rng.integers(pool_d.size))
    else:
        # Shift by the min before exponentiating: exp(-(d - dmin)/tau) is the
        # same distribution and cannot underflow to an all-zero weight vector.
        tau = _MPS_TAU * float(pool_d.max() - pool_d.min())
        if not (tau > 0.0):
            # Degenerate pool (all distances equal): uniform. Must NOT return
            # early -- the _MPS_SCAN_TRACE append below keeps the trace
            # index-aligned with _MPS_SCAN_TRACE_NODES.
            j = int(rng.integers(pool_d.size))
        else:
            w = np.exp(-(pool_d - pool_d.min()) / tau)
            s = w.sum()
            j = (
                int(np.argmin(pool_d))
                if not np.isfinite(s) or s <= 0
                else int(rng.choice(pool_d.size, p=w / s))
            )
    if _MPS_SCAN_TRACE is not None:
        # Distance of the value actually pasted -- this is what an equivalent
        # acceptance threshold would have to sit above.
        _MPS_SCAN_TRACE.append((float(pool_d[j]), False))
    return pool_y[j]


def _scan_window(
    lo,
    win_shape,
    u_start_i,
    max_scan,
    threshold,
    dist_fn,
    scan_path="sequential",
    positions=None,
    n_best=1,
):
    """Chunked vectorized TI window scan (pure-Python path).

    Parameters
    ----------
    lo : numpy.ndarray
        Lower-left anchor of the search window in TI coordinates.
    win_shape : tuple of int
        Shape of the search window.
    u_start_i : float
        The node's pre-drawn uniform in ``[0, 1)``.  ``"sequential"``: start
        position ``int(u_start_i * win_size)``.  ``"random"``: stream seed.
    max_scan : int
        Maximum number of candidates to evaluate.
    threshold : float or None
        Distance threshold for early exit (DS mode). ``None`` → DSBC
        (no early exit).
    dist_fn : callable
        ``dist_fn(y_blk)`` → 1-D distance array for a block of candidate
        anchor coordinates ``y_blk`` of shape ``(b, dim)``.
    scan_path : str, optional
        ``"sequential"`` (default) or ``"random"``.  See the module docstring.

    Returns
    -------
    y : numpy.ndarray, shape (dim,) or None
        Coordinate of the best matching TI anchor, or ``None`` when no candidate
        in the window has a finite distance (every candidate excluded — e.g. an
        all-undefined window on a masked TI). The caller treats ``None`` as the
        empty-neighbourhood case and draws a defined TI cell.
    """
    win_size = int(np.prod(win_shape))
    # "sequential": wrap-around walk from a random start (Mariethoz2010 ¶19,
    # Juda2022 §2). "random": max_scan distinct cells in random order, keyed on
    # the node's own u_start_i so no extra RNG draw is needed.
    #
    # Deliberate exception to the "use gstools.random.RNG" rule: reproducibility
    # is unaffected, since the only entropy is u_start_i, itself drawn from the
    # engine's gstools RNG. gstools RNG cannot serve this call anyway -- it
    # exposes no ``choice``, and its legacy-RandomState ``.random`` implements
    # ``choice(replace=False)`` via a full permutation, i.e. O(win_size)
    # instead of O(max_scan).
    #
    # Positions are built per ``_SCAN_BLOCK`` inside the loop below rather than
    # for all ``max_scan`` candidates upfront, since the greedy DS scan
    # typically exits long before the last block. This is unconditional
    # rather than mode-gated: the sequential walk's block is a cheap arange
    # either way, and it is bit-identical to building everything upfront.
    if scan_path == "random":
        # _scan_for_match already builds this array for the whole-node Rust
        # dispatch from the same (u_start_i, win_size, max_scan); recomputing
        # it here cost ~17% of runtime. Deterministic in those three, so
        # reusing it is bit-identical.
        if positions is None:
            positions = _random_scan_positions(u_start_i, win_size, max_scan)
        start = None
    else:
        positions = None
        start = int(u_start_i * win_size)

    # DSBC has no threshold-based early exit, but an exact match d == 0 is
    # the best attainable candidate, so accept it immediately (retained for
    # categorical, where d == 0 is common — Juda2022 §2). Both modes
    # therefore scan in blocks of ``_SCAN_BLOCK`` and track the running best,
    # so a full-window DSBC scan never materialises one giant distance array.
    # Block argmin + strict ``<`` keep the first-in-scan-order winner,
    # matching a single ``argmin`` over the whole window.
    dsbc = threshold is None
    accept = 0.0 if dsbc else threshold

    if dsbc and n_best > 1:
        return _pool_scan(
            lo,
            win_shape,
            u_start_i,
            max_scan,
            win_size,
            positions,
            start,
            dist_fn,
            n_best,
        )

    # DS mode (threshold > 0): strict acceptance d < t (Mariethoz2010 ¶23).
    # DSBC mode: accept the exact match d == 0, i.e. d <= 0, since distances
    # are non-negative (Juda2022 §2).
    best_d, best_y = np.inf, None
    for b0 in range(0, max_scan, _SCAN_BLOCK):
        b1 = min(b0 + _SCAN_BLOCK, max_scan)
        # Same integers the upfront build produced for this slice: the
        # sequential walk is ``(start + k) % win_size`` elementwise, and
        # ``unravel_index`` is elementwise, so blocking changes nothing.
        pos_blk = (
            (start + np.arange(b0, b1)) % win_size
            if positions is None
            else positions[b0:b1]
        )
        y_blk = lo + np.column_stack(np.unravel_index(pos_blk, win_shape))
        d_blk = dist_fn(y_blk)
        under = d_blk <= accept if dsbc else d_blk < accept
        if np.any(under):
            j = int(np.argmax(under))
            if _MPS_SCAN_TRACE is not None:
                _MPS_SCAN_TRACE.append((float(d_blk[j]), True))
            return y_blk[j]
        k = int(np.argmin(d_blk))
        if d_blk[k] < best_d:
            best_d = float(d_blk[k])
            best_y = y_blk[k]
    # ``best_y is None`` means every candidate had a non-finite (excluded)
    # distance — only reachable on a masked TI where the whole window is
    # undefined. Signal the caller to fall back to a defined TI draw (M10
    # para [15] empty-neighbourhood handling). For a fully-defined TI the loop
    # always sets best_y, so this returns a real anchor unchanged.
    if _MPS_SCAN_TRACE is not None:
        _MPS_SCAN_TRACE.append((float(best_d), False))
    return best_y


def _scan_for_match(
    lo,
    win_shape,
    int_lags,
    de_v,
    cm_v,
    ln_v,
    u_start_i,
    scan_targets,
    scan_fraction,
    cfg,
):
    """Joint multivariate DS scan over a TI search window.

    Scans candidates in randomized order (greedy DS early-exit when the first
    candidate with distance ``< threshold`` is found; DSBC running-best argmin
    when ``threshold`` is ``None``).  The window is the per-variable intersection so
    every ``y + lag`` is in bounds and the gather needs no clipping; the h=0
    lag maps to ``y`` itself.  Returns the single best TI anchor coordinate
    ``y``, or ``None`` when every candidate in the window is undefined (masked
    TI — caller falls back to a random defined cell).

    Parameters
    ----------
    scan_fraction : float
        This pass's scan fraction (main path, or divided by
        ``post_processing_factor`` for a post-pass) — the one field of the
        old ``_ScanConfig`` that varies per pass rather than per domain.
    """
    # Spatially gated threshold: this node's t, falling back to the domain's.
    node_thr = cfg.threshold
    if _MPS_THRESHOLD_GRID is not None and _MPS_CURRENT_NODE is not None:
        v = float(
            _MPS_THRESHOLD_GRID[_MPS_CURRENT_NODE[-2], _MPS_CURRENT_NODE[-1]]
        )
        node_thr = v if v > 0.0 else None

    win_size = int(np.prod(win_shape))
    ti_size = int(np.prod(cfg.ti_shape))
    # Scan fraction is of the TI (Mariethoz2010 ¶24, Juda2022 §2), capped at
    # the valid search window so we never wrap around and re-scan anchors.
    max_scan = max(1, min(win_size, int(scan_fraction * ti_size)))

    active_vars = [v for v in cfg.variables if v in int_lags]
    active_w_total = sum(cfg.weights[v] for v in active_vars)
    precomp_w = {
        v: compute_node_weights(
            len(de_v[v]),
            ln_v[v],
            cfg.distance_power[v],
            cm_v[v],
            cfg.cond_weight[v],
        )
        for v in active_vars
    }
    # Precompute flat lag offsets for each variable: flat_il[v] = int_lags[v] @ strides.
    # Combined with base = y_blk @ strides, all_de_ti = ti_flat[v][base[:,None] + flat_il[v][None,:]].
    # In-bounds invariant: every y + lag is guaranteed in-bounds by window construction
    # (_intersect_search_windows), so the flat take needs no bounds checking.
    flat_il = {v: int_lags[v] @ cfg.ti_strides[v] for v in active_vars}

    _use_rust = (
        _mps_config.USE_GSTOOLS_CORE and _mps_config._GSTOOLS_CORE_AVAIL
    )
    # Both whole-node Rust dispatch paths below take either a sequential-walk
    # start position or an explicit visit order, mirroring _scan_window's own
    # two modes. Python builds the random order so the compiled kernels stay
    # bit-identical to the Python path without reimplementing NumPy's PCG64.
    start = int(u_start_i * win_size)
    if _use_rust and cfg.scan_path == "random":
        scan_positions = _random_scan_positions(u_start_i, win_size, max_scan)
        node_positions = np.ascontiguousarray(scan_positions, dtype=np.int64)
    else:
        scan_positions = None
        node_positions = None

    legacy_categorical_scan = (
        _use_rust
        and _MPS_THRESHOLD_GRID is None
        and cfg.threshold is not None
        and not cfg.ti_has_nan
        and _mps_scan_node_cat_gsc is not None
        and len(active_vars) == 1
        and cfg.var_categorical[active_vars[0]]
        # The Rust categorical kernel is plain binary mismatch only; a
        # per-category penalty matrix must stay on the Python dispatch below.
        and cfg.var_penalty_matrix[active_vars[0]] is None
    )
    if legacy_categorical_scan and not _MPS_FULL_NODE_SCAN_FORCE:
        v = active_vars[0]
        return _mps_scan_node_cat_gsc(
            np.asarray(lo, dtype=np.int64),
            np.asarray(win_shape, dtype=np.int64),
            start,
            max_scan,
            cfg.threshold,
            node_positions,
            de_v[v],
            cfg.ti_flat_f64[v],
            cfg.ti_strides[v].astype(np.int64),
            flat_il[v].astype(np.int64),
            precomp_w[v],
        )

    # Action Plan 6: one serial Rust call owns candidate generation, every
    # per-variable distance, joint weighting, center checks, and winner
    # selection. Python still owns event construction, fallback, retrieval,
    # variation adjustment, and the SG write. Concatenated event arrays support
    # different neighbour counts without padding or a candidate-by-event gather.
    # The measured default policy uses this for positive-threshold DS. DSBC
    # stays on the faster legacy Rust block/specialized scan path unless a
    # validation harness explicitly forces the complete generic scan.
    if (
        _use_rust
        and _MPS_THRESHOLD_GRID is None
        and _mps_scan_node_gsc is not None
        and (
            (cfg.threshold is not None and cfg.threshold > 0.0)
            or (_MPS_FULL_NODE_SCAN_FORCE and cfg.threshold is not None)
        )
        # Like the legacy categorical scan, the joint Rust call only
        # implements plain binary categorical mismatch; any active variable
        # with a penalty matrix forces the per-variable Python/Rust mix below.
        and all(cfg.var_penalty_matrix[v] is None for v in active_vars)
    ):
        lengths = np.fromiter(
            (len(de_v[v]) for v in active_vars),
            dtype=np.int64,
            count=len(active_vars),
        )
        event_offsets = np.empty(len(active_vars) + 1, dtype=np.int64)
        event_offsets[0] = 0
        np.cumsum(lengths, out=event_offsets[1:])
        de_flat = np.concatenate(
            [np.asarray(de_v[v], dtype=np.float64) for v in active_vars]
        )
        lag_flat = np.concatenate(
            [np.asarray(flat_il[v], dtype=np.int64) for v in active_vars]
        )
        node_weights = np.concatenate(
            [np.asarray(precomp_w[v], dtype=np.float64) for v in active_vars]
        )
        active_rows = np.fromiter(
            (cfg.var_index[v] for v in active_vars),
            dtype=np.int64,
            count=len(active_vars),
        )
        variable_weights = np.fromiter(
            (cfg.weights[v] for v in active_vars),
            dtype=np.float64,
            count=len(active_vars),
        )
        metric_kinds = np.fromiter(
            (
                0
                if cfg.var_categorical[v]
                else 1
                if cfg.var_p_norm[v] is not None
                else 2
                for v in active_vars
            ),
            dtype=np.int64,
            count=len(active_vars),
        )
        has_nan = np.fromiter(
            (cfg.var_has_nan[v] for v in active_vars),
            dtype=np.uint8,
            count=len(active_vars),
        )
        d_max = np.fromiter(
            (
                1.0 if cfg.var_d_max[v] is None else cfg.var_d_max[v]
                for v in active_vars
            ),
            dtype=np.float64,
            count=len(active_vars),
        )
        p_norm = np.fromiter(
            (
                1.0
                if cfg.var_categorical[v]
                else cfg.var_p_norm[v]
                if cfg.var_p_norm[v] is not None
                else cfg.var_variation_p[v]
                for v in active_vars
            ),
            dtype=np.float64,
            count=len(active_vars),
        )
        target_rows = np.fromiter(
            (cfg.var_index[v] for v in scan_targets),
            dtype=np.int64,
            count=len(scan_targets),
        )
        return _mps_scan_node_gsc(
            np.asarray(lo, dtype=np.int64),
            np.asarray(win_shape, dtype=np.int64),
            start,
            max_scan,
            cfg.threshold,
            node_positions,
            cfg.ti_matrix_f64,
            np.asarray(cfg.ti_strides[active_vars[0]], dtype=np.int64),
            active_rows,
            event_offsets,
            de_flat,
            lag_flat,
            node_weights,
            variable_weights,
            metric_kinds,
            has_nan,
            d_max,
            p_norm,
            target_rows,
            cfg.ti_has_nan,
        )

    def _dist_block(y_blk):
        d = np.zeros(len(y_blk))
        for v in active_vars:
            base = (y_blk @ cfg.ti_strides[v]).astype(np.int64)
            if cfg.var_has_nan[v]:
                if cfg.var_categorical[v]:
                    # Binary mismatch only -- a penalty matrix must fall
                    # through to the Python vec_categorical_penalty_dist path.
                    rust_kernel_available = (
                        cfg.var_penalty_matrix[v] is None
                        and _mps_dist_block_cat_masked_gsc is not None
                    )
                elif cfg.var_p_norm[v] == 1.0:
                    rust_kernel_available = (
                        _mps_dist_block_l1_masked_gsc is not None
                    )
                elif cfg.var_p_norm[v] == 2.0:
                    rust_kernel_available = (
                        _mps_dist_block_l2_masked_gsc is not None
                    )
                elif cfg.var_p_norm[v] is not None:
                    rust_kernel_available = (
                        _mps_dist_block_lp_masked_gsc is not None
                    )
                else:
                    rust_kernel_available = (
                        _mps_dist_block_variation_masked_gsc is not None
                    )
            else:
                if cfg.var_categorical[v]:
                    rust_kernel_available = (
                        cfg.var_penalty_matrix[v] is None
                        and _mps_dist_block_cat_gsc is not None
                    )
                elif cfg.var_p_norm[v] == 1.0:
                    rust_kernel_available = _mps_dist_block_l1_gsc is not None
                elif cfg.var_p_norm[v] == 2.0:
                    rust_kernel_available = _mps_dist_block_l2_gsc is not None
                elif cfg.var_p_norm[v] is not None:
                    rust_kernel_available = _mps_dist_block_lp_gsc is not None
                else:
                    rust_kernel_available = (
                        _mps_dist_block_variation_gsc is not None
                    )
            if _use_rust and rust_kernel_available:
                lag_i64 = flat_il[v].astype(np.int64)
                if cfg.var_has_nan[v] and cfg.var_categorical[v]:
                    dist_v = _mps_dist_block_cat_masked_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                    )
                elif cfg.var_has_nan[v] and cfg.var_p_norm[v] == 1.0:
                    dist_v = _mps_dist_block_l1_masked_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                        cfg.var_d_max[v],
                    )
                elif cfg.var_has_nan[v] and cfg.var_p_norm[v] == 2.0:
                    dist_v = _mps_dist_block_l2_masked_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                        cfg.var_d_max[v],
                    )
                elif cfg.var_has_nan[v] and cfg.var_p_norm[v] is not None:
                    dist_v = _mps_dist_block_lp_masked_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                        cfg.var_d_max[v],
                        cfg.var_p_norm[v],
                    )
                elif cfg.var_has_nan[v]:
                    dist_v = _mps_dist_block_variation_masked_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                        cfg.var_d_max[v],
                        cfg.var_variation_p[v],
                    )
                elif cfg.var_categorical[v]:
                    categorical_kernel = (
                        _mps_dist_block_cat_rayon_gsc
                        if _MPS_RAYON_CANDIDATES
                        and _mps_dist_block_cat_rayon_gsc is not None
                        and len(base) >= _MPS_RAYON_MIN_CANDIDATES
                        else _mps_dist_block_cat_gsc
                    )
                    dist_v = categorical_kernel(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                    )
                elif cfg.var_p_norm[v] == 1.0:
                    dist_v = _mps_dist_block_l1_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                        cfg.var_d_max[v],
                    )
                elif cfg.var_p_norm[v] == 2.0:
                    dist_v = _mps_dist_block_l2_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                        cfg.var_d_max[v],
                    )
                elif cfg.var_p_norm[v] is not None:
                    dist_v = _mps_dist_block_lp_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                        cfg.var_d_max[v],
                        cfg.var_p_norm[v],
                    )
                else:
                    # Variation distance (NaN-free): p_norm is None but
                    # var_variation_p is set. Mean-shift on the accepted value
                    # stays in Python (TrainingImage.adjust_value).
                    dist_v = _mps_dist_block_variation_gsc(
                        de_v[v],
                        cfg.ti_flat_f64[v],
                        base,
                        lag_i64,
                        precomp_w[v],
                        cfg.var_d_max[v],
                        cfg.var_variation_p[v],
                    )
            else:
                all_de_ti = cfg.ti_flat[v][base[:, None] + flat_il[v][None, :]]
                dist_v = cfg.vec_distance_var(
                    v,
                    de_v[v],
                    all_de_ti,
                    cm_v[v],
                    cfg.cond_weight[v],
                    ln_v[v],
                    weights=precomp_w[v],
                    d_max=cfg.d_max[v],
                    has_nan=cfg.var_has_nan[v],
                )
            contrib = cfg.weights[v] * dist_v
            if _MPS_DIST_TRACE is not None:
                # sum / sumsq / count of the weighted contribution, plus the
                # per-block spread: what drives argmin is how much a variable
                # VARIES across candidates, not its mean level.
                acc = _MPS_DIST_TRACE.setdefault(v, [0.0, 0.0, 0, 0.0, 0])
                acc[0] += float(np.sum(contrib))
                acc[1] += float(np.sum(np.square(contrib)))
                acc[2] += int(np.size(contrib))
                if np.size(contrib) > 1:
                    acc[3] += float(np.std(contrib))
                    acc[4] += 1
            d += contrib
        # Renormalize so the joint distance stays in [0, 1] even when
        # some variables have no data event and are excluded from int_lags.
        # Without this, the threshold fires on a compressed scale and
        # accepts matches that should be rejected.
        if 0.0 < active_w_total < 1.0:
            d /= active_w_total
        if cfg.ti_has_nan:
            # Never select an anchor whose pasted value would be undefined:
            # the matched cell must be defined in every target variable.
            center_ok = np.ones(len(y_blk), dtype=bool)
            ys = tuple(y_blk.T)
            for v in scan_targets:
                cv = cfg.ti_vars[v][ys]
                if np.issubdtype(cv.dtype, np.floating):
                    center_ok &= np.isfinite(cv)
            d = np.where(center_ok, d, np.inf)
        if _MPS_CORE_TRACE is not None:
            pv_ = scan_targets[0] if scan_targets else cfg.variables[0]
            _CORE_SCRATCH.append(
                (
                    np.asarray(d, dtype=np.float64),
                    np.asarray(
                        cfg.ti_vars[pv_][tuple(y_blk.T)], dtype=np.float64
                    ),
                )
            )
        return d

    res = _scan_window(
        lo,
        win_shape,
        u_start_i,
        max_scan,
        node_thr,
        _dist_block,
        cfg.scan_path,
        scan_positions,
        _MPS_N_BEST,
    )
    if _MPS_CORE_TRACE is not None:
        if _CORE_SCRATCH:
            dd = np.concatenate([x[0] for x in _CORE_SCRATCH])
            vv = np.concatenate([x[1] for x in _CORE_SCRATCH])
            ok_ = np.isfinite(dd) & np.isfinite(vv)
            dd, vv = dd[ok_], vv[ok_]
            rec = []
            for lo_, hi_ in zip(_CORE_BINS[:-1], _CORE_BINS[1:]):
                m_ = (vv >= lo_) & (vv < hi_)
                rec.append(
                    (
                        float(dd[m_].min()) if m_.any() else float("nan"),
                        int(m_.sum()),
                    )
                )
            j_ = int(np.argmin(dd)) if dd.size else -1
            pv_ = scan_targets[0] if scan_targets else cfg.variables[0]
            de_pv = np.asarray(de_v[pv_], dtype=np.float64)
            de_fin = de_pv[np.isfinite(de_pv)]
            _MPS_CORE_TRACE.append(
                {
                    "d_win": float(dd[j_]) if j_ >= 0 else float("nan"),
                    "v_win": float(vv[j_]) if j_ >= 0 else float("nan"),
                    "n": int(dd.size),
                    "frontier": rec,
                    "node": _MPS_CORE_NODE,
                    # Already-pasted neighbourhood context for the primary
                    # variable's data event at this node (H6: does a
                    # moderate-valued neighbourhood -- itself a product of
                    # earlier decisions on this same random path -- pull the
                    # match toward moderate candidates before argmin runs?).
                    "de_mean": float(de_fin.mean())
                    if de_fin.size
                    else float("nan"),
                    "de_max": float(de_fin.max())
                    if de_fin.size
                    else float("nan"),
                    "de_n": int(de_fin.size),
                }
            )
        del _CORE_SCRATCH[:]
    return res
