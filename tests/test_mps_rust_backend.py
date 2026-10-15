"""Regression tests for Rust MPS parity, migration, and reproducibility.

Skipped automatically when gstools_core is not installed (e.g. CI without Rust build).
The flag gs.config.USE_GSTOOLS_CORE is restored after each test by the fixture.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

try:
    import gstools_core  # noqa: F401

    _CORE_AVAIL = True
except ImportError:
    _CORE_AVAIL = False

pytestmark = pytest.mark.skipif(
    not _CORE_AVAIL, reason="gstools_core not installed"
)

import gstools as gs
from gstools.mps import DirectSampling, MPSModel, TrainingImage
from gstools.mps.distance import (
    vec_categorical_dist,
    vec_l1_dist,
    vec_l2_dist,
    vec_lp_dist,
    vec_variation_dist,
)
from gstools.mps.training_image import Variable


@pytest.fixture(autouse=True)
def _restore_core_flag():
    """Restore backend and private Rayon experiment flags after each test."""
    from gstools.mps import scan as _scan
    from gstools.mps import simulate as _simulate

    original = gs.config.USE_GSTOOLS_CORE
    original_rayon = _scan._MPS_RAYON_CANDIDATES
    original_full_scan_force = _scan._MPS_FULL_NODE_SCAN_FORCE
    original_engine_enabled = _simulate._MPS_RUST_ENGINE_ENABLED
    original_engine_force = _simulate._MPS_RUST_ENGINE_FORCE
    original_engine_stats_hook = _simulate._MPS_RUST_ENGINE_STATS_HOOK
    # Existing block/full-scan tests isolate the pre-Action-7 routes. Dedicated
    # Action Plan 7 tests below enable the complete engine explicitly.
    _simulate._MPS_RUST_ENGINE_ENABLED = False
    _simulate._MPS_RUST_ENGINE_FORCE = False
    yield
    gs.config.USE_GSTOOLS_CORE = original
    _scan._MPS_RAYON_CANDIDATES = original_rayon
    _scan._MPS_FULL_NODE_SCAN_FORCE = original_full_scan_force
    _simulate._MPS_RUST_ENGINE_ENABLED = original_engine_enabled
    _simulate._MPS_RUST_ENGINE_FORCE = original_engine_force
    _simulate._MPS_RUST_ENGINE_STATS_HOOK = original_engine_stats_hook


def _force_legacy_block_scan(monkeypatch):
    """Disable Action Plan 6 so a test can isolate an older block route."""
    from gstools.mps import scan as _scan

    monkeypatch.setattr(_scan, "_mps_scan_node_gsc", None)
    return _scan


def _cat_ti(shape=(60, 60)):
    gx, gy = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    return ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(np.int32)


def _cont_ti(shape=(60, 60)):
    gx, gy = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    return np.sin(gx / 6.0) * np.cos(gy / 8.0)


def _run(
    ti_data,
    categorical,
    distance,
    sg_shape,
    scan_fraction,
    threshold,
    seed,
    use_rust,
):
    gs.config.USE_GSTOOLS_CORE = use_rust
    ti = TrainingImage(ti_data, categorical=categorical, distance=distance)
    model = MPSModel(ti, scan_fraction=scan_fraction, threshold=threshold)
    ds = DirectSampling(model, seed=seed)
    return ds([np.arange(s, dtype=float) for s in sg_shape], store=False)


def _mixed_clean_masked_ds(num_threads=1):
    """Build the deterministic mixed-TI Stage 2 regression fixture."""
    clean = _cat_ti((32, 32)).astype(float)
    masked = clean.copy()
    masked[9:14, 10:15] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable(
                    "clean",
                    clean,
                    categorical=False,
                    distance="l1",
                    n_neighbors=8,
                    weight=0.5,
                ),
                Variable(
                    "masked",
                    masked,
                    categorical=True,
                    n_neighbors=8,
                    weight=0.5,
                ),
            ]
        )
    ds = DirectSampling(
        MPSModel(ti, scan_fraction=0.4, threshold=None), seed=20260809
    )
    ds.num_threads = num_threads
    # Each of the first two nodes constrains only one variable. The other
    # variable is simulated from the same selected TI anchor.
    ds.set_condition(
        [[1.0, 5.0, 8.0], [2.0, 6.0, 3.0]],
        {
            "clean": np.array([1.0, np.nan, 0.0]),
            "masked": np.array([np.nan, 0.0, 0.0]),
        },
    )
    return ds, [np.arange(12.0), np.arange(12.0)]


def _run_mixed_clean_masked(use_rust, num_threads=1):
    gs.config.USE_GSTOOLS_CORE = use_rust
    ds, pos = _mixed_clean_masked_ds(num_threads=num_threads)
    return ds(pos, store=False)


# ---------------------------------------------------------------------------
# 1. Categorical DSBC (threshold=None) — 20x20 SG, 60x60 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234, 99999])
def test_categorical_dsbc_output_identical(seed):
    """DSBC mode: Rust and Python outputs must be bit-identical."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (20, 20), 0.3, None, seed, False)
    rs = _run(ti, True, "l1", (20, 20), 0.3, None, seed, True)
    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# 2. Categorical DS mode (threshold=0.1) — 20x20 SG, 60x60 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234])
def test_categorical_ds_output_identical(seed):
    """DS (greedy threshold) mode: Rust and Python outputs must be bit-identical."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (20, 20), 0.3, 0.1, seed, False)
    rs = _run(ti, True, "l1", (20, 20), 0.3, 0.1, seed, True)
    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# 2a. Categorical labels may be arbitrary numeric values
# ---------------------------------------------------------------------------


def test_close_non_integer_categorical_labels_output_identical():
    """Distinct labels closer than 0.5 remain distinct in the Rust scan."""
    ti = np.where(_cat_ti((30, 30)) == 0, 0.1, 0.2)
    np.testing.assert_array_equal(np.unique(ti), np.array([0.1, 0.2]))

    py = _run(ti, True, "l1", (15, 15), 0.3, None, 42, False)
    rs = _run(ti, True, "l1", (15, 15), 0.3, None, 42, True)

    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# 3. Continuous L1 — 15x15 SG, 40x40 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234])
def test_l1_output_close(seed):
    """L1 distance: Rust and Python outputs must agree to rtol=1e-10."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "l1", (15, 15), 0.3, None, seed, False)
    rs = _run(ti, False, "l1", (15, 15), 0.3, None, seed, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Continuous L2 — 15x15 SG, 40x40 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234])
def test_l2_output_close(seed):
    """L2 distance: Rust and Python outputs must agree to rtol=1e-10."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "l2", (15, 15), 0.3, None, seed, False)
    rs = _run(ti, False, "l2", (15, 15), 0.3, None, seed, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Continuous Lp (p=3) — 15x15 SG, 40x40 TI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42])
def test_lp_output_close(seed):
    """Lp kernel with p=3: Rust and Python outputs must agree to rtol=1e-10."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "l3", (15, 15), 0.3, None, seed, False)
    rs = _run(ti, False, "l3", (15, 15), 0.3, None, seed, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. Variation distance — NaN-free Rust block kernel must match the Python path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 1234])
@pytest.mark.parametrize("threshold", [None, 0.1])
def test_variation_output_close(seed, threshold):
    """Variation (default p=2), DSBC and DS: Rust kernel matches Python to rtol=1e-10."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, "variation", (12, 12), 0.3, threshold, seed, False)
    rs = _run(ti, False, "variation", (12, 12), 0.3, threshold, seed, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


@pytest.mark.parametrize(
    "distance", ["variation1", "variation1.5", "variation2"]
)
def test_variation_p_variants_output_close(distance):
    """Variation with explicit exponents p=1, 1.5, 2 must match the Python path."""
    ti = _cont_ti((40, 40))
    py = _run(ti, False, distance, (12, 12), 0.3, None, 42, False)
    rs = _run(ti, False, distance, (12, 12), 0.3, None, 42, True)
    np.testing.assert_allclose(py, rs, rtol=1e-10)


def test_variation_wrapper_matches_python_randomized():
    """Compare the public PyO3 wrapper directly with the NumPy oracle."""
    rng = np.random.default_rng(20260809)
    for n in (2, 8, 17):
        for n_candidates in (1, 7, 32):
            for p in (0.5, 1.0, 1.5, 2.0, 3.0):
                de_sim = rng.normal(size=n)
                all_de_ti = rng.normal(size=(n_candidates, n))
                node_weights = rng.random(n)
                node_weights /= node_weights.sum()
                d_max = float(rng.uniform(0.1, 4.0))
                base = np.arange(n_candidates, dtype=np.int64) * n
                lags = np.arange(n, dtype=np.int64)

                expected = vec_variation_dist(
                    de_sim,
                    all_de_ti,
                    node_weights,
                    d_max,
                    p=p,
                    has_nan=False,
                )
                actual = gstools_core.mps_dist_block_variation(
                    de_sim,
                    all_de_ti.ravel(),
                    base,
                    lags,
                    node_weights,
                    d_max,
                    p,
                )
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-12, atol=1e-13
                )


def test_variation_dispatch_calls_rust_kernel(monkeypatch):
    """Prove the Rust variation kernel is actually invoked (not the Python fallback).

    A parity test can pass while silently taking the Python path, so spy on the
    module-level backend function and assert it was called.
    """
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"n": 0}
    original = _scan._mps_dist_block_variation_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    _scan._mps_dist_block_variation_gsc = _spy
    try:
        ti = _cont_ti((40, 40))
        _run(ti, False, "variation", (12, 12), 0.3, None, 42, True)
    finally:
        _scan._mps_dist_block_variation_gsc = original
    assert calls["n"] > 0, "Rust variation kernel was never called"


def test_variation_dispatch_python_when_backend_off():
    """With the backend disabled, the Rust variation kernel must NOT be called."""
    from gstools.mps import scan as _scan

    calls = {"n": 0}
    original = _scan._mps_dist_block_variation_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    _scan._mps_dist_block_variation_gsc = _spy
    try:
        ti = _cont_ti((40, 40))
        _run(ti, False, "variation", (12, 12), 0.3, None, 42, False)
    finally:
        _scan._mps_dist_block_variation_gsc = original
    assert calls["n"] == 0, "Rust variation kernel called despite backend off"


@pytest.mark.parametrize(
    ("distance", "backend_name"),
    [
        ("l1", "_mps_dist_block_l1_gsc"),
        ("l2", "_mps_dist_block_l2_gsc"),
        ("l3", "_mps_dist_block_lp_gsc"),
        ("variation", "_mps_dist_block_variation_gsc"),
    ],
)
def test_missing_continuous_core_export_falls_back(
    distance, backend_name, monkeypatch
):
    """Each missing optional export falls back independently to Python."""
    _scan = _force_legacy_block_scan(monkeypatch)

    ti = _cont_ti((40, 40))
    expected = _run(ti, False, distance, (12, 12), 0.3, None, 42, False)
    monkeypatch.setattr(_scan, backend_name, None)
    actual = _run(ti, False, distance, (12, 12), 0.3, None, 42, True)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("distance", "backend_name"),
    [
        ("l1", "_mps_dist_block_l1_gsc"),
        ("l2", "_mps_dist_block_l2_gsc"),
        ("l3", "_mps_dist_block_lp_gsc"),
    ],
)
def test_continuous_dispatch_calls_expected_kernel(
    distance, backend_name, monkeypatch
):
    """Prove each existing continuous block kernel is reached."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"n": 0}
    original = getattr(_scan, backend_name)

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_scan, backend_name, _spy)
    _run(
        _cont_ti((32, 32)),
        False,
        distance,
        (10, 10),
        0.3,
        None,
        42,
        True,
    )
    assert calls["n"] > 0, f"{backend_name} was never called"


def test_categorical_full_scan_dispatch_calls_rust(monkeypatch):
    """Prove the eligible categorical path reaches the full Rust scan."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"n": 0}
    original = _scan._mps_scan_node_cat_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_scan_node_cat_gsc", _spy)
    # The whole-node kernel is gated on a real threshold: DSBC (None) routes
    # to the block-distance kernel instead, so dispatch is checked in DS mode.
    _run(_cat_ti((32, 32)), True, "l1", (10, 10), 0.3, 0.1, 42, True)
    assert calls["n"] > 0, "full categorical Rust scan was never called"


def test_multivariate_categorical_dispatch_calls_block_kernel(monkeypatch):
    """Prove a joint categorical scan reaches the Rust block kernel."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"n": 0}
    original = _scan._mps_dist_block_cat_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_cat_gsc", _spy)
    a = _cat_ti((32, 32)).astype(float)
    b = (np.indices((32, 32)).sum(axis=0) % 3).astype(float)
    ti = TrainingImage(
        [
            Variable("a", a, categorical=True, n_neighbors=4),
            Variable("b", b, categorical=True, n_neighbors=4),
        ]
    )
    ds = DirectSampling(MPSModel(ti, scan_fraction=0.3), seed=42)
    gs.config.USE_GSTOOLS_CORE = True
    ds([np.arange(10.0), np.arange(10.0)], store=False)
    assert calls["n"] > 0, "categorical Rust block kernel was never called"


def test_categorical_rayon_wrapper_matches_serial_randomized():
    """Experimental Rayon rows preserve exact candidate order and sums."""
    rng = np.random.default_rng(20260811)
    for n_lags in (1, 8, 17):
        for candidate_count in (1, 7, 257, 4095, 4096, 4097):
            de_sim = rng.integers(0, 4, size=n_lags).astype(float)
            candidates = rng.integers(
                0, 4, size=(candidate_count, n_lags)
            ).astype(float)
            base = np.arange(candidate_count, dtype=np.int64) * n_lags
            lags = np.arange(n_lags, dtype=np.int64)
            weights = rng.random(n_lags)
            weights /= weights.sum()
            expected = gstools_core.mps_dist_block_cat(
                de_sim, candidates.ravel(), base, lags, weights
            )
            actual = gstools_core.mps_dist_block_cat_rayon(
                de_sim, candidates.ravel(), base, lags, weights
            )
            np.testing.assert_array_equal(actual, expected)


def test_categorical_rayon_experiment_dispatch(monkeypatch):
    """The private experiment uses Rayon only when explicitly enabled."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"rayon": 0}
    original_rayon = _scan._mps_dist_block_cat_rayon_gsc

    def _rayon_spy(*args, **kwargs):
        calls["rayon"] += 1
        return original_rayon(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_cat_rayon_gsc", _rayon_spy)
    a = _cat_ti((32, 32)).astype(float)
    b = (np.indices((32, 32)).sum(axis=0) % 3).astype(float)
    ti = TrainingImage(
        [
            Variable("a", a, categorical=True, n_neighbors=8),
            Variable("b", b, categorical=True, n_neighbors=8),
        ]
    )
    gs.config.USE_GSTOOLS_CORE = True
    pos = [np.arange(11.0), np.arange(11.0)]
    monkeypatch.setattr(_scan, "_MPS_RAYON_MIN_CANDIDATES", 1)
    monkeypatch.setattr(_scan, "_MPS_RAYON_CANDIDATES", False)
    expected = DirectSampling(MPSModel(ti, scan_fraction=0.4), seed=42)(
        pos, store=False
    )
    monkeypatch.setattr(_scan, "_MPS_RAYON_CANDIDATES", True)
    result = DirectSampling(MPSModel(ti, scan_fraction=0.4), seed=42)(
        pos, store=False
    )
    assert calls["rayon"] > 0
    for variable in ("a", "b"):
        np.testing.assert_array_equal(result[variable], expected[variable])


def test_categorical_rayon_experiment_disabled_by_default(monkeypatch):
    """Normal Rust dispatch must not call the experimental export."""
    _scan = _force_legacy_block_scan(monkeypatch)

    def _unexpected(*args, **kwargs):
        raise AssertionError("experimental Rayon kernel called by default")

    monkeypatch.setattr(_scan, "_mps_dist_block_cat_rayon_gsc", _unexpected)
    assert _scan._MPS_RAYON_CANDIDATES is False
    a = _cat_ti((24, 24)).astype(float)
    b = (np.indices((24, 24)).sum(axis=0) % 3).astype(float)
    ti = TrainingImage(
        [
            Variable("a", a, categorical=True, n_neighbors=4),
            Variable("b", b, categorical=True, n_neighbors=4),
        ]
    )
    gs.config.USE_GSTOOLS_CORE = True
    result = DirectSampling(MPSModel(ti, scan_fraction=0.3), seed=42)(
        [np.arange(8.0), np.arange(8.0)], store=False
    )
    assert all(np.isfinite(field).all() for field in result.values())


def test_large_categorical_near_tie_is_rust_thread_deterministic():
    """Characterize migration-level BLAS ties without weakening Rust output."""
    shape = (96, 96)
    grid = np.indices(shape)
    first = ((grid[0] // 5 + grid[1] // 7) % 3).astype(float)
    second = ((grid[0] // 9 + 2 * grid[1] // 11) % 4).astype(float)
    ti = TrainingImage(
        [
            Variable(
                "facies_a",
                first,
                categorical=True,
                n_neighbors=24,
                weight=0.5,
            ),
            Variable(
                "facies_b",
                second,
                categorical=True,
                n_neighbors=24,
                weight=0.5,
            ),
        ]
    )
    model = MPSModel(ti, scan_fraction=0.6, threshold=None)
    pos = [np.arange(36.0), np.arange(36.0)]

    def _simulate(use_core, threads):
        gs.config.USE_GSTOOLS_CORE = use_core
        ds = DirectSampling(model, seed=20260811)
        ds.num_threads = threads
        return ds(pos, store=False)

    python = _simulate(False, 1)
    rust_a = _simulate(True, 1)
    rust_b = _simulate(True, 1)
    rust_parallel = _simulate(True, 4)

    for variable in ("facies_a", "facies_b"):
        np.testing.assert_array_equal(rust_a[variable], rust_b[variable])
        np.testing.assert_array_equal(
            rust_a[variable], rust_parallel[variable]
        )
        assert set(np.unique(rust_a[variable])) <= set(
            np.unique(ti.variable(variable).data)
        )
        # NumPy BLAS and Rust may resolve an effectively tied categorical
        # candidate differently. Characterize that migration difference while
        # requiring it to remain isolated, rather than making it a product
        # bit-identity contract.
        assert np.count_nonzero(python[variable] != rust_a[variable]) <= 1


def test_variation_nan_ti_uses_only_masked_rust_kernel(monkeypatch):
    """Masked variation must not enter the NaN-free Rust kernel."""
    _scan = _force_legacy_block_scan(monkeypatch)

    ti = _cont_ti((40, 40))
    ti[5:9, 5:9] = np.nan  # undefined patch → ti_has_nan True

    calls = {"plain": 0, "masked": 0}
    original_plain = _scan._mps_dist_block_variation_gsc
    original_masked = _scan._mps_dist_block_variation_masked_gsc

    def _plain_spy(*args, **kwargs):
        calls["plain"] += 1
        return original_plain(*args, **kwargs)

    def _masked_spy(*args, **kwargs):
        calls["masked"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_variation_gsc", _plain_spy)
    monkeypatch.setattr(
        _scan, "_mps_dist_block_variation_masked_gsc", _masked_spy
    )
    py = _run(ti, False, "variation", (12, 12), 0.3, None, 42, False)
    rs = _run(ti, False, "variation", (12, 12), 0.3, None, 42, True)
    assert calls["plain"] == 0
    assert calls["masked"] > 0
    np.testing.assert_allclose(rs, py, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 6a. Stage 2 — per-variable NaN dispatch
# ---------------------------------------------------------------------------


def test_mixed_nan_dispatches_clean_and_masked_rust(monkeypatch):
    """Stage 3 accelerates the masked sibling without changing joint control."""
    _scan = _force_legacy_block_scan(monkeypatch)

    rust_calls = {"clean_l1": 0, "masked_cat": 0}
    python_calls = []
    original_l1 = _scan._mps_dist_block_l1_gsc
    original_cat = _scan._mps_dist_block_cat_masked_gsc
    original_python = TrainingImage.vec_distance_var

    def _l1_spy(*args, **kwargs):
        rust_calls["clean_l1"] += 1
        return original_l1(*args, **kwargs)

    def _cat_spy(*args, **kwargs):
        rust_calls["masked_cat"] += 1
        return original_cat(*args, **kwargs)

    def _python_spy(self, var, *args, **kwargs):
        python_calls.append((var, kwargs.get("has_nan")))
        return original_python(self, var, *args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_l1_gsc", _l1_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_cat_masked_gsc", _cat_spy)
    monkeypatch.setattr(TrainingImage, "vec_distance_var", _python_spy)

    result = _run_mixed_clean_masked(True)

    assert rust_calls["clean_l1"] > 0
    assert rust_calls["masked_cat"] > 0
    assert python_calls == []
    assert np.isfinite(result["clean"]).all()
    assert np.isfinite(result["masked"]).all()


def test_mixed_nan_backend_parity_repeatability_and_parallelism():
    """Stage 2 preserves complete fields, RNG use, and the shared anchor."""
    python = _run_mixed_clean_masked(False, num_threads=1)
    rust_a = _run_mixed_clean_masked(True, num_threads=1)
    rust_b = _run_mixed_clean_masked(True, num_threads=1)
    rust_parallel = _run_mixed_clean_masked(True, num_threads=4)

    for variable in ("clean", "masked"):
        np.testing.assert_array_equal(python[variable], rust_a[variable])
        np.testing.assert_array_equal(rust_a[variable], rust_b[variable])
        np.testing.assert_array_equal(
            rust_a[variable], rust_parallel[variable]
        )
        assert np.isfinite(rust_a[variable]).all()

    # The two TI variables have the same defined values. Equality throughout
    # therefore proves that both targets were copied from one shared anchor,
    # including the two partially conditioned nodes above.
    np.testing.assert_array_equal(rust_a["clean"], rust_a["masked"])


def test_masked_univariate_categorical_uses_masked_block(monkeypatch):
    """Masked categorical uses its block kernel, never the full scan."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"full_scan": 0, "plain_block": 0, "masked_block": 0}
    original_full = _scan._mps_scan_node_cat_gsc
    original_plain = _scan._mps_dist_block_cat_gsc
    original_masked = _scan._mps_dist_block_cat_masked_gsc

    def _full_spy(*args, **kwargs):
        calls["full_scan"] += 1
        return original_full(*args, **kwargs)

    def _plain_spy(*args, **kwargs):
        calls["plain_block"] += 1
        return original_plain(*args, **kwargs)

    def _masked_spy(*args, **kwargs):
        calls["masked_block"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_scan_node_cat_gsc", _full_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_cat_gsc", _plain_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_cat_masked_gsc", _masked_spy)
    data = _cat_ti((24, 24)).astype(float)
    data[7:10, 8:12] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        result = _run(
            data,
            True,
            "l1",
            (10, 10),
            0.4,
            None,
            42,
            True,
        )

    assert calls["full_scan"] == 0
    assert calls["plain_block"] == 0
    assert calls["masked_block"] > 0
    assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# 6b. Stage 3 — masked categorical and L1 block kernels
# ---------------------------------------------------------------------------


def test_masked_wrappers_match_python_randomized():
    """Direct PyO3 masked wrappers agree with the NumPy masked oracle."""
    rng = np.random.default_rng(20260809)
    for n in (2, 8, 17):
        for n_candidates in (1, 7, 32):
            weights = rng.random(n)
            weights /= weights.sum()
            base = np.arange(n_candidates, dtype=np.int64) * n
            lags = np.arange(n, dtype=np.int64)
            mask = rng.random((n_candidates, n)) < 0.25
            mask[0] = True

            de_cat = (rng.integers(0, 3, size=n) + 1) / 10.0
            ti_cat = (rng.integers(0, 3, size=(n_candidates, n)) + 1) / 10.0
            ti_cat[mask] = np.nan
            expected_cat = vec_categorical_dist(
                de_cat, ti_cat, weights, has_nan=True
            )
            actual_cat = gstools_core.mps_dist_block_cat_masked(
                de_cat, ti_cat.ravel(), base, lags, weights
            )
            np.testing.assert_allclose(
                actual_cat, expected_cat, rtol=1e-12, atol=1e-13
            )

            de_l1 = rng.normal(size=n)
            ti_l1 = rng.normal(size=(n_candidates, n))
            ti_l1[mask] = np.nan
            d_max = 4.0
            expected_l1 = vec_l1_dist(
                de_l1, ti_l1, weights, d_max, has_nan=True
            )
            actual_l1 = gstools_core.mps_dist_block_l1_masked(
                de_l1, ti_l1.ravel(), base, lags, weights, d_max
            )
            np.testing.assert_allclose(
                actual_l1, expected_l1, rtol=1e-12, atol=1e-13
            )

            expected_l2 = vec_l2_dist(
                de_l1, ti_l1, weights, d_max, has_nan=True
            )
            actual_l2 = gstools_core.mps_dist_block_l2_masked(
                de_l1, ti_l1.ravel(), base, lags, weights, d_max
            )
            np.testing.assert_allclose(
                actual_l2, expected_l2, rtol=1e-12, atol=1e-13
            )

            expected_lp = vec_lp_dist(
                de_l1, ti_l1, weights, d_max, 3.0, has_nan=True
            )
            actual_lp = gstools_core.mps_dist_block_lp_masked(
                de_l1, ti_l1.ravel(), base, lags, weights, d_max, 3.0
            )
            np.testing.assert_allclose(
                actual_lp, expected_lp, rtol=1e-12, atol=1e-13
            )

            expected_variation = vec_variation_dist(
                de_l1, ti_l1, weights, d_max, 2.0, has_nan=True
            )
            actual_variation = gstools_core.mps_dist_block_variation_masked(
                de_l1, ti_l1.ravel(), base, lags, weights, d_max, 2.0
            )
            np.testing.assert_allclose(
                actual_variation,
                expected_variation,
                rtol=1e-12,
                atol=1e-13,
            )


def test_action6_full_scan_wrapper_multivariate_fixture():
    """The PyO3 full scan returns one shared categorical/L1 anchor."""
    result = gstools_core.mps_scan_node(
        np.array([0], dtype=np.int64),
        np.array([4], dtype=np.int64),
        0,
        4,
        0.0,
        None,
        np.array(
            [[0.0, 1.0, 1.0, 0.0], [0.0, 9.0, 2.0, 5.0]],
            dtype=np.float64,
        ),
        np.array([1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, 2.0]),
        np.array([0, 0], dtype=np.int64),
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
        np.array([0, 1], dtype=np.int64),
        np.array([0, 0], dtype=np.uint8),
        np.array([1.0, 10.0]),
        np.array([1.0, 1.0]),
        np.array([0, 1], dtype=np.int64),
        False,
    )
    np.testing.assert_array_equal(result, np.array([2], dtype=np.int64))


def test_action6_dispatch_uses_one_full_scan_call(monkeypatch):
    """Normal supported dispatch bypasses all legacy block exports."""
    from gstools.mps import scan as _scan

    calls = {"full": 0}
    original_full = _scan._mps_scan_node_gsc

    def _full_spy(*args, **kwargs):
        calls["full"] += 1
        return original_full(*args, **kwargs)

    def _unexpected(*args, **kwargs):
        raise AssertionError(
            "legacy scan/block kernel called by Action Plan 6"
        )

    monkeypatch.setattr(_scan, "_mps_scan_node_gsc", _full_spy)
    for name in (
        "_mps_scan_node_cat_gsc",
        "_mps_dist_block_cat_gsc",
        "_mps_dist_block_cat_masked_gsc",
        "_mps_dist_block_l1_gsc",
        "_mps_dist_block_l1_masked_gsc",
        "_mps_dist_block_l2_gsc",
        "_mps_dist_block_l2_masked_gsc",
        "_mps_dist_block_lp_gsc",
        "_mps_dist_block_lp_masked_gsc",
        "_mps_dist_block_variation_gsc",
        "_mps_dist_block_variation_masked_gsc",
    ):
        monkeypatch.setattr(_scan, name, _unexpected)

    base = _cont_ti((28, 28))
    categorical = (_cat_ti((28, 28)) > 0).astype(float)
    variation = base.copy()
    variation[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable(
                    "categorical",
                    categorical,
                    categorical=True,
                    n_neighbors=6,
                    weight=0.5,
                ),
                Variable(
                    "variation",
                    variation,
                    categorical=False,
                    distance="variation",
                    n_neighbors=6,
                    weight=0.5,
                ),
            ]
        )
    gs.config.USE_GSTOOLS_CORE = True
    result = DirectSampling(
        MPSModel(ti, scan_fraction=0.35, threshold=0.1), seed=20260811
    )([np.arange(9.0), np.arange(9.0)], store=False)

    assert calls["full"] > 0
    assert all(np.isfinite(field).all() for field in result.values())


def test_missing_action6_export_uses_legacy_scan(monkeypatch):
    """An older core without the generic scan retains the existing backend."""
    from gstools.mps import scan as _scan

    data = _cont_ti((30, 30))
    expected = _run(data, False, "l2", (10, 10), 0.35, 0.1, 42, True)
    monkeypatch.setattr(_scan, "_mps_scan_node_gsc", None)
    actual = _run(data, False, "l2", (10, 10), 0.35, 0.1, 42, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_action6_dsbc_policy_keeps_faster_block_scan(monkeypatch):
    """Measured DSBC cases retain the existing block route by default."""
    from gstools.mps import scan as _scan

    calls = {"block": 0}
    original_block = _scan._mps_dist_block_l2_gsc

    def _unexpected(*args, **kwargs):
        raise AssertionError(
            "generic full scan called for default DSBC policy"
        )

    def _block_spy(*args, **kwargs):
        calls["block"] += 1
        return original_block(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_scan_node_gsc", _unexpected)
    monkeypatch.setattr(_scan, "_mps_dist_block_l2_gsc", _block_spy)
    result = _run(
        _cont_ti((30, 30)),
        False,
        "l2",
        (10, 10),
        0.35,
        None,
        42,
        True,
    )
    assert calls["block"] > 0
    assert np.isfinite(result).all()


@pytest.mark.parametrize("threshold", [None, 0.1])
def test_masked_categorical_output_identical(threshold):
    """Masked arbitrary numeric labels preserve DSBC and DS output exactly."""
    data = np.where(_cat_ti((36, 36)) == 0, 0.1, 0.2)
    data[4:10, 7:13] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        python = _run(data, True, "l1", (13, 13), 0.4, threshold, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        rust = _run(data, True, "l1", (13, 13), 0.4, threshold, 42, True)
    np.testing.assert_array_equal(python, rust)


@pytest.mark.parametrize("threshold", [None, 0.1])
def test_masked_l1_output_identical(threshold):
    """Masked L1 preserves complete DSBC and DS fields."""
    data = _cont_ti((36, 36))
    data[4:10, 7:13] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        python = _run(data, False, "l1", (13, 13), 0.4, threshold, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        rust = _run(data, False, "l1", (13, 13), 0.4, threshold, 42, True)
    np.testing.assert_array_equal(python, rust)


def test_masked_l1_dispatch_calls_only_masked_kernel(monkeypatch):
    """Prove a masked L1 variable reaches the new Rust export."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"plain": 0, "masked": 0}
    original_plain = _scan._mps_dist_block_l1_gsc
    original_masked = _scan._mps_dist_block_l1_masked_gsc

    def _plain_spy(*args, **kwargs):
        calls["plain"] += 1
        return original_plain(*args, **kwargs)

    def _masked_spy(*args, **kwargs):
        calls["masked"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_l1_gsc", _plain_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_l1_masked_gsc", _masked_spy)
    data = _cont_ti((28, 28))
    data[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        result = _run(data, False, "l1", (10, 10), 0.4, None, 42, True)
    assert calls["plain"] == 0
    assert calls["masked"] > 0
    assert np.isfinite(result).all()


@pytest.mark.parametrize("threshold", [None, 0.1])
def test_masked_l2_output_matches_migration_oracle(threshold):
    """Masked L2 fields remain scientifically equivalent during migration."""
    data = _cont_ti((36, 36))
    data[4:10, 7:13] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        python = _run(data, False, "l2", (13, 13), 0.4, threshold, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        rust = _run(data, False, "l2", (13, 13), 0.4, threshold, 42, True)
    np.testing.assert_allclose(rust, python, rtol=1e-10, atol=1e-12)


def test_masked_l2_dispatch_calls_only_masked_kernel(monkeypatch):
    """Prove a masked L2 variable reaches its Rust export."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"plain": 0, "masked": 0}
    original_plain = _scan._mps_dist_block_l2_gsc
    original_masked = _scan._mps_dist_block_l2_masked_gsc

    def _plain_spy(*args, **kwargs):
        calls["plain"] += 1
        return original_plain(*args, **kwargs)

    def _masked_spy(*args, **kwargs):
        calls["masked"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_l2_gsc", _plain_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_l2_masked_gsc", _masked_spy)
    data = _cont_ti((28, 28))
    data[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        result = _run(data, False, "l2", (10, 10), 0.4, None, 42, True)
    assert calls["plain"] == 0
    assert calls["masked"] > 0
    assert np.isfinite(result).all()


@pytest.mark.parametrize("threshold", [None, 0.1])
def test_masked_lp_output_matches_migration_oracle(threshold):
    """Masked general-Lp fields remain equivalent during migration."""
    data = _cont_ti((36, 36))
    data[4:10, 7:13] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        python = _run(data, False, "l3", (13, 13), 0.4, threshold, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        rust = _run(data, False, "l3", (13, 13), 0.4, threshold, 42, True)
    np.testing.assert_allclose(rust, python, rtol=1e-10, atol=1e-12)


def test_masked_lp_dispatch_calls_only_masked_kernel(monkeypatch):
    """Prove a masked general-Lp variable reaches its Rust export."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"plain": 0, "masked": 0}
    original_plain = _scan._mps_dist_block_lp_gsc
    original_masked = _scan._mps_dist_block_lp_masked_gsc

    def _plain_spy(*args, **kwargs):
        calls["plain"] += 1
        return original_plain(*args, **kwargs)

    def _masked_spy(*args, **kwargs):
        calls["masked"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_lp_gsc", _plain_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_lp_masked_gsc", _masked_spy)
    data = _cont_ti((28, 28))
    data[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        result = _run(data, False, "l3", (10, 10), 0.4, None, 42, True)
    assert calls["plain"] == 0
    assert calls["masked"] > 0
    assert np.isfinite(result).all()


@pytest.mark.parametrize(
    "distance", ["variation", "variation1", "variation1.5"]
)
@pytest.mark.parametrize("threshold", [None, 0.1])
def test_masked_variation_output_matches_migration_oracle(distance, threshold):
    """Masked variation fields remain equivalent during migration."""
    data = _cont_ti((36, 36))
    data[4:10, 7:13] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        python = _run(
            data, False, distance, (13, 13), 0.4, threshold, 42, False
        )
    with pytest.warns(UserWarning, match="contains NaN"):
        rust = _run(data, False, distance, (13, 13), 0.4, threshold, 42, True)
    np.testing.assert_allclose(rust, python, rtol=1e-10, atol=1e-12)


def test_masked_variation_dispatch_calls_only_masked_kernel(monkeypatch):
    """Prove a masked variation variable reaches its Rust export."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"plain": 0, "masked": 0}
    original_plain = _scan._mps_dist_block_variation_gsc
    original_masked = _scan._mps_dist_block_variation_masked_gsc

    def _plain_spy(*args, **kwargs):
        calls["plain"] += 1
        return original_plain(*args, **kwargs)

    def _masked_spy(*args, **kwargs):
        calls["masked"] += 1
        return original_masked(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_variation_gsc", _plain_spy)
    monkeypatch.setattr(
        _scan, "_mps_dist_block_variation_masked_gsc", _masked_spy
    )
    data = _cont_ti((28, 28))
    data[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        result = _run(data, False, "variation", (10, 10), 0.4, None, 42, True)
    assert calls["plain"] == 0
    assert calls["masked"] > 0
    assert np.isfinite(result).all()


def test_masked_kernels_not_called_when_backend_off(monkeypatch):
    """Explicit opt-out retains the Python masked implementation."""
    from gstools.mps import scan as _scan

    def _unexpected(*args, **kwargs):
        raise AssertionError("masked Rust kernel called with backend disabled")

    monkeypatch.setattr(_scan, "_mps_dist_block_cat_masked_gsc", _unexpected)
    monkeypatch.setattr(_scan, "_mps_dist_block_l1_masked_gsc", _unexpected)
    monkeypatch.setattr(_scan, "_mps_dist_block_l2_masked_gsc", _unexpected)
    monkeypatch.setattr(_scan, "_mps_dist_block_lp_masked_gsc", _unexpected)
    monkeypatch.setattr(
        _scan, "_mps_dist_block_variation_masked_gsc", _unexpected
    )
    cat = _cat_ti((24, 24)).astype(float)
    cont = _cont_ti((24, 24))
    cat[5:9, 7:11] = np.nan
    cont[11:15, 13:17] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        cat_result = _run(cat, True, "l1", (9, 9), 0.4, None, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        l1_result = _run(cont, False, "l1", (9, 9), 0.4, None, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        l2_result = _run(cont, False, "l2", (9, 9), 0.4, None, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        lp_result = _run(cont, False, "l3", (9, 9), 0.4, None, 42, False)
    with pytest.warns(UserWarning, match="contains NaN"):
        variation_result = _run(
            cont, False, "variation", (9, 9), 0.4, None, 42, False
        )
    assert np.isfinite(cat_result).all()
    assert np.isfinite(l1_result).all()
    assert np.isfinite(l2_result).all()
    assert np.isfinite(lp_result).all()
    assert np.isfinite(variation_result).all()


def test_multivariate_different_nan_masks_use_both_kernels(monkeypatch):
    """Different per-variable masks preserve one shared multivariate anchor."""
    _scan = _force_legacy_block_scan(monkeypatch)

    calls = {"categorical": 0, "l1": 0}
    original_cat = _scan._mps_dist_block_cat_masked_gsc
    original_l1 = _scan._mps_dist_block_l1_masked_gsc

    def _cat_spy(*args, **kwargs):
        calls["categorical"] += 1
        return original_cat(*args, **kwargs)

    def _l1_spy(*args, **kwargs):
        calls["l1"] += 1
        return original_l1(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_dist_block_cat_masked_gsc", _cat_spy)
    monkeypatch.setattr(_scan, "_mps_dist_block_l1_masked_gsc", _l1_spy)

    categorical = _cat_ti((30, 30)).astype(float)
    continuous = categorical.copy()
    categorical[4:9, 6:11] = np.nan
    continuous[17:22, 18:23] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable(
                    "categorical",
                    categorical,
                    categorical=True,
                    n_neighbors=8,
                    weight=0.5,
                ),
                Variable(
                    "l1",
                    continuous,
                    categorical=False,
                    distance="l1",
                    n_neighbors=8,
                    weight=0.5,
                ),
            ]
        )
    model = MPSModel(ti, scan_fraction=0.4, threshold=None)
    pos = [np.arange(11.0), np.arange(11.0)]

    gs.config.USE_GSTOOLS_CORE = False
    python = DirectSampling(model, seed=42)(pos, store=False)
    gs.config.USE_GSTOOLS_CORE = True
    rust = DirectSampling(model, seed=42)(pos, store=False)

    assert calls["categorical"] > 0
    assert calls["l1"] > 0
    for variable in ("categorical", "l1"):
        assert np.isfinite(rust[variable]).all()
        np.testing.assert_array_equal(python[variable], rust[variable])
    np.testing.assert_array_equal(rust["categorical"], rust["l1"])


def test_action5_multivariate_masked_kernels_are_thread_deterministic(
    monkeypatch,
):
    """All Action Plan 5 kernels preserve one shared anchor and seed output."""
    _scan = _force_legacy_block_scan(monkeypatch)

    kernel_names = {
        "l2": "_mps_dist_block_l2_masked_gsc",
        "lp": "_mps_dist_block_lp_masked_gsc",
        "variation": "_mps_dist_block_variation_masked_gsc",
    }
    calls = {name: 0 for name in kernel_names}
    for name, attribute in kernel_names.items():
        original = getattr(_scan, attribute)

        def _spy(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(_scan, attribute, _spy)

    base = _cont_ti((34, 34))
    l2_data = base.copy()
    lp_data = base.copy()
    variation_data = base.copy()
    l2_data[3:8, 5:10] = np.nan
    lp_data[13:18, 16:21] = np.nan
    variation_data[23:28, 8:13] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable(
                    "l2",
                    l2_data,
                    categorical=False,
                    distance="l2",
                    n_neighbors=10,
                    weight=1.0,
                ),
                Variable(
                    "lp",
                    lp_data,
                    categorical=False,
                    distance="l3",
                    n_neighbors=10,
                    weight=1.0,
                ),
                Variable(
                    "variation",
                    variation_data,
                    categorical=False,
                    distance="variation",
                    n_neighbors=10,
                    weight=1.0,
                ),
            ]
        )
    model = MPSModel(ti, scan_fraction=0.35, threshold=0.1)
    pos = [np.arange(12.0), np.arange(12.0)]

    def _simulate(use_core, threads):
        gs.config.USE_GSTOOLS_CORE = use_core
        ds = DirectSampling(model, seed=20260811)
        ds.num_threads = threads
        return ds(pos, store=False)

    python = _simulate(False, 1)
    rust_a = _simulate(True, 1)
    rust_b = _simulate(True, 1)
    rust_parallel = _simulate(True, 4)

    assert all(count > 0 for count in calls.values())
    for variable in kernel_names:
        assert np.isfinite(rust_a[variable]).all()
        np.testing.assert_array_equal(rust_a[variable], rust_b[variable])
        np.testing.assert_array_equal(
            rust_a[variable], rust_parallel[variable]
        )
        np.testing.assert_allclose(
            rust_a[variable], python[variable], rtol=1e-9, atol=1e-11
        )
    # L2 and Lp use identical TI values, so equality proves both targets were
    # copied from the one shared multivariate anchor.
    np.testing.assert_array_equal(rust_a["l2"], rust_a["lp"])


def test_action6_full_scan_all_metrics_is_thread_deterministic(monkeypatch):
    """The generic scan is exact across repeats/threads for a mixed TI."""
    from gstools.mps import scan as _scan

    monkeypatch.setattr(_scan, "_MPS_FULL_NODE_SCAN_FORCE", True)
    shape = (30, 30)
    base = _cont_ti(shape)
    categorical = _cat_ti(shape).astype(float)
    l2_data = base.copy()
    lp_data = base.copy()
    variation_data = base.copy()
    categorical[3:7, 4:8] = np.nan
    l2_data[10:14, 11:15] = np.nan
    lp_data[17:21, 18:22] = np.nan
    variation_data[22:26, 6:10] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable(
                    "categorical",
                    categorical,
                    categorical=True,
                    n_neighbors=8,
                    weight=0.25,
                ),
                Variable(
                    "l2",
                    l2_data,
                    categorical=False,
                    distance="l2",
                    n_neighbors=8,
                    weight=0.25,
                ),
                Variable(
                    "lp",
                    lp_data,
                    categorical=False,
                    distance="l3",
                    n_neighbors=8,
                    weight=0.25,
                ),
                Variable(
                    "variation",
                    variation_data,
                    categorical=False,
                    distance="variation1.5",
                    n_neighbors=8,
                    weight=0.25,
                ),
            ]
        )
    model = MPSModel(ti, scan_fraction=0.35, threshold=None)
    pos = [np.arange(10.0), np.arange(10.0)]

    def _simulate(use_core, threads):
        gs.config.USE_GSTOOLS_CORE = use_core
        ds = DirectSampling(model, seed=20260811)
        ds.num_threads = threads
        return ds(pos, store=False)

    python = _simulate(False, 1)
    rust_a = _simulate(True, 1)
    rust_b = _simulate(True, 1)
    rust_parallel = _simulate(True, 4)

    for variable in ("categorical", "l2", "lp", "variation"):
        assert np.isfinite(rust_a[variable]).all()
        np.testing.assert_array_equal(rust_a[variable], rust_b[variable])
        np.testing.assert_array_equal(
            rust_a[variable], rust_parallel[variable]
        )
        np.testing.assert_allclose(
            rust_a[variable], python[variable], rtol=1e-9, atol=1e-11
        )
    np.testing.assert_array_equal(rust_a["l2"], rust_a["lp"])


# ---------------------------------------------------------------------------
# Action Plan 7 — complete stationary Rust engine and level scheduler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("categorical", "distance", "masked", "threshold"),
    [
        (True, "l1", False, None),
        (True, "l1", True, 0.1),
        (False, "l1", False, None),
        (False, "l2", True, 0.1),
        (False, "l3", True, None),
        (False, "variation", False, None),
        (False, "variation1.5", True, 0.1),
    ],
)
def test_action7_stationary_engine_matches_oracle_and_threads(
    categorical, distance, masked, threshold
):
    """The complete engine covers every metric/mask and is deterministic."""
    from gstools.mps import simulate as _simulate

    data = (
        _cat_ti((32, 32)).astype(float) if categorical else _cont_ti((32, 32))
    )
    if masked:
        data[8:12, 9:14] = np.nan
    warning = (
        pytest.warns(UserWarning, match="contains NaN") if masked else None
    )
    if warning is None:
        ti = TrainingImage(
            data,
            categorical=categorical,
            distance=distance,
            n_neighbors=8,
        )
    else:
        with warning:
            ti = TrainingImage(
                data,
                categorical=categorical,
                distance=distance,
                n_neighbors=8,
            )
    model = MPSModel(ti, scan_fraction=0.35, threshold=threshold)
    pos = [np.arange(11.0), np.arange(11.0)]

    def _simulate_once(use_core, threads):
        gs.config.USE_GSTOOLS_CORE = use_core
        _simulate._MPS_RUST_ENGINE_FORCE = use_core
        ds = DirectSampling(model, seed=20260811)
        ds.num_threads = threads
        return ds(pos, store=False)

    python = _simulate_once(False, 1)
    rust_one = _simulate_once(True, 1)
    rust_repeat = _simulate_once(True, 1)
    rust_four = _simulate_once(True, 4)

    np.testing.assert_array_equal(rust_one, rust_repeat)
    np.testing.assert_array_equal(rust_one, rust_four)
    if distance.startswith("variation"):
        np.testing.assert_allclose(rust_one, python, rtol=1e-9, atol=1e-11)
    else:
        # These categorical and continuous L1/L2/Lp fixtures have no effective
        # winner ties. Any field difference is therefore a real regression,
        # not an acceptable Python/Rust summation-order characterization.
        np.testing.assert_array_equal(rust_one, python)


def test_action7_multivariate_masked_engine_preserves_shared_anchor():
    """The scheduler preserves conditions and one multivariate TI anchor."""
    from gstools.mps import simulate as _simulate

    shape = (32, 32)
    base = _cont_ti(shape)
    categorical = _cat_ti(shape).astype(float)
    l2_data = base.copy()
    variation_data = base.copy()
    categorical[4:8, 5:9] = np.nan
    l2_data[12:16, 13:17] = np.nan
    variation_data[21:25, 7:11] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        ti = TrainingImage(
            [
                Variable(
                    "categorical",
                    categorical,
                    categorical=True,
                    n_neighbors=8,
                    weight=0.3,
                ),
                Variable(
                    "l2",
                    l2_data,
                    categorical=False,
                    distance="l2",
                    n_neighbors=8,
                    weight=0.35,
                ),
                Variable(
                    "variation",
                    variation_data,
                    categorical=False,
                    distance="variation",
                    n_neighbors=8,
                    weight=0.35,
                ),
            ]
        )
    model = MPSModel(ti, scan_fraction=0.35, threshold=0.1)
    pos = [np.arange(10.0), np.arange(10.0)]

    def _run_engine(use_core, threads):
        gs.config.USE_GSTOOLS_CORE = use_core
        _simulate._MPS_RUST_ENGINE_FORCE = use_core
        ds = DirectSampling(model, seed=20260811)
        ds.num_threads = threads
        ds.set_condition(
            [[1.0, 7.0], [2.0, 6.0]],
            {
                "categorical": np.array([1.0, np.nan]),
                "l2": np.array([np.nan, base[7, 6]]),
                "variation": np.array([base[1, 2], base[7, 6]]),
            },
        )
        return ds(pos, store=False)

    python = _run_engine(False, 1)
    rust_one = _run_engine(True, 1)
    rust_four = _run_engine(True, 4)
    for variable in ("categorical", "l2", "variation"):
        np.testing.assert_array_equal(rust_one[variable], rust_four[variable])
        np.testing.assert_allclose(
            rust_one[variable], python[variable], rtol=1e-9, atol=1e-11
        )
        assert np.isfinite(rust_one[variable]).all()
    assert rust_one["categorical"][1, 2] == 1.0
    assert rust_one["l2"][7, 6] == base[7, 6]
    assert rust_one["variation"][1, 2] == base[1, 2]


def test_action7_dispatch_calls_engine_once_and_releases_node_pipeline(
    monkeypatch,
):
    """Eligible stationary runs use one engine call, not Python node scans."""
    from gstools.mps import simulate as _simulate

    calls = {"engine": 0}
    original = _simulate._mps_simulate_gsc

    def _engine_spy(*args, **kwargs):
        calls["engine"] += 1
        return original(*args, **kwargs)

    def _unexpected(*args, **kwargs):
        raise AssertionError("Python per-node scan called by Rust engine")

    monkeypatch.setattr(_simulate, "_MPS_RUST_ENGINE_ENABLED", True)
    monkeypatch.setattr(_simulate, "_mps_simulate_gsc", _engine_spy)
    monkeypatch.setattr(_simulate, "_scan_for_match", _unexpected)
    result = _run(_cat_ti((28, 28)), True, "l1", (9, 9), 0.35, None, 42, True)
    assert calls["engine"] == 1
    assert np.isfinite(result).all()


def test_action7_explicit_core_disable_keeps_pure_python_engine(monkeypatch):
    """USE_GSTOOLS_CORE=False remains a supported benchmark/oracle mode."""
    from gstools.mps import simulate as _simulate

    def _unexpected(*args, **kwargs):
        raise AssertionError("disabled core must not enter the Rust engine")

    monkeypatch.setattr(_simulate, "_MPS_RUST_ENGINE_ENABLED", True)
    monkeypatch.setattr(_simulate, "_mps_simulate_gsc", _unexpected)
    result = _run(
        _cont_ti((28, 28)),
        False,
        "l2",
        (9, 9),
        0.35,
        None,
        42,
        False,
    )
    assert np.isfinite(result).all()


def test_action7_missing_export_falls_back_and_nonstationary_uses_rust(
    monkeypatch,
):
    """Older cores fall back, while transformed lags use the Rust engine."""
    from gstools.mps import simulate as _simulate

    original = _simulate._mps_simulate_gsc
    monkeypatch.setattr(_simulate, "_MPS_RUST_ENGINE_ENABLED", True)
    monkeypatch.setattr(_simulate, "_mps_simulate_gsc", None)
    expected = _run(
        _cont_ti((28, 28)), False, "l2", (9, 9), 0.35, None, 42, False
    )
    fallback = _run(
        _cont_ti((28, 28)), False, "l2", (9, 9), 0.35, None, 42, True
    )
    np.testing.assert_array_equal(fallback, expected)

    calls = {"engine": 0}

    def _engine_spy(*args, **kwargs):
        calls["engine"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_simulate, "_mps_simulate_gsc", _engine_spy)
    ti = TrainingImage(_cont_ti((28, 28)), categorical=False, distance="l2")
    ds = DirectSampling(
        MPSModel(ti, scan_fraction=0.35, rotation=np.pi / 6), seed=42
    )
    gs.config.USE_GSTOOLS_CORE = True
    result = ds([np.arange(8.0), np.arange(8.0)], store=False)
    assert calls["engine"] == 1
    assert np.isfinite(result).all()


@pytest.mark.parametrize("dimension", [2, 3])
def test_action7_nonstationary_engine_matches_oracle_and_threads(dimension):
    """Python and Rust transform, deduplicate, and reduce lags alike."""
    from gstools.mps import simulate as _simulate

    rng = np.random.default_rng(20260811)
    ti_shape = (24,) * dimension
    sim_shape = (7,) * dimension
    data = rng.integers(0, 3, ti_shape).astype(float)
    ti = TrainingImage(data, categorical=True, n_neighbors=8)
    # ``scale`` is a full dim-component vector with no axis-0 pin, unlike the
    # CovModel-style ``anis`` ratio this test used before the spec moved onto
    # MPSModel. Ratio ``a`` therefore becomes axis scales ``(1, a)``.
    if dimension == 2:
        rotation = np.linspace(0.0, np.pi / 3, np.prod(sim_shape)).reshape(
            sim_shape
        )
        anis = np.linspace(0.55, 1.0, np.prod(sim_shape)).reshape(sim_shape)
        scale = np.stack([np.ones_like(anis), anis], axis=-1)
    else:
        rotation = np.array([0.1, 0.2, 0.3])
        scale = np.array([1.0, 0.7, 0.9])
    model = MPSModel(
        ti,
        scan_fraction=0.2,
        threshold=None,
        rotation=rotation,
        scale=scale,
    )
    pos = [np.arange(float(size)) for size in sim_shape]

    def _run_transformed(use_core, threads):
        gs.config.USE_GSTOOLS_CORE = use_core
        _simulate._MPS_RUST_ENGINE_FORCE = use_core
        ds = DirectSampling(model, seed=1701)
        ds.num_threads = threads
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return ds(pos, store=False)

    python = _run_transformed(False, 1)
    rust_one = _run_transformed(True, 1)
    rust_four = _run_transformed(True, 4)
    np.testing.assert_array_equal(rust_one, rust_four)
    np.testing.assert_array_equal(rust_one, python)


def test_action7_categorical_migration_difference_is_sparse_and_valid():
    """The known 3-D tie case cannot hide arbitrary categorical divergence."""
    from gstools.mps import simulate as _simulate

    ti_shape = (28, 28, 24)
    sg_shape = (12, 12, 10)
    grid = np.indices(ti_shape)
    data = ((grid[0] // 5 + grid[1] // 7 + grid[2] // 3) % 3).astype(float)
    labels = np.unique(data)
    ti = TrainingImage(
        data,
        categorical=True,
        n_neighbors=12,
        max_radius=4.0,
    )
    model = MPSModel(ti, scan_fraction=0.05, threshold=None)
    pos = [np.arange(float(size)) for size in sg_shape]

    def _simulate_once(use_core, threads):
        gs.config.USE_GSTOOLS_CORE = use_core
        _simulate._MPS_RUST_ENGINE_FORCE = use_core
        ds = DirectSampling(model, seed=20260811)
        ds.num_threads = threads
        return ds(pos, store=False)

    python = _simulate_once(False, 1)
    rust_one = _simulate_once(True, 1)
    rust_four = _simulate_once(True, 4)
    np.testing.assert_array_equal(rust_one, rust_four)
    assert np.isin(python, labels).all()
    assert np.isin(rust_one, labels).all()
    different = int(np.count_nonzero(rust_one != python))
    assert different <= 54
    assert different / rust_one.size <= 0.04


def test_action7_strided_explicit_path_and_progress_fallback(monkeypatch):
    """Strided paths are accepted; a requested callback keeps Python control."""
    from gstools.mps import simulate as _simulate

    shape = (9, 9)
    path = np.argwhere(np.ones(shape, dtype=bool))[::-1]
    data = _cat_ti((28, 28))

    def _run_path(use_core, progress=None):
        gs.config.USE_GSTOOLS_CORE = use_core
        _simulate._MPS_RUST_ENGINE_FORCE = use_core
        ti = TrainingImage(data, categorical=True, n_neighbors=8)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.35), seed=42)
        return ds(
            [np.arange(float(size)) for size in shape],
            path=path,
            progress=progress,
            store=False,
        )

    python = _run_path(False)
    rust = _run_path(True)
    np.testing.assert_array_equal(rust, python)

    calls = []

    def _unexpected(*args, **kwargs):
        raise AssertionError("progress run must retain the Python scheduler")

    monkeypatch.setattr(_simulate, "_mps_simulate_gsc", _unexpected)
    progress_result = _run_path(
        True, progress=lambda done, total: calls.append((done, total))
    )
    np.testing.assert_array_equal(progress_result, python)
    assert calls[-1] == (81, 81)


@pytest.mark.parametrize(
    ("shape", "expected_used_threads"), [((16, 16), 1), ((24, 24), 4)]
)
def test_action7_reports_measured_small_path_thread_policy(
    shape, expected_used_threads, monkeypatch
):
    """Requested and effective thread counts remain visible to the harness."""
    from gstools.mps import simulate as _simulate

    captured = []
    monkeypatch.setattr(_simulate, "_MPS_RUST_ENGINE_ENABLED", True)
    monkeypatch.setattr(
        _simulate, "_MPS_RUST_ENGINE_STATS_HOOK", captured.append
    )
    gs.config.USE_GSTOOLS_CORE = True
    ti = TrainingImage(_cat_ti((48, 48)), categorical=True, n_neighbors=12)
    ds = DirectSampling(MPSModel(ti, scan_fraction=0.3), seed=20260811)
    ds.num_threads = 4
    result = ds(
        [np.arange(float(size)) for size in shape],
        path="sequential",
        store=False,
    )
    assert np.isfinite(result).all()
    assert captured[-1]["requested_threads"] == 4
    assert captured[-1]["used_threads"] == expected_used_threads


def test_action7_scan_path_random_declines_complete_engine(monkeypatch):
    """scan_path="random" must never dispatch through the complete engine.

    The complete Rust engine's node loop hardcodes a sequential walk; only
    the per-node kernels (mps/scan.py) implement random-order scanning. If
    the ``self.scan_path == "sequential"`` guard in ``use_rust_engine``
    (simulate.py) were ever deleted, the engine would silently fall back to
    a sequential walk instead of honouring "random" -- no other test catches
    that, since deleting the guard still leaves the rest of the suite green.
    """
    from gstools.mps import simulate as _simulate

    captured = []
    monkeypatch.setattr(_simulate, "_MPS_RUST_ENGINE_ENABLED", True)
    monkeypatch.setattr(
        _simulate, "_MPS_RUST_ENGINE_STATS_HOOK", captured.append
    )
    gs.config.USE_GSTOOLS_CORE = True
    _simulate._MPS_RUST_ENGINE_FORCE = True
    ti = TrainingImage(_cat_ti((28, 28)), categorical=True, n_neighbors=8)
    ds = DirectSampling(
        MPSModel(ti, scan_fraction=0.35, scan_path="random"), seed=20260811
    )
    result = ds([np.arange(9.0), np.arange(9.0)], store=False)

    assert np.isfinite(result).all()
    # The complete engine's stats hook is only populated by _run_rust_engine;
    # an empty capture list proves the run took the per-node Python/Rust
    # dispatch path (mps/scan.py) instead.
    assert captured == []


@pytest.mark.parametrize(
    ("categorical", "distance", "backend_name"),
    [
        (True, "l1", "_mps_dist_block_cat_masked_gsc"),
        (False, "l1", "_mps_dist_block_l1_masked_gsc"),
        (False, "l2", "_mps_dist_block_l2_masked_gsc"),
        (False, "l3", "_mps_dist_block_lp_masked_gsc"),
        (False, "variation", "_mps_dist_block_variation_masked_gsc"),
    ],
)
def test_missing_masked_export_falls_back(
    categorical, distance, backend_name, monkeypatch
):
    """Older cores without a masked export retain the Python path."""
    _scan = _force_legacy_block_scan(monkeypatch)

    data = (
        _cat_ti((28, 28)).astype(float) if categorical else _cont_ti((28, 28))
    )
    data[8:12, 9:14] = np.nan
    with pytest.warns(UserWarning, match="contains NaN"):
        expected = _run(
            data, categorical, distance, (10, 10), 0.4, None, 42, False
        )
    monkeypatch.setattr(_scan, backend_name, None)
    with pytest.warns(UserWarning, match="contains NaN"):
        actual = _run(
            data, categorical, distance, (10, 10), 0.4, None, 42, True
        )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Conditioned categorical — 3 conditioning points, SG 20x20
# ---------------------------------------------------------------------------


def test_conditioned_categorical():
    """Conditioning data is preserved bit-identically regardless of backend."""
    ti_data = _cat_ti()
    ti = TrainingImage(ti_data, categorical=True)
    model = MPSModel(ti, scan_fraction=0.3, threshold=None)
    sg_shape = (20, 20)
    rng = np.random.RandomState(7)
    xi = rng.randint(0, sg_shape[0], 3).astype(float)
    yi = rng.randint(0, sg_shape[1], 3).astype(float)
    vals = ti_data[xi.astype(int) % 60, yi.astype(int) % 60].astype(float)

    results = {}
    for use_rust in (False, True):
        gs.config.USE_GSTOOLS_CORE = use_rust
        ds = DirectSampling(model, seed=42)
        ds.set_condition([xi, yi], vals)
        results[use_rust] = ds(
            [np.arange(s, dtype=float) for s in sg_shape], store=False
        )

    np.testing.assert_array_equal(results[False], results[True])
    # Verify conditioning positions hold their values in the Rust output
    for x, y, v in zip(xi.astype(int), yi.astype(int), vals):
        assert results[True][x, y] == v, (
            f"Conditioning value at ({x}, {y}) not preserved: "
            f"got {results[True][x, y]}, expected {v}"
        )


# ---------------------------------------------------------------------------
# 8. Rust disabled — simulation still completes with expected shape
# ---------------------------------------------------------------------------


def test_rust_disabled_uses_python():
    """USE_GSTOOLS_CORE=False still produces a correctly shaped output."""
    ti = _cat_ti()
    gs.config.USE_GSTOOLS_CORE = False
    field = _run(ti, True, "l1", (10, 10), 0.3, None, 42, False)
    assert field.shape == (10, 10)
    # All values must come from the TI (0 or 1 for this binary TI)
    assert set(np.unique(field)).issubset({0, 1})


# ---------------------------------------------------------------------------
# 9. Multivariate categorical — two variables, SG 10x10
# ---------------------------------------------------------------------------


def test_multivariate_categorical():
    """Multivariate categorical TI: Rust and Python outputs equal for both variables."""
    shape = (30, 30)
    gx, gy = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
    )
    facies = ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(
        np.int32
    )
    # Second variable: checkerboard pattern
    litho = ((gx + gy) % 3).astype(np.int32)

    ti = TrainingImage(
        [
            Variable(
                "facies", facies.astype(float), categorical=True, n_neighbors=4
            ),
            Variable(
                "litho", litho.astype(float), categorical=True, n_neighbors=4
            ),
        ]
    )
    model = MPSModel(ti, scan_fraction=0.3, threshold=None)
    pos = [np.arange(10, dtype=float)] * 2

    gs.config.USE_GSTOOLS_CORE = False
    ds_py = DirectSampling(model, seed=42)
    py_out = ds_py(pos, store=False)

    gs.config.USE_GSTOOLS_CORE = True
    ds_rs = DirectSampling(model, seed=42)
    rs_out = ds_rs(pos, store=False)

    for var_name in ("facies", "litho"):
        np.testing.assert_array_equal(
            py_out[var_name],
            rs_out[var_name],
            err_msg=f"Mismatch in variable {var_name!r}",
        )


# ---------------------------------------------------------------------------
# 10. Large categorical regression — 120x120 TI, 40x40 SG
# ---------------------------------------------------------------------------


def test_large_categorical_regression():
    """Large-case regression guard: Rust and Python outputs are bit-identical."""
    ti = _cat_ti((120, 120))
    py = _run(ti, True, "l1", (40, 40), 0.3, None, 42, False)
    rs = _run(ti, True, "l1", (40, 40), 0.3, None, 42, True)
    np.testing.assert_array_equal(py, rs)


# ---------------------------------------------------------------------------
# Additional parametrized medium-size tests (from brief)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [None, 0.1])
@pytest.mark.parametrize("seed", [42, 1234, 99999])
def test_categorical_dsbc_medium(seed, threshold):
    """DSBC 30x30 SG, 60x60 TI — bit-identical across seeds."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (30, 30), 0.3, threshold, seed, False)
    rs = _run(ti, True, "l1", (30, 30), 0.3, threshold, seed, True)
    np.testing.assert_array_equal(py, rs)


@pytest.mark.parametrize("seed", [42, 1234])
def test_categorical_ds_mode(seed):
    """DS-mode 30x30 SG — bit-identical with threshold > 0."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (30, 30), 0.3, 0.1, seed, False)
    rs = _run(ti, True, "l1", (30, 30), 0.3, 0.1, seed, True)
    np.testing.assert_array_equal(py, rs)


@pytest.mark.parametrize("threshold", [None, 0.1])
@pytest.mark.parametrize("seed", [42, 1234])
def test_categorical_hiscan(seed, threshold):
    """High scan_fraction (0.8) — bit-identical."""
    ti = _cat_ti()
    py = _run(ti, True, "l1", (30, 30), 0.8, threshold, seed, False)
    rs = _run(ti, True, "l1", (30, 30), 0.8, threshold, seed, True)
    np.testing.assert_array_equal(py, rs)


@pytest.mark.parametrize("threshold", [None, 0.1])
@pytest.mark.parametrize("seed", [42, 1234])
def test_continuous_l1(seed, threshold):
    """L1 atol=1e-12 — 30x30 SG, 60x60 TI."""
    ti = _cont_ti()
    py = _run(ti, False, "l1", (30, 30), 0.3, threshold, seed, False)
    rs = _run(ti, False, "l1", (30, 30), 0.3, threshold, seed, True)
    np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


@pytest.mark.parametrize("threshold", [None, 0.1])
@pytest.mark.parametrize("seed", [42, 1234])
def test_continuous_l2(seed, threshold):
    """L2 atol=1e-12 — 30x30 SG, 60x60 TI."""
    ti = _cont_ti()
    py = _run(ti, False, "l2", (30, 30), 0.3, threshold, seed, False)
    rs = _run(ti, False, "l2", (30, 30), 0.3, threshold, seed, True)
    np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


@pytest.mark.parametrize("threshold", [None, 0.1])
@pytest.mark.parametrize("seed", [42])
def test_continuous_l3(seed, threshold):
    """Lp kernel with p=3 atol=1e-12 — 20x20 SG."""
    ti = _cont_ti()
    py = _run(ti, False, "l3", (20, 20), 0.3, threshold, seed, False)
    rs = _run(ti, False, "l3", (20, 20), 0.3, threshold, seed, True)
    np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


def test_variation_distance_small_output_close():
    """Variation distance on a 20x20 SG agrees across both backends."""
    ti = _cont_ti()
    py = _run(ti, False, "variation", (20, 20), 0.3, None, 42, False)
    rs = _run(ti, False, "variation", (20, 20), 0.3, None, 42, True)
    np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


def test_conditioned_categorical_15pts():
    """15-point conditioning — outputs bit-identical; conditioning values preserved."""
    ti_data = _cat_ti()
    ti = TrainingImage(ti_data, categorical=True)
    model = MPSModel(ti, scan_fraction=0.3, threshold=None)
    sg_shape = (20, 20)
    rng = np.random.RandomState(7)
    xi = rng.randint(0, sg_shape[0], 15).astype(float)
    yi = rng.randint(0, sg_shape[1], 15).astype(float)
    vals = ti_data[xi.astype(int) % 60, yi.astype(int) % 60].astype(float)

    results = {}
    for use_rust in (False, True):
        gs.config.USE_GSTOOLS_CORE = use_rust
        ds = DirectSampling(model, seed=42)
        ds.set_condition([xi, yi], vals)
        results[use_rust] = ds(
            [np.arange(s, dtype=float) for s in sg_shape], store=False
        )
    np.testing.assert_array_equal(results[False], results[True])


@pytest.mark.parametrize("categorical", [True, False])
def test_random_scan_path_rust_matches_python(categorical):
    """scan_path='random' is bit-identical on the Rust kernels."""
    data = _cat_ti((32, 32)) if categorical else _cont_ti((32, 32))
    ti = TrainingImage(
        data, categorical=categorical, distance="l1", n_neighbors=8
    )
    fields = []
    for use_rust in (False, True):
        gs.config.USE_GSTOOLS_CORE = use_rust
        model = MPSModel(
            ti, scan_fraction=0.3, threshold=0.1, scan_path="random"
        )
        ds = DirectSampling(model, seed=20260904)
        fields.append(ds([np.arange(12.0)] * 2, store=False))
    np.testing.assert_array_equal(fields[0], fields[1])


def test_random_scan_path_dispatches_to_rust(monkeypatch):
    """Random mode reaches the whole-node Rust kernel, not just Python."""
    from gstools.mps import scan as _scan

    calls = {"n": 0}
    original = _scan._mps_scan_node_cat_gsc

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_scan, "_mps_scan_node_cat_gsc", _spy)
    gs.config.USE_GSTOOLS_CORE = True
    ti = TrainingImage(_cat_ti((32, 32)), categorical=True, n_neighbors=8)
    model = MPSModel(ti, scan_fraction=0.3, threshold=0.1, scan_path="random")
    DirectSampling(model, seed=42)([np.arange(10.0)] * 2, store=False)
    assert calls["n"] > 0, "random scan path never reached the Rust kernel"


# ---------------------------------------------------------------------------
# 11. Post-processing through the complete Rust engine
# ---------------------------------------------------------------------------


def _cat3_ti(shape):
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    scales = (5.0, 4.0, 3.0)[: len(shape)]
    wave = sum(
        np.sin((g + k) / c) for k, (g, c) in enumerate(zip(grids, scales))
    )
    return np.digitize(wave, np.quantile(wave, [0.3, 0.65])).astype(float)


def _post_run(model, pos, use_engine, threads=1, cond=None):
    """Run ``model`` through the Rust engine (True) or the Python engine."""
    from gstools.mps import simulate as _simulate

    gs.config.USE_GSTOOLS_CORE = use_engine
    _simulate._MPS_RUST_ENGINE_ENABLED = False
    _simulate._MPS_RUST_ENGINE_FORCE = use_engine
    ds = DirectSampling(model, seed=20261005)
    ds.num_threads = threads
    if cond is not None:
        ds.set_condition(*cond)
    return ds(pos, store=False)


def _post_case(name):
    """Return ``(ti, pos, model_kwargs, cond)`` for a post-processing case."""
    shape, sg = (40, 40), (30, 30)
    kwargs = {"scan_fraction": 0.5, "threshold": None, "post_processing": 1}
    cond = None
    if name == "3d":
        shape, sg = (14, 14, 14), (9, 9, 9)
        ti = TrainingImage(_cat3_ti(shape), categorical=True, n_neighbors=10)
    else:
        ti = TrainingImage(_cat3_ti(shape), categorical=True, n_neighbors=16)
    if name == "two_passes":
        kwargs["post_processing"] = 2
    elif name == "factor":
        kwargs.update(post_processing=2, post_processing_factor=2.0)
    elif name in ("same", "sequential"):
        kwargs["post_processing_path"] = name
    elif name == "mask":
        mask = np.zeros(sg, dtype=bool)
        mask[5:20, 8:25] = True
        kwargs["post_processing_mask"] = mask
    elif name == "variables_override":
        kwargs["post_processing_variables"] = [
            ti.variable().replace(n_neighbors=8)
        ]
    elif name == "conditioned":
        kwargs["post_processing"] = 2
        xi = np.array([3.0, 10.0, 17.0, 25.0, 12.0])
        yi = np.array([4.0, 21.0, 9.0, 25.0, 12.0])
        vals = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        cond = ([xi, yi], vals)
    pos = [np.arange(float(s)) for s in sg]
    return ti, pos, kwargs, cond


@pytest.mark.parametrize(
    "name",
    [
        "plain",
        "two_passes",
        "3d",
        "factor",
        "same",
        "sequential",
        "mask",
        "variables_override",
        "conditioned",
    ],
)
def test_post_processing_engine_matches_python_dsbc(name):
    """Categorical DSBC post-processing is bit-identical in both engines."""
    ti, pos, kwargs, cond = _post_case(name)
    model = MPSModel(ti, **kwargs)
    python = _post_run(model, pos, False, cond=cond)
    engine = _post_run(model, pos, True, cond=cond)
    np.testing.assert_array_equal(engine, python)
    unprocessed = _post_run(
        MPSModel(ti, **{**kwargs, "post_processing": 0}), pos, True, cond=cond
    )
    assert not np.array_equal(engine, unprocessed)


def test_post_processing_engine_matches_python_bivariate():
    """A two-variable categorical post-pass (one variable partly conditioned) matches."""
    shape = (30, 30)
    gx, gy = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    facies = _cat3_ti(shape)
    litho = ((gx + gy) % 3).astype(float)
    ti = TrainingImage(
        [
            Variable("facies", facies, categorical=True, n_neighbors=6),
            Variable("litho", litho, categorical=True, n_neighbors=6),
        ]
    )
    model = MPSModel(ti, scan_fraction=0.4, threshold=None, post_processing=2)
    pos = [np.arange(12.0)] * 2
    cond = (
        [[2.0, 7.0], [3.0, 8.0]],
        {"facies": np.array([1.0, np.nan]), "litho": np.array([np.nan, 2.0])},
    )
    python = _post_run(model, pos, False, cond=cond)
    engine = _post_run(model, pos, True, cond=cond)
    for var in ("facies", "litho"):
        np.testing.assert_array_equal(engine[var], python[var])
    assert engine["facies"][2, 3] == 1.0
    assert engine["litho"][7, 8] == 2.0


def test_post_processing_engine_is_thread_independent():
    """Post-pass levels never let same-level nodes read each other."""
    ti, pos, kwargs, cond = _post_case("conditioned")
    pos = [np.arange(30.0)] * 2
    model = MPSModel(ti, **{**kwargs, "threshold": 0.05})
    one = _post_run(model, pos, True, threads=1, cond=cond)
    four = _post_run(model, pos, True, threads=4, cond=cond)
    np.testing.assert_array_equal(one, four)


def test_post_processing_engine_keeps_conditioning_exact():
    """Conditioned cells keep their hard data through every post-pass."""
    ti, pos, kwargs, cond = _post_case("conditioned")
    model = MPSModel(ti, **{**kwargs, "threshold": 0.05})
    out = _post_run(model, pos, True, cond=cond)
    (xi, yi), vals = cond
    np.testing.assert_array_equal(out[xi.astype(int), yi.astype(int)], vals)


def test_post_processing_penalty_matrix_declines_engine(monkeypatch):
    """Penalty distances are Rust-unsupported, so the engine must not run."""
    from gstools.mps import simulate as _simulate

    captured = []
    monkeypatch.setattr(
        _simulate, "_MPS_RUST_ENGINE_STATS_HOOK", captured.append
    )
    penalty = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]])
    plain = TrainingImage(_cat3_ti((30, 30)), categorical=True, n_neighbors=8)
    main_penalty = TrainingImage(
        _cat3_ti((30, 30)),
        categorical=True,
        n_neighbors=8,
        penalty_matrix=penalty,
    )
    pos = [np.arange(10.0)] * 2
    models = {
        "main": MPSModel(main_penalty, scan_fraction=0.3, post_processing=1),
        "post": MPSModel(
            plain,
            scan_fraction=0.3,
            post_processing=1,
            post_processing_variables=[
                plain.variable().replace(penalty_matrix=penalty)
            ],
        ),
    }
    for model in models.values():
        out = _post_run(model, pos, True)
        assert np.isfinite(out).all()
    assert captured == []
    # Control: without a penalty matrix the same setup does use the engine.
    _post_run(MPSModel(plain, scan_fraction=0.3, post_processing=1), pos, True)
    assert len(captured) == 2  # main pass + one post pass
