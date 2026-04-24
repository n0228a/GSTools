# MPS Migration: MPSTools → GSTools

**Date:** 2026-04-24
**Scope:** Migrate the complete MPS implementation from the standalone MPSTools
repo into `gstools/mps/`, following the same conventions as `gstools/field/srf.py`
and `gstools/krige/base.py`.

---

## Context

MPSTools is a standalone repo (step 9+ complete, 83 tests) implementing Direct
Sampling MPS. GSTools already has a thin `gstools/mps/` subpackage (`TI.py` +
`DS.py`) that covers only univariate categorical simulation with no distance
abstraction, no multivariate, no nonstationarity, no Rust dispatch.

The goal is to replace that stub with the full implementation — incrementally,
one manually-verifiable step at a time.

---

## Constraints

- **Pure Python only.** No Rust build infrastructure changes to GSTools. The
  `mpstools-core` Rust crate stays in MPSTools; Rust dispatch is out of scope
  for this migration.
- **No quality metrics.** `quality.py` is out of scope; focus is on
  `TrainingImage` + `DirectSampling`.
- **No tests in this migration.** Tests are a separate follow-up step.
- **No oracle.** MPSTools had a separate `tests/baseline/oracle.py` reference
  implementation. GSTools gets one clean Python implementation only — no
  dual-path, no baseline split.
- **GSTools conventions.** File names stay `TI.py` / `DS.py`. Dispatch pattern
  mirrors `krige/base.py`. No extra helper modules.

---

## File Layout

Only three locations change:

```
gstools/mps/TI.py      ← replace stub (full TrainingImage)
gstools/mps/DS.py      ← replace stub (full DirectSampling + simulation function)
gstools/mps/__init__.py ← untouched
gstools/__init__.py     ← untouched
gstools/config.py       ← untouched (no Rust dispatch in this migration)
```

---

## Design

### TrainingImage (TI.py)

Analogue of `CovModel`. Encapsulates training data and the distance function.
The simulation loop in DS.py calls `ti.distance()` without knowing variable
types — `TrainingImage` is the only place variable-type logic lives.

**Public API additions over current stub:**

| Addition | Purpose |
|---|---|
| `d_max` precomputed per continuous variable | normalises distance to [0,1] |
| `distance(de_sg, de_ti, ...)` | single entry point for all distance types |
| `_distance_single(...)` | per-variable worker |
| `value_at(index)` | uniform uni/multivariate value access |
| `weights` (multivariate) | per-variable contribution to joint distance |
| `angles`, `anis` | stationary rotation/anisotropy, broadcast by DS to uniform maps |
| `distance="l1"/"l2"/"variation"` | Juda Eq. 7 / Mariethoz Eq. 4–5 / Mariethoz Eq. 9 |
| `distance_power` | exponent δ for l2 spatial weighting |
| `adjust_value(ti_val, de_sg, de_ti)` | mean-shift correction for `distance="variation"` |
| `categories(variable)` | unique values for categorical variables |

Multivariate: `data` accepts a `dict` of arrays; `categorical` and `weights`
can be per-variable dicts.

### DirectSampling (DS.py)

Analogue of `SRF` / `Krige`. Subclasses `gstools.field.base.Field` with
`model=None, dim=ti.ndim`. The `__call__` follows the standard pattern:
`pre_pos → _simulate → post_field`.

The pure-Python simulation function lives at module level in `DS.py` (private,
not exported), mirroring how `krige/base.py` has private helper functions above
the `Krige` class.

**Public API additions over current stub:**

| Addition | Purpose |
|---|---|
| `parallel="auto"/"inner"/"outer"/True/False/None` | CpuInner / CpuOuter parallelism |
| `boundary="strict"/"partial"` | strict search window vs. per-candidate lag check |
| `postprocess=int` | sequential re-simulation passes after main loop |
| `set_rotation(rotation_map)` | per-node rotation angles (nonstationarity) |
| `set_affinity(affinity_map)` | per-node affinity ratios (nonstationarity) |
| `extra_fields` property | secondary-variable outputs from multivariate simulation |
| `available_ram_bytes()` | inlined private helper for parallel RAM check |

Conditioning (`set_condition`) already exists in the stub; tie-breaking fix
(Mariethoz2010 §3 ¶12: keep closest-to-node-centre when two points snap to
the same grid node) is applied in step 2.

---

## Migration Steps

Each step targets a specific capability, leaves GSTools in a working state,
and can be verified manually before proceeding.

### Step 1 — `TI.py`: univariate

Replace the 46-line stub. Scope: univariate only, `distance="l1"` only.

Adds: `d_max`, `distance()`, `_distance_single()` (l1 path only), `value_at()`,
`categories()`, `size` property.

`weights`, `angles`, `anis`, `distance_type`, `adjust_value()` are **not** added
yet. Multivariate dict input raises `TypeError` (natural — no dict handling yet).

Verify: construct a `TrainingImage`, call `ti.distance()` for categorical and
continuous arrays, check `ti.value_at()`.

---

### Step 2 — `DS.py`: clean simulation + continuous

Replace the simulation function in `DS.py`. The new function calls
`ti.distance()` instead of hardcoding categorical mismatch logic. Continuous
TIs now work.

Also fixes `_conditions_to_grid` tie-breaking.

No new parameters on `DirectSampling`. Existing `n_neighbors`, `scan_fraction`,
`threshold`, `cond_weight` stay unchanged.

Verify: run a continuous TI simulation and check output is within TI value range.

---

### Step 3 — `DS.py`: parallel

Add `parallel="auto"` parameter to `DirectSampling.__init__`.

Add private `_resolve_parallel_mode(shape) → int` (0=sequential, 1=CpuInner,
2=CpuOuter) and `_available_ram_bytes()` inlined helper.

Warnings for `inner+threshold>0` (non-deterministic early-stop) and
`outer`+RAM-overflow (falls back to auto).

For now the simulation function ignores the parallel mode (single-threaded
Python). The parameter is wired through but has no effect until the Rust
dispatch is added. This is intentional — the API is stable, the implementation
comes later.

Verify: construct DS with various `parallel=` values, check no errors; check
warning fires for `parallel="inner"` with `threshold>0`.

---

### Step 4 — `DS.py`: boundary

Add `boundary="strict"` (default) / `"partial"` parameter to
`DirectSampling.__init__`.

- `"strict"`: search window computed once from bounding box of lag vectors
  (Juda2022 Eq. 5). Correct for non-rotated integer lags.
- `"partial"`: per-candidate check — lags that fall outside the TI after
  rotation are dropped and distance normalised over remaining lags
  (Mariethoz2010 §6.2). Intended for nonstationarity; warn if used without
  rotation/affinity maps.

Verify: construct DS with both boundary values, check warning fires when
`boundary="partial"` is used without rotation/affinity.

---

### Step 5 — `TI.py`: l2/variation distance metrics

Extend `_distance_single()` with l2 and variation paths.

- `"l2"`: spatially-weighted RMS (Mariethoz2010 Eq. 4–5); requires `lag_norms`
  passed through from DS.
- `"variation"`: de-meaned L2 (Mariethoz2010 Eq. 9).
- Add `adjust_value(ti_val, de_sg, de_ti)` for the variation mean-shift.
- Add `distance_power` parameter (exponent δ for l2 spatial weighting).

DS.py passes `lag_norms` to `ti.distance()` when `distance_type="l2"`.

Verify: construct continuous TI with each distance type; call `distance()` and
check range [0, 1]; call `adjust_value()` and check mean shift.

---

### Step 6 — `TI.py` + `DS.py`: multivariate

**TI.py:** `data` accepts `dict` of same-shape arrays. Per-variable
`categorical` (dict or scalar). Per-variable `weights` (default: equal).
`data` property returns dict. `distance()` computes weighted sum across
variables (Mariethoz2010 Eq. 8). `value_at()` returns dict.

**DS.py:** simulation function dispatches on `isinstance(ti_data, dict)`.
Joint distance over all variables. All variables assigned from the same TI
location `y*`. Primary variable returned by `post_field`; secondary variables
stored in `_extra_fields` (accessible via `extra_fields` property).

Verify: construct multivariate TI with two variables; run simulation; check
both fields have correct shape and values come from TI.

---

### Step 7 — `TI.py` + `DS.py`: nonstationarity

**TI.py:** add `angles` and `anis` parameters. DS broadcasts these to uniform
maps (same angle/ratio at every node).

**DS.py:** add `set_rotation(rotation_map)` and `set_affinity(affinity_map)`.
In `__call__`, if `rotation_map` is None and `ti.angles` is set, broadcast to
uniform array. Same for affinity.

Simulation function applies `R = matrix_rotate(dim, angle)` and
`A = matrix_anisotropify(dim, ratio)` to lag vectors before TI lookup:
`L'_i = R · A · L_i` (Mariethoz2010 §6.2). Uses
`gstools.tools.geometric.matrix_rotate` and `matrix_anisotropify`.

Verify: `set_rotation(zeros)` gives same result as no rotation; non-zero
rotation gives different result.

---

### Step 8 — `DS.py`: postprocess

Add `postprocess=0` parameter to `DirectSampling.__init__`.

After the main simulation loop, run `postprocess` sequential re-simulation
passes: for each pass, generate a new random path through all non-conditioning
nodes, erase each node's value, re-simulate using the full informed neighbourhood
(Strebelle & Remy 2005, cited in Mariethoz2010 §7 ¶48).

Verify: `postprocess=1` changes the output vs. `postprocess=0` on the same seed.

---

## What Is Not in This Migration

- Rust dispatch (`mpstools_core`) — stays in MPSTools
- `gstools/config.py` changes — not needed without Rust dispatch
- `quality.py` — separate follow-up
- Tests — separate follow-up
- Recursive syn-processing (Mariethoz2010 §7 ¶50–57) — not yet implemented
  in MPSTools either; out of scope
