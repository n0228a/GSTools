#!/usr/bin/env python3
"""Parse the latest ASV result JSON and write MPS_RUST_SPEEDUP.md.

Run after 'asv run HEAD^!' with BACKENDS = ('python', 'core') in
DirectSamplingBenchmarks (Task 7 of the Rust MPS backend plan).

Usage:
    python benchmarks/tools/generate_speedup_report.py
"""
import glob
import json
import os
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ASV_RESULTS = os.path.join(_REPO_ROOT, ".asv", "results")
_REPORT_PATH = os.path.join(
    os.path.dirname(_REPO_ROOT),
    "mps_benchmark_report",
    "MPS_RUST_SPEEDUP.md",
)

_DS_CASES = (
    "cat_dsbc_small",
    "cat_dsbc_medium",
    "cat_dsbc_large",
    "cat_ds_medium",
    "cont_l1_medium",
    "cat_dsbc_hiscan",
    "cat_dsbc_highk",
    "cat_dsbc_cond",
)

_CONFIGS = {
    "cat_dsbc_small":  "TI 40×40,   SG 20×20,  n=8,  f=0.3, categorical DSBC",
    "cat_dsbc_medium": "TI 60×60,   SG 30×30,  n=12, f=0.3, categorical DSBC",
    "cat_dsbc_large":  "TI 120×120, SG 40×40,  n=16, f=0.3, categorical DSBC",
    "cat_ds_medium":   "TI 60×60,   SG 30×30,  n=12, f=0.3, categorical DS (t=0.1)",
    "cont_l1_medium":  "TI 60×60,   SG 30×30,  n=12, f=0.3, continuous L1",
    "cat_dsbc_hiscan": "TI 60×60,   SG 30×30,  n=12, f=0.8, categorical DSBC high-scan",
    "cat_dsbc_highk":  "TI 60×60,   SG 30×30,  n=24, f=0.3, categorical DSBC high-k",
    "cat_dsbc_cond":   "TI 60×60,   SG 30×30,  n=12, f=0.3, categorical DSBC conditioned",
}

# Pure-Python baseline from f4062e2f (2026-07-29, pre-Rust)
_BASELINE_PY = {
    "cat_dsbc_small":  0.19788,
    "cat_dsbc_medium": 0.79623,
    "cat_dsbc_large":  3.59360,
    "cat_ds_medium":   0.81555,
    "cont_l1_medium":  0.77040,
    "cat_dsbc_hiscan": 1.20300,
    "cat_dsbc_highk":  1.38200,
    "cat_dsbc_cond":   0.54900,
}


def _find_latest_result():
    jsons = [
        f
        for d in glob.glob(os.path.join(_ASV_RESULTS, "*"))
        for f in glob.glob(os.path.join(d, "*.json"))
        if not os.path.basename(f).startswith(("machine", "benchmarks"))
    ]
    if not jsons:
        raise FileNotFoundError(f"No ASV result JSONs in {_ASV_RESULTS}")
    return max(jsons, key=os.path.getmtime)


def _parse_times(entry):
    """Return {(case, backend): time_or_None} from an ASV result entry."""
    times_flat = entry[0]  # flat list, one per param combo; None = skipped
    params = entry[1]      # [[case_values], [backend_values]] or [[case_values]]

    if len(params) == 1:
        # Old single-param format (no BACKENDS yet) — return python times only
        cases = [s.strip("'") for s in params[0]]
        return {(c, "python"): t for c, t in zip(cases, times_flat)}

    cases = [s.strip("'") for s in params[0]]
    backends = [s.strip("'") for s in params[1]]
    n_backends = len(backends)
    out = {}
    for i, case in enumerate(cases):
        for j, backend in enumerate(backends):
            t = times_flat[i * n_backends + j]
            out[(case, backend)] = t
    return out


def _fmt_t(t):
    if t is None:
        return "—"
    return f"{t * 1000:.0f} ms" if t < 1.0 else f"{t:.3f} s"


def _fmt_speedup(py_t, rs_t):
    if py_t is None or rs_t is None or rs_t == 0:
        return "—"
    return f"{py_t / rs_t:.1f}×"


def _generate(times, result_file, commit):
    lines = [
        "# MPS Rust Backend — Benchmark Speedup Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"ASV result: `{os.path.basename(result_file)}`",
        f"Commit: `{commit}`",
        "",
        "## Simulation Timings: Python vs Rust",
        "",
        "| Case | Config | Python (new run) | Python (baseline) | Rust `core` | Speedup |",
        "|---|---|---|---|---|---|",
    ]
    valid = []
    for case in _DS_CASES:
        py_t = times.get((case, "python"))
        rs_t = times.get((case, "core"))
        base = _BASELINE_PY.get(case)
        lines.append(
            f"| `{case}` | {_CONFIGS[case]} "
            f"| {_fmt_t(py_t)} | {_fmt_t(base)} "
            f"| {_fmt_t(rs_t)} | {_fmt_speedup(py_t, rs_t)} |"
        )
        if py_t and rs_t:
            valid.append((case, py_t, rs_t))

    lines += [""]
    if valid:
        avg = sum(p / r for _, p, r in valid) / len(valid)
        peak_case, peak_py, peak_rs = max(valid, key=lambda x: x[1] / x[2])
        lines += [
            f"**Average speedup (measured cases):** {avg:.1f}×  ",
            f"**Peak speedup:** {peak_py / peak_rs:.1f}× (`{peak_case}`)",
            "",
        ]

    lines += [
        "## Hot-Path Coverage by Phase",
        "",
        "| Phase | Function(s) replaced | Field-scale % | Rust kernels |",
        "|---|---|---|---|",
        "| Phase 1 | `_dist_block` + `vec_categorical_dist` | **52.7 %** | `mps_dist_block_cat`, `mps_dist_block_l1`, `mps_dist_block_l2`, `mps_dist_block_lp` |",
        "| Phase 2 | `_scan_window` | **9.9 %** | `mps_scan_node_cat` |",
        "| Phase 3 | `_select_neighbors` | **5.8 %** | deferred |",
        "| Variation dist. | `vec_variation_dist` | — | Python fallback (no Rust kernel) |",
        "",
        "## Baseline Reference (pure Python, pre-Rust, 2026-07-29)",
        "",
        "From `.asv/results/ufz543326.intranet.ufz.de/f4062e2f-*.json`:",
        "",
        "| Case | Time |",
        "|---|---|",
    ]
    for case, t in _BASELINE_PY.items():
        lines.append(f"| `{case}` | {_fmt_t(t)} |")

    return "\n".join(lines) + "\n"


def main():
    result_file = _find_latest_result()
    print(f"Reading: {result_file}")

    with open(result_file) as f:
        data = json.load(f)

    key = "benchmark_mps.DirectSamplingBenchmarks.time_simulate"
    if key not in data.get("results", {}):
        raise KeyError(
            f"{key!r} not found.\n"
            "Run: asv run 'HEAD^!' --bench DirectSamplingBenchmarks"
        )

    times = _parse_times(data["results"][key])
    commit = data.get("commit_hash", "unknown")[:12]

    report = _generate(times, result_file, commit)
    os.makedirs(os.path.dirname(_REPORT_PATH), exist_ok=True)
    with open(_REPORT_PATH, "w") as f:
        f.write(report)

    print(f"Report written to: {_REPORT_PATH}")


if __name__ == "__main__":
    main()
