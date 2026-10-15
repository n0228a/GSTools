"""MPS remains usable when the optional Rust extension is absent or older."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

GSTOOLS_SRC = Path(__file__).resolve().parents[1] / "src"


def _run_isolated(script):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, (
        f"isolated Python failed\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_mps_runs_when_gstools_core_is_not_installed():
    """Block the extension import and exercise the genuine no-core path."""
    _run_isolated(
        f"""
        import importlib.abc
        import sys

        class BlockCore(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "gstools_core" or fullname.startswith(
                    "gstools_core."
                ):
                    raise ModuleNotFoundError("gstools_core intentionally blocked")
                return None

        sys.meta_path.insert(0, BlockCore())
        sys.path.insert(0, {str(GSTOOLS_SRC)!r})

        import numpy as np
        import gstools as gs
        from gstools.mps import DirectSampling, MPSModel, TrainingImage

        assert gs.config._GSTOOLS_CORE_AVAIL is False
        assert gs.config.USE_GSTOOLS_CORE is False
        ti_data = (np.indices((12, 12)).sum(axis=0) % 2).astype(float)
        ti = TrainingImage(ti_data, categorical=True, n_neighbors=4)
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        out = ds([np.arange(6.0), np.arange(6.0)], seed=42, store=False)
        assert out.shape == (6, 6)
        assert np.isfinite(out).all()
        """
    )


def test_older_core_without_mps_exports_uses_python_fallback():
    """A core installed for other GSTools features need not expose MPS."""
    _run_isolated(
        f"""
        import sys
        sys.path.insert(0, {str(GSTOOLS_SRC)!r})

        import gstools_core
        for name in (
            "mps_dist_block_cat",
            "mps_dist_block_cat_masked",
            "mps_dist_block_l1",
            "mps_dist_block_l1_masked",
            "mps_dist_block_l2",
            "mps_dist_block_l2_masked",
            "mps_dist_block_lp",
            "mps_dist_block_lp_masked",
            "mps_dist_block_variation",
            "mps_dist_block_variation_masked",
            "mps_scan_node",
            "mps_scan_node_cat",
            "mps_simulate",
        ):
            if hasattr(gstools_core, name):
                delattr(gstools_core, name)

        import numpy as np
        import gstools as gs
        from gstools.mps import (
            DirectSampling,
            MPSModel,
            TrainingImage,
            scan,
            simulate,
        )

        assert gs.config._GSTOOLS_CORE_AVAIL is True
        assert scan._mps_dist_block_variation_gsc is None
        assert scan._mps_scan_node_gsc is None
        assert simulate._mps_simulate_gsc is None
        data = np.sin(np.indices((16, 16))[0] / 4.0)
        ti = TrainingImage(
            data, categorical=False, distance="variation", n_neighbors=4
        )
        ds = DirectSampling(MPSModel(ti, scan_fraction=0.3))
        out = ds([np.arange(7.0), np.arange(7.0)], seed=42, store=False)
        assert out.shape == (7, 7)
        assert np.isfinite(out).all()
        """
    )
