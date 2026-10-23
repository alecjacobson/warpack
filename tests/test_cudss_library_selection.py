"""Exercise library selection before any process-local cuDSS cache exists."""

import os
import subprocess
import sys
import textwrap

import pytest


def run_fresh(source, override):
    env = os.environ.copy()
    env.pop("CUDSS_LIBRARY_PATH", None)
    if override is not None:
        env["CUDSS_LIBRARY_PATH"] = override
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("override", [None, ""])
def test_default_discovery_preserves_environment_and_replays(override):
    run_fresh(
        """
        import os
        import numpy as np
        import warp as wp
        from tests.test_solver import sparse
        from warpack import CuDSSInverse

        before = os.environ.get("CUDSS_LIBRARY_PATH")
        a = np.diag(np.arange(1., 25.)) + 0.1 * (np.eye(24, k=1) + np.eye(24, k=-1))
        inverse = CuDSSInverse(sparse(a), 3)
        rhs = wp.zeros((3, 24), dtype=wp.float64, device="cuda:0")
        out = wp.empty_like(rhs)
        inverse.apply(rhs, out)
        with wp.ScopedCapture(device="cuda:0") as capture:
            inverse.apply(rhs, out)
        try:
            for seed in (11, 43, 11):
                b = np.random.default_rng(seed).normal(size=(3, 24))
                wp.copy(rhs, wp.array(b, dtype=wp.float64, device="cuda:0"))
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(out.numpy() @ a.T, b, rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(rhs.numpy(), b)
            assert os.environ.get("CUDSS_LIBRARY_PATH") == before
        finally:
            capture.graph = None
            inverse.close()
        """,
        override,
    )


def test_unloadable_explicit_library_does_not_fall_back(tmp_path):
    run_fresh(
        """
        import os
        import numpy as np
        from tests.test_solver import sparse
        from warpack import CuDSSInverse
        from warp_cudss import _bindings

        requested = os.environ["CUDSS_LIBRARY_PATH"]
        try:
            CuDSSInverse(sparse(np.eye(4)), 1)
        except OSError as error:
            assert requested in str(error)
            assert isinstance(error.__cause__, OSError)
        else:
            raise AssertionError("silently loaded a different cuDSS library")
        assert _bindings._lib is None
        assert os.environ["CUDSS_LIBRARY_PATH"] == requested
        """,
        str(tmp_path / "missing-explicit-cudss.so"),
    )
