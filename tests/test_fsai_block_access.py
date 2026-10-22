"""Scalar block views must preserve strided source storage and captured refits."""

from pathlib import Path

import numpy as np
import pytest
import warp as wp
import warp.sparse as sp
from warp.optim import linear

from benchmarks.experimental.fsai_block_access import load, source_patch
from benchmarks.experimental.fsai_current import load as load_current
from benchmarks.upstream.adapters import matrix


def test_block_access_patch_rejects_unknown_source():
    with pytest.raises(ValueError, match="qualified packed source"):
        source_patch("unverified code")


@pytest.mark.parametrize("block", [1, 2, 3, "matrix1"])
@pytest.mark.parametrize("dtype", [wp.float32, wp.float64])
def test_strided_values_and_captured_update(block, dtype, tmp_path):
    if not hasattr(linear, "FSAI") or not Path("build/current_fsai.py").exists():
        pytest.skip("explicit FSAI qualification source required")
    n = 12
    h = (
        np.diag(np.linspace(2, 3, n))
        + np.diag(np.full(n - 1, -0.3), 1)
        + np.diag(np.full(n - 1, -0.3), -1)
    )
    if block == "matrix1":
        rows, columns = np.nonzero(h)
        a = sp.bsr_from_triplets(
            n,
            n,
            wp.array(rows, dtype=int, device="cuda:0"),
            wp.array(columns, dtype=int, device="cuda:0"),
            wp.array(
                h[rows, columns].reshape(-1, 1, 1),
                dtype=wp.types.matrix((1, 1), dtype),
                device="cuda:0",
            ),
        )
    else:
        a = sp.bsr_copy(matrix(h), block_shape=(block, block), scalar_type=dtype)
    a.nnz_sync()
    values = a.values.numpy().copy()
    backing = wp.array(np.repeat(values, 2, axis=0), dtype=a.dtype, device=a.device)
    a.values = backing[::2]
    assert not a.values.is_contiguous
    options = dict(max_row_size=8, max_step_size=3, reuse_pattern=True)
    expected = load_current(packed=True).FSAI(a, **options)
    actual = load("view").FSAI(a, **options)
    actual.update()
    with wp.ScopedCapture() as capture:
        actual.update()
    for scale in (1.0, 1.5, 0.75):
        a.values.assign(values * scale)
        expected.update()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(actual.update_status.numpy(), 0)
        for name in ("G", "GT"):
            old, new = getattr(expected, name), getattr(actual, name)
            count = old.nnz_sync()
            assert count == new.nnz_sync()
            np.testing.assert_array_equal(old.offsets.numpy(), new.offsets.numpy())
            np.testing.assert_array_equal(old.columns.numpy()[:count], new.columns.numpy()[:count])
            np.testing.assert_array_equal(old.values.numpy()[:count], new.values.numpy()[:count])
    path = tmp_path / "strided-refit.dot"
    wp.capture_debug_dot_print(capture.graph, str(path))
    assert not any(x in path.read_text() for x in ("HOST", "HtoD", "DtoH"))
