"""Experimental upstream storage patch preserves adaptive supports and refits."""

import os

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp
from warp.optim import linear

from benchmarks.experimental.fsai_packed_storage import load_candidate, patched_source
from benchmarks.upstream.adapters import matrix


def test_patch_rejects_unpinned_source():
    with pytest.raises(ValueError, match="exact pinned"):
        patched_source("unverified upstream source")


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("permuted", [False, True])
def test_gpu_support_factor_and_refit_equivalence(width, permuted, tmp_path):
    if not hasattr(linear, "FSAI"):
        pytest.skip("pinned upstream FSAI required")
    if width == 64 and os.environ.get("WARPACK_FSAI_LARGE_MEMORY") != "1":
        pytest.skip("set WARPACK_FSAI_LARGE_MEMORY=1 for the ~14 GiB context comparison")
    candidate = load_candidate(tmp_path)
    n = 70
    rng = np.random.default_rng(131)
    off = rng.uniform(-0.2, 0.2, (n, n))
    off = np.tril(off, -1)
    h = off + off.T
    h += np.diag(np.sum(abs(h), axis=1) + 0.5)
    if permuted:
        order = rng.permutation(n)
        h = h[order][:, order]
    a = matrix(sp.csr_matrix(h))
    kwargs = dict(
        max_row_size=width,
        max_step_size=3,
        kap_tolerance=0.0,
        factor_dtype=wp.float32,
        reuse_pattern=True,
    )
    old, new = linear.FSAI(a, **kwargs), candidate.FSAI(a, **kwargs)
    for scale in (1.0, 1.3):
        if scale != 1:
            a.values.assign(a.values.numpy() * scale)
            old.update()
            new.update()
        assert old.truncated_rows == new.truncated_rows
        for name in ("G", "GT"):
            x, y = getattr(old, name), getattr(new, name)
            count = x.nnz_sync()
            assert count == y.nnz_sync()
            np.testing.assert_array_equal(x.offsets.numpy(), y.offsets.numpy())
            np.testing.assert_array_equal(x.columns.numpy()[:count], y.columns.numpy()[:count])
            np.testing.assert_allclose(
                x.values.numpy()[:count], y.values.numpy()[:count], rtol=0, atol=0
            )
