"""Complex iterative realification must track current fixed-topology values."""

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from warpack import ComplexCSR, ComplexShiftInverse


@pytest.mark.parametrize("entry", ["realified", "complex"])
@pytest.mark.parametrize("sigma", [0.2, 0.2 + 0.3j])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("padding", [0, 7])
def test_complex_gmres_changed_coefficients(entry, sigma, in_place, padding, tmp_path):
    n = 8
    capacity = 2 * n * n + padding
    # Unsorted duplicate columns, including explicit zero components that
    # become nonzero. Padding is deliberately nonfinite/outside column bounds.
    columns = np.tile(np.repeat(np.arange(n)[::-1], 2), n)
    columns = np.r_[columns, np.full(padding, 12345)]
    offsets = np.arange(n + 1) * 2 * n
    source = wp.empty(capacity, dtype=wp.vec2d)

    def coefficients(a):
        values = (a[:, ::-1, None] * np.array([0.25, 0.75])).reshape(-1)
        values = np.r_[values, np.full(padding, np.nan + 1j * np.nan)]
        return np.column_stack([values.real, values.imag])

    first = np.diag(np.arange(1.0, n + 1)).astype(complex)
    source.assign(coefficients(first))
    op = ComplexCSR(n, wp.array(offsets, dtype=wp.int32), wp.array(columns, dtype=wp.int32), source)
    inverse = ComplexShiftInverse(op, sigma, backend="gmres", inner_tol=1e-13, inner_maxiter=64)
    dtype = wp.float64 if entry == "realified" else wp.vec2d
    rhs = wp.zeros((3, 2 * n if entry == "realified" else n), dtype=dtype)
    out = rhs if in_place else wp.empty_like(rhs)
    apply = inverse.apply if entry == "realified" else inverse.apply_complex
    apply(rhs, out)
    wp.synchronize()
    graph = None
    if rhs.device.is_cuda:
        with wp.ScopedCapture() as capture:
            apply(rhs, out)
        graph = capture.graph
    rng = np.random.default_rng(623)
    changed = 1.5 * first + 0.02 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    for a in [first, changed, first]:
        source.assign(coefficients(a))
        b = rng.normal(size=(3, n)) + 1j * rng.normal(size=(3, n))
        b[1] = 0
        packed = (
            np.concatenate([b.real, b.imag], axis=1)
            if entry == "realified"
            else np.stack([b.real, b.imag], axis=-1)
        )
        rhs.assign(packed)
        if graph is None:
            apply(rhs, out)
        else:
            wp.capture_launch(graph)
        result = out.numpy()
        x = (
            result[:, :n] + 1j * result[:, n:]
            if entry == "realified"
            else result[..., 0] + 1j * result[..., 1]
        )
        shifted = a - sigma * np.eye(n)
        residual = np.linalg.norm(x @ shifted.T - b) / np.linalg.norm(b)
        assert residual < 2e-12
        np.testing.assert_allclose(x, np.linalg.solve(shifted, b.T).T, atol=1e-12, rtol=1e-12)
        # Validate the representation itself, including all diagonal shift and
        # duplicate-coordinate contributions, independently of solver success.
        m = inverse.matrix
        ends = m.offsets.numpy()[: 2 * n + 1]
        dense = sp.csr_matrix(
            (m.values.numpy()[: ends[-1]], m.columns.numpy()[: ends[-1]], ends),
            shape=(2 * n, 2 * n),
        ).toarray()
        expected = np.block([[shifted.real, -shifted.imag], [shifted.imag, shifted.real]])
        np.testing.assert_allclose(dense, expected, atol=4e-15, rtol=1e-15)
        np.testing.assert_array_equal(source.numpy(), coefficients(a))
        if not in_place:
            np.testing.assert_array_equal(rhs.numpy(), packed)
    if graph is not None:
        dot = tmp_path / "complex-gmres-refresh.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        assert not any(word in dot.read_text() for word in ["HOST", "HtoD", "DtoH"])
