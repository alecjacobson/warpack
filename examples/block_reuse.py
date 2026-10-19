"""Repeated low modes of a clamped P1 elastic bar with consistent mass.

Run from a source checkout: python -m examples.block_reuse
Assembly and both eigensolves use Warp. SciPy is only an independent final
reference; it never supplies a starting vector. This small example uses Warp's
built-in diagonal preconditioner, so it also runs on stock Warp 1.15.
"""

import numpy as np
import scipy.linalg as la
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from warpack import BlockLOBPCG, SparseOperator


@wp.kernel
def bar_values(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    stiffness: wp.array[wp.float64],
    mass: wp.array[wp.float64],
    change: wp.float64,
):
    i = wp.tid()
    n = offsets.shape[0] - 1
    h = wp.float64(1.0) / wp.float64(n)
    # Free node i represents physical node i+1; physical node 0 is clamped.
    left = wp.float64(1.0) + change * wp.float64(i >= n // 2)
    right = wp.float64(1.0) + change * wp.float64(i + 1 >= n // 2)
    for e in range(offsets[i], offsets[i + 1]):
        j = columns[e]
        if i == j:
            stiffness[e] = (left + right * wp.float64(i < n - 1)) / h
            mass[e] = h * (wp.float64(1.0) + wp.float64(i < n - 1)) / wp.float64(3.0)
        else:
            stiffness[e] = -(left if j < i else right) / h
            mass[e] = h / wp.float64(6.0)


def main():
    wp.init()
    n, k, width = 96, 4, 8
    # Topology is setup metadata and remains fixed throughout both solves.
    rows = np.repeat(np.arange(n), 3)
    cols = rows + np.tile([-1, 0, 1], n)
    valid = (cols >= 0) & (cols < n)
    a = ws.bsr_from_triplets(
        n,
        n,
        wp.array(rows[valid], dtype=wp.int32),
        wp.array(cols[valid], dtype=wp.int32),
        wp.ones(int(valid.sum()), dtype=wp.float64),
    )
    a.nnz_sync()
    b = ws.bsr_copy(a)

    def assemble(change):
        wp.launch(bar_values, n, [a.offsets, a.columns, a.values, b.values, change])

    assemble(0.0)
    pre = linear.preconditioner(a, ptype="diag")
    source = BlockLOBPCG(
        SparseOperator(a), k, width=width, mass=SparseOperator(b), preconditioner=pre, tol=1e-10
    )
    wp.capture_launch(source.capture(300))
    assert source.status.numpy()[0] == 0 and source.converged.numpy()[0] == k
    starts = wp.clone(source.x)
    target = BlockLOBPCG(
        SparseOperator(a),
        k,
        width=width,
        mass=SparseOperator(b),
        preconditioner=pre,
        initial_vectors=starts,
        initial_perturbation=1e-3,
        tol=1e-10,
    )
    graph = target.capture(300)
    for change in (0.02, 0.04):
        assemble(change)
        # Reusing this old positive diagonal is valid preconditioning. A larger
        # coefficient change can justify constructing new factors and a graph.
        wp.capture_launch(graph)
        values = target.eigenvalues.numpy()
        vectors = target.eigenvectors.numpy().T
        assert target.status.numpy()[0] == 0 and target.converged.numpy()[0] == k
        # Independent reference/output readbacks are outside capture.
        import scipy.sparse as sp

        ah = sp.csr_matrix((a.values.numpy(), a.columns.numpy(), a.offsets.numpy()), shape=(n, n))
        bh = sp.csr_matrix((b.values.numpy(), b.columns.numpy(), b.offsets.numpy()), shape=(n, n))
        expected = la.eigh(ah.toarray(), bh.toarray(), eigvals_only=True)[:k]
        np.testing.assert_allclose(values, expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(vectors.T @ (bh @ vectors), np.eye(k), atol=1e-9)
        print(f"change={change:.2f}, iterations={target.iterations.numpy()[0]}, values={values}")
        wp.copy(starts, target.x)  # Device copy of computed modes and guard rows.


if __name__ == "__main__":
    main()
