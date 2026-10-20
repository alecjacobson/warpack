"""A real FEM material edit must reach both outer products and captured FP32 CG."""

from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg as la
import warp as wp
from warp.optim import linear

from benchmarks.applications.fem_cases import elastic_matrices, make_fem_case
from benchmarks.applications.runner import accuracy
from benchmarks.upstream.adapters import matrix
from warpack import BatchedCGPreconditioner, BatchedSparseOperator, BlockLOBPCG


@pytest.mark.parametrize("mass_type", ["clamped_lumped", "clamped_consistent"])
def test_public_fast_fem_material_update_and_computed_start_replay(mass_type):
    if not (
        hasattr(linear, "FSAI")
        and all(
            hasattr(linear.LinearOperator, attr) for attr in ("batch_offsets", "max_batch_length")
        )
    ):
        pytest.skip("pinned upstream FSAI and batched CG required")
    case = make_fem_case(mass_type, side=2, modes=6)
    p = case.problem
    selected = case.vertices[case.tets].mean(axis=1)[:, 0] > 1.5
    regional, _ = elastic_matrices(case.vertices, case.tets[selected])
    regional = regional[case.dofs, :][:, case.dofs]
    # Reserve structural zeros too: a material change can un-cancel an entry.
    pattern = (abs(p.a) + abs(regional) + abs(p.b)).tocsr()
    pattern.data.fill(1)
    rows, cols = pattern.tocoo().row, pattern.tocoo().col
    a, shifted, mass = matrix(pattern), matrix(pattern), matrix(p.b)

    def update(host):
        a.values[: len(rows)].assign(np.asarray(host[rows, cols]).ravel())
        shifted.values[: len(rows)].assign(np.asarray((host + 0.01 * p.b)[rows, cols]).ravel())

    update(p.a)
    width = 12
    factor = linear.FSAI(
        shifted, max_row_size=32, max_step_size=3, apply_lanes=8, factor_dtype=wp.float32
    )
    pre = BatchedCGPreconditioner(
        shifted,
        width,
        tol=0.01,
        maxiter=25,
        preconditioner=factor,
        working_dtype=wp.float32,
        fsai_layout="columns",
        matrix_layout="columns",
    )
    starts = wp.array(np.random.default_rng(182).normal(size=(width, p.n)), dtype=wp.float64)
    solver = BlockLOBPCG(
        BatchedSparseOperator(a),
        p.k,
        width,
        mass=BatchedSparseOperator(mass),
        preconditioner=pre,
        initial_vectors=starts,
        tol=1e-10,
        reuse_operator_images=True,
        search_bucket_size=4,
        keep_history=True,
        preconditioner_bucket_size=4,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
    )
    graph = solver.capture(150)
    # Keep the initial SPD FSAI as an approximate preconditioner. The CG matrix
    # and outer stiffness change at fixed addresses; no new capture or factors.
    for iteration, change in enumerate([0.0, 0.02, -0.01, 0.0]):
        if iteration:
            wp.copy(starts[: p.k], solver.eigenvectors)
        host = p.a + change * regional
        update(host)
        wp.capture_launch(graph)
        w, x = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        expected, basis = la.eigh(host.toarray(), p.b.toarray())
        target = replace(p, a=host, expected=expected[: p.k], requested_basis=basis[:, : p.k])
        quality = accuracy(target, w, x)
        assert quality["accepted"], quality
        assert solver.status.numpy()[0] == 0, quality
        assert solver.converged.numpy()[0] == p.k, quality
        assert solver.graph is graph
