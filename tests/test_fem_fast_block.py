"""Public fast composition on independent FEM matrices, including full clusters."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from warp.optim import linear

from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.compare_block import FEM_CASES, prepare
from warpack import BatchedCGPreconditioner, BatchedSparseOperator, BlockLOBPCG


@pytest.mark.parametrize(
    "case,modes",
    [(case, modes) for case in FEM_CASES for modes in (3, 6)]
    + [(case, 30) for case in FEM_CASES if case != "repeated_bending"],
)
def test_fast_fem_requested_space_and_replay(case, modes, tmp_path):
    if not (
        hasattr(linear, "FSAI")
        and all(
            hasattr(linear.LinearOperator, attr) for attr in ("batch_offsets", "max_batch_length")
        )
    ):
        pytest.skip("pinned upstream FSAI and batched CG required")
    args = SimpleNamespace(
        case=case, side=3 if case == "repeated_bending" else 2, modes=modes, shift=0.01
    )
    metadata = {}
    a, mass, shifted, constraints, _, _, k, check = prepare(args, metadata=metadata)
    width = 4 * ((k + 7) // 4)
    pre = linear.FSAI(shifted, max_row_size=32, max_step_size=3, factor_dtype=wp.float32)
    cg = BatchedCGPreconditioner(
        shifted,
        width,
        tol=0.01,
        maxiter=25,
        preconditioner=pre,
        working_dtype=wp.float32,
        fsai_layout="columns",
        matrix_layout="columns",
    )
    start = wp.array(np.random.default_rng(42).normal(size=(width, a.shape[0])), dtype=wp.float64)
    solver = BlockLOBPCG(
        BatchedSparseOperator(a),
        k,
        width,
        mass=BatchedSparseOperator(mass),
        constraints=constraints,
        preconditioner=cg,
        initial_vectors=start,
        tol=1e-11,
        gram_chunk=4096,
        symmetric_products=True,
        fused_updates=True,
        paired_rotation=True,
        reuse_operator_images=True,
        search_bucket_size=4,
        keep_history=True,
        preconditioner_bucket_size=4,
    )
    graph = solver.capture(300)
    for _ in range(2):
        wp.capture_launch(graph)
        quality = check(solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
        assert solver.status.numpy()[0] == 0, quality
        assert solver.converged.numpy()[0] == k, quality
        assert quality["accepted"], quality
        assert quality["recovered_nullity"] == metadata["retained_rigid_modes"]
    if case == "repeated_bending":
        assert k == modes + modes % 2  # Never cut a double eigenvalue in half.
    dot = tmp_path / "fem.dot"
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    assert audit["node_types_including_conditional_body"].get("HOST", 0) == 0
    assert all("DtoD" in kind for kind in audit["memcpy_kinds"])
