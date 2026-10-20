"""First 20 elastic Anvil modes using only installed warpack solver APIs.

Run from a source checkout for the mesh and FEM assembly helpers. The solve
needs the pinned batched-CG/FSAI Warp environment in docs/releases/v0.2.0.md.
Host mesh I/O, reporting and optional export are outside captured computation.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from warp.optim import linear

from examples.problems.elasticity import assemble_rest, mass_normalized, read_mesh, rigid_body_basis
from warpack import BatchedCGPreconditioner, BatchedSparseOperator, BlockLOBPCG


def timed(call):
    wp.synchronize()
    start = time.perf_counter()
    result = call()
    wp.synchronize()
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default="anvil for export-ftetwild.mesh")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="build/anvil-block.json")
    parser.add_argument("--modes-out", help="Optional NPZ with 20 physical displacement fields")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not hasattr(linear, "FSAI") or not all(
        hasattr(linear.LinearOperator, attr) for attr in ("batch_offsets", "max_batch_length")
    ):
        parser.error(
            "install the pinned Warp FSAI/batched-CG environment in docs/releases/v0.2.0.md"
        )
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        start = time.perf_counter()
        vertices, tets = read_mesh(args.mesh)
        vertices = (vertices - vertices.mean(0)) / np.ptp(vertices, axis=0).max()
        hessian, _, mass = assemble_rest(vertices, tets)
        a = mass_normalized(hessian, mass)
        shifted = mass_normalized(hessian, mass, shift=100.0)
        shifted.nnz_sync()
        rigid = rigid_body_basis(wp.array(vertices, dtype=wp.vec3d), mass)
        wp.synchronize()
        assembly_seconds = time.perf_counter() - start
        factor_seconds, factor = timed(
            lambda: linear.FSAI(
                shifted, max_row_size=32, max_step_size=3, apply_lanes=8, factor_dtype=wp.float32
            )
        )
        # The same seed/layout as the comparison harness. This is setup only.
        initial = np.random.default_rng(42).normal(size=(a.shape[0], 24))

        def create_solver():
            preconditioner = BatchedCGPreconditioner(
                shifted,
                24,
                preconditioner=factor,
                tol=0.01,
                maxiter=25,
                working_dtype=wp.float32,
                fsai_layout="columns",
                matrix_layout="columns",
            )
            return BlockLOBPCG(
                BatchedSparseOperator(a),
                20,
                24,
                preconditioner=preconditioner,
                initial_vectors=wp.array(initial.T.copy(), dtype=wp.float64),
                constraints=rigid,
                tol=1e-8,
                gram_chunk=4096,
                symmetric_products=True,
                fused_updates=True,
                paired_rotation=True,
                reuse_operator_images=True,
                search_bucket_size=4,
                keep_history=True,
                preconditioner_bucket_size=4,
            )

        workspace_seconds, solver = timed(create_solver)
        capture_seconds, graph = timed(lambda: solver.capture(200))
        warmup_seconds, _ = timed(lambda: wp.capture_launch(graph))
        rows = []
        for _ in range(args.repeats):
            seconds, _ = timed(lambda: wp.capture_launch(graph))
            row = dict(
                seconds=seconds,
                iterations=int(solver.iterations.numpy()[0]),
                status=int(solver.status.numpy()[0]),
                converged=int(solver.converged.numpy()[0]),
                max_residual=float(solver.residuals.numpy().max()),
                cg_stats=solver.preconditioner.stats.numpy().tolist(),
            )
            rows.append(row)
            if row["status"] != 0 or row["converged"] != 20 or not row["max_residual"] <= 1e-8:
                raise RuntimeError(f"Incomplete or invalid solve: {row}")
        report = dict(
            mesh=args.mesh,
            dofs=a.shape[0],
            assembly_and_io_seconds=assembly_seconds,
            factor_seconds=factor_seconds,
            workspace_seconds=workspace_seconds,
            total_setup_seconds=factor_seconds + workspace_seconds,
            capture_seconds=capture_seconds,
            warmup_seconds=warmup_seconds,
            median_solve_seconds=float(np.median([r["seconds"] for r in rows])),
            rows=rows,
            eigenvalues=solver.eigenvalues.numpy().tolist(),
            timing_scope="Synchronized replay only; setup, capture, warmup and validation excluded.",
            validation_scope="Device original-equation residual/count; independent spectrum and subspace comparisons are in the benchmark report.",
        )
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if args.modes_out:
            destination = Path(args.modes_out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            masses = mass.numpy()
            modes = (
                solver.eigenvectors.numpy().reshape(20, len(vertices), 3)
                / np.sqrt(masses)[None, :, None]
            )
            np.savez_compressed(
                destination,
                vertices=vertices,
                tets=tets,
                mass=masses,
                eigenvalues=solver.eigenvalues.numpy(),
                modes=modes,
            )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
