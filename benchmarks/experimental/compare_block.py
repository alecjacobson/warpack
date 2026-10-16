"""Accuracy-gated experimental block-method comparisons; no production CuPy use."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp
import warp.optim.linear as linear
import warp.sparse as ws

from benchmarks.anvil import validate
from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.applications.fem_cases import make_fem_case
from benchmarks.applications.runner import accuracy
from benchmarks.audit_profile import graph_summary
from benchmarks.experimental.block_lobpcg import BlockLOBPCG, UpstreamPreconditioner
from benchmarks.upstream.adapters import matrix
from examples.problems.elasticity import assemble_rest, mass_normalized, read_mesh, rigid_body_basis
from warpack import CGInverse, SparseOperator


def prepare(args):
    if args.case == "anvil":
        reference = np.load("build/anvil_reference.npz")
        x, t = read_mesh("anvil for export-ftetwild.mesh")
        x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
        h, _, mass = assemble_rest(x, t)
        a = mass_normalized(h, mass)
        a.nnz_sync()
        sa = sp.bsr_matrix(
            (a.values.numpy()[: a.nnz], a.columns.numpy()[: a.nnz], a.offsets.numpy()),
            shape=a.shape,
        ).tocsr()
        constraints = rigid_body_basis(wp.array(x, dtype=wp.vec3d), mass)
        shifted = mass_normalized(h, mass, shift=args.shift)

        def check(w, v):
            quality = validate(
                sa,
                np.concatenate([constraints.numpy(), v.T]),
                (reference["vectors"], reference["values"]),
            )[0]
            # The shared Anvil validator recomputes Rayleigh quotients. Also
            # check the eigenvalues actually returned by the tested solver.
            av = sa @ v
            residual = np.linalg.norm(av - v * w, axis=0) / np.maximum(
                1, np.linalg.norm(av, axis=0) + abs(w) * np.linalg.norm(v, axis=0)
            )
            quality.update(
                returned_eigenvalue_error=float(np.max(abs(w / reference["values"][6:] - 1))),
                returned_pair_residual=float(residual.max()),
            )
            quality["accepted"] &= (
                quality["returned_eigenvalue_error"] < 1e-8
                and quality["returned_pair_residual"] < 1e-8
            )
            return quality

        return a, None, shifted, constraints, sa, None, 20, check
    c = make_fem_case(args.case, args.side, args.modes)
    p = c.problem
    shifted_host = p.a + args.shift * p.b
    blocks = shifted_host.tobsr(blocksize=(3, 3))
    shifted = ws.bsr_from_triplets(
        p.n // 3,
        p.n // 3,
        wp.array(np.repeat(np.arange(p.n // 3), np.diff(blocks.indptr)), dtype=wp.int32),
        wp.array(blocks.indices, dtype=wp.int32),
        wp.array(blocks.data, dtype=wp.mat33d),
    )
    return (matrix(p.a), matrix(p.b), shifted, None, p.a, p.b, p.k, lambda w, v: accuracy(p, w, v))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        choices=["anvil", "clamped_consistent", "clamped_lumped"],
        default="clamped_consistent",
    )
    p.add_argument("--side", type=int, default=6)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--shift", type=float, default=0.01)
    p.add_argument("--fsai-width", type=int, default=64)
    p.add_argument("--preconditioner", choices=["fsai", "bj"], default="fsai")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--cupy", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--keep-graph", action="store_true", help="Retain the potentially large DOT dump"
    )
    p.add_argument("--gram-chunk", type=int, default=256)
    p.add_argument("--native-products", action="store_true")
    p.add_argument("--batched-products", action="store_true")
    p.add_argument("--inner-cg", type=int, default=0)
    p.add_argument("--inner-tol", type=float, default=0.01)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.cupy and args.inner_cg:
        p.error("CuPy adapter currently compares direct FSAI/BJ application only")
    if args.inner_cg < 0 or args.repeats < 1:
        p.error("require inner-cg >= 0 and repeats >= 1")
    before = other_gpu_processes()
    a, mass, shifted, constraints, sa, sb, k, check = prepare(args)
    print("Prepared", args.case, "n=", a.shape[0], flush=True)
    shifted.nnz_sync()
    setup_seconds, pre = timed(
        lambda: (
            linear.FSAI(
                shifted,
                max_row_size=args.fsai_width,
                max_step_size=3,
                apply_lanes=8,
                factor_dtype=wp.float32,
            )
            if args.preconditioner == "fsai"
            else linear.preconditioner(shifted, ptype="block_jacobi_sequential")
        )
    )
    print("Preconditioner ready", setup_seconds, flush=True)
    initial = np.random.default_rng(42).normal(size=(a.shape[0], args.width))
    starts = wp.array(initial.T.copy(), dtype=wp.float64)
    from benchmarks.experimental.block_kernels import BatchedSparseOperator, UpstreamSparseOperator

    operator = UpstreamSparseOperator if args.native_products else SparseOperator
    if args.batched_products:
        operator = BatchedSparseOperator
    preconditioner = (
        CGInverse(shifted, tol=args.inner_tol, atol=0.0, maxiter=args.inner_cg, preconditioner=pre)
        if args.inner_cg
        else UpstreamPreconditioner(pre)
    )
    solver = BlockLOBPCG(
        operator(a),
        k,
        args.width,
        mass=operator(mass) if mass is not None else None,
        preconditioner=preconditioner,
        initial_vectors=starts,
        constraints=constraints,
        tol=args.tol,
        gram_chunk=args.gram_chunk,
    )
    capture_seconds, graph = timed(lambda: solver.capture(args.steps))
    print("Captured", capture_seconds, flush=True)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dot = Path("build/block-eigen") / (destination.stem + ".dot")
    dot.parent.mkdir(parents=True, exist_ok=True)
    wp.capture_debug_dot_print(graph, str(dot))
    audit = graph_summary(dot)
    if not args.keep_graph:
        dot.unlink()
    warmup_seconds, _ = timed(lambda: wp.capture_launch(graph))
    report = dict(
        args=vars(args),
        n=a.shape[0],
        k=k,
        setup_seconds=setup_seconds,
        capture_seconds=capture_seconds,
        warmup_seconds=warmup_seconds,
        graph=audit,
        rows=[],
        versions={
            name: importlib.metadata.version(name)
            for name in ["warp-lang", "numpy", "scipy", "cupy-cuda12x"]
        },
        source_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "benchmarks/experimental/block_lobpcg.py",
                "benchmarks/experimental/block_kernels.py",
                "benchmarks/experimental/compare_block.py",
                "warpack/generalized.py",
            )
        },
    )
    for _ in range(args.repeats):
        seconds, _ = timed(lambda: wp.capture_launch(graph))
        w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        quality = check(w, v)
        quality.update(
            status=int(solver.status.numpy()[0]), converged=int(solver.converged.numpy()[0])
        )
        quality["accepted"] &= quality["status"] == 0 and quality["converged"] == k
        if args.inner_cg:
            quality["preconditioner_cg_stats"] = preconditioner.stats.numpy().tolist()
        report["rows"].append(
            dict(
                method="Warp experimental LOBPCG",
                seconds=seconds,
                iterations=int(solver.iterations.numpy()[0]),
                quality=quality,
            )
        )
        print(report["rows"][-1], flush=True)
    if args.cupy:
        report["rows"].extend(cupy_trials(sa, sb, pre, initial, constraints, args, k, check))
    if args.profile:
        report["profile"] = profile_step(solver)
        report["post_profile_quality"] = check(
            solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
        )
    after = other_gpu_processes()
    report.update(
        other_gpu_processes_before=before,
        other_gpu_processes_after=after,
        timing_eligible=not (before or after),
        timing_note="Warm replay wall times; setup/capture separate. CuPy converges the entire oversampled block; Warp stops on the first k. Contended or numerically rejected runs are not performance evidence.",
    )
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination, flush=True)


def profile_step(solver):
    from benchmarks.applications.skinning_profile import Events
    from benchmarks.experimental import block_lobpcg as implementation

    events = Events()
    saved = []
    for obj, attribute, label in (
        (implementation, "gram", "gram"),
        (implementation, "rotate", "rotate"),
        (solver.preconditioner, "apply", "preconditioner"),
        (solver.op, "apply", "operator"),
        (solver.mass, "apply", "mass"),
        (solver.small, "diagonalize", "rayleigh_ritz_eigh"),
        (solver.metric, "diagonalize", "metric_eigh"),
        *[
            (solver, name, name)
            for name in (
                "project",
                "orthogonalize_x",
                "orthogonalize_search",
                "whiten",
                "finalize",
                "step",
            )
        ],
    ):
        original = getattr(obj, attribute)
        saved.append((obj, attribute, original))
        setattr(obj, attribute, events.wrap(label, original))
    try:
        with wp.ScopedCapture() as capture:
            solver.step()  # Events are legal here: this graph has no conditional body.
    finally:
        for obj, attribute, original in saved:
            setattr(obj, attribute, original)
    samples = []
    for _ in range(5):
        wp.capture_launch(capture.graph)
        wp.synchronize()
        samples.append(events.totals())
    return dict(
        samples_ms=samples,
        medians_ms={key: float(np.median([row[key] for row in samples])) for key in samples[0]},
        note="Instrumented fixed single-step graph, continued from last solve. Nested totals overlap; not additive or equivalent to adaptive whole-solve timings.",
    )


def cupy_trials(a, b, pre, initial, constraints, args, k, check):
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as cl

    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        a = cs.csr_matrix(a)
        b = None if b is None else cs.csr_matrix(b)
        start = cp.asarray(initial)
        y = None if constraints is None else cp.asarray(constraints).T

        def apply(x):
            one = x.ndim == 1
            rows = cp.ascontiguousarray(x.reshape(-1, 1).T if one else x.T)
            out = cp.empty_like(rows)
            wx, wy = wp.from_dlpack(rows), wp.from_dlpack(out)
            for i in range(len(rows)):
                pre.matvec(wx[i], wy[i], wy[i], 1.0, 0.0)
            return out[0] if one else out.T

        preconditioner = cl.LinearOperator(a.shape, matvec=apply, matmat=apply, dtype=cp.float64)

        def solve():
            return cl.lobpcg(
                a,
                start.copy(),
                B=b,
                M=preconditioner,
                Y=y,
                largest=False,
                tol=1e-5 if args.case == "anvil" else 1e-10,
                maxiter=args.steps,
                retResidualNormsHistory=True,
            )

        solve()
        rows = []
        for _ in range(args.repeats):
            seconds, (w, v, history) = timed(solve)
            rows.append(
                dict(
                    method="CuPy LOBPCG + same upstream Warp preconditioner",
                    seconds=seconds,
                    residual_history_length=len(history),
                    quality=check(w[:k].get(), v[:, :k].get()),
                )
            )
            print(rows[-1], flush=True)
        return rows


if __name__ == "__main__":
    main()
