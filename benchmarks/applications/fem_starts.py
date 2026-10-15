"""Related FEM solves: computed block starts versus fresh random starts.

The second system increases stiffness by 2% in the right half of a clamped
beam. Starting vectors come only from a Warpack solve of the first system;
independent reference vectors are used exclusively for validation.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse, GeneralizedEigensolver, SparseOperator

from .compare import other_gpu_processes, timed
from .fem_cases import elastic_matrices, make_fem_case, reference
from .runner import accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["clamped_lumped", "clamped_consistent"], default="clamped_consistent"
    )
    parser.add_argument("--side", type=int, default=6)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--ncv", type=int, default=16)
    parser.add_argument("--change", type=float, default=0.02)
    parser.add_argument("--budgets", default="4,8,16,32,64")
    parser.add_argument("--bootstrap-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cupy", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    budgets = sorted(set(int(x) for x in args.budgets.split(",")))
    if (
        not budgets
        or budgets[0] < 1
        or args.repeats < 1
        or args.bootstrap_steps < 1
        or not np.isfinite(args.change)
        or args.change <= 0
    ):
        parser.error("Positive budgets, repeat count, bootstrap steps and material change required")
    c = make_fem_case(args.case, args.side, args.modes)
    p = c.problem
    if not p.k <= args.ncv < p.n:
        parser.error("Require k <= ncv < n")
    width = args.ncv
    old = CuDSSInverse(matrix(p.a - p.target * p.b), width)
    try:
        bootstrap = GeneralizedEigensolver(
            SparseOperator(matrix(p.a)),
            SparseOperator(matrix(p.b)),
            old,
            p.k,
            ncv=width,
            target=p.target,
            tol=1e-11,
        )
        wp.capture_launch(bootstrap.capture(args.bootstrap_steps, adaptive=False))
        source_quality = accuracy(
            p, bootstrap.eigenvalues.numpy(), bootstrap.eigenvectors.numpy().T
        )
        if not source_quality["accepted"] or bootstrap.status.numpy()[0] != 0:
            raise RuntimeError(f"Bootstrap did not pass independent validation: {source_quality}")
        # Copy before releasing the first inverse. Neither uses reference modes.
        modes = wp.clone(bootstrap.eigenvectors)
        subspace = wp.clone(bootstrap.q)
    finally:
        old.close()
    selected = c.vertices[c.tets].mean(axis=1)[:, 0] > 1.5
    regional, _ = elastic_matrices(c.vertices, c.tets[selected])
    regional = regional[c.dofs, :][:, c.dofs]
    changed = p.a + args.change * regional
    spectrum, vectors, reference_label = reference(changed, p.b, p.k + 8)
    target = replace(p, a=changed, expected=spectrum[: p.k], requested_basis=vectors[:, : p.k])
    contention_before = other_gpu_processes()
    setup_seconds, inverse = timed(lambda: CuDSSInverse(matrix(changed - p.target * p.b), width))
    report = dict(
        arguments=vars(args),
        n=p.n,
        k=p.k,
        ncv=width,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_changes=subprocess.check_output(["git", "status", "--short"], text=True),
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("warpack").glob("*.py"))
        },
        versions={
            name: importlib.metadata.version(name)
            for name in ("warp-lang", "numpy", "scipy", "nvidia-cudss-cu12")
        },
        reference=reference_label,
        source_quality=source_quality,
        start_provenance="Computed Warpack modes/subspace of unmodified beam; no reference starts",
        inverse_setup_seconds=setup_seconds,
        rows=[],
    )
    try:
        op, mass = SparseOperator(matrix(changed)), SparseOperator(matrix(p.b))
        for label, start in (
            ("random", None),
            ("previous_modes", modes),
            ("previous_full_subspace", subspace),
        ):
            solver = GeneralizedEigensolver(
                op,
                mass,
                inverse,
                p.k,
                ncv=width,
                target=p.target,
                tol=1e-8,
                initial_vectors=start,
            )
            solver.initialize()
            # Same normalized starting block for the optional CuPy comparison.
            initial = solver.q.numpy().T.copy()
            for budget in budgets:
                capture_seconds, graph = timed(lambda: solver.capture(budget, adaptive=False))
                samples, quality = [], []
                for _ in range(args.repeats):
                    seconds, _ = timed(lambda: wp.capture_launch(graph))
                    samples.append(seconds)
                    q = accuracy(target, solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T)
                    q.update(
                        status=int(solver.status.numpy()[0]),
                        converged=int(solver.converged.numpy()[0]),
                    )
                    q["accepted"] &= q["status"] == 0 and q["converged"] == p.k
                    quality.append(q)
                row = dict(
                    method="Warp block inverse iteration + cuDSS",
                    start=label,
                    steps=budget,
                    capture_seconds=capture_seconds,
                    seconds=samples,
                    median_seconds=float(np.median(samples)),
                    quality=quality,
                    accepted=all(q["accepted"] for q in quality),
                )
                report["rows"].append(row)
                print(label, budget, row["accepted"], quality[-1]["residual"], flush=True)
            if args.cupy:
                report["rows"].append(cupy_trial(target, inverse, initial, label, args.repeats))
    finally:
        inverse.close()
    contention_after = other_gpu_processes()
    report.update(
        other_gpu_processes_before=contention_before,
        other_gpu_processes_after=contention_after,
        timing_eligible=not (contention_before or contention_after),
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination, "timing eligible:", report["timing_eligible"], flush=True)


def cupy_trial(problem, inverse, initial, label, repeats):
    import cupy as cp
    import cupyx.scipy.sparse as cs
    import cupyx.scipy.sparse.linalg as cl

    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        a, b = cs.csr_matrix(problem.a), cs.csr_matrix(problem.b)
        start = cp.asarray(initial)
        rhs, solution = cp.asarray(inverse.b), cp.asarray(inverse.x)

        def apply(x):
            one = x.ndim == 1
            xx = x[:, None] if one else x
            inverse.b.zero_()
            rhs[: xx.shape[1]] = xx.T
            inverse.apply(inverse.b, inverse.x)
            out = solution[: xx.shape[1]].T.copy()
            return out[:, 0] if one else out

        pre = cl.LinearOperator(a.shape, matvec=apply, matmat=apply, dtype=cp.float64)

        def solve():
            return cl.lobpcg(
                a,
                start.copy(),
                B=b,
                M=pre,
                largest=False,
                tol=1e-10,
                maxiter=200,
                retResidualNormsHistory=True,
            )

        solve()  # warmup; setup and compilation are excluded from replay samples
        samples, quality, lengths = [], [], []
        for _ in range(repeats):
            seconds, (w, v, history) = timed(solve)
            samples.append(seconds)
            quality.append(accuracy(problem, w[: problem.k].get(), v[:, : problem.k].get()))
            lengths.append(len(history))
    return dict(
        method="CuPy LOBPCG + same batched cuDSS preconditioner",
        start=label,
        block_width=initial.shape[1],
        residual_history_lengths=lengths,
        seconds=samples,
        median_seconds=float(np.median(samples)),
        quality=quality,
        accepted=all(q["accepted"] for q in quality),
    )


if __name__ == "__main__":
    main()
