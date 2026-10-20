"""Accuracy-gated replay cost of optional CG residual auditing and in-place RHSs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import CGInverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-replays", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-11)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (
        min(args.side, args.width, args.repeats, args.warmup_replays) < 1
        or not np.isfinite(args.tol)
        or args.tol <= 0
    ):
        parser.error("require positive dimensions, repeats and finite tolerance")
    if args.width < 2:
        parser.error("width>=2 is required for nonzero and zero RHS coverage")
    side = args.side
    one = sp.eye(side, format="csr")
    line = sp.diags(
        [-np.ones(side - 1), np.full(side, 2.0), -np.ones(side - 1)], [-1, 0, 1], format="csr"
    )
    a = (
        sp.kron(sp.kron(line, one), one)
        + sp.kron(sp.kron(one, line), one)
        + sp.kron(sp.kron(one, one), line)
    ).tocsr()
    expected = np.random.default_rng(918).normal(size=(args.width, a.shape[0]))
    expected[-1] = 0.0
    rhs = (a @ expected.T).T.copy()
    inputs = wp.array(rhs, dtype=wp.float64)
    operator = matrix(a)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for values in [a.indptr, a.indices, a.data]:
        digest.update(values.tobytes())
    report = dict(
        args=vars(args),
        gpu=wp.get_device().name,
        warp=wp.__version__,
        n=a.shape[0],
        nnz=a.nnz,
        matrix_sha256=digest.hexdigest(),
        rhs_sha256=hashlib.sha256(rhs.tobytes()).hexdigest(),
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__), Path("warpack/generalized.py"), Path("warpack/iterative.py")]
        },
        protocol="Dirichlet 3D Poisson, manufactured random solution seed918, last RHS zero. Solution only forms/validates RHS, never initializes CG. Upstream diagonal preconditioner, relative tolerance, zero requested absolute tolerance, zero CG start every replay. Fresh host residual and known solution checked for all rows; optional device fresh counters separately mandatory. Warm replay includes resets and necessary in-place copying; setup/capture separate. Component cost, not whole eigensolver timing.",
        rows=[],
    )
    competing = set(other_gpu_processes())
    for inplace in [False, True]:
        for audit in [False, True]:

            def setup():
                return (
                    CGInverse(
                        operator,
                        tol=args.tol,
                        atol=0.0,
                        maxiter=2000,
                        preconditioner="diag",
                        verify_residuals=audit,
                    ),
                    wp.empty_like(inputs),
                )

            setup_seconds, (inverse, output) = timed(setup)

            def body():
                inverse.reset_stats()
                if inplace:
                    wp.copy(output, inputs)
                inverse.apply(output if inplace else inputs, output)

            jit_warmup_seconds, _ = timed(body)

            def record():
                with wp.ScopedCapture() as capture:
                    body()
                return capture.graph

            capture_seconds, graph = timed(record)

            def warmup():
                for _ in range(args.warmup_replays):
                    wp.capture_launch(graph)

            warmup_seconds, _ = timed(warmup)
            samples, checks = [], []
            for _ in range(args.repeats):
                competing.update(other_gpu_processes())
                seconds, _ = timed(lambda: wp.capture_launch(graph))
                competing.update(other_gpu_processes())
                value = output.numpy()
                residual = np.linalg.norm(a @ value.T - rhs.T, axis=0)
                norms = np.linalg.norm(rhs, axis=1)
                relative = np.divide(residual, norms, out=np.zeros_like(residual), where=norms > 0)
                error = float(np.linalg.norm(value - expected) / max(1, np.linalg.norm(expected)))
                stats = inverse.stats.numpy().tolist()
                check = dict(
                    relative_residual=float(relative.max()),
                    solution_error=error,
                    recursive_stats=stats,
                    accepted=bool(
                        relative.max() <= args.tol
                        and error < 1e-8
                        and stats[2] == 0
                        and stats[3] == args.width
                    ),
                )
                if audit:
                    check["fresh_stats"] = inverse.fresh_stats.numpy().tolist()
                    check["residual_maxima"] = inverse.residual_maxima.numpy().tolist()
                    check["accepted"] &= (
                        check["fresh_stats"][2] == 0 and check["fresh_stats"][3] == args.width
                    )
                samples.append(seconds)
                checks.append(check)
            dot = Path("build") / f"{out.stem}-{inplace}-{audit}.dot"
            dot.parent.mkdir(exist_ok=True)
            wp.capture_debug_dot_print(graph, str(dot))
            graph_audit = graph_summary(dot)
            dot.unlink()
            row = dict(
                inplace=inplace,
                verify_residuals=audit,
                setup_seconds=setup_seconds,
                setup_scope="CG/preconditioner/output workspace setup; shared matrix/RHS construction and upload excluded",
                jit_warmup_seconds=jit_warmup_seconds,
                capture_seconds=capture_seconds,
                warmup_seconds=warmup_seconds,
                samples_seconds=samples,
                median_seconds=float(np.median(samples)),
                checks=checks,
                accepted=all(q["accepted"] for q in checks),
                graph=graph_audit,
            )
            report["rows"].append(row)
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(inplace, audit, row["accepted"], row["median_seconds"], flush=True)
    report["accepted"] = all(row["accepted"] for row in report["rows"])
    report["competing_gpu_processes"] = sorted(competing)
    report["timing_eligible"] = not competing and report["accepted"]
    out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["timing_eligible"]:
        raise SystemExit("Rejected accuracy/occupancy gate; evidence retained")


if __name__ == "__main__":
    main()
