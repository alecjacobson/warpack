"""Independent cuDSS setup and validated Anvil solves for either outer solver."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import warp as wp

from benchmarks.anvil import validate
from benchmarks.applications.compare import timed
from benchmarks.experimental.compare_block import prepare
from warpack import CuDSSInverse, KrylovSchur, SymmetricEigensolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["warp", "cupy"], required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full-warmup", action="store_true", help="Control: force pre-#132 warmup")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    _, _, shifted, _, sa, _, _, _ = prepare(argparse.Namespace(case="anvil", shift=1.0))
    shifted.nnz_sync()
    initial = np.load("build/anvil_initial.npy")
    cached = np.load("build/anvil_reference.npz")
    reference = cached["vectors"], cached["values"]
    stream = cp.cuda.Stream.from_external(wp.get_stream())

    phases = {}

    def setup():
        phases["factor_seconds"], inverse = timed(lambda: CuDSSInverse(shifted, 1))
        if args.backend == "warp":
            workspace_start = time.perf_counter()
            solver = KrylovSchur(
                inverse,
                26,
                ncv=64,
                which="LA",
                tol=1e-12,
                initial_vector=wp.array(initial.reshape(1, -1), dtype=wp.float64),
            )
            wp.synchronize()
            phases["workspace_seconds"] = time.perf_counter() - workspace_start
            if args.full_warmup:
                solver._prepare_capture = lambda finalizer: SymmetricEigensolver._prepare_capture(
                    solver, finalizer
                )
            prepare = getattr(solver, "_prepare_capture", None)
            original = prepare if prepare is not None else solver.solve

            def record_warmup(*a, **kw):
                seconds, result = timed(lambda: original(*a, **kw))
                phases["capture_preparation_seconds"] = seconds
                return result

            if prepare is not None:
                solver._prepare_capture = record_warmup
            else:
                solver.solve = record_warmup
            phases["capture_seconds"], graph = timed(lambda: solver.capture(1, adaptive=False))
            if prepare is not None:
                del solver._prepare_capture
            else:
                del solver.solve

            def solve():
                wp.capture_launch(graph)

            def quality():
                result = validate(sa, solver.eigenvectors.numpy(), reference)[0]
                result.update(
                    status=int(solver.status.numpy()[0]),
                    converged=int(solver.converged.numpy()[0]),
                )
                result["accepted"] &= result["status"] == 0 and result["converged"] == 26
                return result

        else:
            with stream:
                start = cp.asarray(initial)

                def mv(v):
                    inverse.apply(wp.from_dlpack(v.reshape(1, -1)), inverse.x)
                    return cp.asarray(inverse.x).reshape(-1).copy()

                op = csl.LinearOperator(sa.shape, matvec=mv, dtype=cp.float64)
            output = []

            def solve():
                with stream:
                    _, v = csl.eigsh(
                        op, k=26, ncv=64, which="LA", tol=1e-12, maxiter=10000, v0=start.copy()
                    )
                    output[:] = [v]

            def quality():
                with stream:
                    return validate(sa, cp.asnumpy(output[0]).T, reference)[0]

        return inverse, solve, quality

    setup_seconds, (inverse, solve, quality) = timed(setup)
    warmup_seconds, _ = timed(solve)
    warmup_quality = quality()
    rows = []
    for _ in range(args.repeats):
        seconds, _ = timed(solve)
        rows.append(dict(seconds=seconds, quality=quality()))
    accepted = warmup_quality["accepted"] and all(row["quality"]["accepted"] for row in rows)
    report = dict(
        backend=args.backend,
        setup_seconds=setup_seconds,
        setup_phases_seconds=phases,
        full_warmup_control=args.full_warmup,
        warmup_seconds=warmup_seconds,
        warmup_quality=warmup_quality,
        rows=rows,
        accepted=bool(accepted),
        setup_scope="cuDSS factor construction plus outer operator/workspace/start setup and Warp graph capture; assembled input is ready. Excludes imports, assembly, outer warmup, validation. Existing JIT disk caches.",
        initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        mesh_sha256=hashlib.sha256(Path("anvil for export-ftetwild.mesh").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(warp=wp.__version__, cupy=cp.__version__),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("Setup", setup_seconds, "solve", [r["seconds"] for r in rows], "accepted", accepted)
    del solve, quality  # Release graph/solver closures before their cuDSS resources.
    inverse.close()
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
