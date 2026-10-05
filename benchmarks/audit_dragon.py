"""Compare the same cuDSS factors and count inverse applications on dragon."""

import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import warp as wp

from benchmarks.audit_profile import graph_summary, profile_restart, replay_timing
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator
from warpack.fem import assemble_rest, mass_normalized, read_mesh


def main():
    x, t = read_mesh("../bbw-comparison/dragon-H/dragon.mesh")
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass)
    shifted = mass_normalized(h, mass, shift=1.0)
    inverse = CuDSSInverse(shifted, 1)
    report = {"n": a.shape[0], "k": 26, "backend": "cuDSS shared factors", "warp": [], "cupy": []}
    for ncv in [48, 64]:
        for cycles in [2, 3]:
            s = KrylovSchur(inverse, 26, ncv=ncv, which="LA", tol=1e-12)
            evaluation = EigenpairEvaluation(SparseOperator(a), s.eigenvectors, tol=1e-7)
            graph = s.capture(cycles, adaptive=False, finalizer=evaluation.run)
            row = {
                "ncv": ncv,
                "cycles": cycles,
                **replay_timing(graph, repeats=3),
                "inverse_applications": s.keep - 1 + cycles * (s.m - s.keep + 1),
                "max_original_residual": float(evaluation.residuals.numpy().max()),
            }
            report["warp"].append(row)
            if ncv == 48 and cycles == 3:
                wp.capture_debug_dot_print(graph, "build/audit/dragon_graph.dot")
                report["graph_audit"] = graph_summary("build/audit/dragon_graph.dot")
                report["restart_stage_ms"] = profile_restart(s)
            print(row, flush=True)
    stream = cp.cuda.ExternalStream(wp.get_stream().cuda_stream)
    scalar = inverse.solver._csr
    with stream:
        from cupyx.scipy.sparse import csr_matrix

        ca = csr_matrix(
            (cp.asarray(scalar.values), cp.asarray(scalar.columns), cp.asarray(scalar.row_offsets)),
            shape=(a.shape[0], a.shape[0]),
        )
        calls = [0]

        def mv(v):
            calls[0] += 1
            inverse.apply(wp.from_dlpack(v.reshape(1, -1)), inverse.x)
            return cp.asarray(inverse.x).reshape(-1).copy()

        op = csl.LinearOperator(ca.shape, matvec=mv, dtype=np.float64)
        for ncv in [48, 64]:
            timings = []
            for _ in range(4):
                calls[0] = 0
                cp.random.seed(42)
                stream.synchronize()
                t0 = time.perf_counter()
                vals, vec = csl.eigsh(op, k=26, ncv=ncv, which="LA", tol=1e-12, maxiter=2000)
                stream.synchronize()
                timings.append(1000 * (time.perf_counter() - t0))
            w = 1 / vals - 1
            err = cp.linalg.norm(ca @ vec - vec * (w + 1), axis=0) / cp.maximum(1, cp.abs(w))
            row = {
                "ncv": ncv,
                "wall_ms": float(np.median(timings[1:])),
                "inverse_applications": calls[0],
                "max_original_residual": float(err.max()),
            }
            report["cupy"].append(row)
            print(row, flush=True)
    Path("results/audit_dragon.json").write_text(json.dumps(report, indent=2) + "\n")
    inverse.close()


if __name__ == "__main__":
    main()
