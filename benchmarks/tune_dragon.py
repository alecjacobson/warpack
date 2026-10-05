"""Select a fixed graph budget using independently evaluated physical residuals."""

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import warp as wp

from benchmarks.audit_profile import replay_timing
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator
from warpack.fem import assemble_rest, mass_normalized, read_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/dragon_tuning.json")
    parser.add_argument("--same-start", action="store_true")
    args = parser.parse_args()
    starting_vector = None
    x, t = read_mesh("../bbw-comparison/dragon-H/dragon.mesh")
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass)
    inverse = CuDSSInverse(mass_normalized(h, mass, shift=1.0), 1)
    report = {
        "backend": "cuDSS shared factors",
        "shared_starting_vector": args.same_start,
        "starting_vector_reset_per_trial": True,
        "residual_threshold": 1e-7,
        "candidates": [],
        "cupy": [],
    }
    for ncv, cycles in [
        (40, 2),
        (40, 3),
        (44, 2),
        (44, 3),
        (48, 2),
        (52, 1),
        (56, 1),
        (60, 1),
        (64, 1),
    ]:
        s = KrylovSchur(inverse, 26, ncv=ncv, which="LA", tol=1e-12)
        evaluation = EigenpairEvaluation(SparseOperator(a), s.eigenvectors, tol=1e-7)
        graph = s.capture(cycles, adaptive=False, finalizer=evaluation.run)
        if args.same_start and starting_vector is None:
            s.initialize()
            starting_vector = s.views[0].numpy().ravel()
        wp.capture_launch(graph)
        err = evaluation.residuals.numpy()
        row = {
            "ncv": ncv,
            "cycles": cycles,
            "inverse_applications": s.keep - 1 + cycles * (ncv - s.keep + 1),
            "residual": float(err.max()),
            "elastic_residual": float(err[6:].max()),
            "accepted": bool(err.max() < 1e-7),
        }
        if row["accepted"]:
            row.update(replay_timing(graph, repeats=5))
            v = s.eigenvectors.numpy()
            row["orthogonality"] = float(np.max(abs(v @ v.T - np.eye(26))))
            row["eigenvalues"] = evaluation.values.numpy().tolist()
        report["candidates"].append(row)
        print(row, flush=True)
    stream = cp.cuda.Stream.from_external(wp.get_stream())
    scalar = inverse.solver._csr
    with stream:
        from cupyx.scipy.sparse import csr_matrix

        ca = csr_matrix(
            (cp.asarray(scalar.values), cp.asarray(scalar.columns), cp.asarray(scalar.row_offsets)),
            shape=(a.shape[0], a.shape[0]),
        )
        v0 = cp.asarray(starting_vector) if args.same_start else None
        calls = [0]

        def mv(v):
            calls[0] += 1
            inverse.apply(wp.from_dlpack(v.reshape(1, -1)), inverse.x)
            return cp.asarray(inverse.x).reshape(-1).copy()

        op = csl.LinearOperator(ca.shape, matvec=mv, dtype=np.float64)
        for ncv, tol in [
            (60, 1e-10),
            (60, 1e-11),
            (60, 1e-12),
            (64, 1e-10),
            (64, 1e-11),
            (64, 1e-12),
        ]:
            samples = []
            for _ in range(6):
                calls[0] = 0
                cp.random.seed(42)
                trial_start = v0.copy() if v0 is not None else None
                stream.synchronize()
                t0 = time.perf_counter()
                w, v = csl.eigsh(
                    op, k=26, ncv=ncv, which="LA", tol=tol, maxiter=2000, v0=trial_start
                )
                stream.synchronize()
                samples.append(1000 * (time.perf_counter() - t0))
            vals = 1 / w - 1
            err = cp.linalg.norm(ca @ v - v * (vals + 1), axis=0) / cp.maximum(1, cp.abs(vals))
            row = {
                "ncv": ncv,
                "tol": tol,
                "wall_ms": float(np.median(samples[1:])),
                "inverse_applications": calls[0],
                "residual": float(err.max()),
                "accepted": bool(float(err.max()) < 1e-7),
            }
            report["cupy"].append(row)
            print(row, flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    inverse.close()


if __name__ == "__main__":
    main()
