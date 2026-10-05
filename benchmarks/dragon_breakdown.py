"""Separate inverse solves, basis work, and validation on the full dragon.

Stage measurements add CUDA events and are diagnostic, not an API trace.
CuPy and Warp share one cuDSS factorization and restored starting vectors.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as csl
import numpy as np
import warp as wp

from benchmarks.audit_profile import replay_timing
from warpack import CuDSSInverse, EigenpairEvaluation, KrylovSchur, SparseOperator
from warpack.fem import assemble_rest, mass_normalized, read_mesh


class Events:
    def __init__(self):
        self.pairs = []

    def wrap(self, name, function):
        def run(*args, **kwargs):
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            external = wp.get_stream().is_capturing
            wp.record_event(start, external=external)
            result = function(*args, **kwargs)
            wp.record_event(end, external=external)
            self.pairs.append((name, start, end))
            return result

        return run

    def totals(self):
        result = defaultdict(float)
        for name, start, end in self.pairs:
            result[name] += wp.get_event_elapsed_time(start, end, False)
        return dict(result)


def main():
    x, t = read_mesh("../bbw-comparison/dragon-H/dragon.mesh")
    x = (x - x.mean(0)) / np.ptp(x, axis=0).max()
    h, _, mass = assemble_rest(x, t)
    a = mass_normalized(h, mass)
    inverse = CuDSSInverse(mass_normalized(h, mass, shift=1.0), 1)
    solver = KrylovSchur(inverse, 26, ncv=64, which="LA", tol=1e-12)
    evaluation = EigenpairEvaluation(SparseOperator(a), solver.eigenvectors, tol=1e-7)
    solve_graph = solver.capture(1, adaptive=False)
    solver.initialize()
    initial = solver.views[0].numpy().ravel()
    evaluation.run()
    with wp.ScopedCapture() as cap:
        evaluation.run()
    report = {
        "backend": "same cuDSS factorization, 64 inverse applications each",
        "n": solver.n,
        "ncv": 64,
        "k": 26,
        "warp_solve_only": replay_timing(solve_graph),
        "warp_original_problem_validation": replay_timing(cap.graph),
    }
    report["warp_residual"] = float(evaluation.residuals.numpy().max())
    events = Events()
    original_apply = inverse.apply
    inverse.apply = events.wrap("inverse", original_apply)
    for name in [
        "initialize",
        "expand",
        "complete_projection",
        "diagonalize",
        "rotate",
        "finalize",
    ]:
        setattr(solver, name, events.wrap(name, getattr(solver, name)))
    with wp.ScopedCapture() as cap:
        solver.solve(1)
        events.wrap("original_problem_validation", evaluation.run)()
    samples = []
    for _ in range(6):
        wp.capture_launch(cap.graph)
        wp.synchronize()
        samples.append(events.totals())
    report["warp_instrumented_ms"] = {
        key: float(np.median([s[key] for s in samples[1:]])) for key in samples[0]
    }
    report["note"] = (
        "initialize/expand include inverse events; these overlapping totals must not be added"
    )
    inverse.apply = original_apply
    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        v0 = cp.asarray(initial)
        events = Events()
        record = [False]
        measured_apply = events.wrap("inverse", original_apply)

        def mv(v):
            apply = measured_apply if record[0] else original_apply
            apply(wp.from_dlpack(v.reshape(1, -1)), inverse.x)
            return cp.asarray(inverse.x).reshape(-1).copy()

        op = csl.LinearOperator((solver.n, solver.n), matvec=mv, dtype=np.float64)
        for instrument in [False, True]:
            samples = []
            inverse_times = []
            record[0] = instrument
            for _ in range(6):
                trial_start = v0.copy()
                events.pairs.clear()
                stream.synchronize()
                start = time.perf_counter()
                values, vectors = csl.eigsh(op, k=26, ncv=64, which="LA", tol=1e-12, v0=trial_start)
                stream.synchronize()
                samples.append(1000 * (time.perf_counter() - start))
                if instrument:
                    inverse_times.append(events.totals()["inverse"])
            if instrument:
                report["cupy_instrumented"] = {
                    "wall_ms": float(np.median(samples[1:])),
                    "inverse_ms": float(np.median(inverse_times[1:])),
                    "inverse_applications": len(events.pairs),
                }
            else:
                report["cupy_solve_only_ms"] = float(np.median(samples[1:]))
        check = EigenpairEvaluation(SparseOperator(a), wp.from_dlpack(vectors.T), tol=1e-7)
        check.run()
        report["cupy_residual"] = float(check.residuals.numpy().max())
    Path("results/dragon_breakdown.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    inverse.close()


if __name__ == "__main__":
    main()
