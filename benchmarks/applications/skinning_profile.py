"""Shared-factor skinning diagnosis; event instrumentation is separate from timing."""

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse.linalg as cl
import numpy as np
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import ConstrainedKrylovSchur, CuDSSInverse, SparseOperator, saddle_point_matrix

from .compare import other_gpu_processes
from .skinning_cases import make_skinning_case, reference, validate


class Events:
    def __init__(self):
        self.pairs = []

    def wrap(self, name, function):
        def call(*args, **kwargs):
            # CUDA conditional bodies do not support external event records.
            # Recovery orthogonalization is included in initialize/step totals.
            if name == "_orthogonalize_column" and not args[1]:
                return function(*args, **kwargs)
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            external = wp.get_stream().is_capturing
            wp.record_event(start, external=external)
            result = function(*args, **kwargs)
            wp.record_event(end, external=external)
            self.pairs.append((name, start, end))
            return result

        return call

    def totals(self):
        totals = defaultdict(float)
        for name, start, end in self.pairs:
            totals[name] += wp.get_event_elapsed_time(start, end, False)
        return dict(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=24)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--ncv", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    case = make_skinning_case(args.side)
    digest = hashlib.sha256()
    for a in (case.h, case.m, case.c):
        for data in (a.indptr, a.indices, a.data):
            digest.update(data.tobytes())
    fingerprint = digest.hexdigest()
    cache = Path("build/skinning-audit") / (
        f"reference-{fingerprint[:16]}-shift{case.shift}-k{args.modes}.npz"
    )
    if cache.exists():
        saved = np.load(cache)
        expected, basis = saved["values"], saved["vectors"]
    else:
        expected, basis, _ = reference(case, args.modes)
    n, r = case.n, case.c.shape[0]
    scale = np.sqrt(case.m.diagonal())
    start = np.random.default_rng(2026).normal(size=n)
    h, m, c = matrix(case.h), matrix(case.m), matrix(case.c)
    inverse = CuDSSInverse(
        saddle_point_matrix(h, m, c, case.shift),
        1,
        mtype="general",
        matching=True,
        refinement_steps=1,
        cooperative_refinement=True,
    )
    solver = ConstrainedKrylovSchur(
        SparseOperator(h),
        SparseOperator(m),
        c,
        inverse,
        args.modes,
        ncv=args.ncv,
        target=case.shift,
        tol=1e-9,
        initial_vector=wp.array(start / scale, dtype=wp.float64),
    )
    report = dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=hashlib.sha256(Path("warpack/constrained.py").read_bytes()).hexdigest(),
        gpu=wp.get_device().name,
        versions={"warp": wp.__version__, "cupy": cp.__version__},
        other_gpu_processes=other_gpu_processes(),
        matrix_hash=fingerprint,
        n=n,
        r=r,
        k=args.modes,
        ncv=args.ncv,
        steps=args.steps,
        shift=case.shift,
        protocol="One shared cuDSS factorization; same physical start. Warm wall times exclude setup/capture and independent validation. Event timings are separate, overlapping diagnostics, not additive. Warp includes physical residuals and multipliers; CuPy checks outside timing.",
    )

    def check_warp():
        result = validate(
            case,
            solver.eigenvalues.numpy(),
            solver.eigenvectors.numpy().T,
            expected,
            basis,
            solver.multipliers.numpy().T,
        )
        result["strict_count"] = int(solver.converged.numpy()[0])
        assert result["accepted"] and result["strict_count"] == args.modes, result
        return result

    graph = solver.capture(args.steps)
    samples, checks = [], []
    for i in range(args.repeats + 1):
        wp.synchronize()
        begin = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize()
        elapsed = 1000 * (time.perf_counter() - begin)
        check = check_warp()
        if i:
            samples.append(elapsed)
            checks.append(check)
    report["warp"] = dict(samples_ms=samples, median_ms=float(np.median(samples)), checks=checks)
    events = Events()
    original_apply = inverse.apply
    inverse.apply = events.wrap("inverse", original_apply)
    originals = {}
    for name in [
        "initialize",
        "step",
        "_orthogonalize_column",
        "diagonalize",
        "rotate",
        "finalize",
    ]:
        originals[name] = getattr(solver, name)
        setattr(solver, name, events.wrap(name, originals[name]))
    with wp.ScopedCapture() as captured:
        solver.solve(args.steps)
    measurements = []
    for _ in range(args.repeats):
        wp.capture_launch(captured.graph)
        wp.synchronize()
        measurements.append(events.totals())
        check_warp()
    report["warp"]["event_ms"] = {
        key: float(np.median([sample[key] for sample in measurements])) for key in measurements[0]
    }
    report["warp"]["calls"] = dict(Counter(p[0] for p in events.pairs))
    inverse.apply = original_apply
    for name, function in originals.items():
        setattr(solver, name, function)

    stream = cp.cuda.Stream.from_external(wp.get_stream())
    with stream:
        scaling, initial = cp.asarray(scale), cp.asarray(start)
        rhs = wp.zeros((1, n + r), dtype=wp.float64)
        output = wp.empty_like(rhs)
        rhs_cp, output_cp = cp.asarray(rhs), cp.asarray(output)
        calls = [0]

        def product(x):
            calls[0] += 1
            rhs_cp.fill(0)
            rhs_cp[0, :n] = scaling * x
            inverse.apply(rhs, output)
            return scaling * output_cp[0, :n]

        operator = cl.LinearOperator((n, n), matvec=product, dtype=np.float64)
        for instrument in (False, True):
            samples, checks, counts, measurements = [], [], [], []
            for i in range(args.repeats + 1):
                events = Events()
                inverse.apply = (
                    events.wrap("inverse", original_apply) if instrument else original_apply
                )
                calls[0] = 0
                cp.random.seed(2026)
                stream.synchronize()
                begin = time.perf_counter()
                theta, vectors = cl.eigsh(
                    operator,
                    k=args.modes,
                    ncv=args.ncv,
                    which="LM",
                    tol=1e-11,
                    maxiter=10000,
                    v0=initial.copy(),
                )
                stream.synchronize()
                elapsed = 1000 * (time.perf_counter() - begin)
                result = validate(
                    case,
                    (case.shift + 1 / theta).get(),
                    (vectors / scaling[:, None]).get(),
                    expected,
                    basis,
                )
                assert result["accepted"], result
                if i:
                    samples.append(elapsed)
                    checks.append(result)
                    counts.append(calls[0])
                    if instrument:
                        measurements.append(events.totals()["inverse"])
            if instrument:
                report["cupy"]["instrumented_wall_ms"] = float(np.median(samples))
                report["cupy"]["inverse_event_ms"] = float(np.median(measurements))
            else:
                report["cupy"] = dict(
                    samples_ms=samples,
                    median_ms=float(np.median(samples)),
                    inverse_calls=counts,
                    checks=checks,
                )
    inverse.apply = original_apply
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["warp", "cupy"]}, indent=2))
    solver.graph = graph = captured.graph = None
    inverse.close()


if __name__ == "__main__":
    main()
