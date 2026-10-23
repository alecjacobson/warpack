"""Accuracy-gated replay samples for Warp residual correction of cuDSS solves."""

import argparse
import gc
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.acceptance import errors_below_tolerance
from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse

from .compare import other_gpu_processes, timed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/refinement_inverse.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    fixture = Path("tests/fixtures/cupy_sm.npz")
    a = np.load(fixture)["a"]
    shift = -np.sqrt(np.finfo(float).eps) * abs(a).max()
    a -= shift * np.eye(len(a))
    report = dict(
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        condition_2=float(np.linalg.cond(a)),
        gpu=wp.get_device().name,
        warp=wp.__version__,
        protocol="Sequential warm synchronized solves; every trial checked; no speedup against rejected accuracy. Correction and residual products use Warp, existing factor solves use cuDSS.",
        cases=[],
    )
    for scale in (1.0, 1e-6, 1e6):
        for mtype in ("symmetric", "general"):
            for steps in (0, 1):
                for width in (1, 3):
                    host = scale * a
                    b = np.random.default_rng(42).normal(size=(width, len(a)))
                    rhs = wp.array(b, dtype=wp.float64)
                    out = wp.empty_like(rhs)
                    mat = matrix(host)
                    setup, inv = timed(
                        lambda: CuDSSInverse(mat, width, mtype=mtype, refinement_steps=steps)
                    )
                    inv.apply(rhs, out)
                    wp.synchronize()
                    with wp.ScopedCapture() as cap:
                        inv.apply(rhs, out)
                    graph = cap.graph
                    warmup, _ = timed(lambda: wp.capture_launch(graph))
                    samples, errors, processes = [], [], set()
                    for _ in range(args.repeats):
                        processes.update(other_gpu_processes())
                        elapsed, _ = timed(lambda: wp.capture_launch(graph))
                        processes.update(other_gpu_processes())
                        residual = np.linalg.norm(out.numpy() @ host.T - b) / np.linalg.norm(b)
                        samples.append(elapsed)
                        errors.append(float(residual))
                    path = Path(f"build/physical-ritz/inverse-{scale}-{mtype}-{steps}-{width}.dot")
                    path.parent.mkdir(exist_ok=True, parents=True)
                    wp.capture_debug_dot_print(graph, str(path))
                    dot = path.read_text()
                    report["cases"].append(
                        dict(
                            scale=scale,
                            mtype=mtype,
                            refinement_steps=steps,
                            width=width,
                            setup_seconds=setup,
                            warmup_seconds=warmup,
                            samples_seconds=samples,
                            solve_seconds=float(np.median(samples)),
                            relative_residuals=errors,
                            accepted=errors_below_tolerance(errors, 1e-12),
                            other_gpu_processes=sorted(processes),
                            timing_uncontended=not processes,
                            host_nodes=len(re.findall(r'label="\{\s*HOST\b', dot)),
                            memcpy_kinds=dict(Counter(re.findall(r"\| \{kind \| ([^}]+)", dot))),
                        )
                    )
                    cap.graph = graph = None
                    gc.collect()
                    inv.close()
                    print(scale, mtype, steps, width, errors, flush=True)
    dest = Path(args.out)
    dest.parent.mkdir(exist_ok=True, parents=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
