"""Same-budget regression for conditional metric QR recovery (GH #46)."""

import argparse
import gc
import importlib.metadata
import json
import os
import subprocess
import types
from pathlib import Path

import numpy as np
import warp as wp

from .compare import other_gpu_processes, timed
from .hodge import make_hodge_case, weak_component_reference
from .nullspace_compare import audit
from .runner import accuracy, warp_setup
from .weak_block_probe import legacy_orthogonalize, shifted_orthogonalize


def run(name, scale, variant, repeats):
    case = make_hodge_case(name, side=41 if name == "large_disconnected" else 4, weak_scale=scale)
    p = case.problem
    expected, space = (
        weak_component_reference(case, scale) if name == "weak_positive" else (None, None)
    )
    solver, inverse = warp_setup(p)
    if variant == "legacy":
        solver.orthogonalize = types.MethodType(legacy_orthogonalize, solver)
    elif variant == "always_shifted":
        solver.relative_shift, solver.shifted_passes = 1e-12, 1
        solver.orthogonalize = types.MethodType(shifted_orthogonalize, solver)
    steps = 20 if name == "large_disconnected" else 100
    capture_seconds, graph = timed(lambda: solver.capture(steps))
    folder = Path("build/weak-block")
    folder.mkdir(parents=True, exist_ok=True)
    row = dict(
        name=name,
        weak_scale=scale,
        variant=variant,
        n=p.n,
        k=p.k,
        ncv=solver.m,
        target=p.target,
        outer_cycles=steps,
        capture_seconds=capture_seconds,
        graph_audit=audit(graph, folder / f"{name}-{scale:g}-{variant}.dot"),
        trials=[],
    )
    competing = set()
    try:
        wp.capture_launch(graph)
        wp.synchronize()
        for _ in range(repeats):
            competing.update(other_gpu_processes())
            seconds, _ = timed(lambda: wp.capture_launch(graph))
            competing.update(other_gpu_processes())
            values, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            finite = bool(np.isfinite(values).all() and np.isfinite(vectors).all())
            trial = dict(
                seconds=seconds,
                finite=finite,
                status=int(solver.status.numpy()[0]),
                count=int(solver.converged.numpy()[0]),
                recoveries=int(solver.orthogonalization_recoveries.numpy()[0]),
            )
            if finite and space is not None:
                remainder = vectors - space @ (space.T @ (p.b @ vectors))
                zero_error = float(abs(np.sort(values)[: p.nullity]).max())
                positive_error = float(
                    abs((np.sort(values)[p.nullity :] - expected[p.nullity :]) / scale).max()
                )
                residual = float(
                    np.linalg.norm(p.a @ vectors - (p.b @ vectors) * values, axis=0).max()
                )
                orthogonality = float(abs(vectors.T @ (p.b @ vectors) - np.eye(p.k)).max())
                subspace_error = float(np.linalg.norm(remainder))
                trial.update(
                    zero_eigenvalue_absolute_error=zero_error,
                    positive_eigenvalue_error_over_weak_scale=positive_error,
                    physical_residual=residual,
                    metric_orthogonality=orthogonality,
                    requested_subspace_error=subspace_error,
                )
                accepted = (
                    zero_error < 1e-14
                    and positive_error < 1e-10
                    and residual < 1e-10
                    and orthogonality < 1e-10
                    and subspace_error < 1e-9
                )
            elif finite:
                trial["physical_validation"] = accuracy(p, values, vectors)
                accepted = trial["physical_validation"]["accepted"]
            else:
                accepted = False
            trial["accepted"] = bool(accepted and trial["status"] == 0 and trial["count"] == p.k)
            row["trials"].append(trial)
        row["solve_seconds"] = float(np.median([r["seconds"] for r in row["trials"]]))
        row["accepted"] = all(r["accepted"] for r in row["trials"])
        row["other_gpu_processes"] = sorted(competing)
        row["timing_uncontended"] = not competing
    finally:
        graph = solver.graph = None
        gc.collect()
        inverse.close()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="results/weak_block_coverage.json")
    args = parser.parse_args()
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            n: importlib.metadata.version(n) for n in ["numpy", "scipy", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        protocol="Sequential synchronized warm replays; setup, capture, independent CPU reference and validation excluded. Same matrices, starts, cuDSS inverse, steps and ncv. Every timed trial validated. Legacy overrides only orthogonalize with the pre-fix two-pass method; always_shifted uses shifted QR followed by QR2. Weak references solve the rescaled annulus independently and include incidence-derived nullspace directions. Zero errors are absolute; positive errors are normalized by the weak component scale. Timings of rejected solves are not wins. DOT inspection includes conditional bodies but is not a host API trace.",
        cases=[],
    )
    for name, scale in [
        ("weak_positive", 1e-4),
        ("weak_positive", 1e-10),
        ("weak_positive", 1e-12),
        ("large_disconnected", 1e-4),
    ]:
        for variant in ["legacy", "conditional", "always_shifted"]:
            row = run(name, scale, variant, args.repeats)
            report["cases"].append(row)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
            print(name, scale, variant, row["accepted"], row["solve_seconds"], flush=True)
    if any(
        not r["accepted"] or not r["timing_uncontended"]
        for r in report["cases"]
        if r["variant"] != "legacy"
    ):
        raise SystemExit("Recovery validation failed or GPU contention detected")


if __name__ == "__main__":
    main()
