"""Accuracy-gated comparison of projected Jacobi launch-fusion candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.applications.compare import timed
from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.jacobi_on_the_fly import OnTheFlyJacobi
from benchmarks.source_provenance import source_manifest
from benchmarks.upstream.adapters import matrix
from warpack import SparseOperator, SymmetricEigensolver


def quality(solver, original, spectrum):
    vectors, values = solver.v.numpy(), solver.ritz.numpy()
    scale = max(abs(spectrum))
    result = dict(
        residual=float(max(np.linalg.norm(original @ vectors - vectors * values, axis=0)) / scale),
        spectrum_error=float(max(abs(np.sort(values) - np.sort(spectrum))) / scale),
        orthogonality=float(np.max(abs(vectors.T @ vectors - np.eye(len(values))))),
        sweeps=None
        if solver._global_jacobi is None
        else int(solver._global_jacobi.sweeps.numpy()[0]),
    )
    result["accepted"] = all(
        np.isfinite(result[k]) and result[k] < 1e-11
        for k in ("residual", "spectrum_error", "orthogonality")
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="32,48,64,72,104,128,208,312")
    parser.add_argument("--variants", default="default,scalar,tile16,tile32")
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sizes, variants = list(map(int, args.sizes.split(","))), args.variants.split(",")
    if min(args.batch, args.repeats) < 1 or any(n < 4 or n % 2 for n in sizes):
        parser.error("positive batch/repeats and even sizes>=4 required")
    if not set(variants) <= {"default", "scalar", "tile16", "tile32"}:
        parser.error("unknown variant")
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        rng = np.random.default_rng(831)
        q = np.linalg.qr(rng.normal(size=(n, n)))[0]
        spectrum = np.geomspace(400, 0.0025, n)
        spectrum[:3], spectrum[-2:] = 400, 0
        original = (q * spectrum) @ q.T
        for variant in variants:
            solver = SymmetricEigensolver(
                SparseOperator(matrix(original)), min(20, n), ncv=n, which="LA"
            )
            if variant != "default":
                solver._global_jacobi = OnTheFlyJacobi(
                    solver.g,
                    solver.v,
                    solver.cs,
                    solver.ss,
                    solver.partners,
                    solver.jtmp,
                    solver.vtmp,
                    variant=variant,
                )
            source = wp.array(original, dtype=wp.float64)
            wp.copy(solver.g, source)
            solver.diagonalize()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                wp.copy(solver.g, source)
                solver.diagonalize()
            wp.capture_launch(capture.graph)
            warmup = quality(solver, original, spectrum)
            samples, checks = [], []
            for _ in range(args.repeats):
                seconds, _ = timed(
                    lambda: [wp.capture_launch(capture.graph) for _ in range(args.batch)]
                )
                samples.append(seconds / args.batch)
                checks.append(quality(solver, original, spectrum))
            dot = args.out.with_suffix(f".{n}-{variant}.dot")
            wp.capture_debug_dot_print(capture.graph, str(dot))
            graph = graph_summary(dot)
            dot.unlink()
            assert not graph["node_types_including_conditional_body"].get("HOST", 0)
            assert all("DtoD" in kind for kind in graph["memcpy_kinds"])
            rows.append(
                dict(
                    n=n,
                    variant=variant,
                    seconds=samples,
                    median_seconds=float(np.median(samples)),
                    warmup=warmup,
                    checks=checks,
                    graph=graph,
                    accepted=warmup["accepted"] and all(c["accepted"] for c in checks),
                )
            )
            print(n, variant, rows[-1]["median_seconds"], rows[-1]["accepted"], flush=True)
    report = dict(
        arguments=vars(args) | dict(out=str(args.out)),
        rows=rows,
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(),
        accepted=all(r["accepted"] for r in rows),
        protocol="Isolated complete projected eigensolves; input restored before every replay. All original equations, complete spectra and orthogonality checked after warmup and every measured batch at 1e-11. Setup/capture/reference excluded; no end-to-end speedup claim. Uncontended external occupancy is required for timing qualification.",
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("Projected accuracy failure; evidence retained")


if __name__ == "__main__":
    main()
