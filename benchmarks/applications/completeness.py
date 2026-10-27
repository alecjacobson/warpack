"""Optional inertia certificate overhead on procedural Newton FEM modes.

Generate references first with benchmarks.applications.newton_ffrf (newton backend).
No upstream mesh/data are used. Historical Newton counts are independently
certified by its pinned CPU implementation; this worker does not retime it.
"""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.applications.fem_cases import rigid_modes
from benchmarks.applications.newton_ffrf import (
    NODE_MASS,
    SPACING,
    SPRING,
    check_pairs,
    gpu_solver,
    lattice,
    versions,
)
from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import LowestModeCertificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=("free", "fixed"), required=True)
    parser.add_argument("--backend", choices=("cudss", "warp"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("positive repeats required")
    stiffness, mass, points, _ = lattice(boundary=args.boundary)
    stiffness.sort_indices()  # Canonical storage, as in the reference report.
    digest = hashlib.sha256(
        stiffness.data.tobytes() + stiffness.indices.tobytes() + stiffness.indptr.tobytes()
    ).hexdigest()
    with np.load(args.reference) as cached:
        assert str(cached["matrix_sha256"]) == digest
        reference = cached["values"], cached["vectors"]
    cpu = json.loads(args.cpu_report.read_text())
    assert cpu["matrix_sha256"] == digest and cpu["accepted"]
    assert cpu["rows"][0]["quality"]["completeness_certified"]
    upper = 0.5 * (reference[0][-1] + cpu["rows"][0]["guard_eigenvalue"])
    rigid = rigid_modes(points, mass) if args.boundary == "free" else None
    sigma = -1e-6 * float(abs(stiffness).sum(axis=0).max()) / NODE_MASS
    if args.boundary == "fixed":
        sigma = 0.0
    solver, graph, values, vectors, inverse, _ = gpu_solver(
        stiffness, points, args.boundary, args.backend, 12, sigma
    )
    try:
        wp.capture_launch(graph)
        wp.synchronize()
        quality = check_pairs(stiffness, mass, values.numpy(), vectors.numpy().T, reference, rigid)
        assert quality["accepted"], quality
        # Scale K and lambda together, retaining the original mass and vectors.
        # In particular rigid residuals need a meaningful absolute scale.
        scale = SPRING / SPACING
        a, m = matrix(stiffness / scale), matrix(mass)
        scaled_values = wp.array(values.numpy() / scale, dtype=wp.float64)
        q = None if rigid is None else wp.array(rigid.T.copy(), dtype=wp.float64)
        rows = []
        for trial in range(args.repeats + 1):
            wp.synchronize()
            start = time.perf_counter()
            certificate = LowestModeCertificate(
                a,
                vectors,
                scaled_values,
                mass=m,
                nullspace=q,
                upper_bound=upper / scale,
                residual_tolerance=1e-8,
            )
            wp.synchronize()
            setup_seconds = time.perf_counter() - start
            certificate.run()
            wp.synchronize()
            with wp.ScopedCapture() as cap:
                certificate.run()
            wp.capture_launch(cap.graph)
            wp.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                wp.capture_launch(cap.graph)
            wp.synchronize()
            checks_seconds = (time.perf_counter() - start) / 100
            start = time.perf_counter()
            certificate.refresh_count()
            wp.synchronize()
            refresh_seconds = time.perf_counter() - start
            assert certificate.certified.numpy()[0] == 0
            wp.capture_launch(cap.graph)
            wp.synchronize()
            row = dict(
                setup_seconds=setup_seconds,
                refactor_and_count_seconds=refresh_seconds,
                device_checks_seconds=checks_seconds,
                inertia=asdict(certificate.inertia),
                certified=int(certificate.certified.numpy()[0]),
                flags=int(certificate.flags.numpy()[0]),
                residual=float(certificate.residuals.numpy().max()),
                orthogonality_error=float(certificate.orthogonality_error.numpy()[0]),
                gap_margin=float(certificate.gap_margin.numpy()[0]),
            )
            assert row["certified"] == 1 and row["flags"] == 0, row
            if trial:
                rows.append(row)
            else:
                warmup = row
        dot = args.out.with_suffix(".dot")
        wp.capture_debug_dot_print(cap.graph, str(dot))
        audit = graph_summary(dot)
        dot.unlink()
        report = dict(
            boundary=args.boundary,
            backend=args.backend,
            n=stiffness.shape[0],
            modes=12,
            nullity=0 if rigid is None else rigid.shape[1],
            upper_bound_physical=upper,
            stiffness_scale=scale,
            matrix_sha256=digest,
            quality=quality,
            warmup=warmup,
            rows=rows,
            graph=audit,
            versions=versions(),
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            reference_report=str(args.cpu_report),
            accepted=True,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(args.out, "accepted", flush=True)
    finally:
        if inverse is not None:
            inverse.close()


if __name__ == "__main__":
    main()
