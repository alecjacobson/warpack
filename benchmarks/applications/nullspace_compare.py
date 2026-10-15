"""Captured nullspace evidence and its incremental cost on Hodge workloads."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import warp as wp

from warpack import NumericalNullspaceDiagnostics, SparseOperator

from ..upstream.adapters import matrix
from .compare import compare_baseline, other_gpu_processes, timed
from .hodge import make_hodge_case
from .runner import accuracy, warp_setup


def host_evidence(p, vectors, lower_bound):
    av, mv = p.a @ vectors, p.b @ vectors
    metric_norms = np.sum(vectors * mv, axis=0)
    values = np.sum(vectors * av, axis=0) / metric_norms
    absolute = np.linalg.norm(av - mv * values, axis=0)
    residuals = absolute / np.maximum(
        1, np.linalg.norm(av, axis=0) + abs(values) * np.linalg.norm(mv, axis=0)
    )
    bounds = absolute / np.sqrt(lower_bound * metric_norms)
    orthogonality = np.max(abs(vectors.T @ mv - np.eye(vectors.shape[1])))
    return values, residuals, bounds, orthogonality


def audit(graph, path):
    wp.capture_debug_dot_print(graph, str(path))
    text = path.read_text()
    return dict(
        host_nodes=len(re.findall(r'label="\{\s*HOST\b', text)),
        memcpy_kinds=dict(Counter(re.findall(r"\| \{kind \| ([^}]+)", text))),
    )


def run(name, side, weak_scale, repeats):
    c = make_hodge_case(name, side=side, weak_scale=weak_scale)
    p = c.problem
    steps = 20 if name == "large_disconnected" else 100
    lower = 0.9  # Valid for all fixtures: T=I or I+0.05*shift, M=T.T T.
    setup, (solver, inverse) = timed(lambda: warp_setup(p))
    diagnostic_setup, d = timed(
        lambda: NumericalNullspaceDiagnostics(
            SparseOperator(matrix(p.a)),
            solver.eigenvectors,
            SparseOperator(matrix(p.b)),
            mass_lower_bound=lower,
            expected_nullity=p.nullity,
        )
    )
    capture_baseline, baseline = timed(lambda: solver.capture(steps))
    capture_combined, combined = timed(lambda: solver.capture(steps, finalizer=d.run))
    with wp.ScopedCapture() as cap:
        d.run()
    diag_graph = cap.graph
    folder = Path("build/nullspace-diagnostics")
    folder.mkdir(parents=True, exist_ok=True)
    label = f"{name}-{side}-{weak_scale:g}"
    digest = hashlib.sha256()
    for mat in (p.a, p.b):
        csr = mat.tocsr()
        for data in (csr.indptr, csr.indices, csr.data):
            digest.update(data.tobytes())
    row = dict(
        name=name,
        side=side,
        target=[float(p.target), 0.0],
        matrix_sha256=digest.hexdigest(),
        n=p.n,
        nnz=p.a.nnz,
        mass_nnz=p.b.nnz,
        k=p.k,
        ncv=solver.m,
        known_topological_nullity=p.nullity,
        weak_scale=weak_scale,
        reference=c.reference,
        mass_lower_bound=lower,
        outer_cycles=steps,
        setup_seconds=setup,
        diagnostic_setup_seconds=diagnostic_setup,
        capture_baseline_seconds=capture_baseline,
        capture_combined_seconds=capture_combined,
        graph_audits={
            key: audit(graph, folder / f"{label}-{key}.dot")
            for key, graph in [
                ("baseline", baseline),
                ("combined", combined),
                ("diagnostic", diag_graph),
            ]
        },
        trials=[],
    )
    competing = set()
    try:
        for graph in (baseline, combined, diag_graph):
            wp.capture_launch(graph)
        wp.synchronize()
        for _ in range(repeats):
            competing.update(other_gpu_processes())
            baseline_time, _ = timed(lambda: wp.capture_launch(baseline))
            baseline_finite = bool(
                np.isfinite(solver.eigenvalues.numpy()).all()
                and np.isfinite(solver.eigenvectors.numpy()).all()
            )
            baseline_status = int(solver.status.numpy()[0])
            combined_time, _ = timed(lambda: wp.capture_launch(combined))
            diag_time, _ = timed(lambda: wp.capture_launch(diag_graph))
            competing.update(other_gpu_processes())
            w, v = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            finite = bool(np.isfinite(w).all() and np.isfinite(v).all())
            physical = (
                accuracy(p, w, v)
                if finite
                else dict(accepted=False, reason="Non-finite eigenvalues or vectors")
            )
            host_time = agreement = None
            if finite:
                start = time.perf_counter()
                reference = host_evidence(p, v, lower)
                host_time = time.perf_counter() - start
                agreement = max(
                    float(np.max(abs(a - b)))
                    for a, b in zip(
                        reference,
                        [
                            d.values.numpy(),
                            d.residuals.numpy(),
                            d.error_bounds.numpy(),
                            d.orthogonality_error.numpy(),
                        ],
                    )
                )
            state = int(d.flags.numpy()[0])
            count, ambiguous, weak = map(int, d.counts.numpy())
            guard = float(d.positive_guard.numpy()[0])
            intentionally_unresolved = name == "weak_positive" and weak_scale < 1e-8
            expected = bool(state & d.COMPLETENESS_UNVERIFIED)
            if finite:
                expected &= agreement < 1e-11
            else:
                expected &= bool(state & d.INVALID_DATA) and count == 0 and not np.isfinite(guard)
            if intentionally_unresolved:
                expected &= bool(state & d.GAP_UNRESOLVED)
            else:
                expected &= (
                    state == d.COMPLETENESS_UNVERIFIED
                    and count == p.nullity
                    and guard > 1e-7
                    and physical["accepted"]
                )
            row["trials"].append(
                dict(
                    baseline_seconds=baseline_time,
                    baseline_finite=baseline_finite,
                    baseline_status=baseline_status,
                    combined_finite=finite,
                    combined_status=int(solver.status.numpy()[0]),
                    combined_seconds=combined_time,
                    diagnostic_seconds=diag_time,
                    host_reference_seconds=host_time,
                    reference_max_absolute_difference=agreement,
                    flags=state,
                    numerical_count=count,
                    ambiguous_count=ambiguous,
                    weak_positive_count=weak,
                    positive_guard=guard if np.isfinite(guard) else None,
                    suggested_capacity=int(d.suggested_capacity.numpy()[0]),
                    physical_validation=physical,
                    reported_converged=int(solver.converged.numpy()[0]),
                    diagnostic_contract_pass=bool(expected),
                )
            )
        for key in ["baseline", "combined", "diagnostic", "host_reference"]:
            samples = [
                r[f"{key}_seconds"] for r in row["trials"] if r[f"{key}_seconds"] is not None
            ]
            row[f"{key}_seconds"] = float(np.median(samples)) if samples else None
        row["combined_over_baseline"] = row["combined_seconds"] / row["baseline_seconds"]
        row["diagnostic_contract_pass"] = all(r["diagnostic_contract_pass"] for r in row["trials"])
        row["solver_accepted"] = all(r["physical_validation"]["accepted"] for r in row["trials"])
        row["other_gpu_processes"] = sorted(competing)
        row["timing_uncontended"] = not competing
        row["completeness_certified"] = False
    finally:
        cap.graph = diag_graph = combined = baseline = solver.graph = None
        gc.collect()
        inverse.close()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases", default="sphere,disk,annulus,torus,disconnected,weak_positive,large_disconnected"
    )
    parser.add_argument("--side", type=int, default=4)
    parser.add_argument("--large-side", type=int, default=41)
    parser.add_argument("--weak-scale", type=float, default=1e-4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="results/nullspace_diagnostics.json")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            n: importlib.metadata.version(n) for n in ["numpy", "scipy", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        protocol="Sequential uncontended synchronized median replays, three trials by default. Independent host residual/metric evidence outside GPU timing. Host reference timing covers Rayleigh values, physical residuals, metric error bounds and Gram independence with vectors already resident on CPU; discrete classification is outside that timing. Solver+diagnostic uses capture(finalizer=diagnostic.run); no intermediate host numerical readback. Known topology checks agreement only; no result certifies complete nullity. Weak cases can deliberately fail the solver counting gate while diagnostics correctly report unresolved evidence. DOT audits include nested bodies, not a host synchronization API trace.",
        cases=[],
    )
    for name in args.cases.split(","):
        row = run(
            name,
            args.large_side if name == "large_disconnected" else args.side,
            args.weak_scale,
            args.repeats,
        )
        report["cases"].append(row)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(
            name,
            row["diagnostic_contract_pass"],
            row["solver_accepted"],
            row["combined_over_baseline"],
            flush=True,
        )
    if args.baseline:
        old = json.loads(Path(args.baseline).read_text())
        if (old["gpu"], old["warp"]) != (report["gpu"], report["warp"]):
            raise ValueError("Baseline GPU/runtime mismatch")
        measured = []
        for row in report["cases"]:
            audit_data = row["graph_audits"]["baseline"]
            measured.append(
                dict(
                    api_class="D",
                    requested_size=row["n"],
                    n=row["n"],
                    method="warp",
                    matrix_sha256=row["matrix_sha256"],
                    target=row["target"],
                    k=row["k"],
                    ncv=row["ncv"],
                    outer_cycles=row["outer_cycles"],
                    accepted=row["solver_accepted"],
                    timing_uncontended=row["timing_uncontended"],
                    solve_seconds=row["baseline_seconds"],
                    device_only_graph=audit_data["host_nodes"] == 0
                    and not any(
                        "HOST" in kind or "Host" in kind for kind in audit_data["memcpy_kinds"]
                    ),
                )
            )
        checks = compare_baseline(measured, old, 1.25)
        report["solver_regression_checks"] = checks
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if not any(c["comparable"] for c in checks) or any(
            c.get("accuracy_regression")
            or c.get("graph_regression")
            or c.get("performance_regression")
            for c in checks
        ):
            raise SystemExit("Saved solver baseline regression or no comparable result")
    if not all(r["diagnostic_contract_pass"] for r in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
