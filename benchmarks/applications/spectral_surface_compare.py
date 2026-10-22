"""Dominant reconstruction pencil: ARPACK, Warp/cuDSS and pure Warp/FSAI-CG."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import warp as wp
from warp.optim import linear

from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.upstream.adapters import matrix
from warpack import CGInverse, CuDSSInverse, GeneralizedKrylovSchur, SparseOperator

from .fem_hierarchy import matrix_hash
from .fem_shapes import elapsed
from .problems import Problem
from .runner import accuracy
from .spectral_surface import reconstruction_quality, surface_pencil


def problem(pencil, modes=1, copies=1):
    """Independent disconnected copies give exact dominant repeated clusters."""
    if copies < 1 or modes < 1 or modes % copies:
        raise ValueError("request complete clusters: modes must be a positive multiple of copies")
    a = sp.block_diag([pencil.a] * copies, format="csr")
    b = sp.block_diag([pencil.b] * copies, format="csr")
    per_copy = modes // copies
    if per_copy + 1 >= pencil.a.shape[0]:
        raise ValueError("reference needs a positive spectral guard")
    if pencil.a.shape[0] <= 1024:
        values, vectors = la.eigh(pencil.a.toarray(), pencil.b.toarray())
        values, vectors = values[::-1], vectors[:, ::-1]
        label = "Dense generalized LAPACK on one independent copy"
    else:
        values, vectors = sla.eigsh(
            pencil.a,
            M=pencil.b,
            k=per_copy + 4,
            which="LA",
            tol=1e-13,
            ncv=min(pencil.a.shape[0], max(64, 4 * (per_copy + 4))),
            v0=np.random.default_rng(136).normal(size=pencil.a.shape[0]),
        )
        order = np.argsort(-values)
        values, vectors = values[order], vectors[:, order]
        label = "Independent ARPACK on one copy, four guard pairs, ncv=max(64,4*(k/copies+4)), tol=1e-13"
    if values[per_copy - 1] - values[per_copy] < 1e-7 * max(1.0, values[0]):
        raise ValueError("requested boundary splits a near cluster; increase modes")
    expected = np.tile(values[:per_copy], copies)
    basis = la.block_diag(*[vectors[:, :per_copy]] * copies)
    order = np.argsort(-expected, kind="stable")
    p = Problem(
        "spectral_surface",
        "A",
        a,
        b,
        modes,
        0.0,
        mode="regular",
        expected=expected[order],
        requested_basis=basis[:, order],
        notes=label,
    )
    return p


class SurfaceSolver:
    """Public generalized solver composition; metric B is regularization, not mass."""

    def __init__(self, p, backend, ncv=32, tol=1e-11, inner_tol=1e-13):
        if backend not in ("cudss", "cg"):
            raise ValueError("backend must be cudss or cg")
        self.a, self.b = matrix(p.a), matrix(p.b)
        self.backend, self.factor = backend, None
        if backend == "cudss":
            self.inverse = CuDSSInverse(self.b, 1, mtype="spd")
        else:
            self.factor = linear.FSAI(
                self.b, max_row_size=32, max_step_size=3, factor_dtype=wp.float32
            )
            self.inverse = CGInverse(
                self.b,
                preconditioner=self.factor,
                tol=inner_tol,
                atol=0.0,
                maxiter=2000,
                adaptive_capture=True,
                verify_residuals=True,
            )
        self.start = wp.array(np.random.default_rng(136).normal(size=p.n), dtype=wp.float64)
        self.solver = GeneralizedKrylovSchur(
            SparseOperator(self.a),
            SparseOperator(self.b),
            self.inverse,
            p.k,
            ncv=min(ncv, p.n),
            tol=tol,
            mode="regular",
            which="LA",
            initial_vector=self.start,
        )
        self.graph = None

    def capture(self, steps=4):
        self.graph = self.solver.capture(steps)
        return self.graph

    def solve(self):
        if self.backend == "cg":
            self.inverse.reset_stats()
        wp.capture_launch(self.graph)

    def output(self):
        return self.solver.eigenvalues.numpy(), self.solver.eigenvectors.numpy().T

    def close(self):
        if self.backend == "cudss":
            self.inverse.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=12)
    parser.add_argument("--modes", type=int, default=1)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--ncv", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--backend", choices=("arpack", "cudss", "cg"), required=True)
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--anisotropy", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1 or args.ncv <= args.modes:
        parser.error("positive repeats/steps and ncv>modes required")
    start = time.perf_counter()
    pencil = surface_pencil(
        args.side, alpha=args.alpha, gamma=args.gamma, anisotropy=args.anisotropy
    )
    assembly_seconds = time.perf_counter() - start
    start = time.perf_counter()
    p = problem(pencil, args.modes, args.copies)
    reference_seconds = time.perf_counter() - start
    driver = None
    capture_seconds, memory = 0.0, None
    if args.backend != "arpack":
        device = wp.get_device("cuda:0")
        before = device.free_memory
        start = time.perf_counter()
        driver = SurfaceSolver(p, args.backend, ncv=args.ncv)
        wp.synchronize()
        setup_seconds = time.perf_counter() - start
        capture_seconds = elapsed(lambda: driver.capture(args.steps))
        memory = before - device.free_memory
    else:
        start = time.perf_counter()
        lu = sla.splu(p.b.tocsc())
        inverse = sla.LinearOperator(p.b.shape, matvec=lu.solve)
        initial = np.random.default_rng(136).normal(size=p.n)
        setup_seconds = time.perf_counter() - start
    rows = []
    for trial in range(args.repeats + 1):
        if driver:
            seconds = elapsed(driver.solve)
            values, vectors = driver.output()
        else:
            start = time.perf_counter()
            values, vectors = sla.eigsh(
                p.a,
                M=p.b,
                Minv=inverse,
                k=p.k,
                ncv=min(args.ncv, p.n),
                which="LA",
                tol=1e-11,
                v0=initial.copy(),
            )
            seconds = time.perf_counter() - start
        q = accuracy(p, values, vectors)
        if driver:
            q.update(
                status=int(driver.solver.status.numpy()[0]),
                converged=int(driver.solver.converged.numpy()[0]),
                iterations=int(driver.solver.iterations.numpy()[0]),
            )
            q["accepted"] &= q["status"] == 0 and q["converged"] == p.k
        # Geometry of individual eigenvectors is only meaningful for the simple dominant mode.
        geometry = (
            reconstruction_quality(pencil, vectors[:, np.argmax(values)]) if p.k == 1 else None
        )
        row = dict(seconds=seconds, quality=q, geometry=geometry, values=values.tolist())
        if driver and args.backend == "cg":
            row.update(
                cg_stats=driver.inverse.stats.numpy().tolist(),
                fresh_cg_stats=driver.inverse.fresh_stats.numpy().tolist(),
                fresh_residual_maxima=driver.inverse.residual_maxima.numpy().tolist(),
            )
        print(trial, row, flush=True)
        if trial:
            rows.append(row)
        else:
            warmup = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    graph = None
    if driver:
        dot = args.out.with_suffix(".dot")
        wp.capture_debug_dot_print(driver.graph, str(dot))
        graph = graph_summary(dot)
        dot.unlink()
    sources = [
        Path(__file__),
        Path("benchmarks/applications/spectral_surface.py"),
        Path("benchmarks/applications/runner.py"),
        *sorted(Path("warpack").glob("*.py")),
    ]
    versions = {}
    for name in ("numpy", "scipy", "nvidia-cudss-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = dict(
        arguments=vars(args) | dict(out=str(args.out)),
        n=p.n,
        nnz_a=p.a.nnz,
        nnz_b=p.b.nnz,
        assembly_seconds=assembly_seconds,
        reference_and_copy_assembly_seconds=reference_seconds,
        reference=p.notes,
        setup_seconds=setup_seconds,
        capture_seconds=capture_seconds,
        setup_capture_device_free_memory_decrease_bytes=memory,
        memory_scope="Device/context free-memory decrease, not peak or live allocations.",
        warmup=warmup,
        rows=rows,
        graph=graph,
        matrix_sha256=matrix_hash(p),
        source_sha256={str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        dependency=warp_dependencies(),
        versions=versions,
        environment=dict(
            platform=platform.platform(),
            gpu=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        ),
        graph_device_only=(
            "HOST" not in graph["node_types_including_conditional_body"]
            and not any(k.startswith(("HtoD", "DtoH")) for k in graph["memcpy_kinds"])
        )
        if graph is not None
        else None,
        accepted=all(r["quality"]["accepted"] for r in [warmup] + rows),
        scope="Independent sampled P1 reconstruction; PSD A, SPD nonmass regularization B, largest algebraic generalized modes. Setup/factorization and capture separate from solve; host assembly/reference excluded. Geometry quality is separate from eigensolver acceptance. cuDSS graph metadata copies remain visible (#29).",
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if driver:
        driver.close()
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
