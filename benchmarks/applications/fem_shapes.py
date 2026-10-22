"""Procedural shape changes with fixed connectivity and changed K and M.

Assembly/references are host benchmark code. Eigensolves, FSAI construction/refit
and CG use upstream Warp and Warpack; no reference eigenvectors seed a solve.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp
from warp.optim import linear

from benchmarks.audit_profile import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import BatchedCGPreconditioner, BatchedSparseOperator, BlockLOBPCG

from .fem_cases import beam_mesh, elastic_matrices, reference
from .problems import Problem
from .runner import accuracy


def shape_family(side=2, modes=6, consistent=True, parameters=(0.0, 0.12, -0.12, 0.0)):
    """Stretch and twist a clamped 3:1:1 tet beam, keeping vertex/DOF identities."""
    base, tets = beam_mesh(side)
    dofs = np.flatnonzero(np.repeat(base[:, 0] > 0, 3))
    states = []
    for parameter in parameters:
        if not np.isfinite(parameter) or abs(parameter) >= 0.5:
            raise ValueError("shape parameter must be finite with abs(parameter)<0.5")
        vertices = base.copy()
        vertices[:, 1:] -= 0.5
        vertices *= np.array([1 + 0.1 * parameter, 1 + parameter, 1 - 0.6 * parameter])
        angle = 0.2 * parameter * base[:, 0] / 3
        y, z = vertices[:, 1].copy(), vertices[:, 2].copy()
        vertices[:, 1] = np.cos(angle) * y - np.sin(angle) * z
        vertices[:, 2] = np.sin(angle) * y + np.cos(angle) * z
        # Reject inverted elements relative to the original orientation.
        x, old = vertices[tets], base[tets]
        assert np.all(
            np.linalg.det(x[:, 1:] - x[:, :1]) * np.linalg.det(old[:, 1:] - old[:, :1]) > 0
        )
        a, b = elastic_matrices(vertices, tets, consistent=consistent)
        a, b = a[dofs, :][:, dofs], b[dofs, :][:, dofs]
        values, vectors, label = reference(a, b, modes + 8)
        if modes >= len(values) or values[modes] - values[modes - 1] < 1e-7 * max(1, values[modes]):
            raise ValueError("requested boundary splits a cluster; increase modes")
        p = Problem(
            f"shape_{parameter:+.3f}",
            "A",
            a,
            b,
            modes,
            -0.01,
            expected=values[:modes],
            requested_basis=vectors[:, :modes],
            notes=label + "; fixed connectivity, updated geometry and mass",
            block_size=3,
        )
        states.append((parameter, p, float(values[modes])))
    return states


def union_pattern(matrices):
    pattern = sum(abs(a) for a in matrices).tocsr()
    pattern.sort_indices()
    pattern.data.fill(1.0)
    return pattern


class ShapeReplay:
    """Benchmark-only fixed-address driver for existing public APIs."""

    def __init__(
        self,
        states,
        width,
        *,
        perturbation=0.0,
        refit=True,
        steps=200,
        start_rows=None,
        tol=1e-11,
    ):
        self.states, self.width = states, width
        p = states[0][1]
        self.k, self.n, self.refit = p.k, p.n, refit
        pattern = union_pattern([a for _, p, _ in states for a in (p.a, p.b)])
        mass_pattern = union_pattern([p.b for _, p, _ in states])
        self.rows, self.cols = pattern.tocoo().row, pattern.tocoo().col
        self.mrows, self.mcols = mass_pattern.tocoo().row, mass_pattern.tocoo().col
        self.a, self.shifted, self.mass = matrix(pattern), matrix(pattern), matrix(mass_pattern)
        self.update(p, factor=False)
        self.factor = linear.FSAI(
            self.shifted,
            max_row_size=32,
            max_step_size=3,
            apply_lanes=8,
            factor_dtype=wp.float32,
            reuse_pattern=refit,
        )
        self.pre = BatchedCGPreconditioner(
            self.shifted,
            width,
            tol=0.01,
            maxiter=25,
            preconditioner=self.factor,
            working_dtype=wp.float32,
            fsai_layout="columns",
            matrix_layout="columns",
        )
        start_rows = p.k if start_rows is None else start_rows
        self.random = wp.array(
            np.random.default_rng(128).normal(size=(start_rows, p.n)), dtype=wp.float64
        )
        self.starts = wp.clone(self.random)
        self.solver = BlockLOBPCG(
            BatchedSparseOperator(self.a),
            p.k,
            width,
            mass=BatchedSparseOperator(self.mass),
            preconditioner=self.pre,
            initial_vectors=self.starts,
            initial_perturbation=perturbation,
            tol=tol,
            reuse_operator_images=True,
            search_bucket_size=4,
            keep_history=True,
            preconditioner_bucket_size=4,
            symmetric_products=True,
            fused_updates=True,
            paired_rotation=True,
        )
        wp.synchronize()
        start = time.perf_counter()
        self.graph = self.solver.capture(steps) if steps is not None else None
        self.capture_seconds = time.perf_counter() - start

    def update(self, p, *, factor=True):
        self.a.values[: len(self.rows)].assign(np.asarray(p.a[self.rows, self.cols]).ravel())
        self.mass.values[: len(self.mrows)].assign(np.asarray(p.b[self.mrows, self.mcols]).ravel())
        self.shifted.values[: len(self.rows)].assign(
            np.asarray((p.a + 0.01 * p.b)[self.rows, self.cols]).ravel()
        )
        if factor and self.refit:
            self.factor.update()

    def solve(self, warm=False):
        if self.graph is None:
            raise ValueError("capture a graph before solve")
        if warm and self.starts.shape != self.solver.eigenvectors.shape:
            raise ValueError("warm solve requires one start row per requested mode")
        wp.copy(self.starts, self.solver.eigenvectors if warm else self.random)
        wp.capture_launch(self.graph)

    def quality(self, p):
        q = accuracy(p, self.solver.eigenvalues.numpy(), self.solver.eigenvectors.numpy().T)
        q.update(
            status=int(self.solver.status.numpy()[0]),
            converged=int(self.solver.converged.numpy()[0]),
            iterations=int(self.solver.iterations.numpy()[0]),
        )
        q["accepted"] &= q["status"] == 0 and q["converged"] == p.k
        return q


def elapsed(fn):
    wp.synchronize()
    start = time.perf_counter()
    fn()
    wp.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--mass", choices=("consistent", "lumped"), default="consistent")
    parser.add_argument("--strategy", choices=("cold", "previous", "perturbed"), required=True)
    parser.add_argument("--factor", choices=("refit", "retain"), default="refit")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.side, args.modes, args.repeats) < 1 or args.width < args.modes:
        parser.error("positive sizes/repeats and width>=modes required")
    begin = time.perf_counter()
    states = shape_family(args.side, args.modes, args.mass == "consistent")
    assembly_reference_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    driver = ShapeReplay(
        states,
        args.width,
        perturbation=1e-3 if args.strategy == "perturbed" else 0.0,
        refit=args.factor == "refit",
    )
    wp.synchronize()
    setup_capture_seconds = time.perf_counter() - begin
    rows = []
    for trial in range(args.repeats + 1):
        sequence = []
        for i, (parameter, p, guard) in enumerate(states):
            update_seconds = elapsed(lambda: driver.update(p))
            solve_seconds = elapsed(lambda: driver.solve(warm=i > 0 and args.strategy != "cold"))
            q = driver.quality(p)
            row = dict(
                parameter=parameter,
                update_seconds=update_seconds,
                solve_seconds=solve_seconds,
                guard=guard,
                quality=q,
                cg_stats=driver.pre.stats.numpy().tolist(),
            )
            sequence.append(row)
            print(trial, parameter, q, flush=True)
        if trial:
            rows.append(sequence)
        else:
            warmup = sequence
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dot = args.out.with_suffix(".dot")
    wp.capture_debug_dot_print(driver.graph, str(dot))
    audit = graph_summary(dot)
    dot.unlink()
    assert "HOST" not in audit["node_types_including_conditional_body"]
    assert not any(kind.startswith(("HtoD", "DtoH")) for kind in audit["memcpy_kinds"])
    matrix_hashes = []
    for _, p, _ in states:
        digest = hashlib.sha256()
        for a in (p.a, p.b):
            a.sort_indices()
            for data in (a.indptr, a.indices, a.data):
                digest.update(data.tobytes())
        matrix_hashes.append(digest.hexdigest())
    report = dict(
        arguments=vars(args) | {"out": str(args.out)},
        n=states[0][1].n,
        assembly_reference_seconds=assembly_reference_seconds,
        setup_capture_seconds=setup_capture_seconds,
        capture_seconds=driver.capture_seconds,
        warmup=warmup,
        rows=rows,
        graph=audit,
        matrix_sha256=matrix_hashes,
        warp=wp.__version__,
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        accepted=all(r["quality"]["accepted"] for seq in [warmup] + rows for r in seq),
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
