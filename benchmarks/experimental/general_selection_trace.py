"""Untimed per-cycle requested-set and projected-solve diagnostics.

Extra device kernels observe the existing restart without changing its policy.
Floating-point trajectories can differ from an uninstrumented run. Traces are
validation evidence, never performance measurements.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.check_general_references import read
from benchmarks.upstream.general_reference import GeneralReference
from benchmarks.upstream.run_benchmarks import StartedGeneral
from warpack.arnoldi import complex_gram, complex_residual, rotate_complex
from warpack.hermitian import cmul, norm2
from warpack.schur import cabs

ROOT = Path(__file__).resolve().parents[2]


@wp.kernel
def projected_residual(
    original: wp.array2d[wp.vec2d],
    vectors: wp.array2d[wp.vec2d],
    values: wp.array[wp.vec2d],
    errors: wp.array[wp.float64],
):
    j = wp.tid()
    radius = wp.float64(0.0)
    error = wp.float64(0.0)
    for r in range(original.shape[0]):
        value = wp.vec2d(0.0)
        for c in range(original.shape[1]):
            value += cmul(original[r, c], vectors[c, j])
            radius += norm2(original[r, c])
        error += norm2(value - cmul(values[j], vectors[r, j]))
    errors[j] = wp.sqrt(error / wp.max(radius, wp.float64(1e-300)))


@wp.kernel
def record(
    values: wp.array[wp.vec2d],
    order: wp.array[wp.int32],
    norms: wp.array2d[wp.float64],
    projected: wp.array[wp.float64],
    gram: wp.array2d[wp.vec2d],
    position: wp.array[wp.int32],
    status: wp.array[wp.int32],
    trace_values: wp.array2d[wp.vec2d],
    trace_residuals: wp.array2d[wp.float64],
    trace_projected: wp.array2d[wp.float64],
    trace_orthogonality: wp.array[wp.float64],
    trace_status: wp.array[wp.int32],
):
    j = wp.tid()
    slot = position[0]
    if slot < trace_values.shape[0]:
        val = values[order[j]]
        trace_values[slot, j] = val
        trace_residuals[slot, j] = wp.sqrt(norms[j, 0]) / wp.max(wp.float64(1.0), cabs(val))
        trace_projected[slot, j] = projected[order[j]]
        if j == 0:
            error = wp.float64(0.0)
            for r in range(gram.shape[0]):
                for c in range(gram.shape[1]):
                    value = gram[r, c]
                    if r == c:
                        value -= wp.vec2d(1.0, 0.0)
                    error += norm2(value)
            trace_orthogonality[slot] = wp.sqrt(error)
            trace_status[slot] = status[0]


@wp.kernel
def advance(position: wp.array[wp.int32]):
    position[0] += 1


class TracedGeneral(StartedGeneral):
    def __init__(self, *args, trace_budget, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_values = wp.zeros((trace_budget, self.m), dtype=wp.vec2d, device=self.device)
        self.trace_residuals = wp.zeros(
            (trace_budget, self.m), dtype=wp.float64, device=self.device
        )
        self.trace_projected = wp.zeros_like(self.trace_residuals)
        self.trace_orthogonality = wp.zeros(trace_budget, dtype=wp.float64, device=self.device)
        self.trace_status = wp.zeros(trace_budget, dtype=wp.int32, device=self.device)
        self.position = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.all_vectors, self.all_images = wp.empty_like(self.q), wp.empty_like(self.q)
        self.all_norms = wp.empty((self.m, 1), dtype=wp.float64, device=self.device)
        self.original_projection, self.basis_gram = wp.empty_like(self.h), wp.empty_like(self.h)
        self.projected_errors = wp.empty(self.m, dtype=wp.float64, device=self.device)

    def initialize(self, seed=42):
        self.position.zero_()
        super().initialize(seed)

    def diagonalize(self):
        wp.copy(self.original_projection, self.h)
        super().diagonalize()

    def select(self):
        super().select()
        wp.launch(
            rotate_complex,
            self.q.shape,
            [self.q, self.projected, self.order, self.all_vectors],
            device=self.device,
        )
        wp.launch(
            rotate_complex,
            self.q.shape,
            [self.z, self.projected, self.order, self.all_images],
            device=self.device,
        )
        self.all_norms.zero_()
        wp.launch_tiled(
            complex_residual,
            dim=(self.m, (self.n + 255) // 256),
            inputs=[self.all_vectors, self.all_images, self.ritz, self.order, self.all_norms],
            block_dim=128,
            device=self.device,
        )
        self.basis_gram.zero_()
        wp.launch_tiled(
            complex_gram,
            dim=(self.m, self.m, (self.n + 255) // 256),
            inputs=[self.q, self.q, self.basis_gram],
            block_dim=128,
            device=self.device,
        )
        wp.launch(
            projected_residual,
            self.m,
            [self.original_projection, self.projected, self.ritz, self.projected_errors],
            device=self.device,
        )
        wp.launch(
            record,
            self.m,
            [
                self.ritz,
                self.order,
                self.all_norms,
                self.projected_errors,
                self.basis_gram,
                self.position,
                self.status,
                self.trace_values,
                self.trace_residuals,
                self.trace_projected,
                self.trace_orthogonality,
                self.trace_status,
            ],
            device=self.device,
        )
        wp.launch(advance, 1, [self.position], device=self.device)


def selected_set(reference, values):
    if len(values) != reference.k:
        return False
    rows, columns = linear_sum_assignment(abs(values[:, None] - reference.values))
    return bool(
        np.max(abs(values[rows] - reference.values[columns])) < reference.margin
        and reference.required.issubset(reference.allowed[columns])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "build/spectra-benchmark-fixtures")
    parser.add_argument("--case", default="benchmark_gen_30")
    parser.add_argument("--ncv", type=int, default=60)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backend", choices=("serial", "tiled"), default="tiled")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in (args.fixtures / "benchmark_gen.jsonl").read_text().splitlines()
    ]
    case = next(case for case in cases if case["id"] == args.case)
    matrix_path, start_path = (args.fixtures / (args.case + suffix) for suffix in (".a", ".v0"))
    a, start = read(matrix_path), read(start_path).ravel()
    reference = GeneralReference(a, case["k"])
    solver = TracedGeneral(
        complex_matrix(a), case["k"], ncv=args.ncv, tol=1e-10, trace_budget=args.budget
    )
    solver.start_vector = wp.array(np.column_stack([start, np.zeros(len(a))]), dtype=wp.vec2d)
    if args.backend == "serial":
        solver.projected_solver = None
    graph = solver.capture(args.budget, adaptive=False)
    report = dict(
        protocol=__doc__,
        case=case,
        ncv=args.ncv,
        keep=solver.keep,
        budget=args.budget,
        tol=solver.tol,
        backend=args.backend,
        actual_tiled=solver.projected_solver is not None,
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/experimental/general_selection_trace.py",
                "warpack/arnoldi.py",
                "warpack/schur.py",
                "warpack/schur_tiled.py",
                "benchmarks/upstream/run_benchmarks.py",
            )
        },
        matrix_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        start_sha256=hashlib.sha256(start_path.read_bytes()).hexdigest(),
        warp=wp.__version__,
        gpu=wp.get_device().name,
        trials=[],
    )
    snapshots = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for replay in range(args.repeats):
        wp.capture_launch(graph)
        values, residuals = output(solver.trace_values), solver.trace_residuals.numpy()
        projected = solver.trace_projected.numpy()
        orthogonality, status = solver.trace_orthogonality.numpy(), solver.trace_status.numpy()
        snapshots.update(
            {
                f"values_{replay}": values,
                f"residuals_{replay}": residuals,
                f"projected_{replay}": projected,
                f"orthogonality_{replay}": orthogonality,
                f"status_{replay}": status,
            }
        )
        cycles = []
        for i in range(args.budget):
            requested_count = int(np.count_nonzero(residuals[i, : solver.k] <= solver.tol))
            accepted_set = selected_set(reference, values[i, : solver.k])
            converged_indices = np.flatnonzero(residuals[i] <= solver.tol)
            candidate_set = selected_set(reference, values[i, converged_indices[: solver.k]])
            cycles.append(
                dict(
                    cycle=i + 1,
                    requested_count=requested_count,
                    selected_spectrum=accepted_set,
                    requested_pass=bool(
                        requested_count == solver.k and accepted_set and status[i] == 0
                    ),
                    complete_set_among_converged_ritz=candidate_set,
                    max_requested_residual=float(residuals[i, : solver.k].max()),
                    max_projected_backward_error=float(projected[i].max()),
                    basis_orthogonality=float(orthogonality[i]),
                    status=int(status[i]),
                )
            )
        passing = [row["cycle"] for row in cycles if row["requested_pass"]]
        lost = [
            row["cycle"]
            for row in cycles
            if passing and row["cycle"] > passing[0] and not row["requested_pass"]
        ]
        final = reference.check(output(solver.eigenvalues), output(solver.eigenvectors).T)
        final.update(
            converged=int(solver.converged.numpy()[0]), status=int(solver.status.numpy()[0])
        )
        final["accepted"] &= final["converged"] == solver.k and final["status"] == 0
        trial = dict(
            replay=replay,
            first_passing_cycle=min(passing) if passing else None,
            last_passing_cycle=max(passing) if passing else None,
            failed_cycles_after_first_pass=lost,
            final=final,
            cycles=cycles,
        )
        report["trials"].append(trial)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(args.out.with_suffix(".npz"), **snapshots)
        print(
            replay,
            "first pass",
            trial["first_passing_cycle"],
            "later failures",
            lost,
            "final",
            final["accepted"],
            flush=True,
        )


if __name__ == "__main__":
    main()
