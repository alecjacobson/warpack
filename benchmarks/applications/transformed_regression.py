"""Accuracy-gated block/Krylov and native-reference comparisons for transformed pencils."""

import argparse
import gc
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import (
    BucklingEigensolver,
    CayleyEigensolver,
    CuDSSInverse,
    GeneralizedEigensolver,
    GeneralizedKrylovSchur,
    SparseOperator,
)

from .compare import other_gpu_processes, timed
from .problems import Problem, make_problem
from .references import spectra_setup
from .runner import accuracy, scipy_setup


def fixtures():
    folder = Path("tests/fixtures")
    cases = []
    for key, file, mode in [
        ("SymGEigsShift_5", "spectra_generalized_shift_5.npz", "shift-invert"),
        ("SymGEigsShift_30", "spectra_buckling_30.npz", "buckling"),
        ("SymGEigsShift_45", "spectra_generalized_matched.npz", "cayley"),
    ]:
        data = np.load(folder / file)
        a, b = (data[key + "_a"], data[key + "_b"]) if mode == "cayley" else (data["a"], data["b"])
        sigma = 1.2345
        values = 1 / la.eigvalsh(b, a) if mode == "buckling" else la.eigvalsh(a, b)
        transformed = (
            values / (values - sigma)
            if mode == "buckling"
            else (values + sigma) / (values - sigma)
            if mode == "cayley"
            else 1 / (values - sigma)
        )
        wanted = values[np.argsort(-abs(transformed))[:10]]
        cases.append(
            (
                Problem(
                    key,
                    "G" if mode == "buckling" else "B",
                    sp.csr_matrix(a),
                    sp.csr_matrix(b),
                    10,
                    sigma,
                    mode,
                    wanted,
                ),
                20,
            )
        )
    cases.append((make_problem("G", 10000), 32))
    return cases


def run(p, ncv, method, repeats, cycles=None):
    inv = solver = graph = None

    def cleanup():
        return None

    row = dict(problem=p.name, n=p.n, k=p.k, ncv=ncv, mode=p.mode, target=p.target, method=method)
    try:
        if method in ("scipy", "spectra"):
            if method == "scipy":
                setup, solve = timed(lambda: scipy_setup(p, ncv=ncv))
            else:
                setup, (solve, cleanup) = timed(lambda: spectra_setup(p, ncv))

            def get(value):
                return value
        else:
            steps = cycles if cycles is not None else (100 if method == "warp-block" else 20)
            row.update(outer_cycles=steps, cudss_matching=True, cudss_refinement_steps=1)

            def setup_warp():
                inverse = CuDSSInverse(
                    matrix(p.a - p.target * p.b),
                    ncv if method == "warp-block" else 1,
                    mtype="general",
                    matching=True,
                    refinement_steps=1,
                )
                a, b = SparseOperator(matrix(p.a)), SparseOperator(matrix(p.b))
                if p.mode == "buckling":
                    s = BucklingEigensolver(
                        a,
                        b,
                        inverse,
                        p.k,
                        p.target,
                        ncv=ncv,
                        tol=1e-12,
                        method="block" if method == "warp-block" else "krylov",
                    )
                elif method == "warp-block":
                    s = (
                        CayleyEigensolver(a, b, inverse, p.k, p.target, ncv=ncv, tol=1e-12)
                        if p.mode == "cayley"
                        else GeneralizedEigensolver(
                            a, b, inverse, p.k, ncv=ncv, target=p.target, tol=1e-12
                        )
                    )
                else:
                    s = GeneralizedKrylovSchur(
                        a, b, inverse, p.k, ncv=ncv, target=p.target, mode=p.mode, tol=1e-12
                    )
                return inverse, s

            setup, (inv, solver) = timed(setup_warp)
            row["capture_seconds"], graph = timed(lambda: solver.capture(steps))

            def solve():
                return wp.capture_launch(graph)

            def get(_):
                return solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T, None

            audit = Path("build/transformed-graphs")
            audit.mkdir(parents=True, exist_ok=True)
            dot_path = audit / f"{p.name}-{method}.dot"
            wp.capture_debug_dot_print(graph, str(dot_path))
            dot = dot_path.read_text()
            row["graph_audit"] = dict(
                host_nodes=len(re.findall(r'label="\{\s*HOST\b', dot)),
                memcpy_kinds=dict(Counter(re.findall(r"\| \{kind \| ([^}]+)", dot))),
                note="DOT inspection, not a CUDA API trace; vendor HtoD copies remain visible",
            )
        row["setup_seconds"] = setup
        row["warmup_seconds"], _ = timed(solve)
        samples, checks, competing = [], [], set()
        for _ in range(repeats):
            competing.update(other_gpu_processes())
            seconds, output = timed(solve)
            competing.update(other_gpu_processes())
            check = accuracy(p, *get(output), reference=p.expected)
            check["accepted"] &= check["max_absolute_residual"] < 1e-9
            if solver is not None:
                check["reported_converged"] = int(solver.converged.numpy()[0])
            samples.append(seconds)
            checks.append(check)
        row.update(
            samples=samples,
            median=float(np.median(samples)),
            checks=checks,
            accepted=all(c["accepted"] for c in checks),
            other_gpu_processes=sorted(competing),
            timing_uncontended=not competing,
        )
    except Exception as e:
        row.update(accepted=False, error=repr(e))
    finally:
        if solver is not None:
            if hasattr(solver, "core"):
                solver.core.graph = None
            solver.graph = None
        graph = solver = None
        gc.collect()
        if inv is not None:
            inv.close()
        cleanup()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/transformed_comparisons.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--krylov-cycles", type=int, default=20)
    parser.add_argument("--problems", help="Comma-separated fixture names; default all")
    parser.add_argument("--methods", default="scipy,spectra,warp-block,warp-krylov")
    args = parser.parse_args()
    problems = fixtures()
    selected = set(args.problems.split(",")) if args.problems else {p.name for p, _ in problems}
    if selected - {p.name for p, _ in problems}:
        parser.error("Unknown problem name")
    methods = args.methods.split(",")
    if set(methods) - {"scipy", "spectra", "warp-block", "warp-krylov"}:
        parser.error("Unknown method")
    if args.krylov_cycles < 1 or args.repeats < 1:
        parser.error("Cycle and repeat counts must be positive")
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        protocol="Sequential backends; setup/capture separate; every timed sample validated against requested spectrum, metric orthogonality and original residual, including absolute 1e-9 bound. Warp block uses 100 cycles; Krylov cycles are recorded per row; both use matched general cuDSS factors and one correction. These are explicit algorithm/budget changes, not identical-iteration timing comparisons. Reference starts differ; native tolerance 1e-11, Warp 1e-12. Internal counts reported separately. CuPy generalized transformed APIs are unsupported by the comparison adapter.",
        cases=[],
    )
    for p, ncv in problems:
        if p.name not in selected:
            continue
        for method in methods:
            row = run(
                p,
                ncv,
                method,
                args.repeats,
                args.krylov_cycles if method == "warp-krylov" else None,
            )
            report["cases"].append(row)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
            print(p.name, method, row["accepted"], row.get("median"), row.get("error"), flush=True)


if __name__ == "__main__":
    main()
