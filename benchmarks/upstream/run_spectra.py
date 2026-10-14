"""Run exported upstream Spectra numerical problems with public Warpack APIs."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp

from warpack import (
    BucklingEigensolver,
    CayleyEigensolver,
    CuDSSInverse,
    EigenpairEvaluation,
    GeneralEigensolver,
    GeneralizedEigensolver,
    HermitianEigensolver,
    KrylovSchur,
    PartialSVD,
    ShiftInvertEigensolver,
    SparseOperator,
)

from .adapters import complex_matrix, execute, matrix, output
from .compare_reports import contracts

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "build/spectra-fixtures"
RULES = {0: "LM", 1: "LR", 2: "LI", 3: "LA", 4: "SM", 5: "SR", 6: "SI", 7: "SA", 8: "BE"}


class Unsupported(Exception):
    pass


def read(path):
    with open(path, "rb") as f:
        shape = np.fromfile(f, dtype=np.int64, count=2)
        a = np.fromfile(f, dtype=np.complex128).reshape(shape)
    return a.real.copy() if not np.any(a.imag) else a


def run(case):
    name = case["family"]
    k = case["k"]
    ncv = case["ncv"] or None
    sel = case["selection"]
    which = RULES[sel]
    if name == "Example4":
        which = "LA"  # Upstream Example4 explicitly uses LargestAlge.
    sigma = complex(case["sigma_real"], case["sigma_imag"])
    a = read(FIX / (case["id"] + ".a"))
    n = a.shape[0]
    bp = FIX / (case["id"] + ".b")
    b = read(bp) if bp.exists() else None
    vp = FIX / (case["id"] + ".v0")
    v0 = wp.array(read(vp).ravel(), dtype=wp.float64) if vp.exists() else None
    inv = None
    evaluation = None
    iterations = 300
    options = {}
    threshold = 1e-12
    margin = (
        1e-8
        if name in ("GenEigsRealShift", "GenEigsComplexShift", "Example2", "Example4")
        else 1e-9
    )
    if name == "DavidsonSymEigs":
        margin = 1e-10
    started = time.perf_counter()
    try:
        if name in ("GenEigs", "ComplexEigs"):
            if sel in (2, 6):
                raise Unsupported(
                    "Spectra selects imaginary-part magnitude; Warpack LI/SI select signed imaginary part"
                )
            if which == "SM":
                solver = ShiftInvertEigensolver(
                    complex_matrix(a), k, 0.0, ncv=ncv, backend="cudss", tol=threshold
                )
                inv = solver
                iterations = 30
            else:
                solver = GeneralEigensolver(
                    complex_matrix(a), k, ncv=ncv, which=which, tol=threshold
                )
        elif name == "GenEigsComplexShift":
            raise Unsupported(
                "Spectra real-part complex-shift transform differs from Warpack complex resolvent"
            )
        elif name == "GenEigsRealShift":
            if which != "LM":
                raise Unsupported(
                    "ShiftInvertEigensolver exposes nearest-eigenvalue selection only"
                )
            solver = ShiftInvertEigensolver(
                complex_matrix(a), k, sigma, ncv=ncv, backend="cudss", tol=threshold
            )
            inv = solver
            # Fixed outer captures cannot nest cuDSS; use an explicitly recorded budget.
            iterations = 30
        elif name == "SVD":
            solver = PartialSVD(matrix(a), k, ncv=ncv, tol=threshold)
        elif b is not None:
            ma, mb = matrix(a), matrix(b)
            if name == "SymGEigsShift":
                if which != "LM":
                    raise Unsupported(
                        "Generalized transforms expose largest transformed magnitude only"
                    )
                inv = CuDSSInverse(
                    matrix(a - sigma.real * b),
                    ncv,
                    mtype="general",
                    matching=True,
                    refinement_steps=1,
                )
                if "buckling" in case["name"]:
                    solver = BucklingEigensolver(
                        SparseOperator(ma),
                        SparseOperator(mb),
                        inv,
                        k,
                        sigma.real,
                        ncv=ncv,
                        tol=threshold,
                    )
                elif "cayley" in case["name"]:
                    solver = CayleyEigensolver(
                        SparseOperator(ma),
                        SparseOperator(mb),
                        inv,
                        k,
                        sigma.real,
                        ncv=ncv,
                        tol=threshold,
                    )
                else:
                    solver = GeneralizedEigensolver(
                        SparseOperator(ma),
                        SparseOperator(mb),
                        inv,
                        k,
                        ncv=ncv,
                        target=sigma.real,
                        tol=threshold,
                    )
            else:
                if which != "SM":
                    raise Unsupported(
                        "GeneralizedEigensolver lacks general B-inverse extremal selection for indefinite A"
                    )
                inv = CuDSSInverse(ma, ncv, mtype="general", matching=True, refinement_steps=1)
                solver = GeneralizedEigensolver(
                    SparseOperator(ma), SparseOperator(mb), inv, k, ncv=ncv, target=0, tol=threshold
                )
            iterations = 100
        elif name in ("SymEigsShift", "Example1Shift") or which == "SM":
            if np.iscomplexobj(a):
                solver = ShiftInvertEigensolver(
                    complex_matrix(a), k, sigma, ncv=ncv, backend="cudss", tol=threshold
                )
                inv = solver
                iterations = 30
            else:
                ma = matrix(a)
                # Keep standard shifted profiles unchanged by this SM fix.
                # Their tight inverse-count sensitivity is tracked separately.
                factor_options = (
                    dict(mtype="symmetric")
                    if name in ("SymEigsShift", "Example1Shift")
                    else dict(mtype="general", matching=True, refinement_steps=1)
                )
                inv = CuDSSInverse(matrix(a - sigma.real * np.eye(n)), 1, **factor_options)
                solver = KrylovSchur(
                    inv,
                    k,
                    ncv=ncv,
                    which="LM" if name not in ("SymEigsShift", "Example1Shift") else which,
                    tol=threshold,
                    initial_vector=v0,
                )
                evaluation = EigenpairEvaluation(
                    SparseOperator(ma), solver.eigenvectors, tol=threshold
                )
                options = {"finalizer": evaluation.run}
                iterations = 100
        elif name == "HermEigs":
            solver = HermitianEigensolver(
                complex_matrix(a, True), k, ncv=2 * ncv, which=which, tol=threshold
            )
        else:
            solver = KrylovSchur(
                SparseOperator(matrix(a)), k, ncv=ncv, which=which, tol=threshold, initial_vector=v0
            )
        setup = time.perf_counter() - started
        timing = execute(solver, iterations, **options)
        if name == "SVD":
            w = output(solver.singular_values)
            u = output(solver.u).T
            v = output(solver.v).T
            ref_u, ref_w, ref_vt = la.svd(a, full_matrices=False)
            errors = dict(
                values=float(np.max(np.abs(w - ref_w[:k]))),
                left_vectors=float(np.max(np.abs(np.abs(u) - np.abs(ref_u[:, :k])))),
                right_vectors=float(np.max(np.abs(np.abs(v) - np.abs(ref_vt[:k].T)))),
            )
            err = max(errors.values())
            errors["reported_converged"] = int(solver.core.converged.numpy()[0])
        else:
            w = output(solver.eigenvalues if evaluation is None else evaluation.values)
            v = output(solver.eigenvectors).T
            residual = a @ v - (v if b is None else b @ v) * w
            err = float(np.max(np.abs(residual)))
            errors = {"max_absolute_residual": err, "returned_pairs": len(w)}
            # Upstream generic tests require residuals and successful convergence.
            # Examples 1/2/4 additionally check the requested eigenvalues.
            if name in ("Example1", "Example1Shift", "Example2", "Example4"):
                ref = la.eigvalsh(a)
                ref = ref[:k] if name == "Example1Shift" else ref[-k:]
                errors["spectrum_error"] = float(np.max(np.abs(np.sort(w.real) - ref)))
                err = max(err, errors["spectrum_error"])
            conv = getattr(
                solver, "converged", getattr(getattr(solver, "real", None), "converged", None)
            )
            if conv is not None:
                count = int(conv.numpy()[0])
                if isinstance(solver, HermitianEigensolver):
                    # The realification solves for 2*k real pairs. Its count
                    # must not be compared directly with k complex pairs.
                    errors["reported_converged_real"] = count
                    errors["extracted_complex_pairs"] = int(solver.count.numpy()[0])
                    count = min(count // 2, errors["extracted_complex_pairs"])
                errors["reported_converged"] = count
        contract = contracts(
            dict(
                **case,
                **errors,
                outcome="failed",  # Evaluated; final status is derived below.
                upstream_absolute_tolerance=margin,
            )
        )
        numerical_pass = contract["checked_numerical_bounds_pass"]
        ok = contract["strict_contract_pass"]
        # Keep the strict historical outcome, but do not conflate original
        # assertions with the solver's stricter/transformed stopping test.
        count = errors.get("reported_converged")
        convergence_space = "original_problem"
        physical_count = count
        if evaluation is not None:
            convergence_space = "inverse_operator"
            physical_count = int(evaluation.converged.numpy()[0])
        elif isinstance(solver, PartialSVD):
            convergence_space = "augmented_operator"
            physical_count = min(
                int(solver.count_left.numpy()[0]), int(solver.count_right.numpy()[0])
            )
        elif isinstance(solver, HermitianEigensolver):
            convergence_space = "realification_and_extraction"
            physical_count = None  # No independent complex residual counter in this API.
        factor = getattr(inv, "inverse", inv)
        factor = getattr(factor, "inverse", factor)
        return dict(
            cudss_mtype=getattr(factor, "mtype", None),
            cudss_reordering=getattr(factor, "reordering", None),
            cudss_matching=getattr(factor, "matching", None),
            cudss_refinement_steps=getattr(factor, "refinement_steps", None),
            outcome="passed" if ok else "failed",
            setup_seconds=setup,
            **timing,
            iteration_budget=iterations,
            upstream_absolute_tolerance=margin,
            checked_numerical_bounds_pass=numerical_pass,
            requested_spectrum_checked=contract["requested_spectrum_checked"],
            internal_tolerance=threshold,
            reported_convergence_space=convergence_space,
            reported_count_pass=None if count is None else count >= k,
            original_evaluation_converged=physical_count,
            **errors,
        )
    except Unsupported as e:
        return dict(outcome="unsupported", reason=str(e))
    except Exception as e:
        import traceback

        return dict(outcome="error", reason=str(e), traceback=traceback.format_exc())
    finally:
        if "solver" in locals():
            if hasattr(solver, "graph"):
                solver.graph = None
            if hasattr(solver, "core") and hasattr(solver.core, "graph"):
                solver.core.graph = None
        if inv is not None:
            inv.close()
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family")
    parser.add_argument("--case")
    parser.add_argument("--out", default="results/upstream_spectra_warp.json")
    args = parser.parse_args()
    cases = [
        json.loads(line) for p in sorted(FIX.glob("*.jsonl")) for line in p.read_text().splitlines()
    ]
    priority = {
        name: i
        for i, name in enumerate(
            ("SymEigsShift", "SymGEigsShift", "GenEigsRealShift", "Example1Shift")
        )
    }
    cases.sort(key=lambda c: priority.get(c["family"], 100))
    if args.family:
        cases = [c for c in cases if c["family"] == args.family]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    result = dict(
        spectra_revision="db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95",
        protocol="Exact exported upstream matrices, k, ncv, selection and residual assertions; graph replay; cuDSS inverses; 300 direct / 100 symmetric inverse / 30 general inverse cycles",
        cases=[],
    )
    for i, c in enumerate(cases):
        r = dict(**c, **run(c))
        result["cases"].append(r)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(
            i + 1,
            len(cases),
            c["id"],
            r["outcome"],
            r.get("max_absolute_residual", r.get("reason", "")),
            flush=True,
        )


if __name__ == "__main__":
    main()
