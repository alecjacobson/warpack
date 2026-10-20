"""Run exported upstream Spectra numerical problems with public Warpack APIs."""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from warpack import (
    BucklingEigensolver,
    CuDSSInverse,
    EigenpairEvaluation,
    GeneralEigensolver,
    GeneralizedKrylovSchur,
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


def svd_errors(a, w, u, v):
    """Original Spectra assertions plus paired equations and leading subspaces."""
    k = len(w)
    ref_u, ref_w, ref_vt = la.svd(a, full_matrices=False)
    return dict(
        values=float(np.max(np.abs(w - ref_w[:k]))),
        left_vectors=float(np.max(np.abs(np.abs(u) - np.abs(ref_u[:, :k])))),
        right_vectors=float(np.max(np.abs(np.abs(v) - np.abs(ref_vt[:k].T)))),
        left_equation_residual=float(np.max(abs(a @ v - u * w))),
        right_equation_residual=float(np.max(abs(a.T @ u - v * w))),
        orthogonality_error=float(
            max(np.max(abs(u.T @ u - np.eye(k))), np.max(abs(v.T @ v - np.eye(k))))
        ),
        subspace_error=float(
            max(
                np.linalg.norm(u - ref_u[:, :k] @ (ref_u[:, :k].T @ u), 2),
                np.linalg.norm(v - ref_vt[:k].T @ (ref_vt[:k] @ v), 2),
            )
        ),
    )


def solver_metadata(solver):
    """Describe the allocated iteration space, after solving and outside timing."""
    core = solver
    while hasattr(core, "core"):
        core = core.core
    if isinstance(core, HermitianEigensolver):
        core = core.real
    return dict(
        effective_ncv=core.m,
        basis_vector_length=core.n,
        basis_scalar_type="complex128" if core.q.dtype == wp.vec2d else "float64",
        basis_solver_type=type(core).__name__,
        outer_iterations=int(core.iterations.numpy()[0]),
    )


def solver_status(solver):
    """Read a device status only after execution, including wrapped solvers."""
    visited = set()
    while solver is not None and id(solver) not in visited:
        visited.add(id(solver))
        status = getattr(solver, "status", None)
        if status is not None:
            return int(status.numpy()[0])
        solver = getattr(solver, "core", getattr(solver, "real", None))
    return None


def select_symmetric_reference(values, k, which, sigma=None):
    """Select on the iteration spectrum, retaining physical reference indices."""
    ranking = values if sigma is None else 1.0 / (values - sigma)
    if which == "LM":
        return np.argsort(-abs(ranking))[:k]
    if which == "SM":
        return np.argsort(abs(ranking))[:k]
    order = np.argsort(ranking)
    if which == "LA":
        return order[-k:]
    if which == "SA":
        return order[:k]
    # BothEnds: the larger transformed end gets the odd extra mode.
    return np.r_[order[: k // 2], order[-(k - k // 2) :]]


def generalized_reference_errors(a, b, values, vectors, k, which, tolerance):
    """Independent SPD-mass reference, including selection ties and multiplicity."""
    reference, basis = la.eigh(a, b)
    score = {
        "LM": abs(reference),
        "SM": -abs(reference),
        "LA": reference,
        "SA": -reference,
    }[which]
    boundary = np.sort(score)[-k]
    allowed_indices = np.flatnonzero(score >= boundary - tolerance)
    mandatory = set(np.flatnonzero(score > boundary + tolerance))
    costs = abs(values[:, None] - reference[allowed_indices])
    rows, cols = linear_sum_assignment(costs)
    covered = mandatory.issubset(set(allowed_indices[cols]))
    allowed = basis[:, allowed_indices]
    # Eigh's reference basis is B-orthonormal, so use the B-inner product in
    # its projector. An ordinary Euclidean projector would test another space.
    projected = allowed @ (allowed.conj().T @ (b @ vectors))
    return dict(
        spectrum_error=float(costs[rows, cols].max()) if covered else float("inf"),
        mandatory_spectrum_covered=covered,
        orthogonality_error=float(np.max(abs(vectors.conj().T @ (b @ vectors) - np.eye(k)))),
        subspace_error=float(np.linalg.norm(vectors - projected, 2)),
    )


def run(
    case, accuracy_only=False, iteration_budget=None, tolerance_override=None, ncv_override=None
):
    if iteration_budget is not None and iteration_budget < 1:
        raise ValueError("iteration_budget must be positive")
    if tolerance_override is not None and (
        not math.isfinite(tolerance_override) or tolerance_override <= 0
    ):
        raise ValueError("tolerance_override must be finite and positive")
    if ncv_override is not None and ncv_override <= 0:
        raise ValueError("ncv_override must be positive")
    name = case["family"]
    k = case["k"]
    ncv = (case["ncv"] or None) if ncv_override is None else ncv_override
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
    threshold = 1e-12 if tolerance_override is None else tolerance_override
    margin = (
        1e-8
        if name in ("GenEigsRealShift", "GenEigsComplexShift", "Example2", "Example4")
        else 1e-9
    )
    if name == "DavidsonSymEigs":
        margin = 1e-10
    started = None if accuracy_only else time.perf_counter()
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
                    1,
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
                        method="krylov",
                    )
                elif "cayley" in case["name"]:
                    solver = GeneralizedKrylovSchur(
                        SparseOperator(ma),
                        SparseOperator(mb),
                        inv,
                        k,
                        ncv=ncv,
                        target=sigma.real,
                        mode="cayley",
                        tol=threshold,
                    )
                else:
                    solver = GeneralizedKrylovSchur(
                        SparseOperator(ma),
                        SparseOperator(mb),
                        inv,
                        k,
                        ncv=ncv,
                        target=sigma.real,
                        tol=threshold,
                    )
            else:
                if which not in ("SM", "LM", "LA", "SA"):
                    raise Unsupported(
                        "Generalized extremal selections are not mapped by this adapter"
                    )
                # Preserve the existing A^-1 M route for SM. The public
                # regular route uses M^-1 A with M-metric orthogonalization.
                inv = CuDSSInverse(
                    ma if which == "SM" else mb,
                    1,
                    mtype="general",
                    matching=True,
                    refinement_steps=1,
                )
                solver = GeneralizedKrylovSchur(
                    SparseOperator(ma),
                    SparseOperator(mb),
                    inv,
                    k,
                    ncv=ncv,
                    target=0,
                    tol=threshold,
                    mode="shift-invert" if which == "SM" else "regular",
                    which="LM" if which == "SM" else which,
                )
            iterations = 100
        elif name in ("SymEigsShift", "Example1Shift") or which == "SM":
            if np.iscomplexobj(a):
                solver = ShiftInvertEigensolver(
                    complex_matrix(a),
                    k,
                    sigma,
                    ncv=ncv,
                    backend="cudss",
                    tol=threshold,
                    hermitian=name == "HermEigs",
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
        if iteration_budget is not None:
            iterations = iteration_budget
        if accuracy_only:
            graph = solver.capture(iterations, **options)
            for _ in range(2):
                wp.capture_launch(graph)
            wp.synchronize_device(solver.device)
            timing = dict(execution="accuracy_only", replays=2)
        else:
            setup = time.perf_counter() - started
            timing = dict(setup_seconds=setup, **execute(solver, iterations, **options))
        if name == "SVD":
            w = output(solver.singular_values)
            u = output(solver.u).T
            v = output(solver.v).T
            errors = svd_errors(a, w, u, v)
            err = max(errors.values())
            errors["reported_converged"] = int(solver.core.converged.numpy()[0])
            errors["left_converged"] = int(solver.count_left.numpy()[0])
            errors["right_converged"] = int(solver.count_right.numpy()[0])
        else:
            w = output(solver.eigenvalues if evaluation is None else evaluation.values)
            v = output(solver.eigenvectors).T
            residual = a @ v - (v if b is None else b @ v) * w
            err = float(np.max(np.abs(residual)))
            errors = {"max_absolute_residual": err, "returned_pairs": len(w)}
            if name in ("SymGEigsCholesky", "SymGEigsRegInv"):
                errors.update(generalized_reference_errors(a, b, w, v, k, which, margin))
            if name in ("HermEigs", "DavidsonSymEigs", "SymEigsShift"):
                ref, basis = la.eigh(a)
                selected = select_symmetric_reference(
                    ref, k, which, sigma.real if name == "SymEigsShift" else None
                )
                expected = ref[selected]
                allowed = basis[:, np.min(abs(ref[:, None] - expected), axis=1) <= 1e-9]
                errors.update(
                    spectrum_error=float(np.max(abs(np.sort(w) - np.sort(expected)))),
                    orthogonality_error=float(np.max(abs(v.conj().T @ v - np.eye(k)))),
                    subspace_error=float(np.linalg.norm(v - allowed @ (allowed.conj().T @ v), 2)),
                )
            if name == "DavidsonSymEigs":
                errors.update(
                    max_normalized_residual=float(
                        np.max(np.linalg.norm(residual, axis=0) / np.maximum(1.0, abs(w)))
                    ),
                    reported_max_normalized_residual=float(np.max(solver.residuals.numpy())),
                    vector_norm_error=float(np.max(abs(np.linalg.norm(v, axis=0) - 1))),
                )
            if name in ("GenEigs", "ComplexEigs") and which == "SM":
                ref, basis = la.eig(a)
                # A real problem may cut through a conjugate pair with equal
                # magnitude. Accept either boundary eigenvalue, but match each
                # returned value to a distinct reference slot.
                cutoff = np.sort(abs(ref))[k - 1] + margin
                selected = np.flatnonzero(abs(ref) <= cutoff)
                costs = abs(w[:, None] - ref[selected])
                rows, cols = linear_sum_assignment(costs)
                allowed = la.orth(basis[:, selected])
                errors.update(
                    spectrum_error=float(np.max(costs[rows, cols])),
                    vector_norm_error=float(np.max(abs(np.linalg.norm(v, axis=0) - 1))),
                    subspace_error=float(np.linalg.norm(v - allowed @ (allowed.conj().T @ v), 2)),
                )
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
        errors["solver_status"] = solver_status(solver)
        errors["solver_status_required"] = True
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
            solver_type=type(solver).__name__,
            requested_ncv=ncv,
            **solver_metadata(solver),
            iteration_method=getattr(solver, "method", None),
            cudss_mtype=getattr(factor, "mtype", None),
            cudss_reordering=getattr(factor, "reordering", None),
            cudss_matching=getattr(factor, "matching", None),
            cudss_refinement_steps=factor.refinement_steps
            if isinstance(factor, CuDSSInverse)
            else None,
            outcome="passed" if ok else "failed",
            **timing,
            iteration_budget=iterations,
            upstream_absolute_tolerance=margin,
            checked_numerical_bounds_pass=numerical_pass,
            requested_spectrum_checked=contract["requested_spectrum_checked"],
            internal_tolerance=threshold,
            reported_convergence_space=convergence_space,
            reported_count_pass=None if count is None else count >= k,
            reported_status_pass=contract["reported_status_pass"],
            original_evaluation_converged=physical_count,
            **errors,
        )
    except Unsupported as e:
        return dict(outcome="unsupported", reason=str(e))
    except Exception as e:
        import traceback

        return dict(outcome="error", reason=str(e), traceback=traceback.format_exc())
    finally:
        if accuracy_only:
            graph = None
        if "solver" in locals():
            if hasattr(solver, "graph"):
                solver.graph = None
            if hasattr(solver, "core") and hasattr(solver.core, "graph"):
                solver.core.graph = None
        if inv is not None:
            inv.close()
        gc.collect()


def main():
    global FIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--family")
    parser.add_argument("--case")
    parser.add_argument("--selection", choices=sorted(set(RULES.values())))
    parser.add_argument("--fixtures", type=Path, default=FIX)
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--iterations", type=int, help="explicit outer-cycle override")
    parser.add_argument("--tol", type=float, help="explicit internal normalized-tolerance override")
    parser.add_argument("--ncv", type=int, help="explicit Krylov-space override")
    parser.add_argument("--out", default="results/upstream_spectra_warp.json")
    args = parser.parse_args()
    if args.iterations is not None and args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.tol is not None and (not math.isfinite(args.tol) or args.tol <= 0):
        parser.error("--tol must be finite and positive")
    if args.ncv is not None and args.ncv < 1:
        parser.error("--ncv must be positive")
    FIX = args.fixtures.resolve()
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
    if args.selection:
        cases = [c for c in cases if RULES[c["selection"]] == args.selection]
    if not cases:
        parser.error("no matching exported fixtures")
    result = dict(
        spectra_revision="db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95",
        protocol="Exact exported upstream matrices, k, selection and assertions; exported ncv unless explicitly overridden; graph replay; cuDSS inverses; default 300 direct / 100 symmetric inverse / 30 general inverse cycles; explicit overrides recorded per case",
        iteration_override=args.iterations,
        tolerance_override=args.tol,
        ncv_override=args.ncv,
        cases=[],
    )
    for i, c in enumerate(cases):
        r = dict(
            **c,
            **run(
                c,
                accuracy_only=args.accuracy_only,
                iteration_budget=args.iterations,
                tolerance_override=args.tol,
                ncv_override=args.ncv,
            ),
        )
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
