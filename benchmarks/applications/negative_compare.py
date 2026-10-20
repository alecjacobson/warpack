"""Bounded, accuracy-gated comparisons for indefinite 3D FEM selection."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import warp as wp
import warp.optim.linear as linear

from .compare import compare_baseline, run
from .negative_foundation import make_negative_foundation

METHODS = [
    "scipy",
    "spectra",
    "cupy",
    "cupy-eigsh",
    "warp",
    "warp-cg",
    "warp-cg-bj",
    "warp-cg-fsai",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=6)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--selections", default="SA,SM,LM")
    parser.add_argument("--masses", default="lumped,consistent")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--ncv", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--lobpcg-tol", type=float, default=1e-8)
    parser.add_argument("--cg-tol", type=float, default=1e-13)
    parser.add_argument("--cg-atol", type=float)
    parser.add_argument("--lobpcg-maxiter", type=int, default=200)
    parser.add_argument("--eigsh-tol", type=float, default=1e-8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    selections, masses, methods = (
        args.selections.split(","),
        args.masses.split(","),
        args.methods.split(","),
    )
    if (
        set(selections) - {"SA", "SM", "LM"}
        or set(masses) - {"lumped", "consistent"}
        or set(methods) - set(METHODS)
    ):
        parser.error("unknown selection, mass or method")
    if min(args.side, args.modes, args.ncv, args.steps, args.repeats, args.timeout) < 1:
        parser.error("positive dimensions, budgets and repetitions required")
    if args.cg_atol is not None and (not math.isfinite(args.cg_atol) or args.cg_atol < 0):
        parser.error("CG absolute tolerance must be finite and nonnegative")
    if args.lobpcg_maxiter < 1 or any(
        not math.isfinite(x) or x <= 0 for x in (args.cg_tol, args.eigsh_tol, args.lobpcg_tol)
    ):
        parser.error("positive finite tolerances and positive LOBPCG iteration limit required")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if any(len(x) != 1 for x in (selections, masses, methods)):
            parser.error("one fixture and method per worker")
        case = make_negative_foundation(
            args.side, args.modes, selections[0], masses[0] == "consistent"
        )
        p, method = case.problem, methods[0]
        if args.ncv <= p.k:
            parser.error("ncv must exceed the complete requested cluster")
        if (
            method == "cupy"
            and case.selection == "LM"
            and not case.largest_magnitude_is_largest_algebraic
        ):
            row = dict(
                method=method,
                outcome="unsupported",
                reason="Generalized CuPy LOBPCG targets algebraic extremes; this LM set differs from LA",
            )
        else:
            row = run(
                p.api_class,
                p.n,
                method,
                args.repeats,
                args.ncv,
                args.steps,
                problem=p,
                lobpcg_tol=args.lobpcg_tol,
                cg_tol=args.cg_tol,
                cg_atol=args.cg_atol,
                lobpcg_maxiter=args.lobpcg_maxiter,
                cupy_which=case.selection,
                eigsh_tol=args.eigsh_tol,
            )
        if row.get("validation"):
            row["physical_accepted"] = row["accepted"]
            row["strict_stopping_accepted"] = all(
                trial.get("solver_full_count", True)
                and trial.get("native_stopping_passed", True)
                and trial.get("solver_status") in (None, 0)
                for trial in row["validation"]
            )
            row["inner_solve_accepted"] = all(
                item["unconverged_rhs"] == 0 for item in row.get("cg_statistics", [])
            )
            row["strict_stopping_accepted"] &= row["inner_solve_accepted"]
            row["accepted"] &= row["strict_stopping_accepted"]
            row["contract_accepted"] &= row["strict_stopping_accepted"]
            row["outcome"] = "passed" if row["accepted"] else "failed"
        row.update(
            selection=case.selection,
            mass=case.mass,
            side=args.side,
            requested_modes=args.modes,
            foundation=case.foundation,
            reference="Analytic complete tensor-product Q1 sine spectrum, independent of eigensolvers",
            stiffness="Symmetric indefinite 3D anisotropic diffusion minus foundation; two repeated polarizations",
            mass_property="SPD sparse consistent"
            if masses[0] == "consistent"
            else "SPD diagonal lumped",
            inverse_property="Symmetric indefinite shifted stiffness"
            if selections[0] == "SM"
            else ("SPD mass" if selections[0] == "LM" else "SPD shifted stiffness"),
            nullity=0,
            negative_modes=int((case.full_spectrum < 0).sum()),
            largest_magnitude_is_largest_algebraic=case.largest_magnitude_is_largest_algebraic,
            selection_contract="SM uses shift-invert at zero; SA uses a shift below the spectrum; LM uses mass-inverse regular iteration. CuPy generalized LM is compared only when its LA target is the same set. No Euclidean eigsh on M^-1 K.",
        )
        destination.write_text(json.dumps(row, indent=2) + "\n")
        return
    report = dict(
        args=vars(args),
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "cupy-cuda12x", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        upstream_source_sha256={
            "linear": hashlib.sha256(
                Path(inspect.getfile(linear.preconditioner)).read_bytes()
            ).hexdigest(),
            "fsai": hashlib.sha256(Path(inspect.getfile(linear.FSAI)).read_bytes()).hexdigest()
            if hasattr(linear, "FSAI")
            else None,
        },
        spectra_binary_sha256=hashlib.sha256(
            Path("build/application_spectra.so").read_bytes()
        ).hexdigest()
        if Path("build/application_spectra.so").exists()
        else None,
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(
                list(Path("warpack").glob("*.py"))
                + [
                    Path(__file__),
                    Path(__file__).with_name("negative_foundation.py"),
                    Path(__file__).with_name("runner.py"),
                    Path(__file__).with_name("references.py"),
                    Path(__file__).with_name("compare.py"),
                ]
            )
        },
        protocol="Sequential isolated workers; warm synchronized samples; every trial checked against the original pencil, complete analytic requested space, eigenvalues and mass orthogonality at unchanged 1e-8 bounds. Setup/capture/reference/validation excluded and recorded separately. Native CPU factorizations are for this cube fixture only. CuPy generalized LOBPCG uses a supplied cuDSS SPD preconditioner; native CuPy eigsh is a separate diagonal-mass SA/LM path with explicit scaling; generalized SM and CG on indefinite shifted systems are explicit unsupported routes. Full Warp counts and fresh native CuPy absolute stopping residuals are required in addition to common physical gates; vendor graph transfers are reported separately. Failed and timed-out results are retained, never counted as speedups.",
        cases=[],
    )
    folder = Path("build/negative-foundation") / destination.stem
    folder.mkdir(parents=True, exist_ok=True)
    for selection in selections:
        for mass in masses:
            for method in methods:
                name = f"{selection}-{mass}-{method}"
                output = folder / f"{name}.json"
                output.unlink(missing_ok=True)
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.applications.negative_compare",
                    "--worker",
                    "--selections",
                    selection,
                    "--masses",
                    mass,
                    "--methods",
                    method,
                    "--out",
                    str(output),
                ]
                for key in (
                    "side",
                    "modes",
                    "ncv",
                    "steps",
                    "repeats",
                    "timeout",
                    "lobpcg_tol",
                    "cg_tol",
                    "eigsh_tol",
                    "lobpcg_maxiter",
                ):
                    cmd += ["--" + key.replace("_", "-"), str(getattr(args, key))]
                if args.cg_atol is not None:
                    cmd += ["--cg-atol", str(args.cg_atol)]
                with output.with_suffix(".log").open("w") as log:
                    try:
                        completed = subprocess.run(
                            cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout
                        )
                        row = (
                            json.loads(output.read_text())
                            if completed.returncode == 0 and output.exists()
                            else dict(
                                method=method, outcome="error", returncode=completed.returncode
                            )
                        )
                    except subprocess.TimeoutExpired:
                        row = dict(method=method, outcome="timeout", seconds=args.timeout)
                row.update(
                    selection=selection,
                    mass=mass,
                    side=args.side,
                    worker_log=str(output.with_suffix(".log")),
                )
                report["cases"].append(row)
                if row.get("api_class") is not None and row.get("n") is not None:
                    (
                        Path("build/issue-work")
                        / f"graph-{row['api_class']}-{row['n']}-{method}.dot"
                    ).unlink(missing_ok=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
                print(name, row["outcome"], row.get("solve_seconds"), flush=True)
    regressions = []
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        for key in ("gpu", "warp", "versions", "blas_threads"):
            if report[key] != baseline[key]:
                raise ValueError(f"baseline environment mismatch: {key}")
        for row in report["cases"]:
            old = next(
                (
                    candidate
                    for candidate in baseline["cases"]
                    if all(
                        candidate.get(key) == row.get(key)
                        for key in ("selection", "mass", "method", "side")
                    )
                ),
                None,
            )
            if old is not None and old.get("accepted"):
                if "api_class" not in row:
                    regressions.append(
                        dict(
                            selection=row["selection"],
                            mass=row["mass"],
                            method=row["method"],
                            accuracy_regression=True,
                        )
                    )
                else:
                    for check in compare_baseline([row], {"cases": [old]}, 1.25):
                        check.update(selection=row["selection"], mass=row["mass"])
                        regressions.append(check)
        report["regression_checks"] = regressions
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if (
            not regressions
            or any(not check.get("comparable", False) for check in regressions)
            or any(
                check.get(key)
                for check in regressions
                for key in ("accuracy_regression", "graph_regression", "performance_regression")
            )
        ):
            raise SystemExit("Regression detected; evidence retained")
    if any(row["outcome"] not in ("passed", "unsupported") for row in report["cases"]):
        raise SystemExit("Rejected/failed profile; evidence retained")


if __name__ == "__main__":
    main()
