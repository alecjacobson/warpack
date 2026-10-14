"""Isolated, accuracy-gated topology comparisons, including failed/unsupported rows."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import warp as wp

from .compare import compare_baseline, run
from .hodge import make_hodge_case

NAMES = [
    "sphere",
    "disk",
    "annulus",
    "torus",
    "disconnected",
    "weak_positive",
    "large_disconnected",
]
METHODS = ["scipy", "spectra", "cupy", "warp", "warp-cg"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(NAMES))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--large-side", type=int, default=41)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ncv", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", default="results/hodge_comparisons.json")
    parser.add_argument("--baseline")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    names, methods = args.cases.split(","), args.methods.split(",")
    if set(names) - set(NAMES) or set(methods) - set(METHODS):
        parser.error("Unknown case or method")
    if args.worker:
        if len(names) != 1 or len(methods) != 1:
            parser.error("Worker needs one case and method")
        c = make_hodge_case(names[0], args.large_side if names[0] == "large_disconnected" else 4)
        p = c.problem
        row = run("D", p.n, methods[0], args.repeats, args.ncv, args.steps, problem=p)
        row.update(
            fixture=names[0],
            expected_nullity=p.nullity,
            reference_positive_guard=float(c.spectrum[p.nullity]),
            nullity_cutoff=1e-8,
            incidence_rank1=c.rank1,
            incidence_rank2=c.rank2,
            vertices=c.boundary1.shape[0],
            faces=c.boundary2.shape[1],
            boundary_composition_nnz=(c.boundary1 @ c.boundary2).nnz,
            topology_reference=c.reference,
        )
        Path(args.out).write_text(json.dumps(row, indent=2) + "\n")
        return
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            n: importlib.metadata.version(n)
            for n in ["numpy", "scipy", "cupy-cuda12x", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        protocol="Sequential isolated workers. Original residual, requested spectrum, mass orthogonality, known nullity, independent nullspace projection and positive guard are required for accepted timings. Cutoff 1e-8 is a benchmark policy, not certified Betti counting. Setup/reference construction/capture/output conversion excluded from solve timing. Every timed trial validated. Warp cuDSS uses SPD factors; pure-Warp CG uses upstream diagonal preconditioning. CuPy is generalized LOBPCG with a cuDSS preconditioner, not eigsh; dense fallback sizes explicitly unsupported. Failed timings retained, not wins.",
        cases=[],
    )
    folder = Path("build/hodge-workers")
    folder.mkdir(parents=True, exist_ok=True)
    destination = Path(args.out)
    for name in names:
        for method in methods:
            out = folder / f"{name}-{method}.json"
            log = out.with_suffix(".log")
            command = [
                sys.executable,
                "-m",
                "benchmarks.applications.hodge_compare",
                "--worker",
                "--cases",
                name,
                "--methods",
                method,
                "--large-side",
                str(args.large_side),
                "--steps",
                str(args.steps),
                "--ncv",
                str(args.ncv),
                "--repeats",
                str(args.repeats),
                "--out",
                str(out),
            ]
            out.unlink(missing_ok=True)
            try:
                with log.open("w") as f:
                    subprocess.run(command, stdout=f, stderr=f, check=True, timeout=args.timeout)
                row = json.loads(out.read_text())
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                p = make_hodge_case(
                    name, args.large_side if name == "large_disconnected" else 4
                ).problem
                row = dict(
                    fixture=name,
                    api_class="D",
                    requested_size=p.n,
                    n=p.n,
                    method=method,
                    accepted=False,
                    outcome="timeout"
                    if isinstance(e, subprocess.TimeoutExpired)
                    else "worker_error",
                    log=str(log),
                )
            row["worker_wall_limit_seconds"] = args.timeout
            report["cases"].append(row)
            destination.write_text(json.dumps(report, indent=2) + "\n")
            print(name, method, row["outcome"], row.get("solve_seconds"), flush=True)
    if args.baseline:
        old = json.loads(Path(args.baseline).read_text())
        if (old["gpu"], old["warp"]) != (report["gpu"], report["warp"]):
            raise ValueError("Baseline device/runtime mismatch")
        index = {(r["fixture"], r["method"]): r for r in old["cases"]}
        checks = []
        for row in report["cases"]:
            baseline = index.get((row["fixture"], row["method"]))
            if baseline is not None:
                for check in compare_baseline([row], dict(cases=[baseline]), 1.25):
                    check["fixture"] = row["fixture"]
                    checks.append(check)
        report["regression_checks"] = checks
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if any(
            c.get("performance_regression") or c["accuracy_regression"] or c["graph_regression"]
            for c in checks
        ):
            raise SystemExit(1)
        if not any(c["comparable"] for c in checks):
            raise SystemExit("No comparable uncontended timings")


if __name__ == "__main__":
    main()
