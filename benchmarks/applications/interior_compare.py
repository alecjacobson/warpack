"""Bounded, isolated nearest-band comparisons on coupled FEM pencils."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

from .compare import compare_baseline, run
from .interior_fem import NAMES, make_interior_case

METHODS = ("scipy", "spectra", "cupy", "warp", "warp-block", "warp-cg-bj", "warp-cg-fsai")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(NAMES))
    parser.add_argument("--methods", default=",".join(m for m in METHODS if m != "warp-block"))
    parser.add_argument("--side", type=int, default=2)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--band-index", type=int, default=6)
    parser.add_argument("--fraction", type=float, default=0.37)
    parser.add_argument("--ncv", type=int, default=48)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", default="results/interior_comparisons.json")
    parser.add_argument("--baseline")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    names, methods = args.cases.split(","), args.methods.split(",")
    if set(names) - set(NAMES) or set(methods) - set(METHODS):
        parser.error("Unknown fixture or method")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(names) != 1 or len(methods) != 1:
            parser.error("One fixture and method per worker")
        c = make_interior_case(names[0], args.side, args.modes, args.band_index, args.fraction)
        p = c.problem
        row = run("B", p.n, methods[0], args.repeats, args.ncv, args.steps, problem=p)
        row.update(
            fixture=names[0],
            side=args.side,
            requested_modes=args.modes,
            band_index=args.band_index,
            gap_fraction=args.fraction,
            reference=c.reference,
            reference_gap=c.band_gap,
            selection_margin=c.selection_margin,
            selected_indices=c.selected_indices.tolist(),
            modes_below_shift=int(np.sum(p.expected < p.target)),
            modes_above_shift=int(np.sum(p.expected > p.target)),
            original_rigid_modes=c.base_nullity,
            mass="Lumped" if names[0] == "clamped_lumped" else "Consistent",
            shifted_matrix="Symmetric indefinite; CG is not applicable",
            shift_units="Eigenvalue / squared angular frequency",
            worker_wall_limit_seconds=args.timeout,
        )
        destination.write_text(json.dumps(row, indent=2) + "\n")
        return
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={
            n: importlib.metadata.version(n)
            for n in ["numpy", "scipy", "cupy-cuda12x", "nvidia-cudss-cu12"]
        },
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        protocol="Sequential isolated workers; three synchronized warm trials by default, every trial checked against original residual, M-orthogonality, nearest physical eigenvalues and complete requested subspace. CPU reference construction, setup and capture are outside solve timing. Reference seed 913 and expanded basis differ from timed native starts 2026 and Warp starts 42. Complete distance-boundary clusters extend k. Native CuPy generalized shift and CG on an indefinite shifted pencil are explicitly unsupported. Internal strict counts and vendor cuDSS graph transfers remain separate from physical numerical acceptance. Failed timings are retained and are not speedups.",
        cases=[],
    )
    folder = Path("build/interior-fem")
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        for method in methods:
            label = f"{destination.stem}-{name}-{method}"
            output, log = folder / f"{label}.json", folder / f"{label}.log"
            command = [
                sys.executable,
                "-m",
                "benchmarks.applications.interior_compare",
                "--worker",
                "--cases",
                name,
                "--methods",
                method,
            ]
            for flag, value in [
                ("side", args.side),
                ("modes", args.modes),
                ("band-index", args.band_index),
                ("fraction", args.fraction),
                ("ncv", args.ncv),
                ("steps", args.steps),
                ("repeats", args.repeats),
                ("timeout", args.timeout),
                ("out", output),
            ]:
                command += ["--" + flag, str(value)]
            output.unlink(missing_ok=True)
            try:
                with log.open("w") as f:
                    subprocess.run(command, stdout=f, stderr=f, check=True, timeout=args.timeout)
                row = json.loads(output.read_text())
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                row = dict(
                    fixture=name,
                    side=args.side,
                    method=method,
                    accepted=False,
                    outcome="timeout"
                    if isinstance(e, subprocess.TimeoutExpired)
                    else "worker_error",
                    log=str(log),
                    worker_wall_limit_seconds=args.timeout,
                )
            row.update(
                requested_modes=args.modes, band_index=args.band_index, gap_fraction=args.fraction
            )
            report["cases"].append(row)
            destination.write_text(json.dumps(report, indent=2) + "\n")
            print(name, method, row["outcome"], row.get("solve_seconds"), flush=True)
    if args.baseline:
        old = json.loads(Path(args.baseline).read_text())
        if (old["gpu"], old["warp"]) != (report["gpu"], report["warp"]):
            raise ValueError("Baseline GPU/runtime mismatch")

        def key(r):
            return tuple(
                r.get(k)
                for k in (
                    "fixture",
                    "side",
                    "method",
                    "requested_modes",
                    "band_index",
                    "gap_fraction",
                )
            )

        index = {key(r): r for r in old["cases"]}
        checks = []
        for row in report["cases"]:
            prior = index.get(key(row))
            if prior is not None and prior.get("accepted"):
                if "n" not in row:
                    checks.append(
                        dict(
                            fixture=row["fixture"],
                            method=row["method"],
                            comparable=False,
                            accuracy_regression=True,
                        )
                    )
                    continue
                checks.extend(
                    dict(c, fixture=row["fixture"])
                    for c in compare_baseline([row], dict(cases=[prior]), 1.25)
                )
        report["regression_checks"] = checks
        destination.write_text(json.dumps(report, indent=2) + "\n")
        if not any(c["comparable"] for c in checks) or any(
            c.get("performance_regression")
            or c.get("accuracy_regression")
            or c.get("graph_regression")
            for c in checks
        ):
            raise SystemExit("Regression or no comparable accepted timings")
    if any(r["outcome"] not in ("passed", "unsupported") for r in report["cases"]):
        raise SystemExit("One or more numerical profiles rejected; evidence preserved")


if __name__ == "__main__":
    main()
