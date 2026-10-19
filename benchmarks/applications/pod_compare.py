"""Bounded large rectangular POD comparisons, including repeated singular spaces."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import warp as wp

from .compare import run
from .problems import Problem


def snapshots(size, variant="tall", modes=4, columns=48):
    if columns < 8 or size < max(96, columns) or not 1 <= modes <= columns - 4:
        raise ValueError(
            "require columns >= 8, size >= max(96, columns), and 1 <= modes <= columns-4"
        )
    if variant == "wave":
        if columns != 48:
            raise ValueError("Wave fixture requires 48 snapshots")
        if modes > 22:
            raise ValueError("Wave fixture has 22 excited spatial modes")
        # Fixed-end unit string: x_i=i/(n+1), t_j=2*j/48, wave speed 1.
        # cos(m*pi*t) pairs with sin(m*pi*x). Spatial/time normalizations
        # make the unweighted sampled snapshot singular spectrum analytic.
        frequency = np.arange(1, 23)
        x, time = np.arange(1, size + 1) / (size + 1), np.arange(48) / 24
        u = np.sin(np.pi * x[:, None] * frequency) * np.sqrt(2 / (size + 1))
        v = np.cos(np.pi * time[:, None] * frequency) * np.sqrt(2 / 48)
        spectrum = np.geomspace(20, 0.05, 22)
        return Problem(
            "fixed_end_string_snapshots",
            "I",
            sp.csr_matrix((u * spectrum) @ v.T),
            None,
            modes,
            0.0,
            mode="svd",
            expected=spectrum[:modes],
            requested_basis=u[:, :modes],
            right_requested_basis=v[:, :modes],
            notes="48 equally spaced snapshots over a full fundamental period of a fixed-end unit string with 22 analytic cosine modal responses; normalized unweighted samples, not a general mass-weighted POD API",
        )
    rng = np.random.default_rng(2026)
    u = la.qr(rng.normal(size=(size, columns)), mode="economic")[0]
    v = la.qr(rng.normal(size=(columns, columns)), mode="economic")[0]
    spectrum = np.geomspace(20, 0.05, columns)
    spectrum[-2:] = 0
    if variant == "clustered":
        if modes < 4:
            raise ValueError("clustered fixture needs at least four requested triplets")
        spectrum[:3] = 20
    elif variant not in ("tall", "wide"):
        raise ValueError(variant)
    a = (u * spectrum) @ v.T
    if variant == "wide":
        a, u, v = a.T, v, u
    return Problem(
        f"snapshot_pod_{variant}",
        "I",
        sp.csr_matrix(a),
        None,
        modes,
        0.0,
        mode="svd",
        expected=spectrum[:modes],
        requested_basis=u[:, :modes],
        right_requested_basis=v[:, :modes],
        notes="Known singular spectrum, two exact zero singular values, independent left/right requested subspaces; top multiplicity three for clustered variant",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="tall,wide,clustered,wave")
    parser.add_argument("--sizes", default="10000,100000")
    parser.add_argument("--methods", default="scipy,spectra,cupy,warp")
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--ncv", type=int, default=32)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", default="results/pod_comparisons.json")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    names, sizes, methods = (
        args.cases.split(","),
        list(map(int, args.sizes.split(","))),
        args.methods.split(","),
    )
    if set(names) - {"tall", "wide", "clustered", "wave"} or set(methods) - {
        "scipy",
        "spectra",
        "cupy",
        "warp",
    }:
        parser.error("Unknown fixture or method")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if len(names) != 1 or len(sizes) != 1 or len(methods) != 1:
            parser.error("One fixture, size and method per worker")
        p = snapshots(sizes[0], names[0], args.modes)
        row = run("I", sizes[0], methods[0], args.repeats, args.ncv, args.steps, problem=p)
        if methods[0] == "warp":
            row["strict_counts_pass"] = (
                row.get("status") == 0
                and all(c.get("solver_full_count", False) for c in row.get("validation", []))
                and bool(row.get("validation"))
            )
        row.update(
            variant=names[0],
            reference="Analytic singular spectrum and independently generated left/right bases",
            worker_wall_limit_seconds=args.timeout,
        )
        destination.write_text(json.dumps(row, indent=2) + "\n")
        return
    report = dict(
        gpu=wp.get_device().name,
        warp=wp.__version__,
        versions={n: importlib.metadata.version(n) for n in ["numpy", "scipy", "cupy-cuda12x"]},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
        protocol="Sequential isolated workers; setup/capture/reference/validation excluded from synchronized warm solve medians; every trial checks requested values, both SVD equations, both orthogonalities and both complete requested subspaces. Independent CPU reference bases use seed 2026, Warp starts seed 42. ncv and maximum Warp restart cycles explicit; no tolerance relaxation. Failed timings are diagnostics, not speedups. Contention inventory before/after every trial. Graph DOT includes nested bodies, not API trace.",
        cases=[],
    )
    folder = Path("build/adaptive-chain")
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        for size in sizes:
            for method in methods:
                output = folder / f"{destination.stem}-{name}-{size}-{method}.json"
                log = output.with_suffix(".log")
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.applications.pod_compare",
                    "--worker",
                    "--cases",
                    name,
                    "--sizes",
                    str(size),
                    "--methods",
                    method,
                ]
                for flag, value in [
                    ("modes", args.modes),
                    ("ncv", args.ncv),
                    ("steps", args.steps),
                    ("repeats", args.repeats),
                    ("timeout", args.timeout),
                    ("out", output),
                ]:
                    command += ["--" + flag, str(value)]
                output.unlink(missing_ok=True)
                try:
                    with log.open("w") as stream:
                        subprocess.run(
                            command, stdout=stream, stderr=stream, check=True, timeout=args.timeout
                        )
                    row = json.loads(output.read_text())
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                    row = dict(
                        variant=name,
                        requested_size=size,
                        method=method,
                        accepted=False,
                        outcome="timeout"
                        if isinstance(error, subprocess.TimeoutExpired)
                        else "worker_error",
                        log=str(log),
                    )
                report["cases"].append(row)
                destination.write_text(json.dumps(report, indent=2) + "\n")
                print(name, size, method, row["outcome"], row.get("solve_seconds"), flush=True)
    if any(
        not r["accepted"]
        or not r.get("timing_uncontended")
        or not r.get("strict_counts_pass", True)
        for r in report["cases"]
    ):
        raise SystemExit("POD numerical or contention gate failed; inspect retained rows")


if __name__ == "__main__":
    main()
