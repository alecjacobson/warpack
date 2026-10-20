"""Accuracy-only native Spectra/ARPACK requested-spectrum audit.

Uses the original exported Spectra matrices and start vectors. No solve timings
are reported: dense reference construction and extraction are diagnostic work.
"""

import argparse
import ctypes as ct
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from .general_reference import GeneralReference

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open("rb") as stream:
        shape = np.fromfile(stream, dtype=np.int64, count=2)
        a = np.fromfile(stream, dtype=np.complex128).reshape(shape)
    if np.any(a.imag):
        raise ValueError("This native Spectra audit requires real input matrices and starts")
    return np.ascontiguousarray(a.real)


def build_native(source, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-I/usr/include/eigen3",
        "-I" + str(source / "include"),
        str(Path(__file__).with_suffix(".cpp").with_name("general_reference.cpp")),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    return dict(command=command, spectra_revision=revision, binary_sha256=digest(output))


def native_function(path):
    library = ct.CDLL(str(path.resolve()))
    function = library.general_eigs
    pointer = ct.c_void_p
    function.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        pointer,
        pointer,
        ct.c_double,
        ct.c_int,
        pointer,
        pointer,
        pointer,
    ]
    function.restype = ct.c_int
    return function


def spectra(function, a, initial, k, ncv, tol, budget):
    values = np.zeros(k + 2, dtype=np.complex128)
    vectors = np.zeros((len(a), k + 2), dtype=np.complex128, order="F")
    stats = np.zeros(5, dtype=np.int32)
    code = function(
        len(a),
        k,
        ncv,
        a.ctypes.data,
        initial.ctypes.data,
        tol,
        budget,
        values.ctypes.data,
        vectors.ctypes.data,
        stats.ctypes.data,
    )
    metadata = dict(
        return_code=code,
        converged=int(stats[0]),
        iterations=int(stats[1]),
        operations=int(stats[2]),
        info=int(stats[3]),
        output_count=int(stats[4]),
    )
    if code != 0:
        raise RuntimeError(f"Native extraction failed: {metadata}")
    return values[: stats[4]], vectors[:, : stats[4]], metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "build/spectra-benchmark-fixtures")
    parser.add_argument("--spectra-source", type=Path, default=ROOT / "build/upstream-spectra")
    parser.add_argument("--library", type=Path, default=ROOT / "build/general_reference.so")
    parser.add_argument("--case", action="append", help="Repeat to select cases; default all 20")
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("original", "wide", "very-wide"),
        default=["original", "wide"],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("spectra", "arpack-real", "arpack-complex"),
        default=["spectra", "arpack-real", "arpack-complex"],
    )
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    native = build_native(args.spectra_source, args.library) if "spectra" in args.backends else None
    function = native_function(args.library) if native else None
    cases = [
        json.loads(line)
        for line in (args.fixtures / "benchmark_gen.jsonl").read_text().splitlines()
    ]
    if args.case:
        cases = [case for case in cases if case["id"] in args.case]
        if len(cases) != len(set(args.case)):
            parser.error("Unknown or missing fixture case")
    report = dict(
        protocol="Accuracy only. Original real Spectra matrices/v0; LM, tol=1e-10, "
        "maxiter=budget for each backend. Original ncv and explicitly widened "
        "ncv=min(n, max(2*original_ncv, 4*k)) are separate profiles. "
        "Very-wide uses 4*original_ncv instead of 2*original_ncv. "
        "Complex ARPACK casts identical real inputs to exercise complex Arnoldi. "
        "Every returned set is checked against an independent dense spectrum, "
        "including mandatory modes above conjugate cutoff ties.",
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        threads={key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        native=native,
        source_sha256={
            name: digest(ROOT / name)
            for name in (
                "benchmarks/upstream/check_general_references.py",
                "benchmarks/upstream/general_reference.cpp",
                "benchmarks/upstream/general_reference.py",
            )
        },
        cases=[],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        matrix_path = args.fixtures / (case["id"] + ".a")
        start_path = args.fixtures / (case["id"] + ".v0")
        a, v0 = read(matrix_path), read(start_path).ravel().copy()
        reference = GeneralReference(a, case["k"])
        for profile in args.profiles:
            ncv = (
                case["ncv"]
                if profile == "original"
                else min(
                    len(a), max((4 if profile == "very-wide" else 2) * case["ncv"], 4 * case["k"])
                )
            )
            for backend in args.backends:
                row = dict(
                    case=case,
                    profile=profile,
                    backend=backend,
                    ncv=ncv,
                    tol=1e-10,
                    budget=args.budget,
                    matrix_sha256=digest(matrix_path),
                    start_sha256=digest(start_path),
                )
                try:
                    if backend == "spectra":
                        w, v, metadata = spectra(
                            function, a, v0.copy(), case["k"], ncv, 1e-10, args.budget
                        )
                    else:
                        dtype = complex if backend == "arpack-complex" else float
                        try:
                            w, v = eigs(
                                a.astype(dtype),
                                k=case["k"],
                                ncv=ncv,
                                v0=v0.astype(dtype),
                                which="LM",
                                tol=1e-10,
                                maxiter=args.budget,
                            )
                            metadata = dict(converged=len(w), info=0)
                        except ArpackNoConvergence as error:
                            w, v = error.eigenvalues, error.eigenvectors
                            metadata = dict(converged=len(w), info=1, exception=str(error))
                    row.update(
                        metadata=metadata,
                        values=np.stack([w.real, w.imag], axis=-1).tolist(),
                        accuracy=reference.check(w, v),
                    )
                    row["accepted"] = bool(
                        row["accuracy"]["accepted"]
                        and metadata["info"] == 0
                        and metadata["converged"] == case["k"]
                    )
                except Exception as error:
                    row.update(accepted=False, exception=repr(error))
                report["cases"].append(row)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    case["id"], profile, backend, "PASS" if row["accepted"] else "FAIL", flush=True
                )
    report["summary"] = {
        f"{profile}/{backend}": dict(
            passed=sum(
                row["accepted"]
                for row in report["cases"]
                if row["profile"] == profile and row["backend"] == backend
            ),
            total=sum(
                row["profile"] == profile and row["backend"] == backend for row in report["cases"]
            ),
        )
        for profile in args.profiles
        for backend in args.backends
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
