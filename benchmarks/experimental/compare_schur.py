"""Matched projected Schur replay comparison, with independent numerical gates.

Timing excludes setup, capture and CPU validation. Each replay resets its input;
the serial and tiled paths solve the same saved projection or random matrix.
This diagnoses a component, not complete sparse eigensolver performance.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp
from scipy.optimize import linear_sum_assignment

from benchmarks.runtime import warp_runtime_hashes
from benchmarks.upstream.adapters import output
from warpack.schur import schur
from warpack.schur_tiled import TiledSchur


def compare(a, repeats, batches):
    n = len(a)
    source = wp.array(np.stack([a.real, a.imag], axis=-1), dtype=wp.vec2d)
    h, q, v = (wp.empty_like(source) for _ in range(3))
    values, u, ss = (wp.empty(n, dtype=wp.vec2d) for _ in range(3))
    cs, status = wp.empty(n, dtype=wp.float64), wp.zeros(1, dtype=wp.int32)
    tiled = TiledSchur(h, q, v, values, u, cs, ss, status)
    reference = la.eigvals(a)
    norm = max(np.linalg.norm(a), np.finfo(float).tiny)
    result = dict(n=n, matrix_sha256=hashlib.sha256(a.tobytes()).hexdigest(), methods={})
    for name in ("serial", "tiled"):

        def run():
            wp.copy(h, source)
            if name == "tiled":
                tiled.run()
            else:
                wp.launch(schur, 1, [h, q, v, values, u, cs, ss, status, n * 100])

        run()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            run()
        wp.capture_launch(capture.graph)
        wp.synchronize()
        times = []
        trials = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(batches):
                wp.capture_launch(capture.graph)
            wp.synchronize()
            times.append((time.perf_counter() - start) / batches)
            hh, qq, vv, ww = (output(x) for x in (h, q, v, values))
            cost = abs(ww[:, None] - reference)
            rows, cols = linear_sum_assignment(cost)
            row = dict(
                status=int(status.numpy()[0]),
                similarity=float(np.linalg.norm(a @ qq - qq @ hh) / norm),
                orthogonality=float(np.linalg.norm(qq.conj().T @ qq - np.eye(n))),
                subdiagonal=float(np.linalg.norm(np.tril(hh, -1)) / norm),
                residual=float(np.max(np.linalg.norm(a @ vv - vv * ww, axis=0)) / norm),
                norm_error=float(abs(np.linalg.norm(vv, axis=0) - 1).max()),
                spectrum=float(cost[rows, cols].max() / norm),
            )
            row["accepted"] = row["status"] == 0 and all(
                value < (1e-10 if key == "spectrum" else 1e-11)
                for key, value in row.items()
                if key != "status"
            )
            trials.append(row)
        result["methods"][name] = dict(
            seconds=times,
            median_seconds=float(np.median(times)),
            trials=trials,
            accepted=all(t["accepted"] for t in trials),
        )
    result["accepted"] = all(m["accepted"] for m in result["methods"].values())
    result["tiled_over_serial"] = (
        result["methods"]["tiled"]["median_seconds"] / result["methods"]["serial"]["median_seconds"]
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--sizes", type=int, nargs="*", default=[6, 16, 32, 50, 64])
    args = p.parse_args()
    if args.repeats < 1 or args.batches < 1 or any(not 1 <= n <= 64 for n in args.sizes):
        p.error("require positive repeats/batches and sizes in 1..64")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    report = dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sha256={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("warpack/schur.py", "warpack/schur_tiled.py")
        },
        runtime_hashes=warp_runtime_hashes("build/application_spectra.so"),
        warp=wp.__version__,
        gpu=wp.get_device().name,
        repeats=args.repeats,
        batches=args.batches,
        timing_note="One warmup; synchronized replay batches; input reset per replay; setup/capture/reference excluded",
        cases={},
    )
    with np.load("tests/fixtures/general_projections.npz") as data:
        matrices = dict(data)
    rng = np.random.default_rng(83)
    matrices.update(
        {f"random{n}": rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) for n in args.sizes}
    )
    for case, matrix in matrices.items():
        report["cases"][case] = compare(matrix, args.repeats, args.batches)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(case, report["cases"][case], flush=True)
    report["accepted"] = all(c["accepted"] for c in report["cases"].values())
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("Projected numerical gate failed")


if __name__ == "__main__":
    main()
