"""Unchanged-value overhead control for the complex GMRES replay repair."""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import complex_matrix
from warpack import ComplexShiftInverse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    n = 20000
    sigma = 0.2 + 0.1j
    host = sp.diags(
        [np.full(n - 1, -0.2), np.linspace(2.0, 3.0, n), np.full(n - 1, -0.2)],
        [-1, 0, 1],
        format="csr",
    ).astype(complex)
    op = complex_matrix(host)
    rng = np.random.default_rng(731)
    b = rng.normal(size=n) + 1j * rng.normal(size=n)
    rhs = wp.array(np.r_[b.real, b.imag].reshape(1, -1), dtype=wp.float64)
    x = wp.empty_like(rhs)
    wp.synchronize()
    start = time.perf_counter()
    inverse = ComplexShiftInverse(op, sigma, backend="gmres", inner_tol=1e-12, inner_maxiter=256)
    wp.synchronize()
    setup = time.perf_counter() - start
    inverse.apply(rhs, x)
    wp.synchronize()
    start = time.perf_counter()
    with wp.ScopedCapture() as cap:
        inverse.apply(rhs, x)
    wp.synchronize()
    capture = time.perf_counter() - start
    wp.capture_launch(cap.graph)
    wp.synchronize()
    trials = []
    shifted = host - sigma * sp.eye(n, format="csr")
    for _ in range(3):
        start = time.perf_counter()
        for j in range(20):
            wp.capture_launch(cap.graph)
        wp.synchronize()
        seconds = (time.perf_counter() - start) / 20
        raw = x.numpy()[0]
        sol = raw[:n] + 1j * raw[n:]
        error = float(np.linalg.norm(shifted @ sol - b) / np.linalg.norm(b))
        trials.append(
            dict(seconds=seconds, relative_original_residual=error, accepted=error < 2e-12)
        )
    destination = Path(args.out)
    dot = destination.with_suffix(".dot")
    wp.capture_debug_dot_print(cap.graph, str(dot))
    audit = graph_summary(dot)
    dot.unlink()
    source = Path(inspect.getsourcefile(ComplexShiftInverse))
    report = dict(
        n=n,
        nnz=host.nnz,
        realified_n=2 * n,
        realified_nnz=int(inverse.matrix.offsets.numpy()[-1]),
        sigma=[sigma.real, sigma.imag],
        setup_seconds=setup,
        capture_seconds=capture,
        trials=trials,
        median_seconds=float(np.median([r["seconds"] for r in trials])),
        warp=wp.__version__,
        gpu=wp.get_device().name,
        source_file=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        graph=audit,
        protocol="Unchanged-value inverse-application overhead control, not an eigenproblem or changed-value correctness test. Three samples average 20 warm graph replays; one RHS, zero-start upstream GMRES. Original complex relative residual checked independently each sample. The initially real source exercises retained imaginary structural zeros in the repaired implementation.",
    )
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(report["median_seconds"], trials, flush=True)


if __name__ == "__main__":
    main()
