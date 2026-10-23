"""Separate Jacobi column-norm drift from off-diagonal orthogonality loss."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp

from benchmarks.anvil_inputs import array_identity
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.normalized_jacobi import LegacyJacobi, NormalizedJacobi
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("new report path required")
    rows = []
    for n in [612, 1024]:
        a = np.random.default_rng(159).normal(size=(n, n))
        a = (a + a.T) * 0.5
        q = la.qr(a)[0]
        expected = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
        expected[: n // 3] = 0
        a = (q * expected) @ q.T
        reference = la.eigvalsh(a)
        source = wp.array(a, dtype=wp.float64)
        for name, constructor in [("legacy", LegacyJacobi), ("normalized", NormalizedJacobi)]:
            h, v, tmp, vt = [wp.empty_like(source) for _ in range(4)]
            cs, ss = [wp.empty(n, dtype=wp.float64) for _ in range(2)]
            eig = constructor(h, v, cs, ss, wp.empty(n, dtype=wp.int32), tmp, vt)
            wp.copy(h, source)
            eig.apply()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                wp.copy(h, source)
                eig.apply()
            for scale in [1.0, 1e-80, 1e80]:
                source.assign(a * scale)
                wp.capture_launch(capture.graph)
                matrix, vectors = h.numpy(), v.numpy()
                values = np.diag(matrix) / scale
                error = vectors.T @ vectors - np.eye(n)
                diagonal = np.diag(error).copy()
                off = error - np.diag(diagonal)
                extended_norms = np.sum(vectors.astype(np.longdouble) ** 2, axis=0) - 1
                record = dict(
                    n=n,
                    backend=name,
                    scale=scale,
                    input=array_identity(a * scale),
                    residual=float(la.norm(a @ vectors - vectors * values) / la.norm(a)),
                    spectrum_error=float(np.max(abs(np.sort(values) - reference)) / la.norm(a)),
                    orthogonality_frobenius=float(la.norm(error)),
                    column_norm_error_frobenius=float(la.norm(diagonal)),
                    column_norm_error_extended_frobenius=float(np.sqrt(np.sum(extended_norms**2))),
                    off_diagonal_orthogonality_frobenius=float(la.norm(off)),
                    orthogonality_max_entry=float(np.max(abs(error))),
                    sweeps=int(eig.sweeps.numpy()[0]),
                    final_max_off=float(np.max(abs(matrix - np.diag(np.diag(matrix))))),
                    stopping_bound=float(2e-15 * np.max(abs(np.diag(matrix)))),
                )
                record["accepted"] = (
                    all(
                        record[k] < 1e-11
                        for k in ["residual", "spectrum_error", "orthogonality_frobenius"]
                    )
                    and record["final_max_off"] <= record["stopping_bound"]
                )
                rows.append(record)
                print(
                    n,
                    name,
                    scale,
                    record["orthogonality_frobenius"],
                    record["column_norm_error_extended_frobenius"],
                    record["off_diagonal_orthogonality_frobenius"],
                    record["accepted"],
                    flush=True,
                )
    report = dict(
        rows=rows,
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(),
        normalized_accepted=all(r["accepted"] for r in rows if r["backend"] == "normalized"),
        protocol="Accuracy diagnostic only; no timing claim. Repeated/null fixture from the1024 regression. Default failures retained. Fixed-loop matrix/eigenvalues unchanged; only final vector normalization differs. Extended-precision independent diagonal norms distinguish norm drift from off-diagonal error; full FP64 equation/spectrum/orthogonality/native criteria checked.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["normalized_accepted"]:
        raise SystemExit("Normalized component rejected")


if __name__ == "__main__":
    main()
