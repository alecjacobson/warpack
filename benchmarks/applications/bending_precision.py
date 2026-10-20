"""Offline high-precision control for ill-conditioned Hermite bending (#130).

Optional analysis dependency: pip install mpmath. This module is not solver code.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la

from benchmarks.applications.fem_cases import bending_matrices


def residuals(a, b, x, values):
    av, bv = a @ x, b @ x
    return np.linalg.norm(av - bv * values, axis=0) / np.maximum(
        1, np.linalg.norm(av, axis=0) + abs(values) * np.linalg.norm(bv, axis=0)
    )


def quality(a, b, x, values):
    return dict(
        values=values.tolist(),
        fp64_residual=residuals(a, b, x, values).tolist(),
        extended_residual=[
            float(v)
            for v in residuals(
                a.astype(np.longdouble),
                b.astype(np.longdouble),
                x.astype(np.longdouble),
                values.astype(np.longdouble),
            )
        ],
    )


def main():
    import mpmath as mp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=24)
    parser.add_argument("--digits", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.elements < 2 or args.digits < 30:
        parser.error("require at least two elements and 30 decimal digits")
    a, b = (m.toarray() for m in bending_matrices(args.elements))
    values, x = la.eigh(a, b)
    scale = 1 / np.sqrt(np.diag(a))
    scaled_a, scaled_b = (scale[:, None] * m * scale[None, :] for m in (a, b))
    scaled_values, scaled_x = la.eigh(scaled_a, scaled_b)
    with mp.workdps(args.digits):
        am, bm = mp.matrix(a.tolist()), mp.matrix(b.tolist())
        lower_inverse = mp.cholesky(bm) ** -1
        precise_values, q = mp.eigsy(lower_inverse * am * lower_inverse.T)
        precise_vectors = lower_inverse.T * q
        rounded_x = np.array(
            [[float(precise_vectors[i, j]) for j in range(3)] for i in range(len(a))]
        )
        rounded_values = np.array([float(precise_values[j]) for j in range(3)])
    report = dict(
        args={**vars(args), "out": str(args.out)},
        n=len(a),
        condition_a=float(np.linalg.cond(a)),
        condition_b=float(np.linalg.cond(b)),
        lapack=quality(a, b, x[:, :3], values[:3]),
        diagonal_congruence_lapack=quality(
            a, b, scale[:, None] * scaled_x[:, :3], scaled_values[:3]
        ),
        high_precision_rounded_fp64=quality(a, b, rounded_x, rounded_values),
        matrix_sha256=hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest(),
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mpmath")},
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("benchmarks/applications/fem_cases.py"))
        },
        longdouble_mantissa_bits=np.finfo(np.longdouble).nmant,
        note="Host reference only, one bending polarization; two copies give the repeated fixture. High-precision solving uses the exact entries of the assembled FP64 matrices. Residuals are evaluated after rounding the reference eigenpairs to FP64. This is evidence of precision sensitivity, not a proof of the smallest attainable residual over all representable vectors.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
