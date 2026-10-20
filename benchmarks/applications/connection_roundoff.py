"""Host-only higher-precision reference for connection linear-residual floors.

SuperLU supplies FP64 correction solves; residuals and reference accumulation
use NumPy extended precision. No production solver imports this diagnostic.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as sla

from .connection import make_connection_case


def norm(x):
    return np.sqrt(np.sum(x.real * x.real + x.imag * x.imag, dtype=np.longdouble))


def run(side):
    case = make_connection_case(side, 6)
    p = case.problem
    a = (p.a - p.target * p.b).tocsc()
    factor = sla.splu(a)
    extended = a.astype(np.clongdouble).tocsr()
    smooth = p.b @ p.requested_basis[:, 0]
    rng = np.random.default_rng(215)
    random = rng.normal(size=p.n) + 1j * rng.normal(size=p.n)
    frequencies = np.indices((side, side)).reshape(2, -1).T
    phase = (2 * np.pi * frequencies + np.asarray(case.angles)) / side
    symbol = 4 - 2 * np.cos(phase[:, 0]) - 2 * np.cos(phase[:, 1]) - p.target * case.mass_symbol
    ratio = abs(case.gauge).max() ** 2 / abs(case.gauge).min() ** 2
    rows = []
    for name, source in (("lowest Fourier mass image", smooth), ("random", random)):
        for scale in (1e-6, 1.0, 1e6):
            b = np.asarray(scale * source, dtype=np.complex128)
            be = b.astype(np.clongdouble)
            x = factor.solve(b).astype(np.clongdouble)
            initial = float(norm(extended @ x - be) / norm(be))
            history = []
            for _ in range(6):
                residual = be - extended @ x
                correction = factor.solve(residual.astype(np.complex128))
                x += correction.astype(np.clongdouble)
                history.append(float(norm(extended @ x - be) / norm(be)))
            rounded = x.astype(np.complex128)
            rows.append(
                dict(
                    rhs=name,
                    scale=scale,
                    initial_factor_relative_residual=initial,
                    extended_reference_residuals=history,
                    extended_reference_accepted=bool(
                        np.isfinite(history[-1]) and history[-1] < 1e-13
                    ),
                    rounded_fp64_extended_residual=float(
                        norm(extended @ rounded.astype(np.clongdouble) - be) / norm(be)
                    ),
                    rounded_fp64_native_residual=float(
                        np.linalg.norm(a @ rounded - b) / np.linalg.norm(b)
                    ),
                )
            )
    return dict(
        n=p.n,
        side=side,
        shift=p.target,
        ideal_condition_lower_bound=float(symbol.max() / symbol.min() / ratio),
        ideal_condition_upper_bound=float(symbol.max() / symbol.min() * ratio),
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", default="64,128")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        raise RuntimeError("this diagnostic needs extended precision beyond FP64")
    report = dict(
        float64_fraction_bits=np.finfo(np.float64).nmant,
        extended_fraction_bits=np.finfo(np.longdouble).nmant,
        note="Condition bounds are for the ideal Fourier/congruence construction. Residuals use the actual stored FP64 matrix and RHS. Extended accumulation with six SuperLU correction solves supplies a reference, not an exact arithmetic proof or a lower bound over every representable FP64 vector. No GPU code is imported or timed.",
        cases=[],
    )
    for side in map(int, args.sides.split(",")):
        report["cases"].append(run(side))
    report["reference_accepted"] = all(
        row["extended_reference_accepted"] for case in report["cases"] for row in case["rows"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["reference_accepted"]:
        raise SystemExit("Extended reference did not meet its declared residual gate")


if __name__ == "__main__":
    main()
