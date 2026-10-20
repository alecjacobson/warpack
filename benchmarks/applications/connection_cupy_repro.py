"""Native-only CuPy complex LOBPCG helper diagnostics; no Warp/cuDSS imports.

The optional corrected-helper experiment replaces functions in this process
only, never edits the installed CuPy package or labels corrected timings native.
"""

import argparse
import inspect
import json
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cs
import cupyx.scipy.sparse.linalg as cl
import cupyx.scipy.sparse.linalg._lobpcg as implementation
import numpy as np

from .connection import make_connection_case


def correct_helpers():
    """Diagnostic only: alter helpers in the current process, never on disk."""
    for name in ("_b_orthonormalize", "_eigh"):
        source = inspect.getsource(getattr(implementation, name))
        updated = source.replace("inv(VBV.T)", "inv(VBV.T.conj())").replace(
            "inv(R.T)", "inv(R.T.conj())"
        )
        if updated == source:
            raise RuntimeError("CuPy helpers differ from the diagnosed implementation")
        exec(compile(updated, "<diagnostic-conjugate-transpose>", "exec"), implementation.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--correct-helpers", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(119)
    z = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    b = cp.asarray(z.conj().T @ z + np.eye(12))
    z = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    a = cp.asarray(z + z.conj().T)
    x = cp.asarray(rng.normal(size=(12, 3)) + 1j * rng.normal(size=(12, 3)))
    if args.correct_helpers:
        correct_helpers()
    q, _ = implementation._b_orthonormalize(lambda v: b @ v, x.copy())
    w, v = implementation._eigh(a, b)
    report = dict(
        cupy=cp.__version__,
        corrected_helpers=args.correct_helpers,
        orthogonalization_error=float(cp.max(cp.abs(q.conj().T @ b @ q - cp.eye(3))).get()),
        generalized_residual=float(
            (cp.linalg.norm(a @ v - (b @ v) * w) / cp.linalg.norm(a @ v)).get()
        ),
        generalized_orthogonality=float(cp.max(cp.abs(v.conj().T @ b @ v - cp.eye(12))).get()),
        cases=[],
    )
    for kind in ("identity", "lumped", "consistent"):
        p = make_connection_case(12, 6, kind).problem
        a = cs.csr_matrix(p.a)
        b = None if p.b is None else cs.csr_matrix(p.b)
        x = cp.asarray(rng.normal(size=(p.n, p.k)) + 1j * rng.normal(size=(p.n, p.k)))
        row = dict(mass=kind)
        try:
            w, v = cl.lobpcg(a, x, B=b, largest=False, tol=1e-9, maxiter=300)
            wh, vh = w.get(), v.get()
            av = p.a @ vh
            bv = vh if p.b is None else p.b @ vh
            row.update(
                residual=float(
                    np.max(
                        np.linalg.norm(av - bv * wh, axis=0)
                        / np.maximum(
                            1, np.linalg.norm(av, axis=0) + abs(wh) * np.linalg.norm(bv, axis=0)
                        )
                    )
                ),
                orthogonality=float(np.max(abs(vh.conj().T @ bv - np.eye(p.k)))),
                spectrum_error=float(np.max(abs(np.sort(wh) - p.expected))),
                outcome="returned",
            )
        except Exception as error:
            row.update(outcome="error", error=repr(error))
        report["cases"].append(row)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
