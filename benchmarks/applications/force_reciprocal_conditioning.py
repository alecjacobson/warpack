"""CPU diagnostics for explicit reciprocal force-covariance accuracy (#153).

This is a comparison study, not an installed solver or a performance benchmark.
Keep native ARPACK and adapter-corrected results separate. Every correction is
checked against both original covariance and reciprocal equations.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from benchmarks.source_provenance import source_manifest

from .fem_hierarchy import beam_problem, matrix_hash
from .force_covariance_compare import quality, reference
from .force_reciprocal_compare import ExplicitReciprocal, reciprocal_residual


def covariance_ritz(host, basis, *, inverse_step=False):
    """Recover covariance Ritz pairs, optionally after one implicit inverse step."""
    if inverse_step:
        basis = host.h_factor.solve(host.prior @ host.h_factor.solve(host.mass @ basis))
    metric = basis.T @ (host.mass @ basis)
    projected = basis.T @ host.apply(basis)
    values, rotation = la.eigh((projected + projected.T) * 0.5, (metric + metric.T) * 0.5)
    return values[::-1].copy(), basis @ rotation[:, ::-1]


def explicit_pairs(q, mass, k, *, equilibrate):
    """A separately labeled CPU control, preserving the physical starting vector."""
    scale = 1 / np.sqrt(q.diagonal()) if equilibrate else np.ones(q.shape[0])
    d = sp.diags(scale)
    a, metric = (d @ q @ d).tocsc(), d @ mass @ d
    factor = sla.splu(a)
    inverse = sla.LinearOperator(a.shape, matvec=factor.solve, dtype=np.float64)
    mu, vectors = sla.eigsh(
        a,
        M=metric,
        k=k,
        ncv=min(q.shape[0], max(64, 2 * k + 1)),
        sigma=0,
        which="LM",
        OPinv=inverse,
        tol=1e-12,
        maxiter=10000,
        v0=np.random.default_rng(915).normal(size=q.shape[0]) / scale,
    )
    order = np.argsort(mu)
    return 1 / mu[order], scale[:, None] * vectors[:, order]


def diagnose(side=10, modes=20, consistent=False):
    p = beam_problem(side, modes, consistent)
    diagonal = p.b.diagonal() * np.linspace(0.2, 2.0, p.n)
    host, ref = reference(p, sp.diags(diagonal), "diagonal")
    explicit = ExplicitReciprocal(p.a, p.b, diagonal)
    mu, basis = explicit.solve(p.k, max(64, 2 * p.k + 1), 1e-12)
    rayleigh = np.sum(basis * host.apply(basis), axis=0) / np.sum(basis * (p.b @ basis), axis=0)
    native_variants = [
        ("native_inverse_ritz", 1 / mu, basis),
        ("native_equilibrated_q", *explicit_pairs(explicit.q, p.b, p.k, equilibrate=True)),
    ]
    corrected_variants = [
        ("covariance_rayleigh_values_only", rayleigh, basis),
        ("covariance_ritz_rotation", *covariance_ritz(host, basis)),
        (
            "implicit_inverse_step_then_covariance_ritz",
            *covariance_ritz(host, basis, inverse_step=True),
        ),
    ]
    # Long double must actually provide additional mantissa bits on this host.
    # Re-evaluate the original FP64 input data; this cannot undo their rounding.
    wider = np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant
    if wider:
        h_wide, m_wide = p.a.astype(np.longdouble), p.b.astype(np.longdouble)
        inv_prior_wide = 1 / diagonal.astype(np.longdouble)
        q_wide = h_wide @ sp.diags(inv_prior_wide) @ h_wide
        for scaled in (False, True):
            native_variants.append(
                (
                    "native_wider_product_rounded_q" + ("_equilibrated" if scaled else ""),
                    *explicit_pairs(q_wide.astype(np.float64), p.b, p.k, equilibrate=scaled),
                )
            )
    rows = []
    for name, values, vectors in native_variants + corrected_variants:
        checks = quality(values, vectors, host, ref)
        checks["explicit_q_residual"] = reciprocal_residual(explicit.q, p.b, vectors, 1 / values)
        if wider:
            v = vectors.astype(np.longdouble)
            reciprocal = 1 / values.astype(np.longdouble)
            checks["wide_assembled_q_residual"] = reciprocal_residual(q_wide, m_wide, v, reciprocal)
            lhs = h_wide @ (inv_prior_wide[:, None] * (h_wide @ v))
            rhs = (m_wide @ v) * reciprocal
            checks["wide_implicit_q_residual"] = float(
                np.max(
                    np.linalg.norm(lhs - rhs, axis=0)
                    / np.maximum(1, np.linalg.norm(lhs, axis=0) + np.linalg.norm(rhs, axis=0))
                )
            )
        rows.append(
            dict(
                method=name,
                quality=checks,
                accepted=bool(
                    checks["accepted"]
                    and all(
                        np.isfinite(value) and 0 <= value < 1e-8
                        for key, value in checks.items()
                        if key.endswith("q_residual")
                    )
                ),
            )
        )
    return dict(
        n=p.n,
        k=p.k,
        matrix_sha256=matrix_hash(p),
        longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant),
        float64_mantissa_bits=int(np.finfo(np.float64).nmant),
        wider_residual_checks=wider,
        versions=dict(
            numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version()
        ),
        rows=rows,
        source_sha256=source_manifest(__file__),
        protocol="Single CPU accuracy diagnostic, no timing claim. Native explicit-Q ARPACK/SuperLU, diagonal equilibration and wider-product-then-FP64 controls, plus three explicitly labeled corrections. Physical gates remain 1e-8; extra-precision evaluation uses the same FP64 input coefficients, not a regenerated higher-precision FEM model.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=10)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--consistent", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.side, args.modes) < 1:
        parser.error("positive mesh size and mode count required")
    result = diagnose(args.side, args.modes, args.consistent)
    result["arguments"] = vars(args) | dict(out=str(args.out))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["rows"]:
        print(row["method"], row["accepted"], row["quality"], flush=True)


if __name__ == "__main__":
    main()
