"""Accuracy-only control of existing cuDSS residual corrections for issue #123.

No kinematic reconstruction, quotient update or eigenvector polishing is used.
Only the fixed correction count inside each shifted sparse solve changes. The
correction products/updates already run in Warp; cuDSS only applies its factors.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from benchmarks.anvil_inputs import array_identity, matrix_identity
from benchmarks.applications.damped import quality
from benchmarks.applications.damped_backends import initial, warp_setup
from benchmarks.applications.damped_pencil import consistent_chain
from benchmarks.audit_profile import graph_summary
from benchmarks.cudss_provenance import cudss_dependencies
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.pencil_assembled_reference import assembled_reference
from benchmarks.source_provenance import source_manifest
from benchmarks.upstream.adapters import output
from warpack import shift


def extended_residual(case, values, vectors):
    """Evaluate the same assembled input in wider precision, outside the solver."""
    x = vectors.astype(np.clongdouble)
    lam = values.astype(np.clongdouble)
    ax = case.a.astype(np.clongdouble) @ x
    bx = case.b.astype(np.clongdouble) @ x
    residual = ax - bx * lam

    def norms(a):
        return np.sqrt(np.sum(a.real * a.real + a.imag * a.imag, axis=0))

    denominator = norms(ax) + np.maximum(1, np.abs(lam)) * norms(bx)
    half = case.a.shape[0] // 2
    return dict(
        residual=float(np.max(norms(residual) / denominator)),
        first_block=float(np.max(norms(residual[:half]) / denominator)),
        second_block=float(np.max(norms(residual[half:]) / denominator)),
        mantissa_bits=int(np.finfo(np.longdouble).nmant),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=16384)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--ncv", type=int, default=64)
    parser.add_argument(
        "--kind", choices=["nonproportional", "gyroscopic"], default="nonproportional"
    )
    parser.add_argument("--refinement-steps", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.refinement_steps < 0 or min(args.repeats, args.cycles) < 1:
        parser.error("require a new output, nonnegative correction count and positive budgets")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    case = consistent_chain(args.nodes, args.k, args.kind)
    stored_reference = assembled_reference(case)
    original = shift.CuDSSInverse

    def configured(*positional, **kwargs):
        kwargs["refinement_steps"] = args.refinement_steps
        return original(*positional, **kwargs)

    with patch.object(shift, "CuDSSInverse", configured):
        solver = warp_setup(case, args.ncv, args.tol)
    try:
        assert solver.inverse.inverse.refinement_steps == args.refinement_steps
        graph = solver.capture(args.cycles)
        rows = []
        for trial in range(args.repeats + 1):
            wp.capture_launch(graph)
            values, vectors = output(solver.eigenvalues), output(solver.eigenvectors).T
            check = quality(case, values, vectors)
            count = int(solver.converged.numpy()[0])
            status = int(solver.status.numpy()[0])
            rows.append(
                dict(
                    trial="first" if trial == 0 else trial,
                    count=count,
                    status=status,
                    native_residual=float(solver.residuals.numpy().max()),
                    quality=check,
                    quality_stored_coefficient_reference=quality(stored_reference, values, vectors),
                    extended=extended_residual(case, values, vectors),
                    accepted=bool(check["accepted"] and count == args.k and status == 0),
                )
            )
        dot = args.out.with_suffix(".dot")
        wp.capture_debug_dot_print(graph, str(dot))
        audit = graph_summary(dot)
        dot.unlink()
        assert not audit["node_types_including_conditional_body"].get("HOST", 0)
        report = dict(
            protocol=__doc__,
            args=vars(args) | {"out": str(args.out)},
            inputs=dict(
                operator=matrix_identity(case.a),
                mass=matrix_identity(case.b),
                start=array_identity(initial(case)),
                reference_values=array_identity(case.values),
                stored_coefficient_reference_values=array_identity(stored_reference.values),
            ),
            reference_coefficient_effect=quality(stored_reference, case.values, case.right),
            reference_scope=(
                "The original analytic fixture reference remains the acceptance gate. "
                "A separate constant-block modal reference uses the actual stored CSR "
                "coefficients and wider-precision symbol assembly. It diagnoses physical "
                "assembly roundoff without replacing the primary gate or native residual."
            ),
            source_sha256=source_manifest(__file__),
            dependency=warp_dependencies(),
            cudss_dependency=cudss_dependencies(),
            graph=audit,
            rows=rows,
            accepted=all(r["accepted"] for r in rows),
            timing_scope="No timings claimed; this is an accuracy diagnostic.",
        )
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"accepted": report["accepted"], "rows": rows}), flush=True)
    finally:
        solver.close()
    if not report["accepted"]:
        raise SystemExit("Strict original-pencil accuracy rejected; all evidence retained")


if __name__ == "__main__":
    main()
