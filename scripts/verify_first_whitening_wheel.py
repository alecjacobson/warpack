"""Installed public wide generalized solves with changing inputs, no reference packages."""

import argparse
import json
import sys
from pathlib import Path

from verify_release_wheel import RejectNonCore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--width", type=int, default=52)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--require-register-triangular", action="store_true")
    parser.add_argument("--gram-chunk", type=int, default=256)
    parser.add_argument("--require-wide-gram", action="store_true")
    parser.add_argument("--require-ritz-qr", action="store_true")
    args = parser.parse_args()
    sys.meta_path.insert(0, RejectNonCore())
    import numpy as np
    import warp as wp
    import warp.sparse as sparse

    import warpack

    assert Path(warpack.__file__).resolve().is_relative_to(args.installed.resolve())
    wp.init()
    n, width, k = args.size, args.width, args.modes
    indices = wp.array(list(range(n)), dtype=wp.int32, device="cuda:0")
    weights = 1.0 + np.arange(n) % 7 / 5.0
    spectrum = np.arange(n) // 2 + 1.0

    def diagonal(values):
        return sparse.bsr_from_triplets(
            n, n, indices, indices, wp.array(values, dtype=wp.float64, device="cuda:0")
        )

    matrix, mass, inverse = [
        diagonal(values) for values in (weights * spectrum, weights, 1 / (weights * (spectrum + 1)))
    ]
    basis = np.eye(n)[:2].copy()
    constraints = wp.array(basis, dtype=wp.float64, device="cuda:0")
    records = []
    for history, compact in [(True, False), (False, False), (False, True)]:
        solver = warpack.BlockLOBPCG(
            warpack.SparseOperator(matrix),
            k,
            width,
            mass=warpack.SparseOperator(mass),
            preconditioner=warpack.SparseOperator(inverse),
            constraints=constraints,
            normalize_constraints=True,
            tol=1e-10,
            symmetric_products=True,
            fused_updates=True,
            paired_rotation=True,
            reuse_operator_images=True,
            search_bucket_size=4,
            keep_history=history,
            compact_projected_graphs=compact,
            gram_chunk=args.gram_chunk,
        )
        graph = solver.capture(200)
        for scale in [1.0, 1.25, 1.0]:
            changed_mass = weights * (2.0 - scale)
            matrix.values.assign(scale * changed_mass * spectrum)
            mass.values.assign(changed_mass)
            inverse.values.assign(1 / (changed_mass * (scale * spectrum + 1)))
            wp.capture_launch(graph)
            values, vectors = solver.eigenvalues.numpy(), solver.eigenvectors.numpy().T
            assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
            np.testing.assert_allclose(values, scale * spectrum[2 : k + 2], atol=1e-9, rtol=0)
            orthogonality = np.linalg.norm(
                vectors.T @ (changed_mass[:, None] * vectors) - np.eye(k)
            )
            residual = np.linalg.norm(
                scale * changed_mass[:, None] * spectrum[:, None] * vectors
                - changed_mass[:, None] * vectors * values,
                axis=0,
            ).max()
            assert orthogonality < 1e-9 and residual < 1e-7
            assert np.linalg.norm(vectors[:2]) < 1e-10
            assert np.linalg.norm(vectors[k + 2 :]) < 1e-7
            usage = solver._first_whitening_usage
            qr_usage = solver._ritz_qr_usage
            if wp.get_device().arch == 89:
                assert usage is not None and usage.numpy()[0] > 0
            if args.require_ritz_qr:
                assert qr_usage is not None and qr_usage.numpy()[0] > 0
            records.append(
                dict(
                    keep_history=history,
                    compact_graphs=compact,
                    scale=scale,
                    max_original_residual=float(residual),
                    mass_orthogonality=float(orthogonality),
                    first_whitening_usage=None if usage is None else usage.numpy().tolist(),
                    ritz_qr_usage=None if qr_usage is None else qr_usage.numpy().tolist(),
                )
            )
        dot = args.installed.parent / f"first-whitening-{history}-{compact}.dot"
        wp.capture_debug_dot_print(graph, str(dot))
        assert not any(kind in dot.read_text() for kind in ("HOST", "HtoD", "DtoH"))
        if args.require_register_triangular:
            assert "register_triangular" in dot.read_text()
        if args.require_wide_gram:
            assert "wide_gram_kernel__locals__compute" in dot.read_text()
        if args.require_ritz_qr:
            assert "metric_defect" in dot.read_text() and "choose_qr" in dot.read_text()
        dot.unlink()
    report = dict(
        installed=str(args.installed.resolve()),
        size=n,
        width=width,
        modes=k,
        require_register_triangular=args.require_register_triangular,
        gram_chunk=args.gram_chunk,
        require_wide_gram=args.require_wide_gram,
        require_ritz_qr=args.require_ritz_qr,
        replays=records,
        forbidden_imports=["benchmarks", "examples", "cupy", "scipy", "warp_cudss"],
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
