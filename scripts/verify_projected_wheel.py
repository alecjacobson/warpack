"""Isolated installed-wheel wide projected replay with analytic spectra.

Run outside the checkout with the extracted wheel on PYTHONPATH. This exercises
private projected components in addition to the separate public-API release
smoke, and forbids benchmark/application/vendor eigensolver imports.
"""

import argparse
import json
import sys
from pathlib import Path

from verify_release_wheel import RejectNonCore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.meta_path.insert(0, RejectNonCore())
    import numpy as np
    import warp as wp

    import warpack
    from warpack._compact_jacobi import CompactJacobi
    from warpack.projected import GlobalJacobi

    assert Path(warpack.__file__).resolve().is_relative_to(args.installed.resolve())
    wp.init()
    rows = []
    for n in [256, 408]:
        original = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
        spectrum = 2 - 2 * np.cos(np.pi * np.arange(1, n + 1) / (n + 1))
        for constructor in [GlobalJacobi, CompactJacobi]:
            h, v, temp, vtemp = [wp.empty((n, n), dtype=wp.float64) for _ in range(4)]
            c, s = [wp.empty(n, dtype=wp.float64) for _ in range(2)]
            partners = wp.empty(n, dtype=wp.int32)
            kwargs = {"rounds_per_loop": 16} if constructor is CompactJacobi else {}
            eig = constructor(h, v, c, s, partners, temp, vtemp, **kwargs)
            assert eig._paired_update == (wp.get_device().arch == 89)
            h.assign(original)
            eig.apply()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                eig.apply()
            checks = []
            for scale in [1.0, 1e-80, 1e80, 1.0]:
                h.assign(original * scale)
                wp.capture_launch(capture.graph)
                values = np.diag(h.numpy()) / scale
                vectors = v.numpy()
                residual = float(
                    np.linalg.norm(original @ vectors - vectors * values) / np.linalg.norm(original)
                )
                orthogonality = float(np.linalg.norm(vectors.T @ vectors - np.eye(n)))
                error = float(np.max(abs(np.sort(values) - spectrum)))
                assert max(residual, orthogonality, error) < 1e-11
                checks.append(
                    dict(
                        scale=scale,
                        residual=residual,
                        orthogonality=orthogonality,
                        spectrum_error=error,
                        sweeps=int(eig.sweeps.numpy()[0]),
                    )
                )
            rows.append(
                dict(
                    n=n, constructor=constructor.__name__, paired=eig._paired_update, checks=checks
                )
            )
    report = dict(
        installed=str(args.installed.resolve()),
        warp=wp.__version__,
        accepted=True,
        rows=rows,
        forbidden_imports=["benchmarks", "examples", "scipy", "cupy", "warp_cudss"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
