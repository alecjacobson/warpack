"""Run outside the checkout with only the extracted wheel on PYTHONPATH.

Exercises installed public APIs while forbidding imports of comparison,
application and optional direct-solver packages. Output is a JSON report.
"""

import argparse
import importlib
import importlib.abc
import importlib.metadata
import json
import pkgutil
import sys
from pathlib import Path


class RejectNonCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in {"benchmarks", "examples", "cupy", "scipy", "warp_cudss"}:
            raise ImportError(f"Release smoke forbids non-core import: {fullname}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--require-fast", action="store_true")
    args = parser.parse_args()
    sys.meta_path.insert(0, RejectNonCore())
    import warp as wp
    import warp.sparse as sparse
    from warp.optim import linear

    import warpack

    assert Path(warpack.__file__).resolve().is_relative_to(args.installed.resolve())
    assert importlib.metadata.version("warpack") == "0.2.0"
    modules = sorted(m.name for m in pkgutil.iter_modules(warpack.__path__))
    for name in modules:
        importlib.import_module(f"warpack.{name}")
    wp.init()
    n, width, k = 64, 6, 3
    indices = wp.array(list(range(n)), dtype=wp.int32, device="cuda:0")
    matrix = sparse.bsr_from_triplets(
        n, n, indices, indices, wp.array([float(i + 1) for i in range(n)], dtype=wp.float64)
    )
    fast = all(
        hasattr(linear.LinearOperator, attr) for attr in ("batch_offsets", "max_batch_length")
    ) and hasattr(linear, "FSAI")
    if args.require_fast and not fast:
        raise RuntimeError("Pinned batched-CG/FSAI environment required")
    kwargs = {}
    if fast:
        factor = linear.FSAI(matrix, max_row_size=8, factor_dtype=wp.float32)
        preconditioner = warpack.BatchedCGPreconditioner(
            matrix,
            width,
            tol=0.01,
            maxiter=25,
            preconditioner=factor,
            working_dtype=wp.float32,
            fsai_layout="columns",
            matrix_layout="columns",
        )
        kwargs.update(preconditioner=preconditioner, preconditioner_bucket_size=2)
    else:
        try:
            warpack.BatchedCGPreconditioner(matrix, width)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Unsupported upstream batched CG was not rejected")
    solver = warpack.BlockLOBPCG(
        warpack.BatchedSparseOperator(matrix),
        k,
        width,
        tol=1e-9,
        reuse_operator_images=True,
        search_bucket_size=2,
        keep_history=True,
        **kwargs,
    )
    graph = solver.capture(200)
    rows = []
    for scale in [1.0, 1.25, 1.0]:
        matrix.values.assign([scale * (i + 1) for i in range(n)])
        wp.capture_launch(graph)
        wp.synchronize()
        values = solver.eigenvalues.numpy().tolist()
        assert solver.status.numpy()[0] == 0 and solver.converged.numpy()[0] == k
        assert max(abs(v - scale * (i + 1)) for i, v in enumerate(values)) < 1e-7
        rows.append(
            dict(
                scale=scale, eigenvalues=values, max_residual=float(solver.residuals.numpy().max())
            )
        )
    report = dict(
        version=importlib.metadata.version("warpack"),
        installed=str(args.installed.resolve()),
        warp=wp.__version__,
        core_modules=["__init__", *modules],
        fast_path=fast,
        replays=rows,
        forbidden_imports=["benchmarks", "examples", "cupy", "scipy", "warp_cudss"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
