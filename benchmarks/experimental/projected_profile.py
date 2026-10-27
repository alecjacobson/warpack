"""Isolate Warp's projected Jacobi dispatch with current-equation validation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.applications.compare import other_gpu_processes, timed
from benchmarks.graph_audit import graph_summary
from benchmarks.upstream.adapters import matrix
from warpack import SparseOperator, SymmetricEigensolver
from warpack import kernels as K
from warpack.projected import GlobalJacobi, jacobi_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", default="48,64,72,96,128")
    p.add_argument("--backend", choices=["default", "shared", "global"], default="default")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    competing = set(other_gpu_processes())
    for n in map(int, args.sizes.split(",")):
        rng = np.random.default_rng(831)
        q = np.linalg.qr(rng.normal(size=(n, n)))[0]
        expected = np.geomspace(400, 0.0025, n)
        expected[:3] = 400
        expected[-2:] = 0
        a = (q * expected) @ q.T
        solver = SymmetricEigensolver(SparseOperator(matrix(a)), 20, ncv=n, which="LA", tol=1e-11)
        if args.backend != "default":
            if n % 2:
                p.error("explicit projected backends require even sizes")
            if args.backend == "shared":
                if n > 64 and not (n == 72 and solver.device.max_shared_memory_per_block >= 98304):
                    p.error("shared benchmark supports <=64, or 72 with sufficient shared memory")

                def diagonalize():
                    wp.launch_tiled(
                        jacobi_kernel(n),
                        dim=1,
                        inputs=[solver.g, solver.v],
                        block_dim=256,
                        device=solver.device,
                    )
                    wp.launch(
                        K.sort_diagonal,
                        n,
                        [solver.g, solver.ritz, solver.order, solver.which],
                        device=solver.device,
                    )
            else:
                solver._global_jacobi = GlobalJacobi(
                    solver.g,
                    solver.v,
                    solver.cs,
                    solver.ss,
                    solver.partners,
                    solver.jtmp,
                    solver.vtmp,
                )

                def diagonalize():
                    solver._global_jacobi.apply()
                    wp.launch(
                        K.sort_diagonal,
                        n,
                        [solver.g, solver.ritz, solver.order, solver.which],
                        device=solver.device,
                    )

            solver.diagonalize = diagonalize
        original = wp.array(a, dtype=wp.float64, device=solver.device)
        wp.copy(solver.g, original)
        solver.diagonalize()
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            wp.copy(solver.g, original)
            solver.diagonalize()
        wp.capture_launch(capture.graph)
        samples, checks = [], []
        for _ in range(args.repeats):
            competing.update(other_gpu_processes())
            elapsed, _ = timed(
                lambda: [wp.capture_launch(capture.graph) for _ in range(args.batch)]
            )
            competing.update(other_gpu_processes())
            v, values = solver.v.numpy(), solver.ritz.numpy()
            residual = float(np.linalg.norm(a @ v - v * values, axis=0).max() / 400)
            orthogonality = float(abs(v.T @ v - np.eye(n)).max())
            error = float(abs(np.sort(values) - np.sort(expected)).max() / 400)
            check = dict(
                residual=residual,
                orthogonality=orthogonality,
                spectrum_error=error,
                sweeps=(
                    int(solver._global_jacobi.sweeps.numpy()[0])
                    if solver._global_jacobi is not None and args.backend != "shared"
                    else None
                ),
            )
            check["accepted"] = residual < 1e-11 and orthogonality < 1e-11 and error < 1e-11
            samples.append(elapsed / args.batch)
            checks.append(check)
        dot = Path("build") / f"projected-{n}.dot"
        dot.parent.mkdir(exist_ok=True)
        wp.capture_debug_dot_print(capture.graph, str(dot))
        rows.append(
            dict(
                n=n,
                backend=args.backend,
                samples=samples,
                seconds=float(np.median(samples)),
                checks=checks,
                graph=graph_summary(dot),
                accepted=all(x["accepted"] for x in checks),
            )
        )
        dot.unlink()
        print(n, rows[-1]["seconds"], rows[-1]["accepted"], flush=True)
        capture.graph = None
    report = dict(
        args=vars(args),
        gpu=wp.get_device().name,
        warp=wp.__version__,
        rows=rows,
        competing_gpu_processes=sorted(competing),
        source_sha256={
            f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
            for f in ["warpack/solver.py", "warpack/projected.py", "warpack/kernels.py", __file__]
        },
        protocol="Isolated projected diagonalization; three warm samples of ten captured solves each. Restore identical input before each solve. Original dense equation, complete spectrum and orthogonality checked after every sample. Setup/capture/host reference excluded. No eigensolver end-to-end speedup claim.",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    if competing or not all(r["accepted"] for r in rows):
        raise SystemExit("Projected solve accuracy/occupancy failure")


if __name__ == "__main__":
    main()
