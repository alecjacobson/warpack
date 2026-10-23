"""Isolated two-pass Ritz-basis orthogonalization controls for #173."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from benchmarks.anvil_inputs import array_identity, matrix_identity
from benchmarks.applications.compare import timed
from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.triangular_block_size import TriangularBlockSize
from benchmarks.experimental.triangular_rotate import TriangularRotateBlock
from benchmarks.source_provenance import source_manifest
from benchmarks.upstream.adapters import matrix
from warpack import BlockLOBPCG, SparseOperator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-sizes", action="store_true")
    parser.add_argument("--widths", default="104,204")
    parser.add_argument("--n", type=int, default=34560)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    widths = list(map(int, args.widths.split(",")))
    if args.out.exists() or args.repeats < 1 or any(b < 4 or 3 * b > args.n for b in widths):
        parser.error("new output, positive repeats and valid block widths required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = args.n
    diagonal = np.linspace(1, 4, n)
    a = sp.diags(diagonal, format="csr")
    mass = sp.diags(
        [-0.1 * np.ones(n - 1), np.linspace(1, 2, n), -0.1 * np.ones(n - 1)],
        [-1, 0, 1],
        format="csr",
    )
    op, metric = SparseOperator(matrix(a)), SparseOperator(matrix(mass))
    rows = []
    for b in widths:
        solvers = []
        backends = [("triangular", BlockLOBPCG), ("inverse_rotate", TriangularRotateBlock)]
        if args.block_sizes:
            backends = [("triangular256", BlockLOBPCG)] + [
                (f"triangular{size}", partial(TriangularBlockSize, triangular_block_dim=size))
                for size in [32, 64, 128, 512]
            ]
        for _, constructor in backends:
            solver = constructor(op, 4, b, mass=metric, reuse_operator_images=True)
            solver._image_reuse_active = True
            solvers.append(solver)
        random = np.random.default_rng(173).normal(size=(b, n))
        solvers[0].sx.assign(random)
        solvers[0]._image_x.assign(random * diagonal)
        solvers[0].orthogonalize_x()
        near = solvers[0].sx.numpy()
        for kind, host in [("random", random), ("near_orthogonal", near)]:
            source = wp.array(host, dtype=wp.float64)
            image = wp.array(host * diagonal, dtype=wp.float64)
            for (backend, _), solver in zip(backends, solvers):

                def run():
                    solver.status.zero_()
                    wp.copy(solver.sx, source)
                    wp.copy(solver._image_x, image)
                    solver.orthogonalize_x()

                run()
                wp.synchronize()
                with wp.ScopedCapture() as capture:
                    run()
                times, checks = [], []
                for trial in range(args.repeats + 1):
                    seconds, _ = timed(lambda: wp.capture_launch(capture.graph))
                    x = solver.sx.numpy()
                    ax = solver._image_x.numpy()
                    bx = solver.bx.numpy()
                    mx = (mass @ x.T).T
                    orthogonality = float(np.linalg.norm(x @ mx.T - np.eye(b)))
                    image_error = float(np.linalg.norm(ax - x * diagonal) / np.linalg.norm(ax))
                    mass_error = float(np.linalg.norm(bx - mx) / np.linalg.norm(mx))
                    status = int(solver.status.numpy()[0])
                    check = dict(
                        orthogonality=orthogonality,
                        image_error=image_error,
                        mass_error=mass_error,
                        status=status,
                    )
                    check["accepted"] = bool(
                        status == 0 and max(orthogonality, image_error, mass_error) < 1e-10
                    )
                    checks.append(check)
                    if trial:
                        times.append(seconds)
                    else:
                        first = seconds
                dot = args.out.with_suffix(f".{b}-{kind}-{backend}.dot")
                wp.capture_debug_dot_print(capture.graph, str(dot))
                audit = graph_summary(dot)
                dot.unlink()
                assert not audit["node_types_including_conditional_body"].get("HOST")
                assert all("DtoD" in k for k in audit["memcpy_kinds"])
                row = dict(
                    width=b,
                    kind=kind,
                    backend=backend,
                    input=array_identity(host),
                    first_seconds=first,
                    seconds=times,
                    median_seconds=float(np.median(times)),
                    checks=checks,
                    graph=audit,
                    accepted=all(c["accepted"] for c in checks),
                )
                rows.append(row)
                print(b, kind, backend, row["median_seconds"], row["accepted"], flush=True)
    report = dict(
        args=vars(args) | {"out": str(args.out)},
        rows=rows,
        accepted=all(r["accepted"] for r in rows),
        inputs={"operator": matrix_identity(a), "mass": matrix_identity(mass)},
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(),
        protocol="Synthetic sparse coupled SPD mass, fixed random and already orthonormal bases. Both complete CholeskyQR passes, cached AX transforms and final M X retained. Stage timing includes resetting X and AX from device copies. Allocation/module preparation/capture/host validation excluded. Fixed baseline-then-candidate order with cached compilation. Every first/measured output checks M-orthogonality, fresh AX and MX, and status; no whole-eigensolve speed claim. External occupancy required. Inverse candidate uses existing gx/bx scratch without adding dense buffers. With --block-sizes, only the existing triangular kernel thread grouping changes; scalar arithmetic and both passes remain identical.",
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("Rejected orthogonalization; evidence retained")


if __name__ == "__main__":
    main()
