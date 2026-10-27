"""Isolated setup/capture/replay controls for experimental block Jacobi."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import scipy.linalg as la
import warp as wp

from benchmarks.anvil_inputs import array_identity
from benchmarks.applications.compare import timed
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.block_jacobi import BlockJacobi, BlockJacobiPolish
from benchmarks.graph_audit import graph_summary
from benchmarks.source_provenance import source_manifest
from warpack.projected import GlobalJacobi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="48,104,208,312,416,612")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--kind", choices=("indefinite", "rank_deficient"), default="indefinite")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sizes = list(map(int, args.sizes.split(",")))
    if args.repeats < 1 or any(n < 2 or n > 1024 or n % 2 for n in sizes):
        parser.error("positive repeats and even dimensions 2..1024 required")
    if args.out.exists():
        parser.error("refusing to overwrite an existing report")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sizes:
        random = np.random.default_rng(159).normal(size=(n, n))
        original = (random + random.T) * 0.5
        if args.kind == "rank_deficient":
            q, _ = la.qr(random)
            values = np.repeat(np.arange((n + 3) // 4), 4)[:n].astype(float)
            values[: n // 3] = 0
            original = (q * values) @ q.T
        reference = la.eigvalsh(original)
        source = wp.array(original, dtype=wp.float64)
        backends = [("global", GlobalJacobi)] + [
            (f"block{size}-local{local}", partial(BlockJacobi, block_size=size, local_sweeps=local))
            for size in (4, 8, 16)
            for local in (1, 30)
        ]
        if args.polish:
            backends = [("global", GlobalJacobi)] + [
                (f"block8-init{count}-polish", partial(BlockJacobiPolish, initializer_sweeps=count))
                for count in (1, 2, 4)
            ]
        for backend, constructor in backends:

            def setup():
                h, v, temp, vtemp = (wp.empty_like(source) for _ in range(4))
                cs, ss = (wp.empty(n, dtype=wp.float64) for _ in range(2))
                partners = wp.empty(n, dtype=wp.int32)
                return constructor(h, v, cs, ss, partners, temp, vtemp)

            setup_seconds, solver = timed(setup)
            h, v = solver.h, solver.v

            def run():
                wp.copy(h, source)
                solver.apply()

            eager_seconds, _ = timed(run)

            def capture_run():
                with wp.ScopedCapture() as capture:
                    run()
                return capture.graph

            capture_seconds, captured = timed(capture_run)
            samples, checks = [], []
            for trial in range(args.repeats + 1):
                seconds, _ = timed(lambda: wp.capture_launch(captured))
                values, vectors = np.diag(h.numpy()), v.numpy()
                norm = max(1, np.linalg.norm(original))
                check = dict(
                    residual=float(np.linalg.norm(original @ vectors - vectors * values) / norm),
                    orthogonality=float(np.linalg.norm(vectors.T @ vectors - np.eye(n))),
                    spectrum_error=float(np.max(abs(np.sort(values) - reference)) / norm),
                    sweeps=int(solver.sweeps.numpy()[0]),
                )
                check["accepted"] = bool(
                    all(
                        np.isfinite(check[k]) and check[k] < 1e-11
                        for k in ("residual", "orthogonality", "spectrum_error")
                    )
                    and 1 <= check["sweeps"] <= 30
                )
                if trial:
                    samples.append(seconds)
                    checks.append(check)
                else:
                    warmup = check
                    first_replay_seconds = seconds
            dot = args.out.with_suffix(f".{n}-{backend}.dot")
            wp.capture_debug_dot_print(captured, str(dot))
            graph = graph_summary(dot)
            dot.unlink()
            assert not graph["node_types_including_conditional_body"].get("HOST", 0)
            assert all("DtoD" in kind for kind in graph["memcpy_kinds"])
            row = dict(
                n=n,
                backend=backend,
                input=array_identity(original),
                setup_seconds=setup_seconds,
                eager_graph_preparation_and_solve_seconds=eager_seconds,
                outer_capture_seconds=capture_seconds,
                first_captured_replay_seconds=first_replay_seconds,
                seconds=samples,
                median_seconds=float(np.median(samples)),
                warmup=warmup,
                checks=checks,
                graph=graph,
                accepted=all(c["accepted"] for c in [warmup, *checks]),
            )
            rows.append(row)
            print(
                n,
                backend,
                "capture",
                capture_seconds,
                "replay",
                row["median_seconds"],
                row["accepted"],
                flush=True,
            )
    report = dict(
        arguments=vars(args) | dict(out=str(args.out)),
        rows=rows,
        accepted=all(r["accepted"] for r in rows),
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(),
        protocol="Isolated even-dimensional symmetric projected eigensolves (indefinite or rank-deficient family in arguments), including device input reset. Original equation, full orthogonality and independent SciPy spectrum checked after one warmup and every measured replay. Same matrix per dimension/backend, with exact identity. Setup includes scratch allocation/construction/module loading where encountered, after input upload. Eager graph preparation plus first solve, subsequent outer capture, first captured replay and warmed replay are reported separately. Same process, cached compilation, fixed global-then-block order; setup is not a cold-install comparison. No complete FEM speed claim. External occupancy required. Polished variants perform the labeled fixed count of block initializer sweeps and then bounded canonical FP64 refinement; sweeps counts refinement only. Production dispatch unchanged.",
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("Rejected projected eigensolve; evidence retained")


if __name__ == "__main__":
    main()
