"""Isolated serial/packed/global factor controls for #155, including workspace cost."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.applications.compare import timed
from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.global_cholesky import global_cholesky
from benchmarks.source_provenance import source_manifest
from warpack import kernels as K
from warpack._projected_cholesky import packed_cholesky_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="24,48,104,128,204,256,512")
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sizes = list(map(int, args.sizes.split(",")))
    if min(args.batch, args.repeats) < 1 or any(not 1 <= n <= 512 for n in sizes):
        parser.error("positive repeats/batch and dimensions1..512 required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for n in sizes:
        rng = np.random.default_rng(1543)
        x = rng.normal(size=(n, n))
        original = x @ x.T + np.eye(n)
        g = wp.array(original, dtype=wp.float64)
        factor = wp.zeros_like(g)
        status = wp.zeros(1, dtype=wp.int32)
        work = wp.empty_like(g)
        for backend in ("serial", "packed", "global"):
            if backend == "packed" and (
                n > 128
                or 8 * (n * (n + 1) // 2 + 1) + 1024 > wp.get_device().max_shared_memory_per_block
            ):
                skipped.append(dict(n=n, backend=backend, reason="packed size/shared-memory guard"))
                continue

            def run():
                status.zero_()
                if backend == "serial":
                    wp.launch(K.cholesky, 1, [g, factor, status])
                elif backend == "packed":
                    wp.launch_tiled(
                        packed_cholesky_kernel(n), dim=1, inputs=[g, factor, status], block_dim=128
                    )
                else:
                    wp.launch_tiled(
                        global_cholesky,
                        dim=1,
                        inputs=[g, factor, status, work],
                        block_dim=256 if n <= 256 else 512,
                    )

            run()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                for _ in range(args.batch):
                    run()
            wp.capture_launch(capture.graph)
            wp.synchronize()
            samples = []
            checks = []
            for _ in range(args.repeats):
                seconds, _ = timed(lambda: wp.capture_launch(capture.graph))
                lower = np.tril(factor.numpy())
                flag = int(status.numpy()[0])
                residual = float(
                    np.linalg.norm(lower @ lower.T - original) / np.linalg.norm(original)
                )
                check = dict(
                    status=flag, residual=residual, positive=bool(np.all(np.diag(lower) > 0))
                )
                check["accepted"] = bool(
                    flag == 0 and check["positive"] and np.isfinite(residual) and residual < 1e-12
                )
                samples.append(seconds / args.batch)
                checks.append(check)
            dot = args.out.with_suffix(f".{n}-{backend}.dot")
            wp.capture_debug_dot_print(capture.graph, str(dot))
            graph = graph_summary(dot)
            dot.unlink()
            assert not graph["node_types_including_conditional_body"].get("HOST", 0)
            assert all("DtoD" in kind for kind in graph["memcpy_kinds"])
            row = dict(
                n=n,
                backend=backend,
                seconds=samples,
                median_seconds=float(np.median(samples)),
                checks=checks,
                graph=graph,
                accepted=all(c["accepted"] for c in checks),
                extra_work_bytes=8 * n * n if backend == "global" else 0,
            )
            rows.append(row)
            print(n, backend, row["median_seconds"], row["accepted"], flush=True)
    report = dict(
        arguments=vars(args) | dict(out=str(args.out)),
        rows=rows,
        skipped=skipped,
        source_sha256=source_manifest(__file__),
        dependency=warp_dependencies(),
        accepted=all(r["accepted"] for r in rows),
        protocol="Isolated factorization; warm captured batches of repeated factorizations of identical SPD input, with batch and repeat counts recorded in arguments. Setup/capture excluded. Status, positive diagonal and independent reconstructed matrix equation checked after every batch. External occupancy is required; not an end-to-end eigen timing.",
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("Factor accuracy failure; evidence retained")


if __name__ == "__main__":
    main()
