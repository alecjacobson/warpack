"""Independent product checks and captured Gram tile screens for #180."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.anvil_inputs import array_identity
from benchmarks.applications.compare import timed
from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.gram_tiles import gram
from benchmarks.source_provenance import source_manifest
from warpack.block import gram as installed_gram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument(
        "--shapes",
        default="34560:204:204:s,34560:408:408:s,34560:612:612:s,34560:408:204:r,34560:120:204:r,34560:24:204:r,34560:8:204:r,70656:120:120:s,70656:240:240:s",
    )
    args = parser.parse_args()
    if args.out.exists() or args.repeats < 1 or args.chunk < 256 or args.chunk % 256:
        parser.error("require new output, positive repeats and chunk a positive multiple of256")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    settings = [(None, 128)] + [
        (tile, threads) for tile in (8, 16, 32) for threads in (64, 128, 256)
    ]
    if args.reverse:
        settings.reverse()
    for shape in args.shapes.split(","):
        n, nr, nc, kind = shape.split(":")
        n, nr, nc = int(n), int(nr), int(nc)
        symmetric = kind == "s"
        if kind not in ("s", "r") or (symmetric and nr != nc):
            parser.error("shape must be n:rows:columns:s|r with square symmetric output")
        rng = np.random.default_rng(180)
        hx = rng.normal(size=(nr, n))
        hy = hx * np.linspace(0.5, 2.0, n) if symmetric else rng.normal(size=(nc, n))
        expected = hx @ hy.T
        # Preserve non-contiguous views and sentinel columns in the timed screen.
        xstore = wp.full((nr, n + 3), 17.0, dtype=wp.float64)
        ystore = wp.full((nc, n + 3), 17.0, dtype=wp.float64)
        x, y = xstore[:, :n], ystore[:, :n]
        x.assign(hx)
        y.assign(hy)
        identity = dict(x=array_identity(hx), y=array_identity(hy))
        for tile, threads in settings:
            store = wp.full((nr + 1, nc + 3), -17.0, dtype=wp.float64)
            out = store[:nr, :nc]
            backend = "installed" if tile is None else f"tile{tile}-threads{threads}"

            def run():
                if tile is None:
                    installed_gram(x, y, out, chunk=args.chunk, symmetric=symmetric)
                else:
                    gram(
                        x, y, out, chunk=args.chunk, tile=tile, threads=threads, symmetric=symmetric
                    )

            run()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                run()
            checks, times = [], []
            for trial in range(args.repeats + 1):
                seconds, _ = timed(lambda: wp.capture_launch(capture.graph))
                actual = out.numpy()
                relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
                tails = bool(
                    np.all(store.numpy()[nr:] == -17) and np.all(store.numpy()[:, nc:] == -17)
                )
                unchanged = bool(
                    np.array_equal(x.numpy(), hx)
                    and np.array_equal(y.numpy(), hy)
                    and np.all(xstore.numpy()[:, n:] == 17)
                    and np.all(ystore.numpy()[:, n:] == 17)
                )
                checks.append(
                    dict(
                        relative_error=relative,
                        tails_intact=tails,
                        inputs_unchanged=unchanged,
                        accepted=bool(relative < 5e-13 and tails and unchanged),
                    )
                )
                if trial:
                    times.append(seconds)
                else:
                    first = seconds
            dot = args.out.with_suffix(".dot")
            wp.capture_debug_dot_print(capture.graph, str(dot))
            audit = graph_summary(dot)
            dot.unlink()
            assert not audit["node_types_including_conditional_body"].get("HOST")
            assert all("DtoD" in k for k in audit["memcpy_kinds"])
            row = dict(
                shape=shape,
                backend=backend,
                inputs=identity,
                checks=checks,
                first_seconds=first,
                seconds=times,
                median_seconds=float(np.median(times)),
                graph=audit,
                accepted=all(c["accepted"] for c in checks),
            )
            rows.append(row)
            report = dict(
                args=vars(args) | {"out": str(args.out)},
                rows=rows,
                accepted=all(r["accepted"] for r in rows),
                source_sha256=source_manifest(__file__),
                dependency=warp_dependencies(),
                protocol="Captured FP64 Gram including output zeroing; cached compilation, allocation, capture and independent CPU products excluded. First plus every timed output checked against CPU product, input/tail preservation and graph HOST/copy audit. Symmetric inputs are X and X times a positive diagonal metric. Strided views, fixed seed, identical inputs for every configuration. External occupancy required; component results do not establish full-solve gains. Rolled candidate loops avoid compiler tile storage expansion at short chunks. No accuracy threshold or installed algorithm change.",
            )
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(shape, backend, row["median_seconds"], row["accepted"], flush=True)
    if not report["accepted"]:
        raise SystemExit("Rejected Gram result; all evidence retained")


if __name__ == "__main__":
    main()
