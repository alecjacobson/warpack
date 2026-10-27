"""Independent paired-product checks and explicit rotation-tile screen (#186)."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.acceptance import errors_below_tolerance
from benchmarks.anvil_inputs import array_identity
from benchmarks.applications.compare import timed
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.paired_rotation_tiles import rotation
from benchmarks.graph_audit import graph_summary
from benchmarks.source_provenance import source_manifest
from warpack.block_products import paired_ritz_rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--shapes",
        default="34560:204:408,34560:204:120,34560:204:24,34560:204:8,34560:104:208,34560:104:8,283857:24:48,283857:24:28,70656:120:240",
    )
    args = parser.parse_args()
    if args.out.exists() or args.repeats < 1:
        parser.error("require a new output and positive repeat count")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    settings = [
        None,
        (4, 32, 8, 128),
        (4, 32, 32, 128),
        (8, 32, 8, 128),
        (8, 32, 32, 128),
        (16, 32, 8, 128),
        (16, 32, 32, 128),
        (8, 64, 32, 128),
        (16, 64, 32, 128),
        (16, 32, 32, 256),
    ]
    if args.reverse:
        settings.reverse()
    rows = []
    for shape in args.shapes.split(","):
        n, width, search = map(int, shape.split(":"))
        if min(n, width, search) < 1:
            parser.error("screen shapes require positive n:width:search")
        rng = np.random.default_rng(186)
        hx, hu = rng.normal(size=(width, n)), rng.normal(size=(search, n))
        hcx = rng.normal(size=(width, width)) / np.sqrt(width)
        hcu = rng.normal(size=(width, search)) / np.sqrt(search)
        expected_p = hcu @ hu
        expected_x = hcx @ hx + expected_p
        basis = wp.full((width + search + 1, n + 3), 17.0, dtype=wp.float64)
        coeff = wp.full((width + 1, width + search + 3), 17.0, dtype=wp.float64)
        sx, su = basis[:width, :n], basis[width : width + search, :n]
        cx, cu = coeff[:width, :width], coeff[:width, width : width + search]
        for dest, source in [(sx, hx), (su, hu), (cx, hcx), (cu, hcu)]:
            dest.assign(source)
        identity = {
            key: array_identity(value)
            for key, value in zip(["sx", "su", "cx", "cu"], [hx, hu, hcx, hcu], strict=True)
        }
        for setting in settings:
            stores = [wp.full((width + 1, n + 5), -17.0, dtype=wp.float64) for _ in range(2)]
            x, p = [store[:width, :n] for store in stores]
            backend = "original" if setting is None else ":".join(map(str, setting))

            def run():
                if setting is None:
                    wp.launch_tiled(
                        paired_ritz_rotation,
                        dim=((width + 3) // 4, (n + 31) // 32),
                        inputs=[sx, su, cx, cu, x, p],
                        block_dim=128,
                        device=x.device,
                    )
                else:
                    nr, nc, nk, threads = setting
                    rotation(sx, su, cx, cu, x, p, rows=nr, columns=nc, inner=nk, threads=threads)

            run()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                run()
            checks, times = [], []
            for trial in range(args.repeats + 1):
                seconds, _ = timed(lambda: wp.capture_launch(capture.graph))
                actual = [store.numpy() for store in stores]
                errors = [
                    float(np.max(abs(a[:width, :n] - e)) / np.max(abs(e)))
                    for a, e in zip(actual, [expected_x, expected_p], strict=True)
                ]
                tails = all(np.all(a[width:] == -17) and np.all(a[:, n:] == -17) for a in actual)
                hb, hc = basis.numpy(), coeff.numpy()
                unchanged = bool(
                    np.array_equal(hb[:width, :n], hx)
                    and np.array_equal(hb[width : width + search, :n], hu)
                    and np.array_equal(hc[:width, :width], hcx)
                    and np.array_equal(hc[:width, width : width + search], hcu)
                    and np.all(hb[width + search :] == 17)
                    and np.all(hb[:, n:] == 17)
                    and np.all(hc[width:] == 17)
                    and np.all(hc[:, width + search :] == 17)
                )
                checks.append(
                    dict(
                        x_error=errors[0],
                        p_error=errors[1],
                        tails_intact=bool(tails),
                        inputs_unchanged=unchanged,
                        accepted=bool(
                            errors_below_tolerance(errors, 5e-13) and tails and unchanged
                        ),
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
                protocol="Captured FP64 Xnew=Cx*X+Cu*U and Pnew=Cu*U. Every first and measured pair checked against independent CPU products, exact input preservation, view-boundary/padding sentinels and graph audit. Compilation, allocation, capture, reference products and validation excluded from replay timing. Same explicit inputs for every tile/thread variant; external occupancy required. Component timings do not establish whole-solve gains; no installed kernel or accuracy policy changed.",
            )
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(shape, backend, row["median_seconds"], row["accepted"], flush=True)
    if not report["accepted"]:
        raise SystemExit("Rejected paired-rotation result; evidence retained")


if __name__ == "__main__":
    main()
