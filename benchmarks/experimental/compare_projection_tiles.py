"""Matched-input captured projection subtraction screen; no solver-policy changes."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.acceptance import errors_below_tolerance
from benchmarks.anvil_inputs import array_identity
from benchmarks.applications.compare import timed
from benchmarks.audit_profile import graph_summary
from benchmarks.dependency_provenance import warp_dependencies
from benchmarks.experimental.projection_tiles import subtract
from benchmarks.source_provenance import source_manifest
from warpack.block_products import subtract_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--shapes",
        default="34560:408:204,34560:120:204,34560:24:204,34560:8:204,34560:208:104,34560:8:104,283857:48:24,283857:28:24",
    )
    args = parser.parse_args()
    if args.out.exists() or args.repeats < 1:
        parser.error("require a new output and positive repeat count")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    settings = [None, 4, 8, 16]
    if args.reverse:
        settings.reverse()
    rows = []
    for shape in args.shapes.split(","):
        n, width, basis = map(int, shape.split(":"))
        if min(n, width, basis) < 1:
            parser.error("require positive n:output:basis dimensions")
        rng = np.random.default_rng(192)
        hq = rng.normal(size=(basis, n))
        hc = rng.normal(size=(width, basis)) / np.sqrt(basis)
        initial = rng.normal(size=(width, n))
        expected = initial - hc @ hq
        qb = wp.full((basis + 1, n + 3), 17.0, dtype=wp.float64)
        cb = wp.full((width + 1, basis + 5), 17.0, dtype=wp.float64)
        q, coeff = qb[:basis, :n], cb[:width, :basis]
        q.assign(hq)
        coeff.assign(hc)
        initial_device = wp.array(initial, dtype=wp.float64)
        identity = {
            name: array_identity(value)
            for name, value in [("q", hq), ("coeff", hc), ("initial", initial)]
        }
        for setting in settings:
            ob = wp.full((width + 1, n + 7), -17.0, dtype=wp.float64)
            out = ob[:width, :n]

            def run():
                if setting is None:
                    wp.launch_tiled(
                        subtract_projection,
                        dim=((width + 3) // 4, (n + 31) // 32),
                        inputs=[q, coeff, out],
                        block_dim=128,
                        device=out.device,
                    )
                else:
                    subtract(q, coeff, out, rows=setting)

            run()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                run()
            checks, times = [], []
            for trial in range(args.repeats + 1):
                # Projection mutates its output; reset outside the timed region.
                wp.copy(out, initial_device)
                seconds, _ = timed(lambda: wp.capture_launch(capture.graph))
                actual = ob.numpy()
                error = float(np.max(abs(actual[:width, :n] - expected)) / np.max(abs(expected)))
                hqb, hcb = qb.numpy(), cb.numpy()
                intact = bool(
                    np.array_equal(hqb[:basis, :n], hq)
                    and np.array_equal(hcb[:width, :basis], hc)
                    and np.all(hqb[basis:] == 17)
                    and np.all(hqb[:, n:] == 17)
                    and np.all(hcb[width:] == 17)
                    and np.all(hcb[:, basis:] == 17)
                    and np.all(actual[width:] == -17)
                    and np.all(actual[:, n:] == -17)
                )
                checks.append(
                    dict(
                        error=error,
                        inputs_and_padding_intact=intact,
                        accepted=bool(intact and errors_below_tolerance([error], 5e-13)),
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
            assert not audit["node_types_including_conditional_body"].get("HOST", 0)
            assert all("DtoD" in key for key in audit["memcpy_kinds"])
            row = dict(
                shape=shape,
                output_rows=setting,
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
                args=vars(args) | dict(out=str(args.out)),
                rows=rows,
                accepted=all(r["accepted"] for r in rows),
                source_sha256=source_manifest(__file__),
                dependency=warp_dependencies(),
                protocol="Captured FP64 out <- out - coeff @ q, 32 output columns, inner tile 8, 128 threads. Original kernel and output-row variants share identical inputs. Reset/capture/compilation/reference/validation excluded. First and every measured output checked independently, including input/padding preservation. External GPU occupancy required. Component screen cannot establish whole-solve gains.",
            )
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(shape, setting, row["median_seconds"], row["accepted"], flush=True)
    if not report["accepted"]:
        raise SystemExit("Rejected subtraction result; evidence retained")


if __name__ == "__main__":
    main()
