"""Diagnose first graph upload overtaking a pending pageable input copy (#61/#198).

No eigensolver is involved. Keep the sequence of sizes/allocations: an isolated
single-size reduction did not reproduce the observed failure. Incorrect outputs
are retained as diagnostics, not treated as a passing correctness test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def snapshot(x: wp.array[wp.float64], out: wp.array[wp.float64]):
    i = wp.tid()
    out[i] = x[i]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pending", "sync", "initialized"], default="pending")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[28320, 33176, 44040, 126276, 1048576]
    )
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.repeats < 1:
        parser.error("sizes and repeats must be positive")
    wp.init()
    failures = []
    for size in args.sizes:
        x = wp.zeros(size, dtype=wp.float64)
        y = wp.empty_like(x)
        status = wp.zeros(1, dtype=wp.int32)
        old, new = np.full(size, 1.25), np.full(size, 7.5)
        wp.launch(snapshot, size, [x, y])
        wp.synchronize()
        for trial in range(args.repeats):
            x.assign(old)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                status.zero_()
                wp.launch(snapshot, size, [x, y])
            if args.mode == "initialized":
                wp.capture_launch(capture.graph)
                wp.synchronize()
            x.assign(new)
            if args.mode == "sync":
                wp.synchronize()
            wp.capture_launch(capture.graph)
            result = y.numpy()
            if not np.array_equal(result, new):
                failures.append(
                    dict(
                        size=size,
                        trial=trial,
                        old_values=int((result == old).sum()),
                        new_values=int((result == new).sum()),
                        max_error=float(np.max(np.abs(result - new))),
                    )
                )
    report = dict(
        version=wp.__version__,
        mode=args.mode,
        sizes=args.sizes,
        repeats=args.repeats,
        failures=failures,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(failures)}/{len(args.sizes) * args.repeats} failures; report: {args.out}")


if __name__ == "__main__":
    main()
