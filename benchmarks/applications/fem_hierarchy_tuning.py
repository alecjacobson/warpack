"""Separate coarse accuracy and start width, charging all hierarchy costs."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import warp as wp

from .fem_hierarchy import BeamTransfer, attach_reference, audit, beam_problem, matrix_hash
from .fem_shapes import ShapeReplay, elapsed

# Source accuracy is an input-quality knob, never a relaxation of the fine solve.
PROFILES = {
    "random": (False, None),
    "partial-strict": (False, 1e-11),
    "partial-loose": (False, 1e-3),
    "full-strict": (True, 1e-11),
    "full-loose": (True, 1e-3),
}


def source_usable(q, count, tolerance):
    """Coarse approximate pairs need native/physical checks, not fine accuracy."""
    return bool(
        q["status"] == 0
        and q["converged"] == count
        and math.isfinite(q["residual"])
        and 0 <= q["residual"] <= 1.01 * tolerance
        and math.isfinite(q["orthogonality"])
        and 0 <= q["orthogonality"] < 1e-8
        and all(math.isfinite(q[key]) for key in ("spectrum_error", "requested_subspace_error"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=int, default=5)
    parser.add_argument("--fine", type=int, default=15)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--mass", choices=("consistent", "lumped"), default="consistent")
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.coarse, args.fine, args.modes, args.repeats) <= 0
        or args.fine <= args.coarse
        or args.fine % args.coarse
        or args.width < args.modes
    ):
        parser.error("positive sizes, fine>coarse, integer refinement and width>=modes required")
    full, tolerance = PROFILES[args.profile]
    count = args.width if full else args.modes
    setup, problems = {}, {}
    levels = [("fine", args.fine, args.modes)]
    if tolerance is not None:
        levels.append(("coarse", args.coarse, count))
    for level, side, k in levels:
        begin = time.perf_counter()
        problems[level] = beam_problem(side, k, args.mass == "consistent")
        setup[level + "_assembly_seconds"] = time.perf_counter() - begin
    begin = time.perf_counter()
    for p in problems.values():
        attach_reference(p)
    reference_seconds = time.perf_counter() - begin
    fine_p = problems["fine"]
    begin = time.perf_counter()
    fine = ShapeReplay([(0, fine_p, 0)], args.width, refit=False, steps=None, start_rows=count)
    wp.synchronize()
    setup["fine_setup_seconds"] = time.perf_counter() - begin
    coarse = None
    transfer = None
    if tolerance is not None:
        begin = time.perf_counter()
        # Same padding as the baseline source solve, also for full-width transfer.
        coarse = ShapeReplay(
            [(0, problems["coarse"], 0)],
            count + args.width - args.modes,
            refit=False,
            tol=tolerance,
        )
        wp.synchronize()
        setup["coarse_setup_capture_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        transfer = BeamTransfer(args.coarse, args.fine, coarse.solver.eigenvectors, fine.starts)
        wp.synchronize()
        setup["transfer_setup_seconds"] = time.perf_counter() - begin
    begin = time.perf_counter()
    fine.graph = fine.solver.capture(200, initializer=transfer.apply if transfer else None)
    wp.synchronize()
    setup["fine_capture_seconds"] = time.perf_counter() - begin
    rows = []
    for trial in range(args.repeats + 1):
        if coarse:
            source_seconds = elapsed(coarse.solve)
        else:
            source_seconds = elapsed(lambda: wp.copy(fine.starts, fine.random))
        fine_seconds = elapsed(lambda: wp.capture_launch(fine.graph))
        q = fine.quality(fine_p)
        source_q = coarse.quality(problems["coarse"]) if coarse else None
        row = dict(
            source_seconds=source_seconds,
            fine_seconds=fine_seconds,
            total_seconds=source_seconds + fine_seconds,
            quality=q,
            source_quality=source_q,
            accepted=q["accepted"]
            and (source_q is None or source_usable(source_q, count, tolerance)),
        )
        print(trial, row, flush=True)
        if trial:
            rows.append(row)
        else:
            warmup = row
    args.out.parent.mkdir(parents=True, exist_ok=True)
    graphs = {"fine": audit(fine.graph, args.out.with_suffix(".fine.dot"))}
    if coarse:
        graphs["coarse"] = audit(coarse.graph, args.out.with_suffix(".coarse.dot"))
    files = [
        Path(__file__),
        Path(__file__).with_name("fem_hierarchy.py"),
        Path(__file__).with_name("fem_shapes.py"),
        Path(__file__).with_name("fem_cases.py"),
        Path("warpack/block.py"),
    ]
    report = dict(
        arguments=vars(args) | {"out": str(args.out)},
        setup=setup,
        reference_seconds=reference_seconds,
        source_tolerance=tolerance,
        source_count=count if coarse else None,
        warmup=warmup,
        rows=rows,
        graphs=graphs,
        n=fine_p.n,
        matrix_sha256={k: matrix_hash(p) for k, p in problems.items()},
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        warp=wp.__version__,
        accepted=all(r["accepted"] for r in [warmup] + rows),
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
