"""Record independent captured replays, including the normally unreported warmup."""

import argparse
import json
import time
from pathlib import Path

import warp as wp

from benchmarks.upstream.adapters import complex_matrix, output
from benchmarks.upstream.run_spectra import FIX, read
from warpack import GeneralEigensolver, ShiftInvertEigensolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=["ComplexEigs_16", "GenEigsRealShift_12"], default="ComplexEigs_16"
    )
    parser.add_argument("--out", default="results/refinement_replay_probe.json")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    cases = [json.loads(line) for p in FIX.glob("*.jsonl") for line in p.read_text().splitlines()]
    c = next(c for c in cases if c["id"] == args.case)
    a = read(FIX / (c["id"] + ".a"))
    if c["family"] == "GenEigsRealShift":
        solver = ShiftInvertEigensolver(
            complex_matrix(a), c["k"], c["sigma_real"], ncv=c["ncv"], tol=1e-12
        )
        graph = solver.capture(30)
        source = "ShiftInvertEigensolver with default cuDSS refinement"
    else:
        assert c["selection"] == 5
        solver = GeneralEigensolver(complex_matrix(a), c["k"], ncv=c["ncv"], which="SR", tol=1e-12)
        graph = solver.capture(300)
        source = "Direct GeneralEigensolver; no inverse or cuDSS calls"
    report = dict(
        case=c,
        source=source,
        replays=[],
        gpu=wp.get_device().name,
        warp=wp.__version__,
        timing_note="Accuracy diagnostic including first replay; not an accuracy-gated performance comparison",
    )
    for i in range(args.repeats):
        wp.synchronize()
        start = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize()
        elapsed = time.perf_counter() - start
        values, vectors = output(solver.eigenvalues), output(solver.eigenvectors).T
        row = dict(
            replay=i,
            seconds=elapsed,
            count=int(solver.converged.numpy()[0]),
            max_absolute_residual=float(abs(a @ vectors - vectors * values).max()),
            residuals=solver.residuals.numpy().tolist(),
            iterations=int(getattr(solver, "core", solver).iterations.numpy()[0]),
        )
        report["replays"].append(row)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(row, flush=True)

    if hasattr(solver, "close"):
        solver.core.graph = solver.graph = graph = None
        solver.close()


if __name__ == "__main__":
    main()
