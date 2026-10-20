"""Untimed fresh-setup/replay diagnostic for the Hodge strict-count profile."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import warp as wp

from benchmarks.applications.hodge import make_hodge_case
from benchmarks.applications.runner import accuracy, warp_output, warp_setup
from benchmarks.runtime import warp_runtime_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budgets", type=int, nargs="+", default=[3, 6])
    p.add_argument("--setups", type=int, default=12)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if min(args.budgets + [args.setups, args.repeats]) < 1:
        p.error("require positive budgets, setups and repeats")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        warp=wp.__version__,
        gpu=wp.get_device().name,
        runtime_hashes=warp_runtime_hashes("build/application_spectra.so"),
        source_sha256={
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("warpack/generalized_krylov.py", "warpack/solver.py")
        },
        note="No timing experiment. Unchanged n294/k6/ncv32, strict tol1e-11 and topology gates.",
        profiles=[],
    )
    for setup in range(args.setups):
        for budget in args.budgets:
            problem = make_hodge_case("large_disconnected", side=7).problem
            solver, inverse = warp_setup(problem, "krylov", ncv=32, force_krylov=True)
            graph = solver.capture(budget)
            rows = []
            try:
                for replay in range(args.repeats):
                    wp.capture_launch(graph)
                    check = accuracy(problem, *warp_output(problem, solver))
                    count, status = int(solver.converged.numpy()[0]), int(solver.status.numpy()[0])
                    rows.append(
                        dict(
                            replay=replay,
                            count=count,
                            status=status,
                            residuals=solver.residuals.numpy().tolist(),
                            accuracy=check,
                            accepted=count == problem.k and status == 0 and check["accepted"],
                        )
                    )
            finally:
                solver.graph = graph = None
                inverse.close()
            report["profiles"].append(dict(setup=setup, budget=budget, rows=rows))
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(setup, budget, [r["count"] for r in rows], flush=True)
    report["accepted"] = all(
        r["accepted"] for profile in report["profiles"] for r in profile["rows"]
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
