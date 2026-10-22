"""Aggregate complete, uncontended shape-reuse runs and gate future regressions."""

import argparse
import json
import math
import statistics
from pathlib import Path

CONFIGURATIONS = [
    ("cold", "refit"),
    ("previous", "refit"),
    ("perturbed", "refit"),
    ("previous", "retain"),
]


def summarize(folder):
    folder = Path(folder)
    rows = []
    common = None
    hashes = {}
    for mass in ("consistent", "lumped"):
        for strategy, factor in CONFIGURATIONS:
            name = f"{mass}-{strategy}-{factor}"
            r = json.loads((folder / f"{name}.json").read_text())
            occupancy = json.loads((folder / f"{name}.occupancy.json").read_text())
            if not occupancy["timing_eligible"] or occupancy["exit_code"] != 0 or not r["accepted"]:
                raise ValueError(f"{name}: rejected or contended run")
            args = r["arguments"]
            if (args["mass"], args["strategy"], args["factor"]) != (mass, strategy, factor):
                raise ValueError(f"{name}: mislabeled configuration")
            signature = {key: r[key] for key in ("source_sha256", "warp", "versions", "n")}
            signature["sizes"] = {key: args[key] for key in ("side", "modes", "width", "repeats")}
            if common is not None and signature != common:
                raise ValueError(f"{name}: mixed source, runtime or problem configuration")
            common = signature
            if mass in hashes and hashes[mass] != r["matrix_sha256"]:
                raise ValueError(f"{name}: matrix mismatch")
            hashes[mass] = r["matrix_sha256"]
            if len(r["rows"]) != args["repeats"] or len(r["rows"]) < 3:
                raise ValueError(f"{name}: need at least three complete repeats")
            for sequence in [r["warmup"]] + r["rows"]:
                if [x["parameter"] for x in sequence] != [0.0, 0.12, -0.12, 0.0]:
                    raise ValueError(f"{name}: incomplete shape sequence")
                for step in sequence:
                    if any(
                        not math.isfinite(step[key]) or step[key] <= 0
                        for key in ("update_seconds", "solve_seconds")
                    ):
                        raise ValueError(f"{name}: invalid timings")
                    q = step["quality"]
                    if not q["accepted"] or q["status"] != 0 or q["converged"] != args["modes"]:
                        raise ValueError(f"{name}: failed numerical/native checks")
            changed = [x for seq in r["rows"] for x in seq[1:]]
            rows.append(
                dict(
                    name=name,
                    setup_capture_seconds=r["setup_capture_seconds"],
                    source_update_solve_seconds=statistics.median(
                        seq[0]["update_seconds"] + seq[0]["solve_seconds"] for seq in r["rows"]
                    ),
                    update_seconds=statistics.median(x["update_seconds"] for x in changed),
                    solve_seconds=statistics.median(x["solve_seconds"] for x in changed),
                    sequence_seconds=statistics.median(
                        sum(x["update_seconds"] + x["solve_seconds"] for x in seq[1:])
                        for seq in r["rows"]
                    ),
                    residual=max(x["quality"]["residual"] for x in changed),
                    subspace_error=max(x["quality"]["requested_subspace_error"] for x in changed),
                    iterations=[
                        min(x["quality"]["iterations"] for x in changed),
                        max(x["quality"]["iterations"] for x in changed),
                    ],
                )
            )
    return dict(
        signature=common,
        matrix_sha256=hashes,
        environment=json.loads((folder / "environment.json").read_text()),
        rows=rows,
    )


def compare(current, baseline, limit=1.25):
    if not math.isfinite(limit) or limit < 1:
        raise ValueError("regression limit must be finite and >=1")
    for field in ("signature", "matrix_sha256", "environment"):
        if current[field] != baseline[field]:
            raise ValueError("incompatible baseline " + field)
    previous = {r["name"]: r for r in baseline["rows"]}
    if set(previous) != {r["name"] for r in current["rows"]}:
        raise ValueError("incompatible baseline rows")
    ratios = {
        r["name"]: r["sequence_seconds"] / previous[r["name"]]["sequence_seconds"]
        for r in current["rows"]
    }
    if any(x > limit for x in ratios.values()):
        raise ValueError(f"update+solve regression above {limit}: {ratios}")
    return ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--regression-limit", type=float, default=1.25)
    args = parser.parse_args()
    report = summarize(args.folder)
    if args.baseline:
        report["baseline_ratios"] = compare(
            report, json.loads(args.baseline.read_text()), args.regression_limit
        )
    (args.folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for r in report["rows"]:
        print(
            r["name"],
            f"setup+capture {r['setup_capture_seconds']:.4f}s, update {1000 * r['update_seconds']:.2f}ms, solve {1000 * r['solve_seconds']:.2f}ms, changed-sequence {r['sequence_seconds']:.4f}s",
        )


if __name__ == "__main__":
    main()
