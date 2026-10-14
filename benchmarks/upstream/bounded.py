"""Isolate GPU contexts so a timed-out case cannot stall the entire inventory."""

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["spectra", "benchmarks"])
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retry-timeouts", action="store_true")
    parser.add_argument("--retry-family", help="Rerun only this family, retaining other results")
    parser.add_argument("--selection", type=int, help="Restrict to this Spectra SortRule value")
    parser.add_argument("--case", help="Restrict to this exported case ID")
    parser.add_argument("--families", help="Comma-separated families for a focused regression run")
    parser.add_argument(
        "--out", type=Path, help="Separate output report; preserves published results"
    )
    parser.add_argument("--fresh", action="store_true", help="Replace the recorded suite results")
    args = parser.parse_args()
    if args.fresh and args.retry_timeouts:
        parser.error("--fresh and --retry-timeouts are mutually exclusive")
    benchmark = args.suite == "benchmarks"
    folder = ROOT / ("build/spectra-benchmark-fixtures" if benchmark else "build/spectra-fixtures")
    target = ROOT / (
        "results/upstream_benchmarks.json" if benchmark else "results/upstream_spectra_warp.json"
    )
    if args.out is not None:
        target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    module = (
        "benchmarks.upstream.run_benchmarks" if benchmark else "benchmarks.upstream.run_spectra"
    )
    cases = [
        json.loads(line)
        for p in sorted(folder.glob("*.jsonl"))
        for line in p.read_text().splitlines()
    ]
    if args.families:
        selected = set(args.families.split(","))
        unknown = selected - {c["family"] for c in cases}
        if unknown:
            parser.error("Unknown families: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["family"] in selected]
    if args.retry_family:
        cases = [c for c in cases if c["family"] == args.retry_family]
        if not cases:
            parser.error("Unknown family: " + args.retry_family)
    if args.selection is not None:
        cases = [c for c in cases if c["selection"] == args.selection]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            parser.error("No matching case: " + args.case)
    report = (
        json.loads(target.read_text()) if target.exists() and not args.fresh else dict(cases=[])
    )
    report["per_case_wall_limit_seconds"] = args.timeout
    report["timeout_scope"] = (
        "Whole worker including import/setup/capture/validation and all replays; subprocess terminated to release CUDA context. Completed earlier cases retained."
    )
    previous = {c["id"]: (i, c) for i, c in enumerate(report["cases"])}
    finished = {
        c["id"]
        for c in report["cases"]
        if not (args.retry_timeouts and c.get("outcome") == "timeout")
        and c["family"] != args.retry_family
    }
    logs = ROOT / "build/upstream-workers"
    logs.mkdir(exist_ok=True)
    pending = [c for c in cases if c["id"] not in finished]
    for attempt, c in enumerate(pending, 1):
        with (folder / (c["id"] + ".a")).open("rb") as matrix_file:
            c["n"] = struct.unpack("qq", matrix_file.read(16))[0]
        suffix = ""
        if args.retry_timeouts or args.retry_family:
            old = previous.get(c["id"], (None, {}))[1]
            number = len(old.get("previous_attempts", [])) + 2
            suffix = f"_attempt{number}_{args.timeout:g}s"
        output = logs / (c["id"] + suffix + ".json")
        log = logs / (c["id"] + suffix + ".log")
        output.unlink(missing_ok=True)
        start = time.perf_counter()
        with log.open("w") as f:
            try:
                p = subprocess.run(
                    [sys.executable, "-m", module, "--case", c["id"], "--out", str(output)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=os.environ,
                    timeout=args.timeout,
                )
                if output.exists():
                    worker = json.loads(output.read_text())
                    for k, v in worker.items():
                        if k != "cases":
                            report.setdefault(k, v)
                    row = worker["cases"][0]
                else:
                    row = dict(
                        **c,
                        outcome="error",
                        reason=f"Worker exited {p.returncode}",
                        log=str(log.relative_to(ROOT)),
                    )
            except subprocess.TimeoutExpired:
                row = dict(
                    **c,
                    outcome="timeout",
                    wall_seconds=time.perf_counter() - start,
                    reason=f"Worker exceeded {args.timeout:g}s wall limit",
                    log=str(log.relative_to(ROOT)),
                )
        row["worker_wall_limit_seconds"] = args.timeout
        if c["id"] in previous:
            index, old = previous[c["id"]]
            row["previous_attempts"] = old.get("previous_attempts", []) + [
                {k: v for k, v in old.items() if k != "previous_attempts"}
            ]
            report["cases"][index] = row
        else:
            report["cases"].append(row)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(
            attempt,
            len(pending),
            c["id"],
            row.get("outcome", [(r["backend"], r["outcome"]) for r in row.get("results", [])]),
            flush=True,
        )


if __name__ == "__main__":
    main()
