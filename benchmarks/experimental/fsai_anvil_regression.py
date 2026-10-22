"""Compare isolated legacy/current FSAI Anvil workers with accuracy and timing gates."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from benchmarks.anvil_inputs import signature as input_signature
from benchmarks.anvil_showcase import command, extract
from benchmarks.dependency_provenance import signature
from scripts.install_fsai_overlay import PROFILES


def summarize(directory, rounds=3, repeats=3):
    summaries, source, config, common = {}, None, None, None
    inputs = None
    for profile, installer_profile in (("legacy", "v0.2.0"), ("fixed", "current")):
        rows, identity = [], None
        for trial in range(1, rounds + 1):
            path = directory / f"{profile}-{trial}.json"
            raw = json.loads(path.read_text())
            current_inputs = input_signature(raw.get("inputs"))
            if inputs is not None and inputs != current_inputs:
                raise ValueError(f"Changed Anvil inputs: {path}")
            inputs = current_inputs
            occupancy = json.loads(path.with_suffix(".occupancy.json").read_text())
            setup, samples, residual, accepted = extract("warp-block", raw)
            if not accepted or not occupancy["timing_eligible"] or len(samples) != repeats:
                raise ValueError(f"Unqualified measurement: {path}")
            current = signature(raw.get("dependency"))
            fsai = current["files"]["warp/_src/optim/fsai.py"]
            if fsai != PROFILES[installer_profile][1] or (identity and current != identity):
                raise ValueError(f"Unexpected dependency profile: {path}")
            identity = current
            other = current | {
                "files": {k: v for k, v in current["files"].items() if not k.endswith("fsai.py")}
            }
            settings = {k: v for k, v in raw["args"].items() if k != "out"}
            if (
                (source and source != raw["source_sha256"])
                or (config and config != settings)
                or (common and common != other)
            ):
                raise ValueError(f"Unmatched implementation/configuration: {path}")
            source, config, common = raw["source_sha256"], settings, other
            rows.append(
                dict(
                    path=str(path),
                    setup_and_capture_seconds=setup,
                    fsai_setup_seconds=raw["preconditioner_setup_seconds"],
                    warm_median_seconds=statistics.median(samples),
                    max_original_residual=residual,
                    setup_capture_memory_bytes=raw[
                        "setup_capture_device_free_memory_decrease_bytes"
                    ],
                )
            )
        summaries[profile] = dict(
            dependency=identity,
            workers=rows,
            **{
                name: statistics.median(r[name] for r in rows)
                for name in (
                    "setup_and_capture_seconds",
                    "fsai_setup_seconds",
                    "warm_median_seconds",
                    "setup_capture_memory_bytes",
                )
            },
        )
    old, new = summaries["legacy"], summaries["fixed"]
    ratios = {
        name: new[name] / old[name]
        for name in (
            "setup_and_capture_seconds",
            "fsai_setup_seconds",
            "warm_median_seconds",
            "setup_capture_memory_bytes",
        )
    }
    accepted = (
        ratios["setup_and_capture_seconds"] <= 1.10
        and ratios["warm_median_seconds"] <= 1.10
        and ratios["setup_capture_memory_bytes"] < 1.0
    )
    return dict(
        inputs=inputs,
        accepted=accepted,
        profiles=summaries,
        candidate_over_legacy=ratios,
        gates="All accuracy and occupancy gates, matched code/configuration/common dependency; median setup+capture and warm replay no more than 10% slower, lower setup/capture device-memory decrease.",
        memory_scope="Device/context free-memory decrease, not peak or live factor bytes.",
        source_sha256=source,
        arguments=config,
        rounds=rounds,
        repeats=repeats,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-overlay", type=Path)
    parser.add_argument("--candidate-overlay", type=Path)
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 3 or args.repeats < 3:
        parser.error("at least three independent workers and three repeats are required")
    if not args.summarize_only:
        if not args.legacy_overlay or not args.candidate_overlay:
            parser.error("both overlay targets are required")
        args.out.mkdir(parents=True, exist_ok=True)
        for trial in range(1, args.rounds + 1):
            for profile, overlay in (
                ("legacy", args.legacy_overlay),
                ("fixed", args.candidate_overlay),
            ):
                target = args.out / f"{profile}-{trial}.json"
                if target.exists():
                    raise ValueError(f"Refusing to overwrite evidence: {target}")
                env = os.environ | {
                    "PYTHONPATH": str(overlay.resolve()) + os.pathsep + str(Path.cwd())
                }
                affinity = ["taskset", "-c", args.cpu_affinity] if args.cpu_affinity else []
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.monitor_gpu",
                        "--out",
                        str(target.with_suffix(".occupancy.json")),
                        "--log",
                        str(target.with_suffix(".log")),
                        "--",
                        *affinity,
                        sys.executable,
                        "-m",
                        *command("warp-block", args.repeats, target),
                    ],
                    env=env,
                    check=True,
                )
    report = summarize(args.out, args.rounds, args.repeats)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["accepted"]:
        raise SystemExit("FSAI Anvil performance regression gate failed")
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
