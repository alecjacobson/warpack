"""Explicit, untimed convergence profile for small exported general SM cases."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import run_spectra as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=Path("build/general-sm-followup/fixtures"))
    parser.add_argument("--compact-cycles", type=int, default=200)
    parser.add_argument("--wide-cycles", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.compact_cycles, args.wide_cycles) < 1:
        parser.error("cycle budgets must be positive")
    runner.FIX = args.fixtures.resolve()
    cases = []
    for path in sorted(runner.FIX.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            case = json.loads(line)
            if case["family"] not in ("GenEigs", "ComplexEigs") or case["selection"] != 4:
                continue
            n = int(np.fromfile(runner.FIX / (case["id"] + ".a"), dtype=np.int64, count=2)[0])
            if n <= 100:
                cases.append(case)
    if not cases:
        parser.error("no small general SM fixtures found")
    report = dict(
        protocol="Accuracy only; matched cuDSS; original k/ncv/tolerances; explicit compact/wide budgets",
        compact_cycles=args.compact_cycles,
        wide_cycles=args.wide_cycles,
        cases=[],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        budget = args.compact_cycles if case["ncv"] <= 6 else args.wide_cycles
        result = runner.run(case, accuracy_only=True, iteration_budget=budget)
        report["cases"].append(dict(**case, **result))
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(case["id"], result["outcome"], result.get("max_absolute_residual"), flush=True)


if __name__ == "__main__":
    main()
