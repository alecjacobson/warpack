"""Recheck issue #6 with explicit inverse-spectrum selection and basis profiles."""

import argparse
import json
from pathlib import Path

from . import run_spectra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("build/shifted-selection/fixtures"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_spectra.FIX = args.fixtures.resolve()
    cases = {
        row["id"]: row
        for row in map(
            json.loads, (run_spectra.FIX / "SymEigsShift.jsonl").read_text().splitlines()
        )
    }
    profiles = [
        ("SymEigsShift_2", None),
        ("SymEigsShift_2", 8),
        ("SymEigsShift_7", None),
        ("SymEigsShift_7", 48),
        ("SymEigsShift_3", None),
        ("SymEigsShift_14", None),
    ]
    report = dict(
        spectra_revision="db1d5cc3279752ca7ea3e33da44ba2a85e4e4a95",
        scope="Accuracy-only: original compact controls, explicit wider SM profiles, unchanged required-success SA/BE cases. Two replays, final result validated; no timing claim.",
        selection="LM/SM/LA/SA/BE on 1/(lambda-sigma); compare physical spectrum, residual, orthogonality, invariant subspace and inverse counts.",
        cases=[],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for key, width in profiles:
        row = dict(
            **cases[key],
            **run_spectra.run(cases[key], accuracy_only=True, ncv_override=width),
            profile="original" if width is None else "wider_basis",
        )
        report["cases"].append(row)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(
            key,
            row.get("requested_ncv", width),
            row["outcome"],
            row.get("max_absolute_residual", row.get("reason")),
            flush=True,
        )


if __name__ == "__main__":
    main()
