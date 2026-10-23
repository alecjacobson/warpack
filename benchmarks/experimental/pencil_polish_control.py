"""Accuracy-only repeated inverse-filter control, preserving original eigenvalues."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.experimental import pencil_refinement as driver
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--polish-steps", type=int, required=True)
    choice, remainder = parser.parse_known_args()
    if choice.polish_steps < 1:
        parser.error("positive explicit eigenvector polish count required")
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    setup, manifest, argv = driver.warp_setup, driver.source_manifest, sys.argv
    error = None
    try:
        driver.warp_setup = lambda *args, **kwargs: setup(
            *args, **kwargs, polish=choice.polish_steps
        )
        driver.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        driver.main()
    except SystemExit as caught:
        error = caught
    finally:
        driver.warp_setup, driver.source_manifest, sys.argv = setup, manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["polish_steps"] = choice.polish_steps
        report["protocol"] = (
            "Accuracy-only repeated inverse filtering of final right eigenvectors. "
            "Each declared polish step applies (A-sigma B)^-1 B then normalizes "
            "the vector; existing Warp kernels perform normalization. Original "
            "eigenvalues are preserved. No quotient update, velocity reconstruction "
            "or mechanics-specific relation is used. The separately declared "
            "refinement_steps controls residual corrections inside sparse solves. "
            "Original physical and native gates remain unchanged; no timing claim."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
