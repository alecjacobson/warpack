"""Whole-FEM controls for final projected eigenvector normalization."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.experimental import compare_block
from benchmarks.experimental.normalized_jacobi import NormalizedJacobi
from benchmarks.source_provenance import source_manifest
from warpack import projected


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--projected-backend", choices=["default", "normalized"], required=True)
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, manifest, argv = projected.GlobalJacobi, compare_block.source_manifest, sys.argv
    dimensions, error = [], None

    def create(h, v, cs, ss, partners, temp, vtemp):
        n = h.shape[0]
        if h.device.is_cuda and 256 <= n <= 1024 and n % 2 == 0:
            dimensions.append(n)
            return NormalizedJacobi(h, v, cs, ss, partners, temp, vtemp)
        return original(h, v, cs, ss, partners, temp, vtemp)

    try:
        if choice.projected_backend != "default":
            projected.GlobalJacobi = create
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        projected.GlobalJacobi, compare_block.source_manifest, sys.argv = original, manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["projected_backend"] = choice.projected_backend
        report["normalized_dimensions"] = dimensions
        report["normalization_scope"] = (
            "Existing even global projected solvers of size256..1024 normalize each "
            "eigenvector column with a pure-Warp reduction after bounded Jacobi. "
            "Projected matrix/eigenvalues, pair/round order, symmetry restoration and "
            "native stopping remain unchanged. Invalid/zero columns are preserved "
            "for existing failure detection. Other projected paths and all outer "
            "physical/requested-space gates unchanged. Experimental wrapper only."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
