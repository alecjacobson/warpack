"""Whole-FEM controls for the qualified FP64 block initializer prototype."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.experimental import compare_block
from benchmarks.experimental.block_jacobi import BlockJacobiPolish
from benchmarks.source_provenance import source_manifest
from warpack import projected


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--projected-backend", choices=("default", "block4"), required=True)
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
        if h.device.is_cuda and 208 <= n <= 1024 and n % 2 == 0:
            dimensions.append(n)
            return BlockJacobiPolish(h, v, cs, ss, partners, temp, vtemp, initializer_sweeps=4)
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
        report["block_initialized_dimensions"] = dimensions
        report["projected_scope"] = (
            "Existing even global projected solves of dimension208..1024 use four FP64 "
            "block-ordered initializer sweeps (block size8, one local sweep), followed by "
            "canonical FP64 Jacobi refinement. Smaller/odd/other projected paths unchanged. "
            "All extra scratch/setup/recording included. No stopping or acceptance threshold "
            "changed. Experimental wrapper only; installed dispatch unchanged."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
