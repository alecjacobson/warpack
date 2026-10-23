"""Run unchanged block comparisons with an explicit projected-Cholesky variant."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.experimental.global_cholesky import GlobalCholeskyBlock
from benchmarks.experimental.parallel_cholesky import SerialCholeskyBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cholesky-backend", choices=("serial", "global", "default"), required=True
    )
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original_class, original_manifest, original_argv = (
        warpack.BlockLOBPCG,
        compare_block.source_manifest,
        sys.argv,
    )
    error = None
    try:
        if choice.cholesky_backend == "global":
            warpack.BlockLOBPCG = GlobalCholeskyBlock
        elif choice.cholesky_backend == "serial":
            warpack.BlockLOBPCG = SerialCholeskyBlock
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv = (
            original_class,
            original_manifest,
            original_argv,
        )
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["cholesky_backend"] = choice.cholesky_backend
        report["cholesky_scope"] = (
            "Only BlockLOBPCG projected Cholesky: global uses a preallocated transposed device workspace through width 512; serial forces the old factor; default retains production shared/serial dispatch. No triangular solve, projected eigensolve, iteration or stopping change. Extra workspace is included in solver setup."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
