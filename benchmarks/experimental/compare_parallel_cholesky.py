"""Run unchanged block comparisons with an explicit projected-Cholesky variant."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.experimental.parallel_cholesky import (
    CooperativeCholeskyBlock,
    SerialCholeskyBlock,
)
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cholesky-backend", choices=("serial", "cooperative", "default"), required=True
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
        if choice.cholesky_backend == "cooperative":
            warpack.BlockLOBPCG = CooperativeCholeskyBlock
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
            "Only BlockLOBPCG projected Cholesky: serial forces the old kernel; cooperative forces dimensions 1..128 when CUDA shared memory suffices; default uses production guarded wide-block dispatch. No triangular solve, matrix product, projected eigenmethod, iteration, or stopping change."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
