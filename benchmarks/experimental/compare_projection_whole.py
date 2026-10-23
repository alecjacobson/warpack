"""Whole FEM controls for explicit projection-subtraction tiles (#192)."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.cached_fem_reference import FEMReferenceCache
from benchmarks.experimental import compare_block
from benchmarks.experimental.projection_tiles import ProjectionBlock
from benchmarks.source_provenance import source_manifest

SETTINGS = {"original": 4, "default": 8, "r4": 4, "r8": 8, "r16": 16}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--projection", choices=SETTINGS, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if (
        "--public-api" not in remainder
        or "--fused-updates" not in remainder
        or "--cupy" in remainder
    ):
        parser.error(
            "require public fused-projection Warp composition without a shared CuPy comparison"
        )
    if destination.out.exists():
        parser.error("require a new output path")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    base = original if choice.projection in ("original", "default") else ProjectionBlock
    error = None

    class Tracking(base):
        _wide_projection_enabled = choice.projection == "default"
        _projection_rows = SETTINGS[choice.projection]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)

    with FEMReferenceCache(choice.reference_cache) as cache:
        try:
            warpack.BlockLOBPCG = Tracking
            compare_block.source_manifest = lambda *_: source_manifest(__file__)
            sys.argv = [sys.argv[0]] + remainder
            compare_block.main()
        except SystemExit as caught:
            error = caught
        finally:
            warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv = original, manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"].update(
            projection=choice.projection,
            compact_graphs=choice.compact_graphs,
            reference_cache=str(choice.reference_cache),
        )
        report["projection_rows"] = SETTINGS[choice.projection]
        report["independent_reference_cache"] = cache.records
        report["projection_scope"] = (
            "Original four-row subtraction, installed guarded default, or benchmark-only "
            "4/8/16-row output tiles. "
            "Inner tile 8, output columns 32 and 128 threads are fixed. Both projection "
            "passes, all installed QR/whitening/paired-rotation kernels, rank/stopping "
            "and physical gates are unchanged. Each worker owns its preconditioner; "
            "only verified independent CPU references may be reused outside timers. "
            "First and every measured output are validated."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
