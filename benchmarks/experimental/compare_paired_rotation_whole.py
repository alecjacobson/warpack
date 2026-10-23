"""Whole FEM controls for explicit paired Ritz-rotation tiles (#186)."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.cached_fem_reference import FEMReferenceCache
from benchmarks.experimental import compare_block
from benchmarks.experimental.paired_rotation_tiles import PairedRotationBlock
from benchmarks.source_provenance import source_manifest

SETTINGS = {
    "original": (4, 32, 8, 128),
    "default": (8, 32, 8, 128),
    "r8k8": (8, 32, 8, 128),
    "r8k32": (8, 32, 32, 128),
    "r16k32t256": (16, 32, 32, 256),
}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rotation", choices=SETTINGS, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if (
        "--public-api" not in remainder
        or "--paired-rotation" not in remainder
        or "--cupy" in remainder
    ):
        parser.error(
            "require public paired-rotation Warp composition without a shared CuPy comparison"
        )
    if destination.out.exists():
        parser.error("require a new output path")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    base = original if choice.rotation in ("original", "default") else PairedRotationBlock
    error = None

    class Tracking(base):
        _wide_paired_rotation_enabled = choice.rotation == "default"
        _rotation_rows, _rotation_columns, _rotation_inner, _rotation_threads = SETTINGS[
            choice.rotation
        ]

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
            rotation=choice.rotation,
            compact_graphs=choice.compact_graphs,
            reference_cache=str(choice.reference_cache),
        )
        report["rotation_tile"] = list(SETTINGS[choice.rotation])
        report["independent_reference_cache"] = cache.records
        report["rotation_scope"] = (
            "Original four-row paired rotation, installed guarded default, or benchmark-only "
            "FP64 tile geometry. Separate bounded X/U inputs and both outputs are retained; "
            "all other installed kernels, QR policies, rank/stopping/physical gates and "
            "settings are identical. Each worker owns its solver and preconditioner. "
            "Only verified independent CPU references may be reused outside timers. "
            "First output and all measured outputs are checked. There is no "
            "accuracy-policy change. The installed default uses eight output rows only on "
            "Ada SM89, n>=32768 and widths24..256; other configurations retain four rows."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
