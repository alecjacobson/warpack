"""Complete FEM controls for refreshing the Ritz operator image after both QR passes."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.cached_fem_reference import FEMReferenceCache
from benchmarks.experimental import compare_block
from benchmarks.experimental.ritz_image_refresh import FreshRitzImageBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ritz-images", choices=["transform", "refresh"], required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("public composition is required")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    base = FreshRitzImageBlock if choice.ritz_images == "refresh" else original
    error = None

    class Tracking(base):
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
            ritz_images=choice.ritz_images,
            compact_graphs=choice.compact_graphs,
            reference_cache=str(choice.reference_cache),
        )
        report["independent_reference_cache"] = cache.records
        report["ritz_image_scope"] = (
            "Both complete mass QR passes remain. Transform uses the installed two "
            "triangular applications to the cached A*X; refresh instead applies A "
            "once to the final X. All other image reuse, installed register "
            "triangular dispatch, whitening, rank, stopping and physical gates "
            "are identical. No new device workspace or host numerical decision. "
            "Only independently verified CPU reference arrays may be reused outside "
            "backend timers; every worker constructs its own preconditioner/solver."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
