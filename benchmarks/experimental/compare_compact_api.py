"""Same-source controls for the public compact projected-graph option."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.source_provenance import source_manifest
from warpack._compact_jacobi import CompactJacobi


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--projected-backend", choices=("default", "enabled"), required=True)
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    dimensions, error = [], None

    class Tracking(original):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args, **kwargs, compact_projected_graphs=choice.projected_backend == "enabled"
            )
            for branch in [self] + ([] if self._search is None else self._search.buckets):
                for projected in (branch.small, branch.metric):
                    if isinstance(projected._global_jacobi, CompactJacobi):
                        dimensions.append(projected.m)

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
        report["args"]["projected_backend"] = choice.projected_backend
        report["compact_projected_dimensions"] = dimensions
        report["projected_scope"] = (
            "Public compact_projected_graphs option only; default remains disabled. "
            "Existing even global Jacobi dimensions96..1024 use16-round device loops. "
            "Other paths/stopping/preconditioning unchanged. Setup/capture charges extra state."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
