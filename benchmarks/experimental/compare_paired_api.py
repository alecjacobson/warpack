"""Same-source controls for installed paired projected Jacobi updates."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.source_provenance import source_manifest
from warpack import _compact_jacobi, projected


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paired", choices=("legacy", "default"), required=True)
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, global_type, compact_type, manifest, argv = (
        warpack.BlockLOBPCG,
        projected.GlobalJacobi,
        _compact_jacobi.CompactJacobi,
        compare_block.source_manifest,
        sys.argv,
    )
    error = None

    class LegacyGlobal(global_type):
        _paired_outputs = False

    class LegacyCompact(compact_type):
        _paired_outputs = False

    class Tracking(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)

    try:
        warpack.BlockLOBPCG = Tracking
        if choice.paired == "legacy":
            projected.GlobalJacobi, _compact_jacobi.CompactJacobi = LegacyGlobal, LegacyCompact
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        (
            warpack.BlockLOBPCG,
            projected.GlobalJacobi,
            _compact_jacobi.CompactJacobi,
            compare_block.source_manifest,
            sys.argv,
        ) = original, global_type, compact_type, manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"].update(paired=choice.paired, compact_graphs=choice.compact_graphs)
        report["paired_projected_scope"] = (
            "Installed Ada89 even global and compact projections256..1024 use64-thread "
            "paired output reuse with immutable partner-derived rounds. Legacy disables "
            "only that update dispatch. FP64 expressions, pair order, normalization, "
            "symmetry restoration,30-sweep cap and2e-15 stopping remain unchanged. "
            "No new buffers, host decisions/readbacks, vendor calls or synchronization. "
            "Other size/device paths retain legacy update. All first/later gates retained."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
