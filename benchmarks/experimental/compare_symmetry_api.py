"""Same-source controls for installed projected symmetry restoration."""

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
    parser.add_argument("--symmetry", choices=("legacy", "default"), required=True)
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
        _restore_symmetry = False

    class LegacyCompact(compact_type):
        _restore_symmetry = False

    class Tracking(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)

    try:
        warpack.BlockLOBPCG = Tracking
        if choice.symmetry == "legacy":
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
        report["args"].update(symmetry=choice.symmetry, compact_graphs=choice.compact_graphs)
        report["projected_symmetry_scope"] = (
            "Installed global and optional compact projected Jacobi restore symmetry after "
            "each sweep. Private legacy control omits only that restoration. Same FP64 "
            "rotation order,30-sweep bound and2e-15 diagonal-scaled stopping criterion. "
            "Small shared/odd paths and outer stopping/acceptance unchanged. Every first "
            "and subsequent complete solve retains all native and original physical gates."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
