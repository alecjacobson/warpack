"""Same-source whole-FEM controls for immutable-round paired Jacobi."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.experimental.paired_jacobi_runtime import RuntimePairedCompact, RuntimePairedJacobi
from benchmarks.source_provenance import source_manifest
from warpack import _compact_jacobi, projected


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--paired", choices=("default", "runtime"), required=True)
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

    dimensions = []

    def factory(original_type, candidate_type):
        def create(h, *args, **kwargs):
            n = h.shape[0]
            if h.device.is_cuda and h.device.arch == 89 and 256 <= n <= 1024 and n % 2 == 0:
                dimensions.append(n)
                return candidate_type(h, *args, **kwargs)
            return original_type(h, *args, **kwargs)

        return create

    class Tracking(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)

    try:
        warpack.BlockLOBPCG = Tracking
        if choice.paired == "runtime":
            projected.GlobalJacobi = factory(global_type, RuntimePairedJacobi)
            _compact_jacobi.CompactJacobi = factory(compact_type, RuntimePairedCompact)
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
        report["paired_output_dimensions"] = dimensions
        report["paired_output_scope"] = (
            "Even Ada89 global/compact projections256..1024 reuse2x2 output blocks, "
            "64 threads per block. Current round derives from immutable partners, "
            "not the compact advancing counter. Same FP64 expressions, pair order, "
            "normalization, symmetry restoration,30-sweep cap and2e-15 stopping. "
            "Other paths unchanged. All first/later native and independent gates retained."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
