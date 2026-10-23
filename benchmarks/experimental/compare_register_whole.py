"""Complete public solves with an explicit register-blocked triangular candidate."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.cached_fem_reference import FEMReferenceCache
from benchmarks.experimental import compare_block
from benchmarks.experimental.register_triangular import RegisterTriangularBlock
from benchmarks.source_provenance import source_manifest
from warpack import kernels as K
from warpack.block import BlockLOBPCG as CoreBlock


class WideRegisterTriangular(RegisterTriangularBlock):
    _register_triangular_enabled = False

    def launch(self, kernel, shape, args):
        if kernel is K.triangular and not 96 <= args[0].shape[0] <= 256:
            CoreBlock.launch(self, kernel, shape, args)
        else:
            super().launch(kernel, shape, args)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--triangular", choices=["canonical", "register", "default"], required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("public composition is required")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    base = WideRegisterTriangular if choice.triangular == "register" else original
    error = None

    class Tracking(base):
        _register_triangular_enabled = choice.triangular == "default"

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
            triangular=choice.triangular,
            compact_graphs=choice.compact_graphs,
            reference_cache=str(choice.reference_cache),
        )
        report["independent_reference_cache"] = cache.records
        report["triangular_scope"] = (
            "Explicit canonical (installed specialization disabled), register "
            "(independent prototype), or default (installed Ada specialization) "
            "triangular application. The eight-row register kernel at widths96..256 "
            "retains the original per-row subtraction order without an inverse. "
            "Other widths retain the original application. Both "
            "complete QR passes, initial scaling, cached AX, final MX, native stopping "
            "and original physical gates remain. No dense workspace added. All other "
            "installed kernels/settings, including first whitening, are identical. "
            "Only verified independent CPU reference arrays may be reused outside "
            "backend setup/capture/solve timers; each backend factor/solver is fresh."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
