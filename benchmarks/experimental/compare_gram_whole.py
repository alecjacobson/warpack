"""Complete public FEM controls for explicit benchmark-only Gram tiles (#180)."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.cached_fem_reference import FEMReferenceCache
from benchmarks.experimental import compare_block
from benchmarks.experimental.gram_tiles import GramTileBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gram-tile", choices=[0, 8, 16, 32], type=int, required=True)
    parser.add_argument("--gram-threads", choices=[64, 128, 256], type=int, default=128)
    parser.add_argument("--installed-default", action="store_true")
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("public composition is required")
    if choice.gram_tile == 0 and choice.gram_threads != 128:
        parser.error("installed control uses128 threads")
    if choice.installed_default and choice.gram_tile != 0:
        parser.error("--installed-default requires --gram-tile 0")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    base = GramTileBlock if choice.gram_tile else original
    error = None

    class Tracking(base):
        _wide_gram_enabled = choice.installed_default
        _gram_output_tile = choice.gram_tile
        _gram_threads = choice.gram_threads

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
            gram_tile=choice.gram_tile,
            gram_threads=choice.gram_threads,
            installed_default=choice.installed_default,
            compact_graphs=choice.compact_graphs,
            reference_cache=str(choice.reference_cache),
        )
        report["independent_reference_cache"] = cache.records
        report["gram_scope"] = (
            "Explicit canonical8x8/128-thread Gram (new installed dispatch disabled), "
            "installed default (with --installed-default), or benchmark-only output tile/thread "
            "choice. Candidate uses a rolled loop to reuse tile storage at short chunks. "
            "FP64 products, symmetric off-diagonal mirroring, diagonal rounding, chunk "
            "length, both complete QR passes, rank/stopping/physical gates and all other "
            "installed kernels/settings are retained. No extra global workspace or "
            "host numerical decision. Only independently verified CPU reference arrays "
            "may be reused outside backend timers; every worker constructs its own "
            "preconditioner and solver."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
