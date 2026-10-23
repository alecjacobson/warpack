"""Same-source controls for installed guarded branch preparation."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.source_provenance import source_manifest
from warpack import _block_search


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preparation", choices=("legacy", "default"), required=True)
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, search, manifest, argv = (
        warpack.BlockLOBPCG,
        _block_search.ActiveSearch,
        compare_block.source_manifest,
        sys.argv,
    )
    error = None

    class LegacySearch(search):
        _force_eager_warmup = True

    class Tracking(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)

    try:
        warpack.BlockLOBPCG = Tracking
        if choice.preparation == "legacy":
            _block_search.ActiveSearch = LegacySearch
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        warpack.BlockLOBPCG, _block_search.ActiveSearch, compare_block.source_manifest, sys.argv = (
            original,
            search,
            manifest,
            argv,
        )
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"].update(preparation=choice.preparation, compact_graphs=choice.compact_graphs)
        report["preparation_scope"] = (
            "Installed default directly records all branches for known library-owned stiffness/mass "
            "applications on conditional-graph-capable CUDA; custom/subclass/instance overrides "
            "retain full eager per-width preparation. Legacy control forces prior warmup. Initial "
            "eager solve, synchronization before capture and all callback order are unchanged. "
            "Every captured branch and all first/later numerical gates remain enforced."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
