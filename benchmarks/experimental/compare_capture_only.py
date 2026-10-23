"""Same-source controls for recording all active-search branches without numerical warmup."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.experimental.capture_only_warmup import CaptureOnlySearch
from benchmarks.source_provenance import source_manifest
from warpack import _block_search


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--preparation", choices=("current", "capture-only"), required=True)
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

    class Tracking(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)

    try:
        warpack.BlockLOBPCG = Tracking
        if choice.preparation == "capture-only":
            _block_search.ActiveSearch = CaptureOnlySearch
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
            "Only setup preparation changes: record all active-search branches in the final graph "
            "without executing an extra numerical warmup for each. The initial eager solve and "
            "pre-capture synchronization remain. CUDA conditional-graph drivers support kernel "
            "loading during capture; cold-cache regression required. Replay arithmetic and "
            "all first/later quality gates remain unchanged. Benchmark-only candidate."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
