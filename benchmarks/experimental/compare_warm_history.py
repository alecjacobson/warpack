"""Complete same-source controls for setup-only active-search history clearing."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.experimental.zero_history_warmup import ZeroHistorySearch
from benchmarks.source_provenance import source_manifest
from warpack import _block_search


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--warm-history", choices=("current", "zero"), required=True)
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
        if choice.warm_history == "zero":
            _block_search.ActiveSearch = ZeroHistorySearch
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
        report["args"].update(
            warm_history=choice.warm_history, compact_graphs=choice.compact_graphs
        )
        report["warm_history_scope"] = (
            "Only setup preparation changes: clear owner.p before the existing complete warmup "
            "of every active-search branch. The captured solve initializes its own search space. "
            "Replay arithmetic/stopping/graphs and initializer/finalizer order are unchanged. "
            "Setup+capture+independently validated first solve and subsequent replays are separate. "
            "Benchmark-only candidate, installed warmup unchanged."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
