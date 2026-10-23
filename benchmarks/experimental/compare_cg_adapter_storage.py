"""Same-source controls for mutually exclusive adapter scratch allocation."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.experimental import compare_block
from benchmarks.experimental.cg_storage_inventory import inventory
from benchmarks.source_provenance import source_manifest
from warpack import _active_cg


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adapter-storage", choices=("shared", "separate"), required=True)
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("public block composition required")
    original, manifest, argv = _active_cg.DeviceBucketedCG, compare_block.source_manifest, sys.argv
    instances, error = [], None

    class Tracking(original):
        _share_adapter_storage = choice.adapter_storage == "shared"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    try:
        _active_cg.DeviceBucketedCG = Tracking
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        _active_cg.DeviceBucketedCG, compare_block.source_manifest, sys.argv = (
            original,
            manifest,
            argv,
        )
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["adapter_storage"] = choice.adapter_storage
        report["adapter_storage_inventory"] = [inventory(instance) for instance in instances]
        report["adapter_storage_scope"] = (
            "Only Warpack adapter-owned b/x and sparse-application buffers change allocation. "
            "Shared uses contiguous prefix views of the maximum-width base buffers. "
            "Outer packing is shared in both controls. Upstream CG internal workspaces, "
            "factors, arithmetic, selection, stopping and statistics remain unchanged. "
            "Named scratch spans exclude matrices/factors/runtime and are not peak memory."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
