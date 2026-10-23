"""Same-source controls for mutually exclusive CG packing-buffer allocation."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.experimental import compare_block
from benchmarks.source_provenance import source_manifest
from warpack import _active_cg


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--packing-backend", choices=("shared", "separate"), required=True)
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("public block composition required")
    original, manifest, argv = _active_cg.DeviceBucketedCG, compare_block.source_manifest, sys.argv
    instances, error = [], None

    class Tracking(original):
        _share_packing_storage = choice.packing_backend == "shared"

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
        report["args"]["packing_backend"] = choice.packing_backend
        allocations = []
        for instance in instances:
            unique = {}
            for array in [*instance.rhs, *instance.outputs]:
                unique[array.ptr] = max(unique.get(array.ptr, 0), array.size * 8)
            allocations.append(
                dict(
                    unique_allocations=len(unique),
                    requested_packing_bytes=sum(unique.values()),
                    width=instance.width,
                    n=instance.n,
                    bucket_widths=instance.widths.numpy().tolist(),
                )
            )
        report["packing_allocations"] = allocations
        report["packing_scope"] = (
            "Only external per-bucket FP64 RHS/output buffers change allocation. Shared uses two maximum-width arrays and prefix views; separate retains old individual allocations. Upstream CG workspaces, factors, arithmetic, width selection, stopping, statistics and solver defaults unchanged. Requested packing bytes deduplicate backing addresses and are not total/peak/context memory. All setup allocations are charged."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
