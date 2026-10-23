"""Complete block controls with guarded shared-FP32 projected initialization."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.experimental import compare_block
from benchmarks.experimental.fp32_shared_jacobi import shared_bytes
from benchmarks.experimental.mixed_jacobi import MixedSharedJacobi
from benchmarks.source_provenance import source_manifest
from warpack import projected


def projected_factory(original, limit, dimensions=None):
    def create(h, v, cs, ss, partners, temp, vtemp):
        n = h.shape[0]
        if (
            h.device.is_cuda
            and h.device.arch == 89
            and 48 <= n <= limit
            and shared_bytes(n) <= h.device.max_shared_memory_per_block
        ):
            if dimensions is not None:
                dimensions.append(n)
            return MixedSharedJacobi(h, v, cs, ss, partners, temp, vtemp)
        return original(h, v, cs, ss, partners, temp, vtemp)

    return create


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--projected-backend", choices=("default", "mixed64", "mixed72"), required=True
    )
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, manifest, argv = projected.GlobalJacobi, compare_block.source_manifest, sys.argv
    dimensions, error = [], None
    try:
        if choice.projected_backend != "default":
            limit = int(choice.projected_backend.removeprefix("mixed"))
            projected.GlobalJacobi = projected_factory(original, limit, dimensions)
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        projected.GlobalJacobi, compare_block.source_manifest, sys.argv = original, manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["projected_backend"] = choice.projected_backend
        report["mixed_projected_dimensions"] = dimensions
        report["projected_scope"] = (
            "On Ada, replace only existing global projected solves at dimensions48..64/72 with shared-FP32 complete-basis initialization and original-matrix FP64 recovery. Other projected paths, outer stopping, preconditioner and physical checks unchanged. Extra workspaces included in setup. FP32 supplies no final values or rank decisions."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
