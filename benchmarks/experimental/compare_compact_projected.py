"""Complete block controls for compact device-loop projected graph construction."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.experimental import compare_block
from benchmarks.experimental.compact_jacobi import CompactJacobi
from benchmarks.source_provenance import source_manifest
from warpack import projected


def projected_factory(original, rounds, dimensions=None):
    def create(h, v, cs, ss, partners, temp, vtemp):
        n = h.shape[0]
        if h.device.is_cuda and 2 <= n <= 1024 and n % 2 == 0:
            if dimensions is not None:
                dimensions.append(n)
            return CompactJacobi(h, v, cs, ss, partners, temp, vtemp, rounds_per_loop=rounds)
        return original(h, v, cs, ss, partners, temp, vtemp)

    return create


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--projected-backend", choices=("default", "compact16", "compact32"), required=True
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
            rounds = int(choice.projected_backend.removeprefix("compact"))
            projected.GlobalJacobi = projected_factory(original, rounds, dimensions)
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
        report["compact_projected_dimensions"] = dimensions
        report["projected_scope"] = (
            "Only existing even-dimensional global projected Jacobi solves use compact device round loops. Same FP64 rotation order, convergence tests, rank and outer stopping. Other projected paths and preconditioner unchanged. Extra loop counters and graph construction included in setup/capture. All physical/spectrum gates unchanged. Benchmark-only, installed dispatch unchanged."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
