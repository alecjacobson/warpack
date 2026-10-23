"""Complete two-pass stage controls with explicit physical-column tile width."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from benchmarks.experimental import triangular_rotate_compare as driver
from benchmarks.experimental.blocked_triangular_columns import ColumnTriangularBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--column-tile", type=int, choices=[32, 64, 128], required=True)
    options, remainder = parser.parse_known_args()
    destination = argparse.ArgumentParser(add_help=False)
    destination.add_argument("--out", type=Path, required=True)
    output, _ = destination.parse_known_args(remainder)
    original, manifest, argv = driver.BlockedTriangularBlock, driver.source_manifest, sys.argv
    error = None
    try:
        driver.BlockedTriangularBlock = partial(
            ColumnTriangularBlock, column_tile=options.column_tile
        )
        driver.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0], "--blocked-triangular", *remainder]
        driver.main()
    except SystemExit as caught:
        error = caught
    finally:
        driver.BlockedTriangularBlock, driver.source_manifest, sys.argv = original, manifest, argv
    if output.out.exists():
        report = json.loads(output.out.read_text())
        report["args"]["column_tile"] = options.column_tile
        output.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
