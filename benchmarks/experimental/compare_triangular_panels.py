"""Complete two-pass stage controls with explicit triangular panel size."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from benchmarks.experimental import triangular_rotate_compare as driver
from benchmarks.experimental.blocked_triangular_panels import PanelTriangularBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--panel-rows", type=int, choices=[8, 16, 32], required=True)
    options, remainder = parser.parse_known_args()
    destination = argparse.ArgumentParser(add_help=False)
    destination.add_argument("--out", type=Path, required=True)
    output, _ = destination.parse_known_args(remainder)
    original, manifest, argv = driver.BlockedTriangularBlock, driver.source_manifest, sys.argv
    error = None
    try:
        driver.BlockedTriangularBlock = partial(PanelTriangularBlock, panel_rows=options.panel_rows)
        driver.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0], "--blocked-triangular", *remainder]
        driver.main()
    except SystemExit as caught:
        error = caught
    finally:
        driver.BlockedTriangularBlock, driver.source_manifest, sys.argv = original, manifest, argv
    if output.out.exists():
        report = json.loads(output.out.read_text())
        report["args"]["panel_rows"] = options.panel_rows
        output.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
