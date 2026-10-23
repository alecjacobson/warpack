"""Two-pass stage control for register-blocked serial triangular recurrence."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.experimental import triangular_rotate_compare as driver
from benchmarks.experimental.register_triangular import RegisterTriangularBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path, required=True)
    options, _ = parser.parse_known_args()
    original, manifest, argv = driver.BlockedTriangularBlock, driver.source_manifest, sys.argv
    error = None
    try:
        driver.BlockedTriangularBlock = RegisterTriangularBlock
        driver.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0], "--blocked-triangular", *sys.argv[1:]]
        driver.main()
    except SystemExit as caught:
        error = caught
    finally:
        driver.BlockedTriangularBlock, driver.source_manifest, sys.argv = original, manifest, argv
    if options.out.exists():
        report = json.loads(options.out.read_text())
        report["args"]["triangular_storage"] = "register_serial_8_rows"
        report["protocol"] = report["protocol"].replace(
            "With --blocked-triangular, direct panel solves replace only the triangular application, without forming an inverse; both QR passes remain.",
            "This control replaces only the triangular application with its original serial subtraction order, interleaving eight RHS rows per thread and reusing each earlier solved value in registers. No tiled matrix product or inverse is formed. Both QR passes remain. The blocked_triangular backend label names this register-blocked candidate here.",
        )
        options.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
