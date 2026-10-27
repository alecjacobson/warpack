"""Run existing accuracy-gated FEM harnesses with a verified CPU reference cache.

This driver changes no solver settings or algorithms. Each worker constructs
its own factors and workspaces; only independent CPU reference pairs are reused.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import warpack
from benchmarks.cached_fem_reference import FEMReferenceCache


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--suite", choices=("block", "direct"), required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    options, remainder = parser.parse_known_args()
    out_parser = argparse.ArgumentParser(add_help=False)
    out_parser.add_argument("--out", type=Path, required=True)
    output, _ = out_parser.parse_known_args(remainder)
    if output.out.exists():
        parser.error("output exists; retain previous measurements")
    if options.suite == "block":
        if "--public-api" not in remainder:
            parser.error("block regression requires --public-api")
        from benchmarks.experimental.compare_block import main as run
    else:
        if "--worker" not in remainder:
            parser.error("direct regression requires one --worker")
        from benchmarks.applications.fem_compare import main as run
    argv, error = sys.argv, None
    with FEMReferenceCache(options.reference_cache) as cache:
        try:
            sys.argv = [argv[0]] + remainder
            run()
        except SystemExit as caught:
            error = caught
        finally:
            sys.argv = argv
    if output.out.exists():
        report = json.loads(output.out.read_text())
        root = Path(warpack.__file__).parent
        report["library_source_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.py"))
        }
        report["independent_reference_cache"] = cache.records
        report["regression_driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        output.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
