"""First-whitening controls with explicit, verified independent CPU reference reuse."""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.cached_fem_reference import FEMReferenceCache
from benchmarks.experimental import compare_first_whitening
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference-cache", type=Path, required=True)
    options, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    manifest, argv = compare_first_whitening.source_manifest, sys.argv
    error = None
    with FEMReferenceCache(options.reference_cache) as cache:
        try:
            compare_first_whitening.source_manifest = lambda *_: source_manifest(__file__)
            sys.argv = [sys.argv[0]] + remainder
            compare_first_whitening.main()
        except SystemExit as caught:
            error = caught
        finally:
            compare_first_whitening.source_manifest, sys.argv = manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["reference_cache"] = str(options.reference_cache)
        report["independent_reference_cache"] = cache.records
        report["reference_cache_scope"] = (
            "Only independently computed CPU reference arrays may be reused after content/provenance and original-pencil validation. Every worker still assembles current matrices, applies current cluster/nullspace policy and builds its own solver/preconditioner. All cache preparation is outside timed backend setup/capture/replay. Anvil uses its established separate fixture protocol."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
