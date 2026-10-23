"""Complete FEM controls for certified second-pass whitening."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.experimental.certified_whitening import CertifiedWhiteningBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--whitening-backend", choices=("certified", "default"), required=True)
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original_class, original_manifest, original_argv = (
        warpack.BlockLOBPCG,
        compare_block.source_manifest,
        sys.argv,
    )
    error = None
    try:
        instances = []
        if choice.whitening_backend == "certified":

            class Tracking(CertifiedWhiteningBlock):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    instances.append(self)

            warpack.BlockLOBPCG = Tracking
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv = (
            original_class,
            original_manifest,
            original_argv,
        )
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["whitening_backend"] = choice.whitening_backend
        report["whitening_scope"] = (
            "Only second search-metric whitening changes. Conservative device spectral/rank certificate permits Cholesky; failed certificate/factorization uses the original canonical eigensolve. First whitening, outer stopping and physical checks unchanged. Extra workspaces included in setup."
        )
        report["whitening_usage_last_replay"] = [
            instance.whitening_usage.numpy().tolist() for instance in instances
        ]
        report["whitening_usage_scope"] = (
            "Last measured Warp replay only: [certified Cholesky, canonical fallback] per solver; not an aggregate over timed repeats."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
