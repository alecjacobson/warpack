"""Whole-solve controls for experimental certified first metric whitening."""

import argparse
import json
import sys
from pathlib import Path

import warpack
from benchmarks.experimental import compare_block
from benchmarks.experimental.first_whitening import FirstWhiteningBlock
from benchmarks.experimental.padded_first_whitening import PaddedFirstWhiteningBlock
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--first-whitening", choices=["canonical", "certified", "padded"], required=True
    )
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    instances, error = [], None
    base = {
        "canonical": original,
        "certified": FirstWhiteningBlock,
        "padded": PaddedFirstWhiteningBlock,
    }[choice.first_whitening]

    class Tracking(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    try:
        warpack.BlockLOBPCG = Tracking
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv = original, manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["first_whitening"] = choice.first_whitening
        report["first_whitening_usage_last_replay"] = [
            instance._first_whitening_usage.numpy().tolist()
            for instance in instances
            if hasattr(instance, "_first_whitening_usage")
        ]
        report["first_whitening_scope"] = (
            "Experimental first normalized metric fast path at CUDA dimensions96..512. Diagonal guard precedes factor/inverse/congruence certificate. Accept only certified full rank above unchanged rank threshold; otherwise canonical rank revelation on the original normalized Gram. Existing second projection/whitening and all outer/native/physical gates remain. Scratch shared with second whitening and existing transform/eigenvector buffers. Usage counts [fast, canonical fallback, early diagonal rejection] on last replay only. Core dispatch unchanged."
        )
        if choice.first_whitening == "padded":
            report["first_whitening_scope"] = (
                "Experimental normalized first metric certificate at CUDA dimensions96..512. "
                "Validate exact zero padding, finite symmetric metric and normalized active diagonal. "
                "Certify auxiliary Gplus=G+identity_on_padding, then verify inverse block separation. "
                "Physical rank/mask retain only original active rows; padding remains zero. "
                "Uncertain/dependent/invalid cases use canonical fallback on unchanged original G. "
                "Second projection/whitening and all native/physical gates retained. Same3shared "
                "dense scratch buffers, inverse/eigenvector scratch reused, work/product temporal alias. "
                "Usage [fast, canonical, early rejection] on last replay; core dispatch unchanged."
            )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
