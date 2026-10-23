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
        "--first-whitening", choices=["canonical", "default", "certified", "padded"], required=True
    )
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    instances, error = [], None

    class CanonicalBlock(original):
        _first_whitening_enabled = False

    base = {
        "canonical": CanonicalBlock,
        "default": original,
        "certified": FirstWhiteningBlock,
        "padded": PaddedFirstWhiteningBlock,
    }[choice.first_whitening]

    class Tracking(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)
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
        report["args"].update(
            first_whitening=choice.first_whitening, compact_graphs=choice.compact_graphs
        )
        report["first_whitening_usage_last_replay"] = [
            instance._first_whitening_usage.numpy().tolist()
            for instance in instances
            if getattr(instance, "_first_whitening_usage", None) is not None
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
        if choice.first_whitening in ("canonical", "default"):
            report["first_whitening_scope"] = (
                "Installed first-whitening dispatcher: " + choice.first_whitening + ". "
                "Default uses the certified exact-padding route on Ada metrics96..512; "
                "canonical explicitly disables it. Both whitening passes, original rank "
                "threshold, canonical fallback and native/physical gates remain. "
                "Usage [fast, canonical fallback, early rejection] on last replay."
            )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise SystemExit(error.code)


if __name__ == "__main__":
    main()
