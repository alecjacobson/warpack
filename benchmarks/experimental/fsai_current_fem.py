"""Qualify the isolated current FSAI module on the existing FEM comparison."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from warp.optim import linear

from benchmarks.applications import fem_hierarchy_tuning
from benchmarks.experimental.fsai_current import SHA256, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--row-size", type=int, choices=(32, 64), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    module = load(packed=args.packed)
    original = linear.FSAI

    def factor(a, **kwargs):
        kwargs.pop("apply_lanes", None)
        kwargs["max_row_size"] = args.row_size
        return module.FSAI(a, **kwargs)

    linear.FSAI = factor
    sys.argv = [
        "fem_hierarchy_tuning",
        "--profile",
        "random",
        "--fine",
        "10",
        "--out",
        str(args.out),
    ]
    try:
        fem_hierarchy_tuning.main()
    finally:
        linear.FSAI = original
        if args.out.exists():
            raw = json.loads(args.out.read_text())
            raw["arguments"].update(
                factor_variant="current-packed" if args.packed else "current",
                factor_rows=args.row_size,
            )
            raw["source_sha256"]["fsai.py"] = hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            raw["base_fsai_sha256"] = SHA256
            for file in (
                __file__,
                "warpack/batched_sparse.py",
                "warpack/_coalesced_fsai.py",
                "benchmarks/experimental/fsai_current.py",
            ):
                raw["source_sha256"][Path(file).name] = hashlib.sha256(
                    Path(file).read_bytes()
                ).hexdigest()
            args.out.write_text(json.dumps(raw, indent=2) + "\n")


if __name__ == "__main__":
    main()
