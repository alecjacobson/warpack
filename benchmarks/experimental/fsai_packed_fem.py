"""Fresh-worker FEM comparison of pinned versus experimentally packed FSAI.

Only this process substitutes the constructor. Installed modules and production
Warpack defaults are unchanged. Raw output records both source identities.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from warp.optim import linear

from benchmarks.applications import fem_hierarchy_tuning
from benchmarks.experimental.fsai_packed_storage import BASE_SHA256, load_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("upstream", "packed"), required=True)
    parser.add_argument("--row-size", type=int, choices=(32, 64), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    module = load_candidate(Path("build/fsai-packed-fem"))
    original = linear.FSAI
    constructor = module.FSAI if args.variant == "packed" else original

    def factor(a, **kwargs):
        kwargs["max_row_size"] = args.row_size
        return constructor(a, **kwargs)

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
            raw["arguments"].update(factor_variant=args.variant, factor_rows=args.row_size)
            raw["source_sha256"]["fsai_packed_fem.py"] = hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
            raw["source_sha256"]["fsai.py"] = (
                hashlib.sha256(
                    Path("build/fsai-packed-fem/fsai_packed_candidate.py").read_bytes()
                ).hexdigest()
                if args.variant == "packed"
                else BASE_SHA256
            )
            for module_file in ("warpack/batched_sparse.py", "warpack/_coalesced_fsai.py"):
                raw["source_sha256"][module_file] = hashlib.sha256(
                    Path(module_file).read_bytes()
                ).hexdigest()
            args.out.write_text(json.dumps(raw, indent=2) + "\n")


if __name__ == "__main__":
    main()
