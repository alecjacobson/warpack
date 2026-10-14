"""Isolate transfers in a single captured cuDSS solve."""

import gc
import json
import re
from collections import Counter
from pathlib import Path

import scipy.sparse as sp
import warp as wp

from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse


def main():
    Path("build/issue-work").mkdir(parents=True, exist_ok=True)
    rows = []
    for kind in ("spd", "symmetric", "general"):
        inv = CuDSSInverse(matrix(sp.diags([2.0, 3.0, 4.0, 5.0])), 1, mtype=kind)
        inv.x.fill_(1.0)
        rhs = wp.ones_like(inv.x)
        inv.apply(rhs, inv.x)
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            inv.apply(rhs, inv.x)
        wp.capture_launch(cap.graph)
        wp.synchronize()
        path = f"build/issue-work/cudss-only-{kind}.dot"
        wp.capture_debug_dot_print(cap.graph, path)
        text = open(path).read()
        rows.append(
            dict(
                mtype=kind,
                memcpy_kinds=dict(Counter(re.findall(r"\| \{kind \| ([^}]+)", text))),
                host_nodes=len(re.findall(r'label="\{\s*HOST\b', text)),
                solution=inv.x.numpy().tolist(),
            )
        )
        del cap
        gc.collect()
        inv.close()
    open("build/issue-work/cudss-graph.json", "w").write(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
