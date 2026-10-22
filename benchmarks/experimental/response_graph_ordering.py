"""Diagnostic for #148/#61: queued cuDSS response and first snapshot-graph upload.

Deliberately retains the failing lazy-upload policy as a control. This is not
an application recipe; the weighted-SVD example initializes graphs at setup.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.applications.fem_hierarchy import beam_problem
from benchmarks.applications.force_dual import localized_forces
from benchmarks.applications.weighted_snapshot_svd import weight_snapshots
from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p, f = beam_problem(2, 4, False), localized_forces(2, 12)
    mass = wp.array(p.b.diagonal(), dtype=wp.float64)
    rows = []
    for policy in ("lazy", "sync-response", "setup-initialize"):
        forces = wp.array(f.T.copy(), dtype=wp.float64)
        response = wp.empty_like(forces)
        weighted = wp.empty(f.size, dtype=wp.float64)
        invalid = wp.zeros(1, dtype=wp.int32)
        inverse = CuDSSInverse(matrix(p.a), f.shape[1])

        def prepare():
            invalid.zero_()
            wp.launch(
                weight_snapshots, f.shape, [response, mass, weighted, invalid], device=mass.device
            )

        try:
            inverse.apply(forces, response)
            prepare()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                inverse.apply(forces, response)
            response_graph = capture.graph
            with wp.ScopedCapture() as capture:
                prepare()
            weighted_graph = capture.graph
            if policy == "setup-initialize":
                for graph in (response_graph, weighted_graph):
                    wp.capture_launch(graph)
                    wp.synchronize()
            for trial in range(3):
                current = f * (1 + 0.3 * trial)
                forces.assign(current.T.copy())
                wp.synchronize()  # Exclude pending host upload from this diagnostic.
                wp.capture_launch(response_graph)
                if policy == "sync-response":
                    wp.synchronize()
                wp.capture_launch(weighted_graph)
                actual = weighted.numpy().reshape(f.shape)
                g = response.numpy().T
                rows.append(
                    dict(
                        policy=policy,
                        trial=trial,
                        source_residual=float(
                            np.linalg.norm(p.a @ g - current) / np.linalg.norm(current)
                        ),
                        weighting_error=float(
                            np.max(np.abs(actual - np.sqrt(p.b.diagonal())[:, None] * g))
                        ),
                    )
                )
        finally:
            response_graph = weighted_graph = capture = None
            inverse.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(rows)


if __name__ == "__main__":
    main()
