"""Diagnostic attribution of block graph preparation (not baseline timings).

Extra device synchronization separates eager preparation phases. Branch times
are nested inside warm time and must not be added to it again.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import warp as wp

import warpack
from benchmarks.experimental import compare_block
from benchmarks.source_provenance import source_manifest


def measured(device, action):
    start = time.perf_counter()
    wp.synchronize_device(device)
    result = action()
    wp.synchronize_device(device)
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compact-graphs", action="store_true")
    choice, remainder = parser.parse_known_args()
    destination_parser = argparse.ArgumentParser(add_help=False)
    destination_parser.add_argument("--out", type=Path, required=True)
    destination, _ = destination_parser.parse_known_args(remainder)
    if "--public-api" not in remainder:
        parser.error("the public block composition is required")
    original, manifest, argv = warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv
    profiles, error = [], None

    class Tracked(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, compact_projected_graphs=choice.compact_graphs)

        def capture(self, *args, **kwargs):
            profile = dict(solve_preparation_seconds=0.0, search_warm_seconds=0.0, buckets=[])
            solve, search = self.solve, self._search
            warm = None if search is None else search.warm
            step = None if search is None else search.step_bucket
            warming = False

            def prepare_solve(*a, **kw):
                seconds, result = measured(self.device, lambda: solve(*a, **kw))
                profile["solve_preparation_seconds"] += seconds
                return result

            def prepare_bucket(branch, index):
                if not warming:
                    return step(branch, index)
                seconds, result = measured(self.device, lambda: step(branch, index))
                profile["buckets"].append(
                    dict(
                        index=index,
                        metric_size=branch.metric.m,
                        ritz_size=branch.small.m,
                        seconds=seconds,
                    )
                )
                return result

            def prepare_search():
                nonlocal warming
                warming = True
                try:
                    seconds, result = measured(self.device, warm)
                    profile["search_warm_seconds"] += seconds
                    return result
                finally:
                    warming = False

            self.solve = prepare_solve
            if search is not None:
                search.warm, search.step_bucket = prepare_search, prepare_bucket
            start = time.perf_counter()
            try:
                result = super().capture(*args, **kwargs)
                wp.synchronize_device(self.device)
                profile["total_seconds"] = time.perf_counter() - start
                profile["remaining_capture_seconds"] = (
                    profile["total_seconds"]
                    - profile["solve_preparation_seconds"]
                    - profile["search_warm_seconds"]
                )
                profiles.append(profile)
                return result
            finally:
                self.solve = solve
                if search is not None:
                    search.warm, search.step_bucket = warm, step

    try:
        warpack.BlockLOBPCG = Tracked
        compare_block.source_manifest = lambda *_: source_manifest(__file__)
        sys.argv = [sys.argv[0]] + remainder
        compare_block.main()
    except SystemExit as caught:
        error = caught
    finally:
        warpack.BlockLOBPCG, compare_block.source_manifest, sys.argv = original, manifest, argv
    if destination.out.exists():
        report = json.loads(destination.out.read_text())
        report["args"]["compact_graphs"] = choice.compact_graphs
        report["capture_preparation_profiles"] = profiles
        report["capture_profile_scope"] = (
            "Diagnostic only: extra device synchronization around eager solve, search warm and each warm bucket. "
            "Bucket samples are inclusive children of search_warm_seconds; do not sum both. "
            "Remaining capture time includes recorded outer-graph construction/instantiation and intervening setup. "
            "Module loading and lazy child-graph preparation stay charged to the phase where they occur. "
            "Not a CUDA API trace or uninstrumented performance baseline. All first/later numerical gates remain enforced."
        )
        destination.out.write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
