"""Profile wide certified whitening, including retained-rank diagnostics.

Additional device reads happen after the profiled graph is synchronized, outside
CUDA event intervals. This changes neither solver arithmetic nor branch selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.experimental import profile_active_block as profile
from benchmarks.source_provenance import source_manifest


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", type=Path, required=True)
    args, _ = parser.parse_known_args()
    original_create, original_totals, original_manifest = (
        profile.create_solver,
        profile.Events.totals,
        profile.source_manifest,
    )
    instances, diagnostics = [], []

    def create(config):
        result = original_create(config)
        instances.append(result[0])
        return result

    def totals(events):
        result = original_totals(events)
        solver = instances[-1]
        selected = np.flatnonzero(solver._search.flags.numpy())
        if len(selected) != 1:
            raise ValueError("diagnostic requires exactly one selected search branch")
        branch = solver._search.buckets[int(selected[0])]
        rank = int(solver.rank.numpy()[0]) - solver.b
        if not 0 <= rank <= branch.metric.m:
            raise ValueError("invalid retained search rank")
        helper = branch._second_whitener
        diagnostics.append(
            dict(
                retained_search_rank=rank,
                certified_second_pass_flags=None
                if helper is None
                else helper.flags.numpy().tolist(),
                retained_rank_scope="After the selected step: total Ritz rank minus the full X block width. Flags are candidate/success/fallback/previous active count.",
            )
        )
        return result

    try:
        profile.create_solver, profile.Events.totals = create, totals
        profile.source_manifest = lambda *_: source_manifest(__file__)
        profile.main()
    finally:
        profile.create_solver, profile.Events.totals, profile.source_manifest = (
            original_create,
            original_totals,
            original_manifest,
        )
    report = json.loads(args.out.read_text())
    # The base profiler owns numerical checks, graph audit and timing fields.
    samples = report["samples"]
    for sample, diagnostic in zip(samples, diagnostics, strict=True):
        sample.update(diagnostic)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
