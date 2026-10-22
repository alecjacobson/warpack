"""Rejected solves and incompatible baselines must not become speedup claims."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from benchmarks.applications.fem_shapes_report import compare, summarize

RESULTS = Path(__file__).resolve().parents[1] / "results" / "fem_shapes"


def test_archived_batch_and_baseline_gate():
    summary = summarize(RESULTS)
    assert len(summary["rows"]) == 8
    assert set(compare(summary, summary).values()) == {1.0}
    slower = copy.deepcopy(summary)
    slower["rows"][0]["sequence_seconds"] *= 1.3
    with pytest.raises(ValueError, match="regression"):
        compare(slower, summary)
    incompatible = copy.deepcopy(summary)
    incompatible["matrix_sha256"]["lumped"][1] = "different geometry"
    with pytest.raises(ValueError, match="incompatible"):
        compare(incompatible, summary)


@pytest.mark.parametrize("failure", ["contention", "native_count", "matrix", "nan_time"])
def test_reject_invalid_batch(tmp_path, failure):
    for p in RESULTS.glob("*.json"):
        shutil.copyfile(p, tmp_path / p.name)
    name = "consistent-cold-refit"
    path = tmp_path / (name + (".occupancy.json" if failure == "contention" else ".json"))
    data = json.loads(path.read_text())
    if failure == "contention":
        data["timing_eligible"] = False
    elif failure == "native_count":
        data["rows"][0][1]["quality"]["converged"] = 0
        # Keep both accepted booleans true: fresh native status still matters.
    elif failure == "matrix":
        data["matrix_sha256"][1] = "different geometry"
    else:
        data["rows"][0][1]["solve_seconds"] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summarize(tmp_path)
