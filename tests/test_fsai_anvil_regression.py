"""A dependency promotion must pass accuracy, matched-input and performance gates."""

import json

import pytest

from benchmarks.experimental.fsai_anvil_regression import summarize
from scripts.install_fsai_overlay import PROFILES


@pytest.mark.parametrize(
    "failure",
    [None, "contention", "source", "dependency", "accuracy", "inputs", "slow_setup", "slow_replay"],
)
def test_promotion_gates(tmp_path, failure):
    for profile, name in (("legacy", "v0.2.0"), ("fixed", "current")):
        fixed = profile == "fixed"
        rows = [
            dict(
                seconds=1.2 if fixed and failure == "slow_replay" else 1.0,
                quality=dict(max_original_residual=1e-9),
            )
            for _ in range(3)
        ]
        raw = dict(
            inputs=dict(
                schema=1,
                common=dict(
                    fixture="changed" if fixed and failure == "inputs" else "shared",
                    reference="same",
                ),
                method=dict(initial="same", matrix="same"),
            ),
            dependency=dict(
                warp_version="1.18.0",
                manifest_status="verified",
                files={
                    "warp/_src/optim/fsai.py": {
                        "sha256": "unknown"
                        if fixed and failure == "dependency"
                        else PROFILES[name][1]
                    }
                },
            ),
            accepted=not (fixed and failure == "accuracy"),
            rows=rows,
            total_setup_seconds=2.0 if fixed and failure == "slow_setup" else 0.5,
            capture_seconds=0.1,
            preconditioner_setup_seconds=0.4,
            setup_capture_device_free_memory_decrease_bytes=80 if fixed else 100,
            args={"out": str(tmp_path / profile), "width": 24},
            source_sha256={"core.py": "changed" if fixed and failure == "source" else "same"},
        )
        for trial in range(1, 4):
            path = tmp_path / f"{profile}-{trial}.json"
            path.write_text(json.dumps(raw))
            path.with_suffix(".occupancy.json").write_text(
                json.dumps(dict(timing_eligible=not (fixed and failure == "contention")))
            )
    if failure in ("contention", "source", "dependency", "accuracy", "inputs"):
        with pytest.raises(ValueError):
            summarize(tmp_path)
    else:
        report = summarize(tmp_path)
        assert report["accepted"] == (failure is None)
