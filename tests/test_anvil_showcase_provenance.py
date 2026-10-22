"""Never combine accurate timings from different numerical dependency overlays."""

import json

import pytest

from benchmarks import anvil_showcase


@pytest.mark.parametrize("mixed", [False, True])
def test_showcase_requires_one_verified_dependency(tmp_path, monkeypatch, mixed):
    monkeypatch.setattr(anvil_showcase, "METHODS", ("warp-block", "cupy-block"))
    for method in anvil_showcase.METHODS:
        dependency = dict(
            warp_version="1.18.0",
            manifest_status="verified",
            files={"fsai": {"sha256": "changed" if mixed and method == "cupy-block" else "same"}},
        )
        rows = [dict(seconds=1.0, quality=dict(max_original_residual=1e-9)) for _ in range(3)]
        raw = dict(
            dependency=dependency,
            accepted=True,
            rows=rows,
            validation_rows=rows,
            total_setup_seconds=0.5,
            capture_seconds=0.1,
            setup_seconds=0.6,
        )
        file = tmp_path / f"{method}-1.json"
        file.write_text(json.dumps(raw))
        file.with_suffix(".occupancy.json").write_text(json.dumps(dict(timing_eligible=True)))
    if mixed:
        with pytest.raises(RuntimeError, match="Mixed dependency"):
            anvil_showcase.summarize(tmp_path, 1, 3)
        assert not (tmp_path / "summary.json").exists()
    else:
        anvil_showcase.summarize(tmp_path, 1, 3)
        result = json.loads((tmp_path / "summary.json").read_text())
        assert result["dependency"]["files"]["fsai"] == "same"
