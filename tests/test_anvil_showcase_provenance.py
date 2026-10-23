"""Never combine accurate timings from different numerical dependency overlays."""

import json

import pytest

from benchmarks import anvil_showcase


@pytest.mark.parametrize("changed", [None, "binary", "binding", "runtime", "missing"])
def test_direct_rows_require_matching_loaded_backend(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(anvil_showcase, "METHODS", ("warp-direct", "cupy-direct"))
    for method in anvil_showcase.METHODS:
        identity = dict(
            runtime_version=[0, 8, 0],
            library=dict(sha256="binary", path=method),
            binding_files={"solver.py": dict(sha256="source")},
        )
        if method == "cupy-direct":
            if changed == "binary":
                identity["library"]["sha256"] = "changed"
            elif changed == "binding":
                identity["binding_files"]["solver.py"]["sha256"] = "changed"
            elif changed == "runtime":
                identity["runtime_version"] = [1, 0, 0]
            elif changed == "missing":
                identity = None
        raw = dict(
            inputs=dict(
                schema=1, common=dict(fixture="same", reference="same"), method=dict(initial="same")
            ),
            dependency=dict(warp_version="1.18", manifest_status="absent", files={}),
            cudss_dependency=identity,
            accepted=True,
            setup_seconds=0.5,
            rows=[dict(seconds=1.0, quality=dict(max_original_residual=1e-9))],
        )
        path = tmp_path / f"{method}-1.json"
        path.write_text(json.dumps(raw))
        path.with_suffix(".occupancy.json").write_text(json.dumps(dict(timing_eligible=True)))
    if changed is not None:
        with pytest.raises((ValueError, RuntimeError), match="cuDSS"):
            anvil_showcase.summarize(tmp_path, 1, 1)
        assert not (tmp_path / "summary.json").exists()
    else:
        anvil_showcase.summarize(tmp_path, 1, 1)
        result = json.loads((tmp_path / "summary.json").read_text())
        assert result["cudss_dependency"]["library_sha256"] == "binary"


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
            inputs=dict(
                schema=1,
                common=dict(fixture="shared", reference="same"),
                method=dict(initial="same", matrix="same"),
            ),
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


@pytest.mark.parametrize("failure", [None, "matrix", "mass", "reference", "start", "missing"])
def test_showcase_input_identity_and_method_specific_starts(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(anvil_showcase, "METHODS", ("warp-block", "cupy-block"))
    for method in anvil_showcase.METHODS:
        for trial in (1, 2):
            inputs = dict(
                schema=1,
                common=dict(fixture=dict(K="same", M="same"), reference="same"),
                method=dict(initial=method, shifted_matrix="same"),
            )
            if method == "cupy-block" and trial == 2:
                if failure == "matrix":
                    inputs["common"]["fixture"]["K"] = "changed"
                if failure == "mass":
                    inputs["common"]["fixture"]["M"] = "changed"
                if failure == "reference":
                    inputs["common"]["reference"] = "changed"
                if failure == "start":
                    inputs["method"]["initial"] = "changed"
                if failure == "missing":
                    inputs = None
            rows = [dict(seconds=1.0, quality=dict(max_original_residual=1e-9))] * 3
            raw = dict(
                inputs=inputs,
                dependency=dict(warp_version="1.18", manifest_status="absent", files={}),
                accepted=True,
                rows=rows,
                validation_rows=rows,
                total_setup_seconds=0.5,
                capture_seconds=0.1,
                setup_seconds=0.6,
            )
            file = tmp_path / f"{method}-{trial}.json"
            file.write_text(json.dumps(raw))
            file.with_suffix(".occupancy.json").write_text(json.dumps(dict(timing_eligible=True)))
    if failure:
        with pytest.raises((ValueError, RuntimeError), match="Anvil|input provenance"):
            anvil_showcase.summarize(tmp_path, 2, 3)
    else:
        anvil_showcase.summarize(tmp_path, 2, 3)
        assert (
            json.loads((tmp_path / "summary.json").read_text())["common_inputs"]["fixture"]["M"]
            == "same"
        )
