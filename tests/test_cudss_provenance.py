"""Benchmark identity must follow the cached binary, never preload a backend."""

import ctypes as ct
import sys
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

from benchmarks import cudss_provenance as provenance


@pytest.mark.parametrize("module", [None, SimpleNamespace(_lib=None)])
def test_unused_backend_is_not_loaded(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "warp_cudss._bindings", module)
    assert provenance.cudss_dependencies() is None


def fake_backend(monkeypatch, tmp_path, status=0):
    tmp_path.mkdir(exist_ok=True)
    binding = tmp_path / "_bindings.py"
    binding.write_text("# binding identity\n")
    binary = tmp_path / "libcudss.so.1"
    binary.write_bytes(b"candidate binary")

    def get_property(index, output):
        ct.cast(output, ct.POINTER(ct.c_int))[0] = (1, 0, 0)[index]
        return status

    library = SimpleNamespace(cudssGetProperty=get_property, _name="libcudss.so.1")
    module = SimpleNamespace(_lib=library, __file__=str(binding))
    monkeypatch.setitem(sys.modules, "warp_cudss._bindings", module)
    monkeypatch.setattr(provenance, "_loaded_library_path", lambda lib: binary)
    monkeypatch.setattr(provenance, "version", lambda name: "0.8.0.10")
    return binary, binding


def test_cached_runtime_wins_over_package_and_changed_override(monkeypatch, tmp_path):
    fake_backend(monkeypatch, tmp_path)
    monkeypatch.setenv("CUDSS_LIBRARY_PATH", "/different/not-loaded.so")
    result = provenance.cudss_dependencies()
    assert result["runtime_version"] == [1, 0, 0]
    assert result["installed_package_version"] == "0.8.0.10"
    assert result["library"]["path"] == str(tmp_path / "libcudss.so.1")
    assert result["environment_at_reporting"]["CUDSS_LIBRARY_PATH"] == "/different/not-loaded.so"


def test_content_comparison_and_absent_package_metadata(monkeypatch, tmp_path):
    fake_backend(monkeypatch, tmp_path / "first")
    first = provenance.cudss_dependencies()
    binary, binding = fake_backend(monkeypatch, tmp_path / "second")

    def no_package(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(provenance, "version", no_package)
    second = provenance.cudss_dependencies()
    assert second["installed_package_version"] is None
    assert provenance.cudss_signature(first) == provenance.cudss_signature(second)
    binary.write_bytes(b"changed binary")
    changed = provenance.cudss_dependencies()
    assert provenance.cudss_signature(first) != provenance.cudss_signature(changed)
    binding.write_text("# changed binding\n")
    assert provenance.cudss_signature(changed) != provenance.cudss_signature(
        provenance.cudss_dependencies()
    )


def test_runtime_query_failure_cannot_be_accepted(monkeypatch, tmp_path):
    fake_backend(monkeypatch, tmp_path, status=4)
    with pytest.raises(RuntimeError, match="property 0 failed with status 4"):
        provenance.cudss_dependencies()


def test_missing_historical_provenance_is_not_invented():
    with pytest.raises(ValueError, match="missing loaded cuDSS provenance"):
        provenance.cudss_signature(None)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux dynamic-loader path resolution")
def test_loaded_symbol_resolves_soname_without_search_path_guess():
    library = SimpleNamespace(cudssGetProperty=ct.CDLL(None).strlen, _name="not-a-path.so")
    path = provenance._loaded_library_path(library)
    assert path.is_absolute() and path.is_file()
    assert "libc" in path.name


def test_unresolved_soname_is_rejected_but_explicit_file_is_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance.ct, "CDLL", lambda *_: SimpleNamespace())
    with pytest.raises(ValueError, match="SONAME"):
        provenance._loaded_library_path(SimpleNamespace(_name="libcudss.so.0"))
    binary = tmp_path / "explicit.so"
    binary.write_bytes(b"test")
    assert provenance._loaded_library_path(SimpleNamespace(_name=str(binary))) == binary


@pytest.mark.parametrize("changed", [None, "binary", "binding", "runtime", "missing"])
def test_application_comparison_requires_matching_actual_backend(changed):
    from copy import deepcopy

    from benchmarks.applications.compare import compare_baseline

    identity = dict(
        runtime_version=[0, 8, 0],
        library=dict(sha256="binary"),
        binding_files={"solver.py": dict(sha256="source")},
    )
    old = dict(
        api_class="A",
        n=80,
        method="warp",
        accepted=True,
        timing_uncontended=True,
        solve_seconds=1.0,
        cudss_used=True,
        cudss_dependency=identity,
    )
    current = deepcopy(old)
    current["solve_seconds"] = 2.0
    if changed == "binary":
        current["cudss_dependency"]["library"]["sha256"] = "changed"
    elif changed == "binding":
        current["cudss_dependency"]["binding_files"]["solver.py"]["sha256"] = "changed"
    elif changed == "runtime":
        current["cudss_dependency"]["runtime_version"] = [1, 0, 0]
    elif changed == "missing":
        old.pop("cudss_dependency")
    result = compare_baseline([current], {"cases": [old]}, 1.25)[0]
    assert result["comparable"] == (changed is None)
    assert result["cudss_dependency_comparable"] == (changed is None)
    if changed is None:
        assert result["performance_regression"]
    else:
        assert "time_ratio" not in result
