"""A helper/core edit must change benchmark identity without executing imports."""

from pathlib import Path

import pytest

from benchmarks.source_provenance import source_manifest


def test_recursive_relative_namespace_and_package_imports(tmp_path):
    files = {
        "bench/run.py": "from .helpers import operation\nimport pkg.child\nimport numpy\n",
        "bench/__init__.py": "from . import entry_initializer\n",
        "bench/entry_initializer.py": "value = 0\n",
        "bench/helpers.py": "from . import leaf\noperation = 1\n",
        "bench/leaf.py": "raise RuntimeError('must not execute')\n",
        "pkg/__init__.py": "from . import init_dependency\n",
        "pkg/child.py": "value = 3\n",
        "pkg/init_dependency.py": "value = 4\n",
        "pkg/unrelated.py": "value = 5\n",
        "warpack/dynamic_backend.py": "value = 6\n",
    }
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    manifest = source_manifest("bench/run.py", root=tmp_path)
    assert set(manifest) == set(files) - {"pkg/unrelated.py"}
    for name in ("bench/leaf.py", "warpack/dynamic_backend.py"):
        p = tmp_path / name
        p.write_text(p.read_text() + "# changed\n")
        changed = source_manifest("bench/run.py", root=tmp_path)
        assert {key for key in manifest if manifest[key] != changed[key]} == {name}
        manifest = changed
    (tmp_path / "pkg/unrelated.py").write_text("# unrelated\n")
    assert source_manifest("bench/run.py", root=tmp_path) == manifest


@pytest.mark.parametrize("driver", ["force_dual", "fem_shapes"])
def test_repository_workers_cover_helpers_and_solver(driver):
    manifest = source_manifest(f"benchmarks/applications/{driver}.py")
    assert {
        f"benchmarks/applications/{driver}.py",
        "benchmarks/applications/fem_cases.py",
        "benchmarks/applications/fem_shapes.py",
        "benchmarks/upstream/adapters.py",
        "benchmarks/source_provenance.py",
        "warpack/block.py",
    } <= set(manifest)
    if driver == "force_dual":
        assert "benchmarks/applications/weighted_snapshot_reference.py" in manifest
    assert all(
        not Path(name).is_absolute() and len(value) == 64 for name, value in manifest.items()
    )


def test_external_entrypoint_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "outside.py"
    external.write_text("value = 1\n")
    with pytest.raises(ValueError, match="inside the repository"):
        source_manifest(external, root=root)


def test_wide_block_manifest_covers_previously_omitted_inputs_and_validation():
    manifest = source_manifest("benchmarks/experimental/compare_block.py")
    assert {
        "benchmarks/experimental/compare_block.py",
        "benchmarks/applications/hodge.py",
        "benchmarks/applications/problems.py",
        "benchmarks/upstream/adapters.py",
        "benchmarks/anvil.py",
        "benchmarks/anvil_inputs.py",
        "benchmarks/applications/fem_cases.py",
        "benchmarks/experimental/batched_cg.py",
        "warpack/batched_cg.py",
    } <= set(manifest)
