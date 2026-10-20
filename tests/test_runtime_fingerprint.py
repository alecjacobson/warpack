"""Forwarded exports must fingerprint their actual implementations."""

import importlib.util

from benchmarks.runtime import complete_warp_fingerprint, source_hash, warp_runtime_hashes


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_implementation_change_with_unchanged_public_module(tmp_path):
    source = tmp_path / "implementation.py"
    source.write_text("def operation():\n    return 1\n")
    implementation = load_module(source, "test_runtime_implementation")
    forward = tmp_path / "forward.py"
    forward.write_text("# public forwarding interface\n")
    public = load_module(forward, "test_runtime_forward")
    public.operation = implementation.operation
    before = source_hash(public), source_hash(public.operation)
    source.write_text("def operation():\n    return 2\n")
    after = source_hash(public), source_hash(public.operation)
    assert before[0] == after[0]
    assert before[1] != after[1]


def test_runtime_records_actual_exports_and_missing_binary(tmp_path):
    import warp.optim.linear as linear

    hashes = warp_runtime_hashes(tmp_path / "absent.so")
    assert complete_warp_fingerprint(hashes)
    assert hashes["spectra_binary"] is None
    assert hashes["warp_cg_implementation"] == source_hash(linear.cg)
    if hasattr(linear, "FSAI"):
        assert hashes["warp_fsai_implementation"] == source_hash(linear.FSAI)
    else:
        assert hashes["warp_fsai_implementation"] is None
