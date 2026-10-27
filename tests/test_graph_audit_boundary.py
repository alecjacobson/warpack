"""DOT inspection must not load a GPU runtime or competitor library."""

import json
import subprocess
import sys

from benchmarks.graph_audit import graph_summary


def test_dot_parser_is_standalone_and_retains_transfer_types(tmp_path):
    dot = tmp_path / "graph.dot"
    dot.write_text(
        "digraph G {\n"
        'a [label="{ KERNEL | operation }"];\n'
        'b [label="{ MEMCPY | {kind | DtoD }"] ;\n'
        'c [label="{ HOST | callback }"];\n'
        'd [label="{ MEMCPY | {kind | HtoD }"] ;\n'
        "}\n"
    )
    expected = graph_summary(dot)
    assert expected["node_types_including_conditional_body"] == {
        "KERNEL": 1,
        "MEMCPY": 2,
        "HOST": 1,
    }
    assert expected["memcpy_kinds"] == {"DtoD ": 1, "HtoD ": 1}
    code = """
import importlib.abc, json, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'warp', 'warpack', 'cupy', 'cupyx', 'numpy', 'scipy'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Reject())
from benchmarks.graph_audit import graph_summary
print(json.dumps(graph_summary(sys.argv[1])))
"""
    output = subprocess.check_output([sys.executable, "-c", code, str(dot)], text=True)
    assert json.loads(output) == expected


def test_legacy_helper_import_does_not_load_cupy():
    code = """
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'cupyx', 'scipy'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Reject())
from benchmarks.audit_profile import graph_summary
from benchmarks.graph_audit import graph_summary as standalone
assert graph_summary is standalone
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_warp_benchmark_utilities_do_not_require_cupy():
    code = """
import importlib.abc, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'cupyx'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Reject())
from benchmarks.experimental.compare_block import prepare
from benchmarks.upstream.run_benchmarks import StartedGeneral
from benchmarks.anvil import validate
from benchmarks.validate_dragon_meshes import mode_quality
"""
    subprocess.run([sys.executable, "-c", code], check=True)
