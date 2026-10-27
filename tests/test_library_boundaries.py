"""Keep application assembly and comparison dependencies out of the package."""

import ast
import sys
from pathlib import Path

import warpack


def test_core_imports_only_warp_optional_cudss_and_standard_library():
    allowed = set(sys.stdlib_module_names) | {"warpack", "warp", "warp_cudss"}
    root = Path(warpack.__file__).parent
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module.split(".")[0] not in allowed:
                    violations.append(f"{path.name}:{node.lineno}: {module}")
    assert not violations, "Non-core imports:\n" + "\n".join(violations)
