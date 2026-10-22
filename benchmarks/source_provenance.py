"""Repository source identities for application benchmark regression gates.

Follow static local imports without importing their modules. Include the full
solver package as a conservative guard for runtime-selected kernels/backends.
Third-party installed code is covered separately by dependency provenance.
"""

import ast
import hashlib
from pathlib import Path


def source_manifest(*entrypoints, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    pending = [Path(p) if Path(p).is_absolute() else root / p for p in entrypoints]
    pending.extend((root / "warpack").rglob("*.py"))
    seen = set()

    def add_module(parts):
        if not parts:
            return
        # Importing a child also executes any parent package initializers.
        for depth in range(1, len(parts) + 1):
            package = root.joinpath(*parts[:depth]) / "__init__.py"
            if package.is_file():
                pending.append(package)
        module = root.joinpath(*parts).with_suffix(".py")
        if module.is_file():
            pending.append(module)

    while pending:
        path = pending.pop().resolve()
        if not path.is_relative_to(root):
            raise ValueError("source must be inside the repository")
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(root)
        package = list(relative.parent.parts)
        add_module(package)
        for node in ast.walk(ast.parse(path.read_text(), filename=str(relative))):
            if isinstance(node, ast.Import):
                for name in node.names:
                    add_module(name.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                base = prefix + (node.module.split(".") if node.module else [])
                add_module(base)
                # A from-import may name a module or an ordinary attribute.
                for name in node.names:
                    if name.name != "*":
                        add_module(base + [name.name])
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(seen)
    }
