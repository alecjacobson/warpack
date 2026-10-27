"""Build the Markdown API reference from public Python signatures/docstrings.

Run from a source checkout with Warp installed. No CUDA device, cuDSS, SciPy,
CuPy, or benchmark imports are needed. Use --check to reject stale reference pages.
"""

import argparse
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import warpack  # noqa: E402

GROUPS = {
    "symmetric": (
        "Symmetric and generalized eigensolvers",
        [
            "BlockLOBPCG",
            "KrylovSchur",
            "GeneralizedKrylovSchur",
            "SymmetricEigensolver",
            "GeneralizedEigensolver",
            "BucklingEigensolver",
            "CayleyEigensolver",
        ],
    ),
    "complex": (
        "Complex and nonsymmetric eigensolvers",
        [
            "HermitianEigensolver",
            "GeneralEigensolver",
            "ShiftInvertEigensolver",
            "GeneralizedShiftInvertEigensolver",
        ],
    ),
    "svd": ("Singular value decomposition", ["PartialSVD", "NormalSVD"]),
    "operators": (
        "Operators, sparse inverses and preconditioners",
        [
            "SparseOperator",
            "BatchedSparseOperator",
            "HermitianCSR",
            "ComplexCSR",
            "CuDSSInverse",
            "CGInverse",
            "GMRESInverse",
            "BatchedCGPreconditioner",
            "ComplexShiftInverse",
        ],
    ),
    "constraints": (
        "Constraints and subspace projection",
        [
            "ConstrainedKrylovSchur",
            "saddle_point_matrix",
            "independent_constraints",
            "OrthogonalComplementOperator",
            "MassOrthogonalComplementInverse",
        ],
    ),
    "diagnostics": (
        "Residuals, nullspaces and completeness",
        [
            "EigenpairEvaluation",
            "NumericalNullspaceDiagnostics",
            "LowestModeCertificate",
            "InertiaResult",
        ],
    ),
}
METHODS = (
    "solve",
    "capture",
    "initialize",
    "step",
    "finalize",
    "apply",
    "apply_complex",
    "run",
    "refresh",
    "refresh_count",
    "reset_stats",
    "reset_history",
    "for_width",
    "inertia",
    "close",
)
PAGES = {name: slug for slug, (_, names) in GROUPS.items() for name in names}


def signature(obj):
    sig = inspect.signature(obj)
    sig = sig.replace(parameters=[p for p in sig.parameters.values() if p.name != "self"])
    return re.sub(r"<class 'warp\.(?:_src\.)?types\.(\w+)'>", r"wp.\1", str(sig)).replace("'", '"')


def signature_block(name, obj):
    sig = signature(obj)
    if len(name + sig) <= 96:
        return f"```python\n{name}{sig}\n```\n"
    # Parameter annotations/defaults here contain no nested commas except types,
    # so use the Signature parameters rather than splitting the rendered string.
    original = inspect.signature(obj)
    rows = []
    keyword_marker = False
    for param in original.parameters.values():
        if param.name == "self":
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_marker:
            rows.append("    *,")
            keyword_marker = True
        rendered = re.sub(r"<class 'warp\.(?:_src\.)?types\.(\w+)'>", r"wp.\1", str(param))
        rendered = rendered.replace("'", '"')
        rows.append(f"    {rendered},")
    return f"```python\n{name}(\n" + "\n".join(rows) + "\n)\n```\n"


def render_doc(obj):
    doc = inspect.getdoc(obj) or ""
    # Warp-style Args/Returns/Attributes in source, readable Markdown on GitHub.
    section = None
    output = []
    for line in doc.splitlines():
        if line in ("Args:", "Returns:", "Attributes:"):
            section = line[:-1]
            output.extend(["", f"**{'Parameters' if section == 'Args' else section}**", ""])
        elif section in ("Args", "Attributes") and re.match(r"    \w+: ", line):
            name, description = line.strip().split(": ", 1)
            output.append(f"- `{name}`: {description}")
        elif section in ("Args", "Attributes") and line.startswith("        "):
            output.append("  " + line.strip())
        elif section == "Returns" and line.startswith("    "):
            output.append(line[4:])
        else:
            output.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(output).replace("``", "`").strip()) + "\n"


def validate_docs(obj):
    doc = inspect.getdoc(obj) or ""
    if not doc:
        raise ValueError(f"Missing documentation: {obj}")
    parameters = [
        p for p in inspect.signature(obj).parameters if p != "self" and not p.startswith("_")
    ]
    args_section = re.search(r"(?:^|\n)Args:\n(.*?)(?=\n[^ \n]|\Z)", doc, re.DOTALL)
    for name in parameters:
        if args_section is None or not re.search(
            rf"^    {re.escape(name)}: ", args_section[1], re.MULTILINE
        ):
            raise ValueError(f"Missing parameter documentation: {obj.__qualname__}.{name}")
    if not inspect.isclass(obj) and "Returns:" not in doc:
        raise ValueError(f"Missing return documentation: {obj.__qualname__}")


def build():
    if set(PAGES) != set(warpack.__all__):
        raise ValueError("Reference inventory must match warpack.__all__ exactly")
    banner = "<!-- Generated by scripts/generate_api_docs.py; edit source docstrings. -->\n\n"
    index = [
        banner + "# API reference\n",
        "[Documentation](../index.md) · [Usage guide](../guide.md) · [Execution contracts](../library_contracts.md)\n",
        "This reference describes the current source tree. Import public entry points from `warpack`.\n"
        "It follows Warp's `Args`/`Returns` docstring convention; signatures, parameter descriptions\n"
        "and output attributes below are generated from the library source. The released v0.2.0\n"
        "wheel predates some current-main options; use documentation from your checkout/tag.\n",
        "Constructors allocate persistent workspace; they do not return an eigensystem.\n"
        "`solve()` returns the solver instance and writes device outputs. `capture()` returns a\n"
        "CUDA graph which must be launched. Eigenvectors and singular vectors are **rows**.\n"
        "See the [result and residual contracts](../guide.md#results-and-acceptance) before\n"
        "interpreting convergence; the Hermitian, buckling and SVD wrappers have different fields.\n",
        "Invalid parameter metadata is reported by Python exceptions during setup or dispatch.\n"
        "Numerical failures may instead set device status/count fields: inspect the output\n"
        "contract after execution. Construction and graph capture may allocate or synchronize;\n"
        "device replay and explicit host result inspection are separate operations.\n",
        "Only the entry points and methods listed here form this documented frontend.\n"
        "Other exposed implementation stages and names beginning with `_` are internal.\n"
        "In particular, `_working_matrix` and `_storage` are internal adapter plumbing,\n"
        "retained in signatures for accuracy; callers should not supply them.\n",
    ]
    files = {}
    for slug, (title, names) in GROUPS.items():
        index.append(f"## [{title}]({slug}.md)\n")
        parts = [
            banner + f"# {title}\n",
            "[API index](README.md) · [Usage guide](../guide.md) · [Execution contracts](../library_contracts.md)\n",
        ]
        for name in names:
            obj = getattr(warpack, name)
            validate_docs(obj)
            index.append(
                f"- [`{name}`]({slug}.md#{name.lower()}): {inspect.getdoc(obj).splitlines()[0].replace(chr(96) * 2, chr(96))}\n"
            )
            parts.append(f"## {name}\n")
            parts.append(signature_block(f"warpack.{name}", obj))
            source = Path(inspect.getfile(obj)).relative_to(ROOT)
            parts.append(f"[Source](../../{source.as_posix()})\n")
            parts.append(render_doc(obj))
            if inspect.isclass(obj):
                constants = [
                    (key, value)
                    for key, value in vars(obj).items()
                    if key.isupper() and isinstance(value, int)
                ]
                if constants:
                    parts.append("**Flag constants**\n")
                    parts.append(
                        "\n".join(f"- `{key} = {value}`" for key, value in constants) + "\n"
                    )
                for method in METHODS:
                    fn = getattr(obj, method, None)
                    if fn is None:
                        continue
                    validate_docs(fn)
                    parts.append(f"### {name}.{method}\n")
                    parts.append(signature_block(f"{name}.{method}", fn))
                    owner = next(base for base in obj.__mro__ if method in base.__dict__)
                    if owner.__name__ != name and owner.__name__ in PAGES:
                        parts.append(
                            f"Inherited contract: [{owner.__name__}.{method}]({PAGES[owner.__name__]}.md#{owner.__name__.lower()}{method}).\n"
                        )
                    else:
                        parts.append(render_doc(fn))
        files[f"{slug}.md"] = "\n".join(parts).rstrip() + "\n"
    files["README.md"] = "\n".join(index).rstrip() + "\n"
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()
    destination = ROOT / "docs/api"
    files = build()
    stale = []
    for name, text in files.items():
        path = destination / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            destination.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if stale:
        raise SystemExit(
            "Stale API reference; run python scripts/generate_api_docs.py:\n" + "\n".join(stale)
        )
    print(
        f"{'Checked' if args.check else 'Generated'} {len(files)} reference pages for {len(PAGES)} public entry points"
    )


if __name__ == "__main__":
    main()
